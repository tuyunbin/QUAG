import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
import sys
import kornia
from copy import deepcopy

sys.path.append("./clip4caption")
from modules.tokenization import BertTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CaptionGenerator
from train import collect_hypothesis_and_scores, collate_active_info, beam_decode_step, \
    get_inst_idx_to_tensor_position_map
from modules.beam import Beam

alpha = 0.0003
class VAL(nn.Module):
    def __init__(self, embed_dim):
        super(VAL, self).__init__()
        self.fc1 = nn.Linear(embed_dim * 2, embed_dim)
        self.att_ch_fc = nn.Linear(embed_dim, embed_dim)
        self.att_tp_fc = nn.Linear(1, 1)

        # the number of heads is tunable
        self.mh_att = nn.MultiheadAttention(embed_dim, num_heads=12, bias=False)
        # self.mh_att2 = nn.MultiheadAttention(embed_dim, num_heads=2, bias=False)
        self.dp = nn.Dropout(0.1)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # weight parameter
        # self.a = torch.nn.Parameter(torch.tensor([1.0, 0.0]))
    #
    def forward(self, img_feat, text_feat):
        # img_feat : B x L x D
        # text_feat : B x D
        B, L, D = img_feat.size()
        text_feat = text_feat.unsqueeze(1).repeat(1, L, 1)
        v1_feat = torch.cat([img_feat, text_feat], -1)  # B x L x 2D
        v1_feat = self.fc1(v1_feat)  # B x L x D
        # v1_feat = self.dp(v1_feat)
        ##############################
        gate_sqz = v1_feat.mean(1, keepdim=True)  # B x 1 x D
        att_ch = self.att_ch_fc(gate_sqz)  # B x 1 x D

        gate_sqz = v1_feat.mean(-1, keepdim=True)  # B x L x 1
        att_sp = self.att_tp_fc(gate_sqz)  # B x L x 1

        joint_att = torch.sigmoid(att_ch) * torch.sigmoid(
            att_sp)  # B x L x D

        ##############################
        v1_feat = v1_feat.permute(1, 0, 2)  # L x B x D
        q = v1_feat
        self_att, _ = self.mh_att(v1_feat, v1_feat, v1_feat)  # L x B x D
        self_att = self.norm(q + self.dp(self_att))
        self_att = self.fc2(self_att)

        self_att = self_att.permute(1, 0, 2)  # B x L x D

        ##############################
        # composite_features = self.a[0] * joint_att * img_feat + self.a[1] * self_att
        composite_features = joint_att * img_feat + self_att
        # composite_features = composite_features.mean((2, 3))  # B x D
        return composite_features


class CrossTransformer(nn.Module):
  """
  Cross Transformer layer
  """
  def __init__(self, dropout, d_model = 768, n_head = 12):
    """
    :param dropout: dropout rate
    :param d_model: dimension of hidden state
    :param n_head: number of heads in multi head attention
    """
    super(CrossTransformer, self).__init__()
    self.attention = nn.MultiheadAttention(d_model, n_head, dropout = dropout)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)

    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.dropout3 = nn.Dropout(dropout)

    self.activation = nn.ReLU()

    self.linear1 = nn.Linear(d_model, d_model * 4)
    self.linear2 = nn.Linear(d_model * 4, d_model)

  def forward(self, input1, input2):
    attn_output, attn_weight = self.attention(input1, input2, input2)
    output = input1 + self.dropout1(attn_output)
    output = self.norm1(output)
    ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output))))
    output = output + self.dropout3(ff_output)
    output = self.norm2(output)

    return output

class Align(nn.Module):
  """
  Cross Transformer layer
  """
  def __init__(self, dropout, d_model = 768, n_head = 12):
    """
    :param dropout: dropout rate
    :param d_model: dimension of hidden state
    :param n_head: number of heads in multi head attention
    """
    super(Align, self).__init__()
    self.attention = nn.MultiheadAttention(d_model, n_head, dropout = dropout)

    self.norm1 = nn.LayerNorm(d_model)
    # self.norm2 = nn.LayerNorm(d_model)

    self.dropout1 = nn.Dropout(dropout)
    # self.dropout2 = nn.Dropout(dropout)
    # self.dropout3 = nn.Dropout(dropout)
    #
    # self.activation = nn.ReLU()
    #
    # self.linear1 = nn.Linear(d_model, d_model * 4)
    # self.linear2 = nn.Linear(d_model * 4, d_model)

  def forward(self, input1, input2):
    attn_output, attn_weight = self.attention(input1, input2, input2)
    output = input1 + self.dropout1(attn_output)
    output = self.norm1(output)
    # ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output))))
    # output = output + self.dropout3(ff_output)
    # output = self.norm2(output)

    return output

class MomentModel(nn.Module):

    def __init__(self, n_frames=-1, asr_dim=-1, args=None):
        super(MomentModel, self).__init__()

        self.args = args
        self.n_frames = n_frames

        embed_dim = 768

        self.asr_dim = asr_dim
        self.use_asr = asr_dim > 0
        if self.use_asr:
            self.asr_enc_layer = nn.Sequential(
                nn.LayerNorm(asr_dim),
                nn.Linear(asr_dim, embed_dim)
            )

        # map timestamp to embedding
        # scalar in [0, 1] ->
        self.temporal_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 0: frames outside of moment
        # 1: frames inside of moment
        self.mask_embed = nn.Embedding(2, embed_dim)

        self.boundary_embed = nn.Embedding(2, embed_dim)

        dropout = 0.1
        self.input_dropout = nn.Dropout(dropout)

        # Moment Retrieval

        kernel_size = 5
        padding = kernel_size // 2

        self.moment_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=3,
                padding=1,
            )
        )

        # Moment Segmentation

        embed_dim_2 = 768

        self.start_predictor = nn.Sequential(
            nn.Linear(embed_dim_2, 1),
        )

        self.end_predictor = nn.Sequential(
            # nn.LayerNorm(embed_dim),
            # nn.Linear(embed_dim, embed_dim),
            # nn.GELU(),
            nn.Linear(embed_dim_2, 1),
        )

        self.segment_predictor = nn.Sequential(
            # nn.LayerNorm(embed_dim),
            # nn.Linear(embed_dim, embed_dim),
            # nn.GELU(),
            nn.Linear(embed_dim_2, 1),
        )


        self.tokenizer = BertTokenizer.from_pretrained("/data1/yunbin_tu/HiREST/pretrained_weights/bert-base-uncased", do_lower_case=True)
        model_state_dict = torch.load("/data1/yunbin_tu/HiREST/pretrained_weights/clip4caption_vit-b-32_model.bin", map_location='cpu')
        args.d_model = embed_dim
        args.video_dim = embed_dim
        args.max_frames = args.max_frames_step_captioning

        cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')

        self.clip4cap_model = CaptionGenerator.from_pretrained("/data1/yunbin_tu/HiREST/pretrained_weights/bert-base-uncased", "visual-base", "decoder-base",
                                                               cache_dir=cache_dir, state_dict=model_state_dict,
                                                               task_config=args, max_position_embeddings_override=2048)

        self.clip_g_map = nn.Linear(1024, embed_dim)
        self.clip_g_map_text = nn.Linear(1024, embed_dim)

        self.align = nn.ModuleList([Align(dropout, embed_dim, 12) for i in range(1)])

        self.transformer = nn.ModuleList([CrossTransformer(dropout, embed_dim, 12) for i in range(2)])

        self.val = VAL(embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * 0.07)

        # self.mlp = nn.Sequential(
        #        nn.Linear(embed_dim, embed_dim),
        #        # nn.Dropout(0.1),
        #        # nn.ReLU(),
        #        # nn.Linear(embed_dim, embed_dim)
        # )

        self.fc = nn.Sequential(
            # nn.LayerNorm(embed_dim),
            # nn.Linear(embed_dim, embed_dim),
            # nn.GELU(),
            nn.Linear(embed_dim*2, embed_dim),
            # nn.ReLU(),
            # nn.Dropout(0.1)
        )

        sys.path.append("./EVA_clip")
        from eva_clip import build_eva_model_and_transforms
        self.clip_model, self.clip_preprocess = build_eva_model_and_transforms("EVA_CLIP_g_14",
                                                                               pretrained="./pretrained_weights/eva_clip_psz14.pt")
        print("Loaded EVA CLIP G")

        self.clip_model = self.clip_model.float()
        self.clip_model.eval()

        self.freeze_clip()

    def freeze_clip(self):
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

    def train_step(self, batch):
        task = batch['tasks'][0]

        if task == 'moment_retrieval':
            return self.train_moment_retrieval(batch)
        elif task == 'moment_segmentation':
            return self.train_moment_segmentation(batch)
        elif task == 'step_captioning':
            return self.train_step_captioning(batch)
        else:
            raise NotImplementedError

    def test_step(self, batch, **kwargs):
        task = batch['tasks'][0]

        if task == 'moment_retrieval':
            return self.test_moment_retrieval(batch, **kwargs)
        elif task == 'moment_segmentation':
            return self.test_moment_segmentation(batch, **kwargs)
        elif task == 'step_captioning':
            return self.test_step_captioning(batch, **kwargs)
        else:
            raise NotImplementedError

    def foward_moment_shared(self, video_feats, text_feat, video_mask=None, moment_mask=None, asr_feats=None,
                             boundary_mask=None, text_mask=None):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        B, max_n_frames, embed_dim = video_feats.size()
        # text_feat = text_feats[:, 0, :]

        video_feats = self.clip_g_map(video_feats)
        text_feat = self.clip_g_map_text(text_feat)
        video_feats = self.clip4cap_model.normalize_video(video_feats)

        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        if self.use_asr:
            asr_feats = self.asr_enc_layer(asr_feats)

        mask_sum = video_mask.sum(-1)
        video_global = torch.sum(video_feats, dim=1) / mask_sum.unsqueeze(1)
        asr_global = torch.sum(asr_feats, dim=1) / mask_sum.unsqueeze(1)

        # video_global = self.mlp(video_global)
        # asr_global = self.mlp(asr_global)

        video_feats_proj = F.normalize(video_global, dim=-1)
        asr_feats_proj = F.normalize(asr_global, dim=-1)

        # video_feats_proj = self.mlp(video_global)
        # asr_feats_proj = self.mlp(asr_global)
        # video_feats_proj = F.normalize(video_feats_proj, dim=-1)
        # asr_feats_proj = F.normalize(asr_feats_proj, dim=-1)
        # video_feats_proj = video_feats_proj.view(B, max_n_frames, 12, 64)
        # asr_feats_proj = asr_feats_proj.view(B, max_n_frames, 12, 64)
        # video_feats_proj = video_feats_proj.permute(0, 2, 1, 3)
        # asr_feats_proj = asr_feats_proj.permute(0, 2, 1, 3)
        #
        # retrieve_logits = torch.einsum("ahld,bhmd->ablm", video_feats_proj, asr_feats_proj)  # (B,B,Nq,Nk)
        # v2a_logits, max_idx1 = retrieve_logits.max(dim=-1)  # B,B,Nq,Nk -> B,B,Nq
        # a2v_logits, max_idx2 = retrieve_logits.max(dim=-2)  # B,B,Nq,Nk -> B,B,Nk
        # v2a_logits = torch.sum(v2a_logits, dim=2) / (mask_sum.unsqueeze(1))
        # a2v_logits = torch.sum(a2v_logits, dim=2) / (mask_sum.unsqueeze(0))
        # retrieve_logits = (v2a_logits + a2v_logits) / 2.0
        # retrieve_logits = retrieve_logits / self.temp

        v2a_logits = video_feats_proj @ asr_feats_proj.t()
        a2v_logits = asr_feats_proj @ video_feats_proj.t()
        v2a_logits = v2a_logits / self.temp
        a2v_logits = a2v_logits / self.temp

        sim_targets = torch.zeros_like(v2a_logits)
        sim_targets[:, :] = torch.eye(B)

        loss_b2a = -torch.sum(F.log_softmax(v2a_logits, dim=1) * sim_targets, dim=1).mean()
        loss_a2b = -torch.sum(F.log_softmax(a2v_logits, dim=1) * sim_targets, dim=1).mean()
        loss_con = (loss_b2a + loss_a2b) / 2

        video_feats = video_feats.transpose(0, 1)
        asr_feats = asr_feats.transpose(0, 1)
        for l in self.align:
            video_feats, asr_feats = l(video_feats, asr_feats), l(asr_feats, video_feats)

        video_feats = video_feats.transpose(0, 1)
        asr_feats = asr_feats.transpose(0, 1)

        feats = torch.cat([video_feats, asr_feats], -1)
        feats = self.fc(feats)

        feats = self.val(feats, text_feat)

        if boundary_mask is not None:
            boundary_emb = self.boundary_embed(boundary_mask)
            feats += boundary_emb

        # [batch_size]
        if video_mask is None:
            video_mask = torch.ones((B, max_n_frames), device=video_feats.device, dtype=torch.long)
        n_frames_batch = video_mask.sum(dim=-1).long()

        # time representation normlized in [-1, 1]
        normalized_times = []
        max_n_frames = max(n_frames_batch)
        for n_frames in n_frames_batch:
            # [0, 1] -> [-0.5, 0.5] ->  [-1, 1]
            normalized_time = (torch.linspace(0, 1, n_frames) - 0.5) * 2

            n_pad = max_n_frames - n_frames
            padding = torch.zeros(n_pad)
            normzlied_time = torch.cat([normalized_time, padding]).view(1, max_n_frames, 1)
            normalized_times.append(normzlied_time)

        normalized_times = torch.cat(normalized_times, dim=0).to(video_feats.device)

        temporal_embed = self.temporal_embed(normalized_times)
        feats += temporal_embed

        mask_embed = self.mask_embed(moment_mask)
        feats += mask_embed

        assert video_mask.dim() == 2, video_mask.shape
        extended_attention_mask = video_mask[:, None, None, :]

        dtype = feats.dtype
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min

        # feats = self.clip4cap_model.get_visual_output(feats, torch.zeros((B, max_n_frames)).long().to(feats.device),
        #                                               shaped=True)
        feats = feats.transpose(0, 1)
        for l in self.transformer:
            feats = l(feats, feats)

        feats = feats.transpose(0, 1)

        return feats, loss_con

    def forward_moment_retrieval(self, video_feats, text_feat, video_mask=None, moment_mask=None, asr_feats=None,
                                 text_mask=None):

        B, max_n_frames, embed_dim = video_feats.size()

        feats, red_loss = self.foward_moment_shared(video_feats, text_feat, video_mask, moment_mask=moment_mask,
                                                    asr_feats=asr_feats, text_mask=text_mask)

        start_logits = self.start_predictor(feats).squeeze(2)
        end_logits = self.end_predictor(feats).squeeze(2)

        return {
            'start_logits': start_logits,
            'end_logits': end_logits,
            'red_loss': red_loss
        }

    def train_moment_retrieval(self, batch):
        device = next(self.parameters()).device
        video_feats = batch['vis_feats'].to(device)

        video_mask = batch['vis_mask'].to(device)
        moment_mask = batch['moment_mask'].to(device)

        start_target = batch['moment_retrieval_start_target'].to(device)
        end_target = batch['moment_retrieval_end_target'].to(device)

        asr_feats = None
        if self.use_asr:
            asr_feats = batch['asr_feats'].to(device)

        with torch.no_grad():
            clip_text_ids = batch['clip_text_ids'].to(device)
            clip_text_masks = batch['clip_text_masks'].to(device)
            text_feat = self.clip_model.encode_text(clip_text_ids).float()

        out = self.forward_moment_retrieval(
            video_feats, text_feat, video_mask=video_mask, moment_mask=moment_mask, asr_feats=asr_feats,
            text_mask=clip_text_masks)
        start_logits = out['start_logits']
        end_logits = out['end_logits']
        red_loss = out['red_loss']

        _start_target = torch.zeros(start_logits.size(), device=start_logits.device)
        _end_target = torch.zeros(end_logits.size(), device=end_logits.device)

        _start_target.scatter_(1, start_target.unsqueeze(1), 1)
        _end_target.scatter_(1, end_target.unsqueeze(1), 1)

        start_loss = F.binary_cross_entropy_with_logits(start_logits, _start_target, reduction='none')
        end_loss = F.binary_cross_entropy_with_logits(end_logits, _end_target, reduction='none')

        start_loss = start_loss * moment_mask
        end_loss = end_loss * moment_mask

        start_loss = start_loss.sum() / moment_mask.sum().clamp(min=1)
        end_loss = end_loss.sum() / moment_mask.sum().clamp(min=1)

        loss = (start_loss + end_loss) / 2
        loss = loss + alpha * red_loss

        result = {
            'loss': loss,
        }

        return result

    @torch.no_grad()
    def test_moment_retrieval(self, batch, **kwargs):
        device = next(self.parameters()).device
        video_feats = batch['vis_feats'].to(device)

        video_mask = batch['vis_mask'].to(device)
        moment_mask = batch['moment_mask'].to(device)

        if self.use_asr:
            asr_feats = batch['asr_feats'].to(device)

        # text_feat = batch['text_feat'].to(device)
        with torch.no_grad():
            clip_text_ids = batch['clip_text_ids'].to(device)
            clip_text_masks = batch['clip_text_masks'].to(device)
            text_feat = self.clip_model.encode_text(clip_text_ids).float()

        out = self.forward_moment_retrieval(
            video_feats, text_feat, video_mask=video_mask, moment_mask=moment_mask, asr_feats=asr_feats,
            text_mask=clip_text_masks)

        start_logits = out['start_logits']
        end_logits = out['end_logits']

        start_logits[video_mask == 0] = -1e10
        end_logits[video_mask == 0] = -1e10

        start = start_logits.argmax(dim=1)
        end = end_logits.argmax(dim=1)

        pred_boundaries = torch.stack([start, end], dim=-1)

        start_target = batch['moment_retrieval_start_target']
        end_target = batch['moment_retrieval_end_target']

        result = {
            'prediction': pred_boundaries.detach().tolist(),
        }

        return result

    def forward_moment_segmentation(self, video_feats, text_feat, video_mask, moment_mask, asr_feats=None,
                                    boundary_mask=None, text_mask=None):

        B, max_n_frames, embed_dim = video_feats.size()

        feats, red_loss = self.foward_moment_shared(
            video_feats, text_feat, video_mask, moment_mask=moment_mask, asr_feats=asr_feats,
            boundary_mask=boundary_mask, text_mask=text_mask)

        out = self.segment_predictor(feats).squeeze(2)

        return out, red_loss

    def train_moment_segmentation(self, batch):
        device = next(self.parameters()).device
        video_feats = batch['vis_feats'].to(device)

        video_mask = batch['vis_mask'].to(device)
        moment_mask = batch['moment_mask'].to(device)

        if self.use_asr:
            asr_feats = batch['asr_feats'].to(device)

        with torch.no_grad():
            clip_text_ids = batch['clip_text_ids'].to(device)
            clip_text_masks = batch['clip_text_masks'].to(device)
            text_feat = self.clip_model.encode_text(clip_text_ids).float()

        prev_boundary_mask = batch['prev_boundary_mask'].to(device)

        moment_segmentation_target = batch['moment_segmentation_target'].to(device)

        moment_segmentation_logits, red_loss = self.forward_moment_segmentation(
            video_feats, text_feat, video_mask, moment_mask, asr_feats=asr_feats, boundary_mask=prev_boundary_mask,
            text_mask=clip_text_masks)

        moment_segmentation_logits[moment_mask == 0] = -torch.finfo(moment_segmentation_logits.dtype).max
        moment_segmentation_loss = F.cross_entropy(moment_segmentation_logits, moment_segmentation_target)
        moment_segmentation_loss = moment_segmentation_loss + alpha * red_loss
        result = {
            'loss': moment_segmentation_loss,
        }

        return result

    @torch.no_grad()
    def test_moment_segmentation(self, batch, threshold=0.15, **kwargs):
        device = next(self.parameters()).device
        video_feats = batch['vis_feats'].to(device)
        video_mask = batch['vis_mask'].to(device)

        if self.use_asr:
            asr_feats = batch['asr_feats'].to(device)

        with torch.no_grad():
            clip_text_ids = batch['clip_text_ids'].to(device)
            clip_text_masks = batch['clip_text_masks'].to(device)
            text_feat = self.clip_model.encode_text(clip_text_ids).float()

        B = video_feats.shape[0]

        step_predictions = []

        for i in range(B):
            step_predictions.append(list())

        moment_start_boundaries = batch['moment_bound_frames'][:, 0].tolist()
        moment_last_boundaries = batch['moment_bound_frames'][:, 1].tolist()

        moment_mask = torch.zeros(B, video_feats.shape[1], device=video_feats.device, dtype=torch.long)
        for b in range(B):
            moment_mask[b, moment_start_boundaries[b]:moment_last_boundaries[b] + 1] = 1

        prev_boundary_mask = torch.zeros(B, video_feats.shape[1], device=video_feats.device, dtype=torch.long)
        for b in range(B):
            prev_boundary_mask[b, moment_start_boundaries[b]] = 1

        softmax = nn.Softmax(dim=1)

        for b in range(B):
            step_predictions[b].append([moment_start_boundaries[b], moment_start_boundaries[b]])

        PERCENT_THRESHOLD = self.args.moment_segmentation_difference_threshold
        n_max_iteration = self.args.moment_segmentation_max_iterations

        for i in range(n_max_iteration):
            moment_segmentation_logits, _ = self.forward_moment_segmentation(video_feats, text_feat, video_mask,
                                                                             moment_mask, asr_feats=asr_feats,
                                                                             boundary_mask=prev_boundary_mask,
                                                                             text_mask=clip_text_masks)
            moment_segmentation_logits[moment_mask == 0] = -torch.finfo(moment_segmentation_logits.dtype).max
            moment_segmentation_logits = softmax(moment_segmentation_logits)

            max_frame_idxs = moment_segmentation_logits.argmax(dim=1)

            for b in range(B):
                scores = moment_segmentation_logits[b].cpu().tolist()

                max_idx = max_frame_idxs[b].item()
                max_score = scores[max_idx]

                if max_score < 0.00001:
                    continue

                left_bound = max_idx
                right_bound = max_idx

                while (scores[left_bound] / max_score) > PERCENT_THRESHOLD:
                    if left_bound == 0:
                        break

                    left_bound -= 1

                while (scores[right_bound] / max_score) > PERCENT_THRESHOLD:
                    if right_bound == (len(scores) - 1):
                        break

                    right_bound += 1

                current_step_prediction = [left_bound, right_bound]

                if left_bound == 0 or right_bound == 0:
                    continue

                moment_mask[b, current_step_prediction[0]:current_step_prediction[1] + 1] = 0

                prev_boundary_mask[b, current_step_prediction[0]] = 1
                prev_boundary_mask[b, current_step_prediction[1]] = 1

                step_predictions[b].append(current_step_prediction)

        for b in range(B):
            step_predictions[b].append([moment_last_boundaries[b], moment_last_boundaries[b]])

            step_predictions[b].sort(key=lambda x: x[0], reverse=False)

            temp = []
            for x in step_predictions[b]:
                temp.extend(x)
            step_predictions[b] = temp

            while step_predictions[b][-1] > moment_last_boundaries[b]:
                step_predictions[b].pop(-1)

            step_predictions[b] = list(set(step_predictions[b]))
            step_predictions[b].sort()

            temp = deepcopy(step_predictions[b])
            step_predictions[b] = []

            current_bound = temp[0]
            step_predictions[b].append(current_bound)

            for i in range(1, len(temp) - 1):
                next_bound = temp[i]

                if next_bound - current_bound >= 5:
                    step_predictions[b].append(next_bound)
                    current_bound = next_bound

        step_predictions = np.array(step_predictions, dtype=object).tolist()
        raw_predictions = deepcopy(step_predictions)

        result = {
            'raw_predictions': raw_predictions,
            'prediction': step_predictions,
        }

        return result

    def train_step_captioning(self, batch):
        device = next(self.parameters()).device
        video_feats = batch['vis_feats'].to(device)
        video_mask = batch['vis_mask'].to(device)

        moment_mask = batch['moment_mask'].to(device)

        if self.use_asr:
            asr_feats = batch['asr_feats'].to(device)
        else:
            asr_feats = None

        with torch.no_grad():
            clip_text_ids = batch['clip_text_ids'].to(device)
            clip_text_masks = batch['clip_text_masks'].to(device)
            text_feat = self.clip_model.encode_text(clip_text_ids).float()

        target_text = batch['target_text']

        B = len(target_text)
        pairs_input_caption_ids = torch.zeros((B, self.args.max_words))
        pairs_decoder_mask = torch.zeros((B, self.args.max_words))
        pairs_output_caption_ids = torch.zeros((B, self.args.max_words))

        for i in range(B):
            (_, _, _, _, _, a, b, c, _) = target_text[i]
            pairs_input_caption_ids[i] = torch.tensor(a)
            pairs_decoder_mask[i] = torch.tensor(b)
            pairs_output_caption_ids[i] = torch.tensor(c)

        v_mask = torch.zeros((B, self.args.max_frames)).long().to(video_feats.device)

        input_caption_ids = pairs_input_caption_ids.view(-1, pairs_input_caption_ids.shape[-1])
        decoder_mask = pairs_decoder_mask.view(-1, pairs_decoder_mask.shape[-1])

        video_feats = self.trim_feats(video_feats, moment_mask, B, video_feats.device)
        asr_feats = self.trim_feats(asr_feats, moment_mask, B, video_feats.device)

        visual_output, red_loss = self.foward_moment_shared(
            video_feats, text_feat, video_mask=torch.ones((B, self.args.max_frames)).long().to(video_feats.device),
            moment_mask=torch.ones((B, self.args.max_frames)).long().to(video_feats.device), asr_feats=asr_feats,
            text_mask=clip_text_masks)

        decoder_scores, res_tuples = self.clip4cap_model._get_decoder_score(visual_output.to(video_feats.device),
                                                                            v_mask,
                                                                            input_caption_ids.long().to(
                                                                                video_feats.device),
                                                                            decoder_mask.long().to(video_feats.device),
                                                                            shaped=True)

        pairs_output_caption_ids = pairs_output_caption_ids.view(-1, pairs_output_caption_ids.shape[-1]).long().to(
            video_feats.device)

        loss = self.clip4cap_model.decoder_loss_fct(
            decoder_scores.view(-1, self.clip4cap_model.bert_config.vocab_size).to(video_feats.device),
            pairs_output_caption_ids.view(-1).long().to(video_feats.device))
        loss = loss + alpha * red_loss
        result = {
            'loss': loss,
        }

        return result

    def trim_feats(self, visual_output, moment_mask, B, device):
        f = []
        for i in range(B):
            z = visual_output[i][moment_mask[i] == 1]

            if self.args.max_frames < z.shape[0]:
                z = z[:self.args.max_frames]
            else:
                x = torch.zeros((self.args.max_frames, z.shape[1]))
                count_embeds = [0] * self.args.max_frames
                N: int = z.shape[0]

                count_embeds = [count_embeds[(j * self.args.max_frames) // N: ((j + 1) * self.args.max_frames) // N] for
                                j in range(N)]

                j = 0
                for k in range(len(count_embeds)):
                    for _ in count_embeds[k]:
                        x[j] = z[k]
                        j += 1

                z = x.clone()
            f.append(z.to(device))

        visual_output = torch.stack(f).to(device)

        return visual_output

    @torch.no_grad()
    def test_step_captioning(self, batch, **kwargs):
        device = next(self.parameters()).device
        video_feats = batch['vis_feats'].to(device)
        video_mask = batch['vis_mask'].to(device)
        moment_mask = batch['moment_mask'].to(device)

        if self.use_asr:
            asr_feats = batch['asr_feats'].to(device)

        with torch.no_grad():
            clip_text_ids = batch['clip_text_ids'].to(device)
            clip_text_masks = batch['clip_text_masks'].to(device)
            text_feat = self.clip_model.encode_text(clip_text_ids).float()

        B, max_n_frames, embed_dim = video_feats.size()

        video_feats = self.trim_feats(video_feats, moment_mask, B, video_feats.device)
        asr_feats = self.trim_feats(asr_feats, moment_mask, B, video_feats.device)

        feats, _ = self.foward_moment_shared(
            video_feats, text_feat, video_mask=torch.ones((B, self.args.max_frames)).long().to(video_feats.device),
            moment_mask=torch.ones((B, self.args.max_frames)).long().to(video_feats.device), asr_feats=asr_feats,
            text_mask=clip_text_masks)

        beam_size = 5
        if 'num_beams' in kwargs:
            beam_size = kwargs['num_beams']

        generated_text = []

        visual_output = feats

        n_inst, len_v, v_h = visual_output.size()

        decoder = self.clip4cap_model.decoder_caption

        video_mask = torch.zeros((B, self.args.max_frames)).long().to(feats.device)

        visual_output_rpt = visual_output.repeat(1, beam_size, 1).view(n_inst * beam_size, len_v, v_h)
        video_mask_rpt = video_mask.repeat(1, beam_size).view(n_inst * beam_size, len_v)

        inst_dec_beams = [Beam(beam_size, device=device, tokenizer=self.tokenizer) for _ in range(n_inst)]

        active_inst_idx_list = list(range(n_inst))
        inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        for len_dec_seq in range(1, self.args.max_words + 1):
            active_inst_idx_list = beam_decode_step(decoder, inst_dec_beams,
                                                    len_dec_seq, inst_idx_to_position_map, beam_size, device,
                                                    (visual_output_rpt, video_mask_rpt))

            if not active_inst_idx_list:
                break

            (visual_output_rpt, video_mask_rpt), \
            inst_idx_to_position_map = collate_active_info((visual_output_rpt, video_mask_rpt),
                                                           inst_idx_to_position_map, active_inst_idx_list, beam_size,
                                                           device)
        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)
        result_list = [batch_hyp[i][0] for i in range(n_inst)]

        for re_idx, re_list in enumerate(result_list):
            decode_text_list = self.tokenizer.convert_ids_to_tokens(re_list)
            if "[SEP]" in decode_text_list:
                SEP_index = decode_text_list.index("[SEP]")
                decode_text_list = decode_text_list[:SEP_index]
            if "[PAD]" in decode_text_list:
                PAD_index = decode_text_list.index("[PAD]")
                decode_text_list = decode_text_list[:PAD_index]
            decode_text = ' '.join(decode_text_list)
            decode_text = decode_text.replace(" ##", "").strip("##").strip()

            generated_text.append(str(decode_text))

        result = {
            'prediction': generated_text,
        }

        return result