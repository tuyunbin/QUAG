# Query-centric Audio-Visual Cognition Network for  Moment Retrieval, Segmentation and Step-Captioning
This package contains the accompanying code for the following paper:

Tu, Yunbin, et al. ["Query-centric Audio-Visual Cognition Network for  Moment Retrieval, Segmentation and Step-Captioning"](https://arxiv.org/pdf/2412.13543), which has appeared as a long paper in AAAI 2025. 

## We illustrate the training and testing details as follows:

## Installation


```bash
# Requires torch<1.13.0
# You need this only for step captioning evaluation (evaluate.py)
pip install allennlp_models

pip install -r requirements.txt
python -c "import language_evaluation; language_evaluation.download('coco')"
```

## Prepare Features

You can 1) download the pre-extracted visual (EVA-CLIP), and ASR (Whisper) and ASR embedding (MiniLM) from HiREST, or 2) extract features by yourself.


### 1) Download pre-extracted features
Download feature files extract them into the `./data/` directory.

For video retrieval task
```bash
# Visual features (EVA-CLIP) - 32 frames per video, including features from negative distractors
wget https://huggingface.co/j-min/HiREST-baseline/resolve/main/eva_clip_features_32_frame.zip
unzip -q eva_clip_features_32_frame.zip
mv eva_clip_features_32_frame/ data/
```

For moment retrieval / moment segmentation / step captioning tasks
```bash
# Speech transcripts (ASR with Whisper)
wget https://huggingface.co/j-min/HiREST-baseline/resolve/main/ASR.zip
# Speech embedding (MiniLM)
wget https://huggingface.co/j-min/HiREST-baseline/resolve/main/ASR_feats_all-MiniLM-L6-v2.zip
# Visual features (EVA-CLIP) - 1 frame per second
wget https://huggingface.co/j-min/HiREST-baseline/resolve/main/eva_clip_features.zip

unzip -q ASR.zip
unzip -q ASR_feats_all-MiniLM-L6-v2.zip
unzip -q eva_clip_features.zip

mv ASR/ data/
mv ASR_feats_all-MiniLM-L6-v2/ data/
mv eva_clip_features/ data/
```

Afterwards the `./data/` directory should look like: 
```bash
data/
    ASR/
    ASR_feats_all-MiniLM-L6-v2/
    eva_clip_features/
    eva_clip_features_32_frame/
    evaluation/
    splits/
```

### 2) Feature extraction (Optional)
Check out [extraction/README.md](extraction/README.md) for details.

<hr>

## Training on Single GPU

Before training, you need to download the weights of CLIP4Caption (to initialize multimodal encoder/decoder parameters) and EVA-CLIP-G (visual encoder and text query encoder).
As the [CLIP4Caption](https://github.com/liupeng0606/clip4caption) and [EVA CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP) repositories have been updated since our model development and also could be updated later, we uploaded the versions we used at (`clip4caption/` and `EVA_clip/`).


### Download Clip4Caption weights
Please download the clip4caption pretrained weights [here](https://drive.google.com/file/d/17p476sL5_KZoQ2h4e1TU7-qWH4VqzReT/view?usp=sharing) and put them in `./pretrained_weights/`.

```bash
wget https://huggingface.co/j-min/HiREST-baseline/resolve/main/clip4caption_vit-b-32_model.bin
mv clip4caption_vit-b-32_model.bin ./pretrained_weights/clip4caption_vit-b-32_model.bin
```

### Download EVA-CLIP weights
Please download the EVA-CLIP pretrained weights [here](https://huggingface.co/BAAI/EVA/blob/main/eva_clip_psz14.pt) and put them in `./pretrained_weights/`.

```bash
wget https://huggingface.co/BAAI/EVA/resolve/main/eva_clip_psz14.pt
mv eva_clip_psz14.pt ./pretrained_weights/eva_clip_psz14.pt
```


### Run Training
Change `output` variable at the top of `scripts/run.sh` to change model checkpoint path.
We used an RTX 3090 GPU (24GB memory) with batch size 5.

To run training (and automatically run inference) run:
```bash
bash scripts/run_quag.sh --train
```



## Inference & Evaluation

### Moment Retrieval / Moment Segmentation / Step Captioning

```bash
# Inference
bash scripts/run_val_quag.sh

# Evaluation
bash scripts/score_val_quag.sh
```


# Acknowledgements

Our project borrows code from the following repository:
- https://github.com/j-min/HiREST

We thank for the authors of the repositories for their public release of dataset and code.

# Citation
If you find this helps your research, please consider citing:
```
@inproceedings{tu2025query,
  title={Query-centric Audio-Visual Cognition Network for Moment Retrieval, Segmentation and Step-Captioning},
  author={Tu, Yunbin and Li, Liang and Su, Li and Huang, Qingming},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={7},
  pages={7464--7472},
  year={2025}
}
```

## Contact
My email is tuyunbin1995@foxmail.com

Any discussions and suggestions are welcome!
