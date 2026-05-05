# Replicating Mellow Under Real World Constraints

**Advanced Topics in Audio Processing using Deep Learning — Final Project**
Tel Aviv University, School of Computer Science

---

## Overview

We reimplement [Mellow](https://arxiv.org/abs/2501.xxxxx), a 167M-parameter Small Audio-Language Model (SALM) designed for audio reasoning. Mellow takes up to two audio recordings and a text prompt as input and generates a natural-language response.

Since the full ReasonAQA training set requires AudioCaps (which was not accessible to us), we train exclusively on the **Clotho portion of ReasonAQA** (292,768 examples across 6 task subtypes). We document all six distributed training and gradient stability failures encountered, and evaluate the trained model on the Clotho validation set using standard captioning metrics and multiple-choice accuracy. We also run a Self-Consistency decoding analysis on both our model and the original Mellow weights.

**Key results:**
- Stable training for 10 epochs on 4× NVIDIA Titan GPUs (~17 hours)
- CIDEr = 0.027, BLEU-1 = 0.052, ROUGE-L = 0.086 on the mixed validation set
- Self-Consistency decoding yields +19.7% absolute accuracy on closed-ended tasks for our model

The full paper is in [`paper/main.tex`](paper/main.tex).

---

## Repository Structure

```
mellow_apdl_project/
├── config/              # Training and evaluation YAML configs
├── data/                # Dataset modules
│   ├── ClothoV21/       # Clotho V2.1 audio files (see Dataset Setup)
│   ├── ClothoAQA/       # ClothoAQA audio files
│   └── AudioCapsLarger/ # (not used in our experiments)
├── datafiles/           # JSON metadata files for train/val splits
├── distributed/         # DDP / NCCL utilities
├── htsat/               # HTSAT audio encoder
├── metrics/             # COCO captioning metrics (BLEU, ROUGE-L, CIDEr)
├── models/              # Mellow, HTSAT encoder, decoder
├── paper/               # LaTeX source + figures
├── scripts/             # Evaluation scripts and SLURM configs
│   ├── eval_mc_accuracy.py       # Multiple-choice accuracy evaluation
│   ├── eval_debiased.py          # Debiased inference evaluation
│   └── extract_htsat_embeddings.py  # t-SNE / UMAP embedding extraction
├── training/            # Trainer and logging
├── utils/               # Launch utilities, helpers
├── train.py             # Main entry point (train + evaluate)
└── requirements.txt     # Dependencies
```

---

## Installation

> Requires **Python 3.10** and a CUDA-capable GPU.

```bash
pip install -r requirements.txt
```

The `requirements.txt` lists all dependencies with pinned versions (conda-style). For a clean pip install, the core packages are:

```
torch==1.12.1
torchaudio==0.12.1
transformers==4.47.1
librosa==0.11.0
pyyaml
tqdm
scikit-learn
matplotlib
```

---

## Dataset Setup

### Clotho V2.1 Audio Files

Download Clotho V2.1 audio files from the [official Clotho page](https://zenodo.org/record/4743815) and place them under:

```
data/ClothoV21/
    development/   # training audio (.wav)
    validation/    # validation audio (.wav)
    evaluation/    # evaluation audio (.wav)
```

### ClothoAQA Audio Files

Download ClothoAQA audio files and place them under `data/ClothoAQA/`.

### Dataset JSON Metadata

The JSON datafiles (train/val splits with question-answer pairs) are too large for GitHub.
Download them from **Zenodo**: https://zenodo.org/records/15036628

Place them in the `datafiles/` directory:

```
datafiles/
    train_clotho.json
    val_clotho.json
    val_captioning.json
    val_captioning_500.json
    val_comparison.json
    val_mc.json
    test_clotho.json
```

### HTSAT Pretrained Checkpoint

Download `HTSAT_AudioSet_Saved_1.ckpt` from the [HTSAT Google Drive](https://drive.google.com/drive/folders/1cZhMO7qLXTeifXVPP7PdM1NRYCG5cx28) and note its path — you will need to set it in the config file.

---

## Audio Samples

Sample audio files from the Clotho V2.1 dataset are included under:

```
audio_samples/
    train/     # 5 sample clips from the development (training) split
    val/       # 5 sample clips from the validation split
```

These are representative environmental sound recordings used as input to the model. Sample names include: hill creek, vinyl noise, harvest festival, lake beach, trolley bus (train); tornado siren, footsteps on gravel, industrial noise, pug breathing, cat snoring (val).

---

## Training

### 1. Configure

Copy the example config and fill in your paths:

```bash
cp config/train_4gpu_example.yaml config/my_train.yaml
```

Edit `config/my_train.yaml` — at minimum update:

| Field | Description |
|-------|-------------|
| `data.datapath` | Root path to your Clotho audio files |
| `data.datafiles` | List of training JSON files (e.g. `datafiles/train_clotho.json`) |
| `model.encoder.pretrained_audioencoder_path` | Path to `HTSAT_AudioSet_Saved_1.ckpt` |

Key training hyperparameters (already set in the provided configs):

| Parameter | Value |
|-----------|-------|
| Learning rate | 5e-5 (cosine schedule) |
| LR warmup | 10,000 steps |
| Batch size | 2 per GPU (8 total with 4 GPUs) |
| Gradient clip | max_norm = 0.5 |
| Epochs | 10 |

### 2. Single GPU

```bash
python train.py --conf config/train_example.yaml --save_dir outputs/
```

### 3. Multi-GPU (4 GPUs, DDP)

```bash
torchrun --nproc_per_node=4 train.py --conf config/train_4gpu.yaml --save_dir outputs/
```

Checkpoints are saved to `outputs/<timestamp>/` every epoch. Training logs are written to the same directory.

### Resuming from a Checkpoint

Set `resume_checkpoint` in the config (or pass `--resume_checkpoint /path/to/model.ckpt`) to resume.

---

## Evaluation

### Captioning Metrics (BLEU, ROUGE-L, CIDEr)

```bash
python train.py --conf config/eval_example.yaml --save_dir outputs/eval/
```

Set `mode: evaluate_checkpoint` and `resume_checkpoint: /path/to/model.ckpt` in the config.

### Multiple-Choice Accuracy

```bash
python scripts/eval_mc_accuracy.py \
    --checkpoint /path/to/model.ckpt \
    --datafile datafiles/val_mc.json \
    --datapath /path/to/ClothoV21 \
    --n_samples 500
```

### Debiased Inference

```bash
python scripts/eval_debiased.py \
    --checkpoint /path/to/model.ckpt \
    --datafile datafiles/val_clotho.json \
    --datapath /path/to/ClothoV21
```

### HTSAT Embedding Visualisation (t-SNE / UMAP)

```bash
python scripts/extract_htsat_embeddings.py \
    --checkpoint /path/to/model.ckpt \
    --datafile datafiles/val_captioning.json \
    --datapath /path/to/ClothoV21 \
    --output_dir paper/figures/
```

---

## Results

### Captioning Metrics (epoch-10 checkpoint, Clotho validation set)

| Condition | BLEU-1 | BLEU-4 | ROUGE-L | CIDEr |
|-----------|--------|--------|---------|-------|
| Mixed (captioning + comparison) | 0.052 | 0.005 | 0.086 | 0.027 |
| Captioning only | 0.096 | 0.000 | 0.084 | 0.011 |
| Comparison only | 0.003 | 0.000 | 0.079 | 0.003 |
| Debiased inference | 0.015 | 0.001 | 0.057 | 0.025 |

### Multiple-Choice Accuracy (500 samples/task)

| Task | Accuracy | Random Baseline |
|------|----------|----------------|
| Clotho-MCQ (4-choice) | 28.0% | 25.0% |
| CLE hypothesis (4-choice) | 21.2% | 25.0% |
| ClothoAQA yes/no | 4.6% | 50.0% |

### Self-Consistency Decoding (n=2000, MCQ + ClothoAQA)

| N | Temperature | Our Model | Original Mellow |
|---|-------------|-----------|-----------------|
| 1 | 1.0 (baseline) | 27.15% | 63.65% |
| 1 | 0.5 | 41.25% | **66.8%** |
| 5 | 0.5 | **46.85%** | 66.75% |

Self-Consistency yields +19.7% absolute accuracy for our model (27.15% → 46.85%) and only marginal gains on the well-calibrated original.

---

## Key Training Challenges

Six non-trivial failures were resolved during training:

1. **Checkpoint I/O contention** — 4 DDP ranks saturated NFS; fixed by rank-0 copying to `/dev/shm`
2. **Optimizer broadcast deadlock** — asymmetric broadcast counts; fixed by symmetric loading
3. **Corrupted audio (NaN/Inf waveforms)** — propagated to NaN loss; fixed with `all_reduce(isfinite)` guard
4. **OOD amplitude** — waveforms with |x|≈100 cause log-mel overflow; fixed by rejecting `max|x| ≥ 1.5`
5. **Unfrozen `c2l` layer** — adjacent linear layer grew unbounded; fixed by freezing `self.base` entirely
6. **LayerNorm scale drift** — γ grew to 6.2, causing 99% skip rate on resume; fixed by resetting γ=1.0, clamping projections to [-2, 2], and replacing skip-on-NaN with `nan_to_num`

---

## Paper

The full report is in [`paper/main.tex`](paper/main.tex) (Interspeech format, compiled with Overleaf).

Figures are generated by scripts in `paper/`:
- `make_figures.py` — loss curves, skip rate plots
- `make_tsne.py` — t-SNE and UMAP visualisations
- `make_wordcloud.py` — output token word cloud

---

## Citation

If you find this work useful, please cite the original Mellow paper:

```
@article{deshmukh2025mellow,
  title={Mellow: a Small Audio Language Model for Reasoning},
  author={Deshmukh, Soham and ...},
  year={2025}
}
```

---

## Acknowledgements

- [Mellow](https://github.com/soham97/mellow) — original model and ReasonAQA dataset
- [HTSAT](https://github.com/RetroCirce/HTS-Audio-Transformer) — audio encoder
- [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) — language model backbone
- [Clotho V2.1](https://zenodo.org/record/4743815) — audio dataset
