================================================================
Replicating Mellow Under Real World Constraints
Advanced Topics in Audio Processing using Deep Learning - Final Project
Tel Aviv University, School of Computer Science
================================================================


REQUIREMENTS
------------
Install all dependencies using pip:

    pip install -r requirements.txt

Requires Python 3.10 and a CUDA-capable GPU.


DATASET SETUP
-------------
1. Download Clotho V2.1 audio files from:
      https://zenodo.org/record/4743815
   Place them under:
      data/ClothoV21/development/   (training audio)
      data/ClothoV21/validation/    (validation audio)

2. Download dataset JSON metadata files from:
      https://zenodo.org/records/15036628
   Place them in the datafiles/ directory.

3. Download the HTSAT pretrained checkpoint (HTSAT_AudioSet_Saved_1.ckpt) from:
      https://drive.google.com/drive/folders/1cZhMO7qLXTeifXVPP7PdM1NRYCG5cx28
   Note the path — you will need it in the config file.


CONFIGURATION
-------------
Copy the example config and update the following fields:

    cp config/train_4gpu_example.yaml config/my_train.yaml

Fields to update in the config:
  - data.datapath: path to your Clotho audio files root directory
  - data.datafiles: list of training JSON files (e.g. datafiles/train_clotho.json)
  - model.encoder.pretrained_audioencoder_path: path to HTSAT_AudioSet_Saved_1.ckpt


================================================================
TRAIN SCRIPT
================================================================

Single GPU:
    python train.py --conf config/train_example.yaml --save_dir outputs/

Multi-GPU (4 GPUs):
    torchrun --nproc_per_node=4 train.py --conf config/train_4gpu.yaml --save_dir outputs/

Checkpoints are saved to outputs/<timestamp>/ every epoch.
To resume from a checkpoint, set resume_checkpoint in the config.


================================================================
EVALUATION SCRIPT
================================================================

1. Captioning metrics (BLEU, ROUGE-L, CIDEr):

    Edit config/eval_example.yaml:
      - Set mode: evaluate_checkpoint
      - Set resume_checkpoint: /path/to/model.ckpt
      - Set data.datafiles to the validation JSON (e.g. datafiles/val_captioning.json)

    Then run:
        python train.py --conf config/eval_example.yaml --save_dir outputs/eval/


2. Multiple-choice accuracy (Clotho-MCQ, CLE, ClothoAQA):

    python scripts/eval_mc_accuracy.py \
        --checkpoint /path/to/model.ckpt \
        --datafile datafiles/val_mc.json \
        --datapath /path/to/ClothoV21 \
        --n_samples 500


3. Debiased inference (blocks MC option tokens during decoding):

    python scripts/eval_debiased.py \
        --checkpoint /path/to/model.ckpt \
        --datafile datafiles/val_clotho.json \
        --datapath /path/to/ClothoV21


4. HTSAT embedding visualisation (t-SNE / UMAP):

    python scripts/extract_htsat_embeddings.py \
        --checkpoint /path/to/model.ckpt \
        --datafile datafiles/val_captioning.json \
        --datapath /path/to/ClothoV21 \
        --output_dir paper/figures/


================================================================
AUDIO SAMPLES
================================================================
Sample audio files from the Clotho V2.1 dataset are included in:
    audio_samples/train/   (5 clips from the training split)
    audio_samples/val/     (5 clips from the validation split)
