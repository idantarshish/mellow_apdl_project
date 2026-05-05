#!/bin/bash
#SBATCH --job-name=mellow_eval_epo7
#SBATCH --partition=studentkillable
#SBATCH --nodes=1
#SBATCH --gres=gpu:titan:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/logs/slurm-%j_inference.out
#SBATCH --error=/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/logs/slurm-%j_inference.err

# Project and venv paths
PROJECT_DIR=/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project
VENV=/home/yandex/APDL2526a/sapirtalmi/mellow_env
PYTHON=$VENV/bin/python

# HuggingFace cache on fast storage (not home dir)
export HF_HOME=/home/yandex/APDL2526a/sapirtalmi/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

echo "=== SLURM Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Python: $($PYTHON --version)"
echo "======================"

cd $PROJECT_DIR

$PYTHON train.py \
    --config config/eval_1gpu.yaml \
    --checkpoint_path /home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/outputs/2026-03-31_10-38-33/model--epo-7.ckpt \
    --save-dir /home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/outputs
