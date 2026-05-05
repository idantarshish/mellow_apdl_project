#!/bin/bash
#SBATCH --job-name=mellow_debiased
#SBATCH --partition=studentkillable
#SBATCH --nodes=1
#SBATCH --gres=gpu:titan:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/logs/slurm-%j_debiased.out
#SBATCH --error=/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/logs/slurm-%j_debiased.err

PROJECT_DIR=/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project
VENV=/home/yandex/APDL2526a/sapirtalmi/mellow_env
PYTHON=$VENV/bin/python

export HF_HOME=/home/yandex/APDL2526a/sapirtalmi/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

echo "=== SLURM Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Python: $($PYTHON --version)"
echo "======================"

cd $PROJECT_DIR

$PYTHON scripts/eval_debiased.py \
    --checkpoint /home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/outputs/2026-04-01_12-23-01/model--epo-10.ckpt
