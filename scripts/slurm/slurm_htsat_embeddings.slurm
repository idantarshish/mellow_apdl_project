#!/bin/bash
#SBATCH --job-name=htsat_embed
#SBATCH --partition=studentkillable
#SBATCH --nodes=1
#SBATCH --gres=gpu:titan:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/logs/slurm-%j_htsat_embed.out
#SBATCH --error=/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/logs/slurm-%j_htsat_embed.err

PROJECT_DIR=/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project
VENV=/home/yandex/APDL2526a/sapirtalmi/mellow_env
PYTHON=$VENV/bin/python

export HF_HOME=/home/yandex/APDL2526a/sapirtalmi/hf_cache

echo "=== SLURM Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Python: $($PYTHON --version)"
echo "======================"

cd $PROJECT_DIR

$PYTHON scripts/extract_htsat_embeddings.py
