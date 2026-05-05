#!/bin/bash
#SBATCH --job-name=fig5_gen
#SBATCH --partition=studentkillable
#SBATCH --nodes=1
#SBATCH --gres=gpu:titan:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/logs/slurm-%j_fig5.out
#SBATCH --error=/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/logs/slurm-%j_fig5.err

PROJECT_DIR=/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project
VENV=/home/yandex/APDL2526a/sapirtalmi/mellow_env
PYTHON=$VENV/bin/python

export HF_HOME=/home/yandex/APDL2526a/sapirtalmi/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES ==="
cd $PROJECT_DIR

$PYTHON paper/make_fig5_v2.py
echo "=== Done ==="
