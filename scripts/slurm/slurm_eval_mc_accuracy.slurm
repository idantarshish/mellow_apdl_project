#!/bin/bash
#SBATCH --job-name=mellow_mc_acc
#SBATCH --partition=studentkillable
#SBATCH --nodes=1
#SBATCH --gres=gpu:titan:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/logs/slurm-%j_mc_acc.out
#SBATCH --error=/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/logs/slurm-%j_mc_acc.err

PROJECT_DIR=/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project
VENV=/home/yandex/APDL2526a/sapirtalmi/mellow_env
PYTHON=$VENV/bin/python

export HF_HOME=/home/yandex/APDL2526a/sapirtalmi/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

echo "=== SLURM Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "======================"

cd $PROJECT_DIR

# Run on epoch-10 checkpoint, limit 500 per subtype for speed (~1.5h total)
$PYTHON scripts/eval_mc_accuracy.py \
    --checkpoint /home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/outputs/2026-04-01_12-23-01/model--epo-10.ckpt \
    --limit 500

echo "=== Done ==="
