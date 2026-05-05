#!/bin/bash
#SBATCH --job-name=mellow_grid
#SBATCH --partition=studentkillable
#SBATCH --nodes=1
#SBATCH --gres=gpu:titan:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
# -------------------------------------------------------------------------
# The --array flag defines the range of jobs.
# If you have 3 values for N and 3 for TEMP, you have 9 combinations (0-8).
# Adjust the number below (Total combinations - 1)
#SBATCH --array=0-8
# -------------------------------------------------------------------------
# Use %A (Job ID) and %a (Array Index) to keep logs separate
#SBATCH --output=/home/yandex/APDL2526a/idantarshish/mellow_apdl_project/logs/grid-%A_%a.out
#SBATCH --error=/home/yandex/APDL2526a/idantarshish/mellow_apdl_project/logs/grid-%A_%a.err

# 1. Define your grid parameters here
N_VALUES=(1 3 5)
TEMP_VALUES=(1.0 0.7 0.5)

# 2. Calculate which combination this specific task should run
NUM_TEMP=${#TEMP_VALUES[@]}
N_IDX=$((SLURM_ARRAY_TASK_ID / NUM_TEMP))
T_IDX=$((SLURM_ARRAY_TASK_ID % NUM_TEMP))

CURRENT_N=${N_VALUES[$N_IDX]}
CURRENT_TEMP=${TEMP_VALUES[$T_IDX]}

# Project and venv paths
PROJECT_DIR=/home/yandex/APDL2526a/idantarshish/mellow_apdl_project
VENV=/home/yandex/MLWG2025/idantarshish/anaconda3/envs/my_mellow_env
PYTHON=$VENV/bin/python

export HF_HOME=/home/yandex/APDL2526a/idantarshish/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

echo "=== SLURM Grid Job Info ==="
echo "Master Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running Grid: N=$CURRENT_N, TEMP=$CURRENT_TEMP"
echo "Node: $SLURMD_NODENAME"
echo "==========================="

cd $PROJECT_DIR

# Run the python script with the selected parameters
$PYTHON evaluate_sc.py \
    --config config/eval_sc_1gpu.yaml \
    --checkpoint_path /home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/outputs/2026-04-01_12-23-01/model--epo-10.ckpt \
    --save-dir "/home/yandex/APDL2526a/idantarshish/mellow_apdl_project/outputs/grid_n${CURRENT_N}_t${CURRENT_TEMP}" \
    --sc_n "$CURRENT_N" \
    --sc_temperature "$CURRENT_TEMP"