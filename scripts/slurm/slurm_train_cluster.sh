#!/bin/bash
#SBATCH --job-name=mellow_train
#SBATCH --partition=studentkillable
#SBATCH --nodes=1
#SBATCH --gres=gpu:titan:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/logs/slurm-%j.out
#SBATCH --error=/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/logs/slurm-%j.err

# Project and venv paths
PROJECT_DIR=/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project
VENV=/home/yandex/APDL2526a/sapirtalmi/mellow_env
PYTHON=$VENV/bin/python

# HuggingFace cache on fast storage (not home dir)
export HF_HOME=/home/yandex/APDL2526a/sapirtalmi/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

export NCCL_DEBUG=WARN

echo "=== SLURM Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Python: $($PYTHON --version)"
echo "======================"

cd $PROJECT_DIR

# Checkpoint path is set in config/train_cluster.yaml (resume_checkpoint field).
# trainer.py handles the checkpoint copy: rank 0 copies to /dev/shm (RAM disk),
# barrier, then all ranks load from /dev/shm — no NFS contention.

$PYTHON -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    --config config/train_cluster.yaml \
    --distributed-backend nccl \
    --save-dir /home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/outputs
