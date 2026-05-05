#!/bin/bash
#SBATCH --job-name=mellow_epoch_curve
#SBATCH --partition=studentkillable
#SBATCH --nodes=1
#SBATCH --gres=gpu:titan:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/logs/slurm-%j_epoch_curve.out
#SBATCH --error=/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/logs/slurm-%j_epoch_curve.err

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

CHECKPOINTS=(
    "4:/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/outputs/2026-03-31_10-38-33/model--epo-4.ckpt"
    "5:/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/outputs/2026-03-31_10-38-33/model--epo-5.ckpt"
    "6:/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/outputs/2026-03-31_10-38-33/model--epo-6.ckpt"
    "7:/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/outputs/2026-03-31_10-38-33/model--epo-7.ckpt"
    "8:/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/outputs/2026-04-01_12-23-01/model--epo-8.ckpt"
    "9:/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/outputs/2026-04-01_12-23-01/model--epo-9.ckpt"
    "10:/home/yandex/APDL2526a/sapirtalmi/mellow_apdl_project/outputs/2026-04-01_12-23-01/model--epo-10.ckpt"
)

RESULTS_FILE=$PROJECT_DIR/outputs/epoch_curve_results.json
echo "[]" > $RESULTS_FILE

for ENTRY in "${CHECKPOINTS[@]}"; do
    EPOCH="${ENTRY%%:*}"
    CKPT="${ENTRY#*:}"
    echo ""
    echo "=== Evaluating epoch $EPOCH ==="
    echo "Checkpoint: $CKPT"

    OUT=$($PYTHON train.py \
        --config config/eval_per_epoch.yaml \
        --checkpoint_path "$CKPT" \
        --save-dir $PROJECT_DIR/outputs 2>&1)

    echo "$OUT"

    CIDER=$(echo "$OUT" | grep "cider:" | tail -1 | awk '{print $NF}')
    BLEU1=$(echo "$OUT" | grep "bleu_1:" | tail -1 | awk '{print $NF}')
    ROUGE=$(echo "$OUT" | grep "rouge_l:" | tail -1 | awk '{print $NF}')

    echo "=== Epoch $EPOCH: CIDEr=$CIDER BLEU-1=$BLEU1 ROUGE-L=$ROUGE ==="

    # Append to results JSON
    $PYTHON -c "
import json, sys
results = json.load(open('$RESULTS_FILE'))
results.append({'epoch': $EPOCH, 'cider': $CIDER, 'bleu_1': $BLEU1, 'rouge_l': $ROUGE})
json.dump(results, open('$RESULTS_FILE', 'w'), indent=2)
"
done

echo ""
echo "=== All epochs done. Results saved to $RESULTS_FILE ==="
$PYTHON -c "
import json
results = json.load(open('$RESULTS_FILE'))
print('Epoch | CIDEr  | BLEU-1 | ROUGE-L')
for r in results:
    print(f\"{r['epoch']:5d} | {r['cider']:.4f} | {r['bleu_1']:.4f} | {r['rouge_l']:.4f}\")
"
