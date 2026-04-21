#!/bin/bash
# =============================================================================
# SACRED — Experiment 5: Cosine concept-deletion (anchor-matched, gated)
# Primary: results/paper/table_cosine_concept_deletion.csv
# Validation: results/paper/table_cosine_exp1_validation.csv (Exp1 English)
#
# Submit: sbatch scripts/run_exp5.sh
# Requires: exp1 outputs, exp2/exp4 JSON under results/, vectors, stimuli
# =============================================================================

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J "exp5"
#SBATCH --output=logs/%x_%j.out
#SBATCH --qos=matrix

echo "========================================================="
echo "SLURM JOB ID   : $SLURM_JOB_ID"
echo "JOB NAME       : $SLURM_JOB_NAME"
echo "Node           : $SLURM_NODELIST"
echo "GPUs allocated : $SLURM_GPUS_ON_NODE"
echo "========================================================="

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SACRED_DIR="/home/$(whoami)/CS601R-interpretability/finalproject/SACRED"
source "${SACRED_DIR}/scripts/config.sh"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

source "${SACRED_DIR}/.venv/bin/activate"
cd "${SACRED_DIR}"

mkdir -p logs results/paper

echo "Starting Experiment 5: cosine concept-deletion..."
# Optional: validate Exp1 only first, then full run:
# srun "${PYTHON}" ./experiments/exp5_cosine_supplement.py --validate-exp1-only --device cuda

srun "${PYTHON}" ./experiments/exp5_cosine_supplement.py \
    --results-dir results \
    --vectors-dir outputs/vectors \
    --stimuli-dir outputs/stimuli \
    --exp1-json-dir outputs \
    --output-csv results/paper/table_cosine_concept_deletion.csv \
    --validation-output-csv results/paper/table_cosine_exp1_validation.csv \
    --log-every 10 \
    --generation-batch-size 8
    # For faster exploratory runs, also add: --max-random-trials 20

echo "Experiment 5 complete."
