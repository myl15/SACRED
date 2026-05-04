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
#SBATCH --qos=gpu

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

echo "Starting Experiment 5: token-level max cosine concept-deletion..."

echo "Phase 1/2: calibration and coherence checks..."
srun "${PYTHON}" ./experiments/exp5_cosine_supplement.py \
    --results-dir results \
    --vectors-dir outputs/vectors \
    --stimuli-dir outputs/stimuli \
    --exp1-json-dir outputs \
    --calibration-output-csv results/paper/table_cosine_token_max_calibration.csv \
    --coherence-output-csv results/paper/table_cosine_token_max_coherence.csv \
    --validate-exp1-only \
    --debug-calibration \
    --debug-sentence-cap 200 \
    --blocked-token-strings "▁The,▁the,▁a,▁an,▁of,▁to,▁in,▁and" \
    --log-every 10 \
    --presence-threshold 0.25 \
    --deletion-threshold 0.15 \
    --generation-batch-size 8

echo "Phase 2/2: full Exp2 + Exp4 evaluation..."
srun "${PYTHON}" ./experiments/exp5_cosine_supplement.py \
    --results-dir results \
    --vectors-dir outputs/vectors \
    --stimuli-dir outputs/stimuli \
    --exp1-json-dir outputs \
    --output-csv results/paper/table_cosine_token_max.csv \
    --calibration-output-csv results/paper/table_cosine_token_max_calibration.csv \
    --coherence-output-csv results/paper/table_cosine_token_max_coherence.csv \
    --pivot-comparison-output-csv results/paper/table_cosine_token_max_pivot_comparison.csv \
    --log-every 10 \
    --generation-batch-size 8 \
    --presence-threshold 0.25 \
    --deletion-threshold 0.15 \
    --blocked-token-strings "▁The,▁the,▁a,▁an,▁of,▁to,▁in,▁and" \
    --random-trials 20 \
    --min-valid-random-trials 10

echo "Experiment 5 complete."
