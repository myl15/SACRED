#!/bin/bash
# =============================================================================
# SACRED — Experiment 4: Full N×N Cross-Lingual Transfer Matrix
# Loads concept vectors from Experiment 1, computes deletion rates for all
# (source vector, target language) pairs, and saves the transfer heatmap.
#
# Requires: exp1 outputs in outputs/vectors/ and outputs/stimuli/kinship_pairs.json
#
# Submit: sbatch scripts/run_exp4.sh [domain]
#   domain defaults to "kinship"; pass "sacred" for sacred domain.
# =============================================================================

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J "sacred_exp4_transfer"
#SBATCH --output=logs/%x_%j.out
#SBATCH --qos=matrix

echo "========================================================="
echo "SLURM JOB ID   : $SLURM_JOB_ID"
echo "JOB NAME       : $SLURM_JOB_NAME"
echo "Node           : $SLURM_JOB_NODELIST"
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

mkdir -p logs results

# Args: domain (default kinship), alpha (default 0.25)
# Pass "both" as domain to run sacred + kinship and generate comparison chart
DOMAIN="${1:-kinship}"
ALPHA="${2:-0.25}"

echo "Starting Experiment 4: N×N transfer matrix (domain=${DOMAIN}, alpha=${ALPHA})..."
echo "NOTE: alpha=${ALPHA} avoids ceiling effects. Run run_calibration.sh to tune."

if [ "${DOMAIN}" = "both" ]; then
    srun "${PYTHON}" experiments/exp4_transfer_matrix.py \
        --both-domains --alpha "${ALPHA}" --results-dir results
    echo "Exp 4 complete (both domains)."
    echo "  Sacred matrix  : results/transfer_matrix_sacred_calibrated.png"
    echo "  Kinship matrix : results/transfer_matrix_kinship_calibrated.png"
    echo "  Comparison     : results/transfer_comparison_sacred_vs_kinship.png"
else
    srun "${PYTHON}" experiments/exp4_transfer_matrix.py \
        --domain "${DOMAIN}" --alpha "${ALPHA}" --results-dir results
    echo "Exp 4 complete (${DOMAIN})."
    echo "  Matrix   : results/transfer_matrix_${DOMAIN}_calibrated.png"
    echo "  Prob map : results/transfer_matrix_${DOMAIN}_prob_reduction.png"
    echo "  Summary  : results/exp4_transfer_summary_${DOMAIN}.json"
fi
