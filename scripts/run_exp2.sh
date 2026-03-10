#!/bin/bash
# =============================================================================
# SACRED — Experiment 2: Pivot Language Diagnosis
# Runs the 4-condition pivot test for all non-English translation pairs.
# Requires Experiment 1 outputs (outputs/vectors/, outputs/stimuli/).
#
# Submit: sbatch scripts/run_exp2.sh
# Sacred domain:   sbatch scripts/run_exp2.sh sacred
# Both domains:    sbatch scripts/run_exp2.sh both
# =============================================================================

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J "sacred_exp2_pivot"
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

# Positional args: domain (default kinship; use "both" for all domains), alpha (default 0.25)
DOMAIN="${1:-kinship}"
ALPHA="${2:-0.25}"

echo "Starting Experiment 2: Pivot diagnosis (domain=${DOMAIN}, alpha=${ALPHA})..."
echo "NOTE: alpha=${ALPHA} avoids ceiling effects. Run run_calibration.sh to tune."

if [ "${DOMAIN}" = "both" ]; then
    srun "${PYTHON}" -c "
from experiments.exp2_pivot import run_both_domains
run_both_domains(
    vectors_dir='outputs/vectors',
    alpha=${ALPHA},
    results_dir='results',
)
"
    echo "Exp 2 (both domains) complete."
    echo "  JSON  : results/exp2_pivot_sacred.json"
    echo "          results/exp2_pivot_kinship.json"
    echo "  Plots : results/exp2_pivot_{sacred,kinship}_{binary,continuous}.png"
    echo "          results/exp2_pivot_index_summary_both.png"
else
    srun "${PYTHON}" -c "
from experiments.exp2_pivot import run_exp2
run_exp2(
    domain='${DOMAIN}',
    alpha=${ALPHA},
    vectors_dir='outputs/vectors',
    test_sentences_path='outputs/stimuli/${DOMAIN}_pairs.json',
    results_dir='results',
)
"
    echo "Exp 2 complete."
    echo "  JSON  : results/exp2_pivot_${DOMAIN}.json"
    echo "  Plots : results/exp2_pivot_${DOMAIN}_binary.png"
    echo "          results/exp2_pivot_${DOMAIN}_continuous.png"
    echo "          results/exp2_pivot_index_summary_${DOMAIN}.png"
fi
