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
export SACRED_DEBUG_RUN_ID=post-fix

SACRED_DIR="/home/$(whoami)/CS601R-interpretability/finalproject/SACRED"
source "${SACRED_DIR}/scripts/config.sh"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

source "${SACRED_DIR}/.venv/bin/activate"
cd "${SACRED_DIR}"

mkdir -p logs results

# Positional args: domain (default both), alpha (default 0.25), mode (default standard)
DOMAIN="${1:-both}"
DEFAULT_ALPHA="0.25"
ALPHA="${2:-$DEFAULT_ALPHA}"
MODE="${3:-standard}"
N_RANDOM_CONTROLS="${4:-20}"

echo "Starting Experiment 2: Pivot diagnosis (domain=${DOMAIN}, alpha=${ALPHA}, mode=${MODE})..."
if [ -z "${2}" ]; then
    echo "NOTE: Using default calibrated alpha=${ALPHA}. Pass arg2 to override."
else
    echo "NOTE: Using user-provided alpha=${ALPHA}. Run run_calibration.sh to tune."
fi

if [ "${MODE}" = "grid" ]; then
    srun "${PYTHON}" -m experiments.exp2_pivot \
        --domain "${DOMAIN/both/sacred}" \
        --vector-method both \
        --sensitivity-grid \
        --alpha-grid "0.25,0.35,0.5" \
        --n-per-concept-grid "15,20,30" \
        --results-dir results/grid \
        --n-random-controls ${N_RANDOM_CONTROLS}
    echo "Exp 2 sensitivity grid complete."
    echo "  JSON  : results/json/exp2_sensitivity_${DOMAIN/both/sacred}.json"
elif [ "${DOMAIN}" = "both" ]; then
    srun "${PYTHON}" -c "
from experiments.exp2_pivot import run_both_domains
run_both_domains(
    vectors_dir='outputs/vectors',
    alpha=${ALPHA},
    results_dir='results',
    n_random_controls=${N_RANDOM_CONTROLS},
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
    n_random_controls=${N_RANDOM_CONTROLS},
)
"
    echo "Exp 2 complete."
    echo "  JSON  : results/exp2_pivot_${DOMAIN}.json"
    echo "  Plots : results/exp2_pivot_${DOMAIN}_binary.png"
    echo "          results/exp2_pivot_${DOMAIN}_continuous.png"
    echo "          results/exp2_pivot_index_summary_${DOMAIN}.png"
fi
