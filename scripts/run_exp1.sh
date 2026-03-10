#!/bin/bash
# =============================================================================
# SACRED — Experiment 1: Concept Vector Extraction
# Generates contrastive pairs, extracts concept vectors across all encoder
# layers for 4 languages, and runs same-language deletion tests.
#
# Submit: sbatch scripts/run_exp1.sh            (kinship only)
#         sbatch scripts/run_exp1.sh sacred     (sacred only)
#         sbatch scripts/run_exp1.sh both       (both domains, shared model load)
# =============================================================================

#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J "sacred_exp1_kinship"
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

mkdir -p logs outputs/vectors outputs/stimuli

# Positional arg: domain (default kinship; "sacred" or "both" also accepted)
DOMAIN="${1:-kinship}"

echo "Starting Experiment 1: concept vector extraction (domain=${DOMAIN})..."

if [ "${DOMAIN}" = "both" ]; then
    srun "${PYTHON}" experiments/exp1_kinship.py --both-domains
    echo "Exp 1 (both domains) complete."
    echo "  Vectors : outputs/vectors/sacred_*.pt  outputs/vectors/kinship_*.pt"
    echo "  Stimuli : outputs/stimuli/sacred_pairs.json  outputs/stimuli/kinship_pairs.json"
else
    srun "${PYTHON}" experiments/exp1_kinship.py --domain "${DOMAIN}"
    echo "Exp 1 complete."
    echo "  Vectors : outputs/vectors/${DOMAIN}_*.pt"
    echo "  Stimuli : outputs/stimuli/${DOMAIN}_pairs.json"
fi
echo "Next: sbatch scripts/run_exp2.sh both"
