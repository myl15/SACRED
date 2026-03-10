#!/bin/bash
# =============================================================================
# SACRED — Full Sacred Circuit Discovery Pipeline
# Runs: stimulus generation → circuit discovery → necessity test → statistics
#       → visualizations
#
# Submit: sbatch scripts/run_main.sh
# Skip circuit discovery (use cache): sbatch scripts/run_main.sh --skip-discovery
# =============================================================================

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J "sacred_main"
#SBATCH --output=logs/%x_%j.out
#SBATCH --qos=matrix

echo "========================================================="
echo "SLURM JOB ID   : $SLURM_JOB_ID"
echo "JOB NAME       : $SLURM_JOB_NAME"
echo "Node           : $SLURM_JOB_NODELIST"
echo "GPUs allocated : $SLURM_GPUS_ON_NODE"
echo "========================================================="

# --- Environment ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SACRED_DIR="/home/$(whoami)/CS601R-interpretability/finalproject/SACRED"
source "${SACRED_DIR}/scripts/config.sh"

# --- Offline mode (compute nodes have no internet) ---
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

source "${SACRED_DIR}/.venv/bin/activate"
cd "${SACRED_DIR}"

echo "Starting sacred pipeline..."

srun "${PYTHON}" main.py "$@"

echo "Job complete."
