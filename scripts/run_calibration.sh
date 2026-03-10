#!/bin/bash
# =============================================================================
# SACRED — Calibration: Find optimal alpha for scaled vector subtraction
#
# Run this BEFORE experiments 2, 4, and 5 to determine the alpha that puts
# concept deletion rate in the diagnostic range (0.2–0.8).
#
# Default: sacred concept, English test sentences, Arabic concept vector,
#          targeting residual stream layers [5, 6, 7, 8].
#
# Submit: sbatch scripts/run_calibration.sh
# =============================================================================

#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -J "sacred_calibration"
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

echo "Starting calibration..."
echo "  Domain   : sacred"
echo "  Src lang : eng_Latn (test sentences)"
echo "  Vec lang : arb_Arab (concept vector)"
echo "  Tgt lang : spa_Latn (translation output)"
echo "  Layers   : [5, 6, 7, 8]"

srun "${PYTHON}" experiments/run_calibration.py

echo "Calibration complete."
echo "  Curve  : results/calibration_curve.png"
echo "  JSON   : results/calibration_results.json"
echo "  Use the recommended alpha value for run_exp2.sh, run_exp4.sh."
