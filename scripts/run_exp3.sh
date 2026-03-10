#!/bin/bash
# =============================================================================
# SACRED — Experiment 3: Layer-Wise Representation Convergence
# Computes CKA similarity curves, English-centricity index, silhouette scores,
# and t-SNE panels across all 24 encoder layers.
# This experiment is independent of Experiments 1 & 2.
#
# Submit: sbatch scripts/run_exp3.sh
# =============================================================================

#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J "sacred_exp3_layerwise"
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

# Compute nodes are fully offline. FLORES+ must be pre-downloaded on a login node:
#   bash scripts/download_cache.sh
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

source "${SACRED_DIR}/.venv/bin/activate"
cd "${SACRED_DIR}"

mkdir -p logs results

echo "Starting Experiment 3: Layer-wise convergence (FLORES-200, 100 sentences/lang)..."
echo "  Panel layers: [0, 8, 16, 23]"
echo "  Outputs: t-SNE panels, UMAP panels, silhouette trajectory"

srun "${PYTHON}" experiments/exp3_layer_wise.py --results-dir results

echo "Exp 3 complete."
echo "  JSON              : results/exp3_layer_wise.json"
echo "  t-SNE panels      : results/tsne_panels_scaled.png"
echo "  UMAP panels       : results/umap_panels_scaled.png"
echo "  Silhouette traj.  : results/silhouette_trajectory.png"
echo "  CKA curves        : results/exp3_cka_curves.png"
echo "  English centricity: results/exp3_english_centricity.png"
