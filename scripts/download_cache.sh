#!/bin/bash
# =============================================================================
# Download all models and datasets to the project cache.
#
# Run this ONCE on the login node (internet required) before submitting jobs:
#   bash scripts/download_cache.sh
#
# Optional flags (passed through to download_models.py):
#   --model facebook/nllb-200-1.3B   # use larger model instead
#   --skip-flores                     # skip FLORES+ dataset
# =============================================================================

set -euo pipefail

SACRED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SACRED_DIR}/scripts/config.sh"

echo "========================================================="
echo "SACRED — Model & Dataset Cache Download"
echo "Project root : ${SACRED_DIR}"
echo "Cache dir    : ${HF_HOME}"
echo "Model        : ${MODEL_NAME}"
echo "========================================================="

# Ensure the venv exists
if [ ! -f "${PYTHON}" ]; then
    echo "ERROR: venv not found at ${PYTHON}"
    echo "Run: cd ${SACRED_DIR} && uv venv .venv && uv pip install -e . --python .venv/bin/python"
    exit 1
fi

# Make sure logs dir exists (SLURM jobs write here)
mkdir -p "${SACRED_DIR}/logs"

cd "${SACRED_DIR}"

"${PYTHON}" scripts/download_models.py \
    --cache-dir "${HF_HOME}" \
    --model "${MODEL_NAME}" \
    "$@"

echo ""
echo "Done. Cached:"
echo "  Model   : ${MODEL_NAME}"
echo "  Dataset : openlanguagedata/flores_plus  (eng_Latn, arb_Arab, zho_Hant, spa_Latn)"
echo ""
echo "Submit jobs in order:"
echo "  sbatch scripts/run_calibration.sh       # find optimal alpha (run first)"
echo "  sbatch scripts/run_exp1.sh              # extract concept vectors"
echo "  sbatch scripts/run_exp2.sh              # pivot diagnosis"
echo "  sbatch scripts/run_exp3.sh              # layer-wise + t-SNE/UMAP"
echo "  sbatch scripts/run_exp4.sh both         # transfer matrix (both domains)"
echo "  sbatch scripts/run_main.sh              # sacred circuit necessity"
