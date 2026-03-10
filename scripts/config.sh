#!/bin/bash
# =============================================================================
# Shared configuration for all SACRED job scripts.
# Source this file at the top of every script:
#   source "${SACRED_DIR}/scripts/config.sh"
# =============================================================================

SACRED_DIR="/home/$(whoami)/CS601R-interpretability/finalproject/SACRED"

# HuggingFace cache — stored on fast scratch storage
export HF_HOME="/home/myl15/nobackup/autodelete/.hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"   # legacy alias

# Model / data identifiers
export MODEL_NAME="facebook/nllb-200-1.3B"
export FLORES_DATASET="openlanguagedata/flores_plus"

# Python interpreter inside the project venv
export PYTHON="${SACRED_DIR}/.venv/bin/python"
