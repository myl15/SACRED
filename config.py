"""
Shared constants and configuration for SACRED experiments.
"""

import os

# HuggingFace cache — honour HF_HUB_CACHE env var (set by scripts/config.sh),
# falling back to the project-level autodelete scratch directory.
HF_CACHE_DIR = os.environ.get(
    "HF_HUB_CACHE",
    "/home/myl15/nobackup/autodelete/.hf_cache/hub",
)

# Model configuration
# nllb-200-1.3B has 24 encoder layers (nllb-200-distilled-600M has 12)
MODEL_NAME = "facebook/nllb-200-1.3B"
ENCODER_HIDDEN_DIM = 1024   # Residual stream dimension (same for both 600M and 1.3B)
MLP_INTERMEDIATE_DIM = 8192  # fc1 intermediate dimension (1.3B uses 8192, 600M uses 4096)
NUM_ENCODER_LAYERS = 24

# Languages used in experiments
LANGUAGES = ["eng_Latn", "spa_Latn", "arb_Arab", "zho_Hant", "qul_Latn"]
EXPERIMENT_LANGUAGES = ["eng_Latn", "arb_Arab", "zho_Hant", "spa_Latn"]  # 4-language set for transfer matrix

# Layer ranges (proportionally scaled for 24-layer model)
DEFAULT_LAYERS = list(range(NUM_ENCODER_LAYERS))
EARLY_LAYERS = list(range(8))
MID_LAYERS = list(range(8, 16))
LATE_LAYERS = list(range(16, 24))

# Layers used for all concept-vector interventions AND calibration.
# Calibration and experiments MUST use the same set — applying a calibrated
# alpha over a different number of layers changes the effective intervention
# strength (more layers = more aggressive, even at the same alpha).
# Covers the semantic mid-late layers where concept directions are most stable.
INTERVENTION_LAYERS = list(range(10, 16))   # 6 layers out of 24

# Experiment defaults
SEED = 42
N_PER_CONDITION = 50
N_CONTRASTIVE_PAIRS = 15   # pairs per concept per language (project plan target)
TRAIN_RATIO = 0.8

# Device
import torch
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Output paths
OUTPUT_DIR = "outputs"
VECTORS_DIR = "outputs/vectors"
FIGURES_DIR = "outputs/figures"
STIMULI_DIR = "outputs/stimuli"
