# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SACRED** (Semantic Analysis of Cross-lingual Representations in Encoder-Decoders) is a mechanistic interpretability project investigating whether NLLB-200 encodes concepts ("sacred", "kinship") as transferable directions in activation space across languages, and whether translation routes through English as an implicit pivot.

Model: `facebook/nllb-200-1.3B` — 24 encoder layers, 1024-dim residual stream, 8192-dim MLP intermediate.

## Setup

```bash
uv venv .venv
uv pip install -e . --python .venv/bin/python
source .venv/bin/activate
```

## Running Experiments

```bash
# Individual experiments (run in order: exp1 → exp2 → exp4; exp3 is independent)
python experiments/exp1_kinship.py      # Kinship vectors + same-language deletion
python experiments/exp2_pivot.py        # Pivot language diagnosis (4-condition test)
python experiments/exp3_layer_wise.py   # CKA curves, t-SNE/UMAP, English-centricity
python experiments/exp4_transfer_matrix.py --both-domains  # Full NxN transfer matrix

# Full sacred baseline pipeline
python main.py                  # Circuit discovery + necessity + stats + visualizations
python main.py --skip-discovery # Use cached circuit (skip slow discovery step)

# On SLURM cluster (compute nodes are fully offline — pre-download first):
bash scripts/download_cache.sh          # login node only; downloads model + FLORES+
sbatch scripts/run_calibration.sh       # find optimal alpha (run before exp2/exp4)
sbatch scripts/run_exp1.sh
sbatch scripts/run_exp2.sh [domain] [alpha]   # default: kinship 0.25
sbatch scripts/run_exp3.sh                    # uses cached FLORES+ (HF_DATASETS_OFFLINE=1)
sbatch scripts/run_exp4.sh both [alpha]       # default alpha: 0.25
sbatch scripts/run_main.sh
```

## Architecture

Data flows through four stages:

1. **Data generation** (`data/`) — `ContrastivePairGenerator.generate_pairs()` produces `{lang: {concept: [{positive, negative, concept_token_pos}]}}` dicts. Legacy `StimulusGenerator` (3-way: sacred/secular/inanimate) is preserved for `main.py`.

2. **Extraction** (`extraction/`) — `extract_concept_vectors()` registers forward hooks on `encoder.layers[i]` output (residual stream, 1024-dim) and computes `mean(pos_acts - neg_acts)` per layer → `{layer_idx: tensor[1024]}`. The older `circuit_discovery.py` hooks `fc1` (4096-dim) instead.

3. **Intervention & measurement** (`intervention/`) — `InterventionHook.register_vector_subtraction_hook()` subtracts a concept vector from encoder activations during forward pass. `measure_concept_deletion()` translates sentences and checks concept presence in output. `run_pivot_diagnosis()` runs 4-condition test (source vector, target vector, English vector, random).

4. **Analysis & visualization** (`analysis/`, `visualization/`) — `analysis/statistical.py` is the canonical source for all statistical primitives (Cohen's d, permutation test, bootstrap CI, H1–H4 hypothesis tests). Transfer matrix (`analysis/transfer_matrix.py`) computes full NxN deletion rates across language pairs.

## Key Design Decisions

- **Residual stream over fc1**: Concept vectors extracted from 1024-dim residual stream (not 8192-dim `fc1`) for cleaner causal subtraction.
- **Mean-pooling default**: `extract_concept_vectors()` uses `pooling="mean"` over sequence positions. Switch to `pooling="token_aligned"` for noisy concepts.
- **Deduplicated statistics**: Always import `compute_cohens_d`, `permutation_test`, `bootstrap_confidence_interval` from `analysis/statistical.py` — do not reimplement elsewhere.
- **Experiment ordering**: exp2 and exp4 load vectors saved by exp1. Run exp1 first.
- **Calibrated alpha**: Default intervention alpha is 0.25 (not 1.0) to avoid ceiling effects. Run `experiments/run_calibration.py` to find the optimal value for your data. The diagnostic range is deletion_rate ∈ [0.2, 0.8]. **Critical**: calibration and experiments must use the same `INTERVENTION_LAYERS` from `config.py` — calibrating on 4 layers and running on 24 makes the alpha 6× too large.
- **FLORES+ dataset**: exp3 loads `openlanguagedata/flores_plus` (set via `FLORES_DATASET` env var in `scripts/config.sh`). This must be pre-downloaded via `download_cache.sh` before running on offline compute nodes. **Language code mismatch:** FLORES+ uses `cmn_Hant` (ISO 639-3) while NLLB uses `zho_Hant` (BCP-47). `load_flores200()` handles this via a `NLLB_TO_FLORES` dict — add new mappings there if other language codes diverge. `download_models.py` downloads `cmn_Hant` directly.
- **Domain-matched stimuli**: `run_exp2()` and `run_exp4()` default `test_sentences_path=None`, which resolves to `outputs/stimuli/{domain}_pairs.json` at runtime. Never hardcode `kinship_pairs.json` as a default — using kinship sentences for the sacred domain makes baseline concept presence = 0 and produces deletion_rate = 1.000 regardless of alpha. `run_both_domains()` derives the path per domain internally and does not accept `test_sentences_path`.
- **Outputs split**: New experiment outputs (calibrated runs) go to `results/`; original pipeline outputs stay in `outputs/figures/`.

## Configuration (`config.py` / `scripts/config.sh`)

- `EXPERIMENT_LANGUAGES = ["eng_Latn", "arb_Arab", "zho_Hant", "spa_Latn"]` — 4-language set for transfer matrix
- `LANGUAGES` also includes `"qul_Latn"` (Quechua) for broader experiments
- `HF_CACHE_DIR` respects `HF_HUB_CACHE` env var (set by `scripts/config.sh` for cluster runs)
- `FLORES_DATASET = "openlanguagedata/flores_plus"` — set in `config.sh`, read by `exp3_layer_wise.py`
- Outputs: `outputs/{vectors,figures,stimuli}/` (original pipeline), `results/` (calibrated runs)
