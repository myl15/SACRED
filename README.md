# SACRED

**S**emantic **A**nalysis of **C**ross-lingual **R**epresentations in **E**ncoder-**D**ecoders

A mechanistic interpretability project investigating whether NLLB-200 encodes concepts (sacred, kinship) as transferable directions in activation space across languages, and whether translation routes through English as an implicit pivot.

---

## Repository Structure

```
SACRED/
├── config.py                    # Shared constants (model name, languages, layers)
├── main.py                      # Full pipeline orchestrator
├── pyproject.toml               # UV project + dependencies
│
├── data/
│   ├── concept_vocabularies.py  # CONCEPT_VOCABULARIES + CONCEPT_DOMAINS dicts
│   └── contrastive_pairs.py     # ContrastivePairGenerator (new) + StimulusGenerator (legacy)
│
├── extraction/
│   ├── activation_capture.py    # Forward-hook management (mlp / attn / residual stream)
│   ├── circuit_discovery.py     # Circuit dataclasses + discover_sacred_circuit()
│   └── concept_vectors.py       # extract_concept_vectors() via residual-stream hooks
│
├── intervention/
│   ├── hooks.py                 # InterventionHook + register_vector_subtraction_hook()
│   ├── necessity.py             # measure_concept_deletion(), test_circuit_necessity()
│   └── pivot_diagnosis.py       # run_pivot_diagnosis() — 4-condition pivot test
│
├── analysis/
│   ├── journal_stats.py         # Bootstrap CIs + multiple-comparison helpers (paper-facing)
│   ├── statistical.py           # Cohen's d, permutation test, bootstrap CI, H1–H4 tests
│   ├── layer_wise.py            # CKA, English-centricity index, silhouette score
│   └── transfer_matrix.py       # compute_transfer_matrix() — full NxN deletion rates
│
├── journal/
│   ├── run_manifest.py          # Run manifest schema (git hash, env, config snapshot)
│   ├── ablation_runner.py       # Thin wrapper for wrong-domain / layers / matching-mode ablations
│   ├── hyperparam_sweep.py      # Sweep planner (plan JSON; optional execute)
│   └── validate_claims.py       # External validation (light): checksums + JSON structure
│
├── visualization/
│   ├── circuits.py              # Circuit maps, universal heatmap, report figure
│   ├── interventions.py         # Necessity/sufficiency plots, statistical summary
│   ├── layer_analysis.py        # CKA curves, t-SNE panels, English-centricity plot
│   └── transfer_heatmap.py      # Transfer matrix heatmap, pivot diagnosis bar chart
│
├── experiments/
│   ├── exp1_kinship.py          # Concept vector extraction + deletion test (kinship/sacred)
│   ├── exp2_pivot.py            # Pivot language diagnosis for all language pairs
│   ├── exp3_layer_wise.py       # Layer-wise convergence: CKA, centricity, silhouette
│   └── exp4_transfer_matrix.py  # Full NxN cross-lingual transfer matrix
│
└── nnlb_analysis.py             # Legacy monolith (kept for reference; superseded above)
```

---

## Setup

Requires Python ≥ 3.10 and [uv](https://github.com/astral-sh/uv).

```bash
# Clone / navigate to project
cd CS601R-interpretability/finalproject/SACRED

# Create virtual environment
uv venv .venv

# Install all dependencies into the venv
uv pip install -e . --python .venv/bin/python

# Activate
source .venv/bin/activate
```

Dependencies include: `torch`, `transformers`, `numpy`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`, `pandas`, `sacrebleu`, `nltk`, `tqdm`, `scikit-learn`, `umap-learn`, `datasets`.

---

## Phase 1 — Running Instructions

Phase 1 focuses on generating **multilingual stimuli**, extracting concept vectors, and validating deletion/transfer metrics before running sweeps/ablations.

Default intervention strength is **alpha=0.25** (calibrated to avoid ceiling effects). See `experiments/run_calibration.py`.

### Step 1: Generate stimuli + vectors (Experiment 1)

```bash
# Kinship
python experiments/exp1_kinship.py --domain kinship

# Sacred
python experiments/exp1_kinship.py --domain sacred

# Both domains (recommended before exp2/exp4 --both-domains)
python experiments/exp1_kinship.py --both-domains
```

This produces (per domain):

- **Stimuli**: `outputs/stimuli/{domain}_pairs.json`
- **Vectors**: `outputs/vectors/{domain}_{lang}.pt` and `outputs/vectors/{domain}_{lang}_pca.pt`
- **Diff matrices (for exp3 geometry)**: `outputs/vectors/{domain}_{lang}_diffs.pt`
- **Deletion results + manifest**: `outputs/exp1_{domain}_deletion.json` (now includes a top-level `run_manifest`) and `outputs/manifests/exp1_{domain}.json`

### Step 2: Pivot diagnosis (Experiment 2)

```bash
# One domain
python experiments/exp2_pivot.py --domain kinship --alpha 0.25 --vector-method both

# Both domains + combined pivot-index summary
python experiments/exp2_pivot.py --both-domains --alpha 0.25 --vector-method both

# Sensitivity grid for underpowered pairs (writes results/json/exp2_sensitivity_<domain>.json)
python experiments/exp2_pivot.py --domain kinship --sensitivity-grid --alpha-grid 0.25,0.35,0.5 --n-per-concept-grid 15,20,30

# Ablations / sensitivity knobs
python experiments/exp2_pivot.py --domain sacred --vector-source-domain kinship            # wrong-domain vectors
python experiments/exp2_pivot.py --domain sacred --matching-mode token_only                # output matching sensitivity
python experiments/exp2_pivot.py --domain sacred --layers 10,11,12,13,14,15                # layer subset
```

Outputs land in `results/`:

- `results/json/exp2_pivot_{domain}_{vector_method}.json` (includes `run_manifest` + `statistics`)
- `results/figures/exp2_pivot_{domain}_{vector_method}_continuous.png` (primary)
- `results/figures/exp2_pivot_index_summary_*.png`
- `results/json/exp2_sensitivity_{domain}.json` (when `--sensitivity-grid` is used)

### Step 3: Full transfer matrix (Experiment 4)

```bash
# One domain
python experiments/exp4_transfer_matrix.py --domain kinship --alpha 0.25 --vector-method both --output-lang eng_Latn

# Both domains + comparison chart
python experiments/exp4_transfer_matrix.py --both-domains --alpha 0.25 --vector-method both --output-lang eng_Latn

# Ablations / sensitivity knobs
python experiments/exp4_transfer_matrix.py --domain sacred --vector-source-domain kinship
python experiments/exp4_transfer_matrix.py --domain sacred --matching-mode word_boundary
python experiments/exp4_transfer_matrix.py --domain sacred --layers 10,11,12,13,14,15

```

Outputs land in `results/<output_lang>/` and include `run_manifest` + bootstrap `statistics` in the summary JSON:

- `results/<output_lang>/json/exp4_transfer_summary_{domain}_{vector_method}.json`
- `results/<output_lang>/figures/exp4_transfer_matrix_{domain}_{vector_method}_*.png`
  - Summary now includes relative (`english_hub_score`) and absolute hub metrics plus ceiling diagnostics.

---

## Experiments

| Script | Description | Status |
|---|---|---|
| `exp1_kinship.py` | Concept vectors + same-language deletion (kinship/sacred) + manifests | Ready |
| `exp2_pivot.py` | Pivot language diagnosis (4-condition test per pair) | Ready |
| `exp3_layer_wise.py` | CKA curves, t-SNE panels, English-centricity by layer | Ready |
| `exp4_transfer_matrix.py` | Full NxN cross-lingual transfer matrix | Ready |
| `main.py` | Sacred baseline circuit discovery + necessity + stats | Ready |

Run experiments in order: exp1 → exp2 → exp4 (exp2 and exp4 load vectors from exp1), exp3 is independent.

---

## Journal-readiness tooling (ablations, sweeps, external validation)

### Ablation runner (thin wrapper)

```bash
# Wrong-domain vector ablation
python -m journal.ablation_runner exp2 --domain sacred --vector-source-domain kinship --alpha 0.25
python -m journal.ablation_runner exp4 --domain sacred --vector-source-domain kinship --alpha 0.25 --output-lang eng_Latn

# Matching-mode sensitivity
python -m journal.ablation_runner exp2 --domain kinship --matching-mode token_only
python -m journal.ablation_runner exp4 --domain kinship --matching-mode hybrid
```

### Hyperparameter sweep planner

```bash
# Write a sweep plan JSON (does not run jobs)
python -m journal.hyperparam_sweep --experiment exp2 --domain kinship --alphas 0.1,0.25,0.5 --vector-methods mean,pca --out results/journal/sweep_exp2.json

# Optional: execute each command (GPU heavy)
python -m journal.hyperparam_sweep --experiment exp4 --domain sacred --alphas 0.1,0.25 --vector-methods mean --output-langs eng_Latn --execute
```

### External validation (light mode)

```bash
# Checksums + basic JSON structure (no model load)
python -m journal.validate_claims --stimuli outputs/stimuli/kinship_pairs.json --vectors-glob 'outputs/vectors/kinship_*.pt' --light-only
python -m journal.validate_claims --manifest outputs/manifests/exp1_kinship.json

# Build deterministic manuscript-facing tables + checksum index
python -m journal.build_paper_artifacts --results-dir results
```

---

## Key Design Decisions

**Residual stream over fc1** — Concept vector extraction hooks `encoder.layers[i]` output (1024-dim residual stream) rather than `fc1` (8192-dim MLP intermediate for 1.3B). This allows cleaner causal subtraction. The original sacred circuit analysis (still available via `extraction/circuit_discovery.py`) continues to use `fc1` as supplementary evidence.

**Mean-pool by default** — `extract_concept_vectors()` uses sequence mean-pooling (`pooling="mean"`). Switch to `pooling="token_aligned"` if results are noisy for kinship concepts.

**Paired output format** — `ContrastivePairGenerator` outputs `{positive, negative, concept_token_pos}` dicts. The legacy `StimulusGenerator` (three-way sacred/secular/inanimate) is preserved for backward compatibility with `main.py`.

**Deduplicated statistics** — `analysis/statistical.py` is the canonical source for `compute_cohens_d`, `permutation_test`, and `bootstrap_confidence_interval`. All other modules import from there.
