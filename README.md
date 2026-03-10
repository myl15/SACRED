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
│   ├── statistical.py           # Cohen's d, permutation test, bootstrap CI, H1–H4 tests
│   ├── layer_wise.py            # CKA, English-centricity index, silhouette score
│   └── transfer_matrix.py       # compute_transfer_matrix() — full NxN deletion rates
│
├── visualization/
│   ├── circuits.py              # Circuit maps, universal heatmap, report figure
│   ├── interventions.py         # Necessity/sufficiency plots, statistical summary
│   ├── layer_analysis.py        # CKA curves, t-SNE panels, English-centricity plot
│   └── transfer_heatmap.py      # Transfer matrix heatmap, pivot diagnosis bar chart
│
├── experiments/
│   ├── exp1_kinship.py          # Kinship concept vector extraction + deletion test
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

Phase 1 focuses on extracting kinship concept vectors and validating the same-language deletion pipeline before running cross-lingual experiments.

### Step 1: Generate contrastive pairs

```python
from data.contrastive_pairs import ContrastivePairGenerator

gen = ContrastivePairGenerator(seed=42)
pairs = gen.generate_pairs(
    domain="kinship",
    n_per_concept=15,
    languages=["eng_Latn", "arb_Arab", "zho_Hant", "spa_Latn"],
    output_path="outputs/stimuli/kinship_pairs.json",
)
```

Each pair has the form:
```json
{"positive": "My mother taught me.", "negative": "My teacher taught me.", "concept_token_pos": 1}
```

### Step 2: Extract concept vectors

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from extraction.concept_vectors import extract_concept_vectors, save_concept_vectors

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-1.3B").to("cuda")
model.eval()

# Extract for one language / concept
vectors = extract_concept_vectors(
    contrastive_pairs=pairs["eng_Latn"]["mother"],
    model=model,
    tokenizer=tokenizer,
    lang_code="eng_Latn",
    layers=list(range(24)),        # all 24 encoder layers (1.3B)
    component="encoder_hidden",    # residual stream (1024-dim)
    pooling="mean",
    device="cuda",
)
# vectors = {layer_idx: tensor[1024]}
```

Or run the full Experiment 1 script:

```bash
python experiments/exp1_kinship.py
```

This generates pairs, extracts vectors for all languages, runs same-language deletion tests, and saves results to `outputs/`.

### Step 3: Test same-language concept deletion

```python
from intervention.hooks import InterventionHook
from intervention.necessity import measure_concept_deletion
from data.contrastive_pairs import load_independent_sacred_tokens

# Load kinship token IDs
token_ids = load_independent_sacred_tokens("eng_Latn", tokenizer, domain="kinship")

# Apply concept vector subtraction hook
hook = InterventionHook()
hook.register_vector_subtraction_hook(model, vectors[12], layers=[12], alpha=1.0)

result = measure_concept_deletion(
    sentences=["My mother taught me everything."],
    model=model,
    tokenizer=tokenizer,
    source_lang="eng_Latn",
    target_lang="spa_Latn",
    concept_token_ids=token_ids,
    intervention=hook,
    device="cuda",
)
print(result["concept_present_rate"])   # fraction of outputs still containing the concept
hook.cleanup()
```

### Step 4: Full pipeline (sacred baseline)

```bash
# Run full sacred circuit discovery + necessity test + visualizations
python main.py

# Skip circuit discovery (use cached circuit)
python main.py --skip-discovery
```

Outputs land in `outputs/figures/`.

---

## Experiments

| Script | Description | Status |
|---|---|---|
| `exp1_kinship.py` | Kinship concept vectors + same-language deletion | Ready |
| `exp2_pivot.py` | Pivot language diagnosis (4-condition test per pair) | Ready |
| `exp3_layer_wise.py` | CKA curves, t-SNE panels, English-centricity by layer | Ready |
| `exp4_transfer_matrix.py` | Full NxN cross-lingual transfer matrix | Ready |
| `main.py` | Sacred baseline circuit discovery + necessity + stats | Ready |

Run experiments in order: exp1 → exp2 → exp4 (exp2 and exp4 load vectors from exp1), exp3 is independent.

---

## Key Design Decisions

**Residual stream over fc1** — Concept vector extraction hooks `encoder.layers[i]` output (1024-dim residual stream) rather than `fc1` (8192-dim MLP intermediate for 1.3B). This allows cleaner causal subtraction. The original sacred circuit analysis (still available via `extraction/circuit_discovery.py`) continues to use `fc1` as supplementary evidence.

**Mean-pool by default** — `extract_concept_vectors()` uses sequence mean-pooling (`pooling="mean"`). Switch to `pooling="token_aligned"` if results are noisy for kinship concepts.

**Paired output format** — `ContrastivePairGenerator` outputs `{positive, negative, concept_token_pos}` dicts. The legacy `StimulusGenerator` (three-way sacred/secular/inanimate) is preserved for backward compatibility with `main.py`.

**Deduplicated statistics** — `analysis/statistical.py` is the canonical source for `compute_cohens_d`, `permutation_test`, and `bootstrap_confidence_interval`. All other modules import from there.
