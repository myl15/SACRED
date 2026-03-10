# Phase 1 Modification Outline: Refactoring `nllb_causal_intervention.py`

**Date:** March 6, 2026
**Goal:** Transform the sacred-circuit-specific monolith into a reusable, domain-agnostic pipeline for the final project experiments.

---

## 1. Architectural Decision: Split vs. Refactor-In-Place

The current script is a ~3,400-line monolith converted from a Colab notebook with multiple `if __name__ == "__main__"` blocks and duplicated imports. For a code repository deliverable (required by the course), splitting into proper modules is the right move.

**Recommended module structure:**

```
project/
├── config.py                    # Shared constants, model/language config
├── data/
│   ├── contrastive_pairs.py     # Domain-agnostic contrastive pair generation
│   └── concept_vocabularies.py  # Token lists per concept domain per language
├── extraction/
│   ├── activation_capture.py    # ActivationCapture class (mostly unchanged)
│   ├── concept_vectors.py       # NEW: Concept vector extraction via activation differencing
│   └── circuit_discovery.py     # Existing circuit discovery (kept for sacred baseline)
├── intervention/
│   ├── hooks.py                 # InterventionHook + NEW vector subtraction hooks
│   ├── necessity.py             # Necessity/sufficiency testing
│   └── pivot_diagnosis.py       # NEW: Pivot language diagnosis framework
├── analysis/
│   ├── statistical.py           # Hypothesis testing, power analysis
│   ├── layer_wise.py            # NEW: CKA, English-centricity, silhouette scores
│   └── transfer_matrix.py       # NEW: Full NxN directional transfer computation
├── visualization/
│   ├── circuits.py              # Existing circuit visualization
│   ├── interventions.py         # Intervention result plots
│   ├── layer_analysis.py        # NEW: CKA curves, t-SNE panels, heatmaps
│   └── transfer_heatmap.py      # NEW: Transfer matrix heatmap
├── experiments/
│   ├── exp1_kinship.py          # Experiment 1 runner
│   ├── exp2_pivot.py            # Experiment 2 runner
│   ├── exp3_layer_wise.py       # Experiment 3 runner
│   └── exp4_transfer_matrix.py  # Experiment 4 runner
└── main.py                      # Orchestrator
```

You don't need to build all of this on day one. Start by extracting the core reusable pieces and build out the experiment scripts as you go.

---

## 2. Critical Methodological Gap: Concept Vectors vs. Circuit Neurons

This is the most important issue to address before writing any code.

**Your current script** discovers *circuits* — sets of individual neurons that show statistically significant activation differences between sacred and secular sentences. It identifies *which neurons* matter.

**Your project plan** calls for *concept vectors* — directions in activation space that encode a concept. These are computed by averaging the activation *difference* between contrastive sentence pairs across many examples.

These are fundamentally different objects:

| Property | Circuit Neurons | Concept Vectors |
|---|---|---|
| Representation | Sparse set of neuron indices | Dense direction in activation space |
| Intervention | Zero out specific neurons | Subtract/add a vector from full activation |
| Transfer operation | Check if same neurons fire cross-lingually | Apply vector from Language A to Language B |
| What your proposal describes | Background discovery work | Primary experimental method |

**Your initial experiments (the "sacred vector erases God" finding from the proposal, Figure 2)** used concept vectors — you subtracted an Arabic divine vector from Chinese activations. That's the approach your experiments need.

**What to build:** A `ConceptVectorExtractor` class that:

1. Takes contrastive sentence pairs (concept-present vs. concept-absent)
2. Runs both through the encoder
3. Computes the activation difference **at the concept token position** (not mean-pooled across the whole sequence — see Section 3)
4. Averages across all pairs to get a robust concept vector `v[layer, language, concept]`

The existing circuit discovery code is still valuable as a supplementary analysis (it tells you *which individual neurons* the concept vectors are spread across), but the primary pipeline needs to work with vectors.

---

## 3. Critical Methodological Gap: Token-Level vs. Sequence-Level Extraction

The current `capture_all_activations` function mean-pools across the entire sequence dimension:

```python
pooled = acts.mean(dim=1)  # [batch, intermediate_dim]
```

This is appropriate for circuit discovery (you want a sentence-level representation to ask "does this sentence contain sacred content?"), but **not ideal for concept vector extraction** via contrastive pairs.

When you have:
- **Positive:** "My **mother** taught me to cook"
- **Negative:** "My **teacher** taught me to cook"

The activation difference at the "mother"/"teacher" token position is your concept signal. Mean pooling dilutes this with all the shared context tokens.

**Two approaches, with a practical recommendation:**

**Approach A — Token-aligned extraction (theoretically cleaner):**
- Identify the position of the concept token in each sentence
- Extract activations specifically at that position
- Compute the difference

**Approach B — Sequence-mean extraction (simpler, may still work):**
- Mean pool as now
- The contrastive design means shared tokens cancel out in the difference
- Faster, no token alignment needed

**Recommendation:** Start with Approach B since your preliminary results already show it works (the sacred vector findings used something similar). Then, if results are noisy for kinship concepts, switch to Approach A for those. This is a reasonable research choice — document it in your methods section as a deliberate decision you can analyze.

The important new function signature would be:

```python
def extract_concept_vector(
    positive_sentences: List[str],   # Sentences containing the concept
    negative_sentences: List[str],   # Matched sentences without the concept
    model,
    tokenizer,
    lang_code: str,
    layer: int,
    pooling: str = "mean",  # "mean" or "token_aligned"
    concept_token_positions: Optional[List[int]] = None,  # For token_aligned
    device: str = "cuda"
) -> torch.Tensor:
    """
    Extract a concept vector via contrastive activation differencing.
    
    Returns:
        Concept vector of shape [hidden_dim] or [intermediate_dim]
    """
```

---

## 4. Modifications by Component

### 4.1 Data Generation (`StimulusGenerator` → `ContrastivePairGenerator`)

**Current state:** Generates sacred/secular/inanimate sentences from templates. English-only (non-English is a pass-through). Uses `_fill_template` with random sampling.

**What needs to change:**

**(a) Rename and generalize the concept domain system.** The current three-way (sacred/secular/inanimate) design is sacred-specific. For kinship, the contrastive structure is different:
- **Kinship present:** "My **mother** taught me to cook"
- **Kinship absent:** "My **teacher** taught me to cook"

This is a *minimal pair* design, not a three-way comparison. The generator needs to support:

```python
CONCEPT_DOMAINS = {
    "sacred": {
        "concepts": ["God", "Allah", "the Divine", ...],
        "controls": ["the king", "the president", ...],  # semantic controls
        "templates": [
            ("{concept} watches over humanity.", "{control} watches over humanity."),
            ...
        ]
    },
    "kinship": {
        "concepts": ["mother", "father", "family", "child"],
        "controls": ["teacher", "leader", "group", "student"],
        "templates": [
            ("My {concept} taught me everything.", "My {control} taught me everything."),
            ("The {concept} is important to me.", "The {control} is important to me."),
            ...
        ]
    }
}
```

**(b) Add multilingual sentence data.** The current generator only produces English and pretends non-English languages work. For the final project, you need actual contrastive pairs in Chinese, Arabic, and Spanish. Two realistic options:

- **Option 1 (faster):** Use NLLB itself to translate your English contrastive pairs, then manually verify a subset. Circular dependency concern is minimal for contrastive pairs (unlike the sacred token validation issue, which your script already addresses).
- **Option 2 (cleaner):** Write contrastive pairs directly in each language (or use a fluent speaker/resource). This is better for Arabic and Chinese where translation may not preserve the minimal-pair property.

**Recommendation:** For your ~15 pairs per concept per language (per the project plan), hand-crafting or at least hand-verifying is feasible and produces cleaner data. Budget this as part of the "sentence construction: 2h" in your timeline.

**(c) Output format change.** Instead of `{lang: {condition: [sentences]}}`, output paired data:

```python
{
    "kinship": {
        "eng_Latn": {
            "mother": [
                {"positive": "My mother taught me.", "negative": "My teacher taught me.", "concept_token_pos": 1},
                ...
            ],
            "father": [...],
        },
        "arb_Arab": {...}
    }
}
```

### 4.2 Concept Vector Extraction (NEW)

This is entirely new code. The current script has nothing that computes concept vectors as dense directions.

**Core function:**

```python
def extract_concept_vectors(
    contrastive_pairs: List[Dict],  # List of {positive, negative} sentence dicts
    model, tokenizer, lang_code: str,
    layers: List[int],
    component: str = "encoder_hidden",  # or "mlp"
    device: str = "cuda"
) -> Dict[int, torch.Tensor]:
    """
    Returns: {layer_idx: concept_vector} where concept_vector is shape [hidden_dim]
    """
    pos_acts = collect_activations(pairs["positive"], ...)
    neg_acts = collect_activations(pairs["negative"], ...)
    return {layer: (pos_acts[layer] - neg_acts[layer]).mean(dim=0) for layer in layers}
```

**Important design choice: Which activations to extract?**

The current script hooks into `model.model.encoder.layers[i].fc1` (the MLP intermediate projection). This gives you the 8192-dim intermediate representation. But for concept vector subtraction during translation, you probably want to intervene on the **residual stream** (the hidden states between layers), which is the 1024-dim representation that flows through the network.

The reason: Subtracting a vector from the residual stream is a cleaner causal intervention because it affects all downstream computation. Subtracting from MLP internals only affects that layer's MLP output.

**Recommendation:** Extract and intervene on `encoder.layers[i]` output (the residual stream), not `fc1`. This aligns with how Meng et al. (2023) and the ROME/MEMIT literature does causal tracing. If your sacred circuit results used fc1, you can still keep that analysis as supplementary.

To capture residual stream activations, hook the full layer output instead of fc1:

```python
# Instead of:
target_module = model.model.encoder.layers[layer_idx].fc1

# Use:
target_module = model.model.encoder.layers[layer_idx]
# The output of a transformer layer IS the residual stream
```

### 4.3 Cross-Lingual Intervention Hooks (MODIFY `InterventionHook`)

**Current state:** `InterventionHook` supports:
- `register_ablation_hook` — zeroes out specific neuron indices
- `register_random_ablation_hook` — zeroes out random neurons
- `register_patching_hook` — patches in clean activations

**What needs to be added:**

```python
def register_vector_subtraction_hook(
    self,
    model,
    concept_vector: torch.Tensor,  # Shape: [hidden_dim]
    layers: List[int],
    alpha: float = 1.0  # Scaling factor for intervention strength
):
    """
    Subtract a concept vector from encoder hidden states at specified layers.
    
    This is the core intervention for cross-lingual transfer testing:
    - Extract sacred vector from Arabic
    - Subtract it from English sentence activations during translation
    - Measure whether sacred concept disappears from output
    """
    self.cleanup()
    self.intervention_type = "vector_subtraction"
    
    for layer_idx in layers:
        target_module = model.model.encoder.layers[layer_idx]
        handle = target_module.register_forward_hook(
            self._make_subtraction_hook(concept_vector, alpha)
        )
        self.handles.append(handle)

def _make_subtraction_hook(self, vector: torch.Tensor, alpha: float):
    def hook_fn(module, input, output):
        # output is (hidden_states, ...) tuple or just hidden_states
        if isinstance(output, tuple):
            hidden = output[0]
            modified = hidden - alpha * vector.to(hidden.device)
            return (modified,) + output[1:]
        else:
            return output - alpha * vector.to(output.device)
    return hook_fn
```

This is the single most important new piece of infrastructure, because it's used by Experiments 1, 2, and 4.

### 4.4 Concept Deletion Measurement (MODIFY `measure_translation_quality`)

**Current state:** Measures sacred token probability and presence in translation output. Uses a fixed `sacred_token_ids` list.

**What needs to change:**

Generalize to work with arbitrary concept domains:

```python
def measure_concept_deletion(
    sentences: List[str],
    model, tokenizer,
    source_lang: str,
    target_lang: str,
    concept_token_ids: List[int],  # Renamed from sacred_token_ids
    intervention: Optional[InterventionHook] = None,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Measure whether a concept survives translation under intervention.
    
    Returns:
        {
            "concept_present_rate": float,  # Fraction of outputs containing concept
            "mean_concept_probability": float,  # Mean token probability
            "translations": List[str],  # Actual translations for inspection
            "per_sentence": List[Dict]  # Per-sentence metrics
        }
    """
```

Also extend `concept_vocabularies.py` to include kinship terms:

```python
CONCEPT_VOCABULARIES = {
    "sacred": {
        "eng_Latn": ["God", "Lord", "Creator", ...],
        "arb_Arab": ["الله", "الرب", ...],
        ...
    },
    "kinship": {
        "eng_Latn": ["mother", "father", "family", "child", ...],
        "zho_Hant": ["母親", "父親", "家庭", "孩子", ...],
        "arb_Arab": ["أم", "أب", "عائلة", "طفل", ...],
        "spa_Latn": ["madre", "padre", "familia", "niño", ...],
    }
}
```

### 4.5 Pivot Language Diagnosis (NEW)

This is entirely new and is your Tier 1 priority experiment.

```python
def run_pivot_diagnosis(
    translation_pair: Tuple[str, str],  # e.g., ("arb_Arab", "zho_Hant")
    concept_vectors: Dict[str, Dict[int, torch.Tensor]],  
        # {lang: {layer: vector}}
    test_sentences: List[str],  # Source language sentences with concept
    concept_token_ids: Dict[str, List[int]],  # {lang: [token_ids]}
    model, tokenizer,
    layers: List[int],
    device: str = "cuda"
) -> Dict[str, Dict[str, float]]:
    """
    Run the four-condition pivot diagnosis experiment.
    
    Conditions:
        A: Subtract source language vector
        B: Subtract target language vector
        C: Subtract English vector (the pivot test)
        D: Subtract random vector (control)
    
    Returns:
        {
            "condition_A_source": {"deletion_rate": float, "mean_prob": float},
            "condition_B_target": {"deletion_rate": float, "mean_prob": float},
            "condition_C_english": {"deletion_rate": float, "mean_prob": float},
            "condition_D_random": {"deletion_rate": float, "mean_prob": float},
            "pivot_index": float,  # effect(C) / mean(effect(A), effect(B))
            "interpretation": str
        }
    """
```

### 4.6 Layer-Wise Convergence Analysis (NEW)

This is entirely new infrastructure for Experiment 3.

```python
def extract_parallel_representations(
    parallel_sentences: Dict[str, List[str]],  # {lang: [sentences]}
    model, tokenizer,
    layers: List[int],
    device: str = "cuda"
) -> Dict[str, Dict[int, torch.Tensor]]:
    """Extract encoder hidden states at each layer for parallel sentences."""

def compute_cka_similarity(
    reps_a: torch.Tensor,  # [n_sentences, hidden_dim]
    reps_b: torch.Tensor   # [n_sentences, hidden_dim]
) -> float:
    """Compute Centered Kernel Alignment between two representation matrices."""

def compute_english_centricity(
    reps_by_lang: Dict[str, torch.Tensor],  # {lang: [n_sentences, hidden_dim]}
    english_key: str = "eng_Latn"
) -> float:
    """
    Compute English-centricity index:
    distance_to_english_centroid / distance_to_global_centroid
    
    < 1.0 means representations are closer to English than to the global mean
    > 1.0 means representations are more language-neutral
    """

def compute_silhouette_by_language(
    reps_by_lang: Dict[str, torch.Tensor]
) -> float:
    """Silhouette score for language clustering (high = well-separated by language)."""
```

### 4.7 Transfer Matrix (NEW, but builds on existing)

```python
def compute_transfer_matrix(
    concept_vectors: Dict[str, Dict[int, torch.Tensor]],  # {lang: {layer: vec}}
    test_sentences: Dict[str, List[str]],  # {lang: [sentences with concept]}
    concept_token_ids: Dict[str, List[int]],  # {lang: [token_ids]}
    model, tokenizer,
    layers: List[int],
    device: str = "cuda"
) -> np.ndarray:
    """
    Compute full NxN transfer matrix where entry [i,j] = deletion rate
    when applying language_i's concept vector to language_j's sentences.
    
    Returns: N×N numpy array of deletion rates
    """
```

### 4.8 Visualization Additions

The existing visualization code can stay but needs additions:

- **`plot_transfer_heatmap(matrix, languages)`** — 4×4 heatmap with asymmetry highlighting
- **`plot_pivot_diagnosis(results, translation_pairs)`** — Grouped bar chart per condition
- **`plot_cka_curves(cka_by_pair_by_layer)`** — Line plot, 6 pairs × 24 layers
- **`plot_tsne_panels(reps_by_lang, layers)`** — 2×2 panel at layers 0, 8, 16, 23
- **`plot_english_centricity(centricity_by_layer)`** — Line plot across layers
- **`plot_concept_domain_comparison(sacred_transfer, kinship_transfer)`** — Side-by-side bars

---

## 5. What to Reuse Without Modification

Not everything needs to change. These components are solid:

- **`ActivationCapture` class** — Hook management is clean; just add a new hook target option for full layer output (not just fc1)
- **`compute_cohens_d`, `permutation_test`, `bootstrap_confidence_interval`** — Statistical utilities are fine
- **`validate_intervention_execution`** — Intervention validation logic is good practice; extend it to verify vector subtraction too
- **`StatisticalReport` and hypothesis testing framework** — Keep as-is, add new hypothesis tests for pivot diagnosis
- **Most visualization utilities** — The matplotlib infrastructure is reusable; just add new plot functions

---

## 6. Suggested Implementation Order (Mapping to Week 9 Timeline)

**Friday March 6 (~3 hours):**
1. Create the module directory structure
2. Extract `ActivationCapture` and `InterventionHook` into their own files
3. Add `register_vector_subtraction_hook` to `InterventionHook`
4. Write `extract_concept_vectors()` — the core new function
5. Write `measure_concept_deletion()` — the generalized measurement function

**Saturday March 7 (~2 hours):**
1. Build kinship contrastive sentence pairs in all 4 languages
2. Create `concept_vocabularies.py` with kinship token lists
3. Store as `data/kinship_pairs.json`

**Monday March 9 (~2 hours):**
1. Wire up the pipeline: pairs → extraction → vectors
2. Run kinship concept vector extraction across all layers
3. Save vectors per layer/language

**Tuesday March 10 (~1.5 hours):**
1. Run same-language kinship deletion tests
2. Compare to sacred baseline deletion rates
3. Debug any pipeline issues

**Wednesday March 11 (~1.5 hours):**
1. Begin layer-wise visualization setup (t-SNE/UMAP)
2. This is lower priority — get the extraction pipeline solid first

---

## 7. Dependencies and Environment

**New packages needed (beyond what's already in the script):**
- `sklearn` — for CKA computation, silhouette scores, t-SNE
- `umap-learn` — for UMAP visualizations  
- `datasets` — for loading FLORES parallel data (Experiment 3)

**Already present:**
- `torch`, `transformers`, `numpy`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`, `pandas`, `sacrebleu`, `nltk`, `tqdm`

---

## 8. Risk Note: fc1 vs. Residual Stream

Your existing sacred circuit results were obtained by hooking `fc1` (MLP intermediate activations, 8192-dim). The concept vectors for cross-lingual transfer experiments should probably operate on the residual stream (1024-dim) for cleaner causal semantics.

This means your sacred circuit analysis and your cross-lingual transfer experiments may operate on different representational spaces. That's actually fine — you can frame it as complementary evidence: the circuit analysis tells you *which components* encode the concept, while the vector subtraction tells you *whether the overall representational direction* transfers across languages.

Just be explicit about this distinction in your methods section.