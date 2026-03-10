# Phase 1 Results Fix-Forward: 5 Priority Tasks

## Context

I just completed Phase 1 experiments for my NLLB interlingua hypothesis project. The results revealed a critical methodological problem: **ceiling effects from overpowered interventions** are masking the signal I need for my two core research questions (RQ1: semantic universality, RQ2: interlingua vs. pivot). The pivot diagnosis (Experiment 2), kinship transfer matrix (Experiment 4), and potentially the sacred transfer results all show 1.00 deletion rates across ALL conditions including random controls, which means I can't distinguish between hypotheses.

The working codebase is `nllb_causal_intervention.py` (~3,400 lines). Key existing infrastructure:
- `InterventionHook` class with methods: `register_ablation_hook()`, `register_random_ablation_hook()`, `register_patching_hook()`, `cleanup()`
- All ablation hooks target `model.model.encoder.layers[layer_idx].fc1` (MLP intermediate, 8192-dim)
- `ActivationCapture` class for extracting activations at specified layers
- `measure_translation_quality()` returns `QualityMetrics` (sacred_token_present, sacred_token_probability, bleu_score, avg_token_prob, perplexity)
- Concept vectors are extracted via contrastive activation differencing (positive - negative sentence pairs, mean-pooled across sequence)
- `Circuit` dataclass with `NeuronComponent` list and `get_critical_layers()`
- Hypothesis testing functions: `test_h1_necessity()`, `test_h2_specificity()`, `test_h3_universality()` with Bonferroni correction at α=0.01

The model is NLLB-600M (12 encoder layers, 1024-dim residual stream, 8192-dim MLP intermediate). Languages: English (eng_Latn), Arabic (arb_Arab), Chinese (zho_Hans), Spanish (spa_Latn).

Execute the following 5 tasks in priority order. Each task builds on the previous. Do NOT proceed to task N+1 until task N compiles and runs correctly.

---

## Task 1: Implement Scaled Vector Subtraction (CRITICAL — unlocks everything else)

**Problem:** The current intervention approach subtracts the full concept vector at full magnitude. This is too strong — it saturates the deletion metric at 1.00 for all conditions including random controls, destroying all diagnostic value.

**What to build:**

Add a new method to `InterventionHook`:

```python
def register_scaled_vector_subtraction_hook(
    self,
    model,
    concept_vector: torch.Tensor,  # shape: [hidden_dim] or [intermediate_dim]
    layers: List[int],
    alpha: float = 1.0,  # scaling factor — THIS IS THE KEY PARAMETER
    target: str = "residual"  # "residual" for layer output, "fc1" for MLP intermediate
):
```

This hook should:
1. Accept a concept vector and a scaling factor `alpha`
2. During the forward pass, subtract `alpha * concept_vector` from the specified activation
3. Support two targets:
   - `"residual"`: Hook on `model.model.encoder.layers[layer_idx]` output (1024-dim). This is the full layer output / residual stream. For NLLB's M2M100 architecture, you need to hook the layer itself and modify its output. The layer forward returns `(hidden_states, attn_weights, ...)` — modify `hidden_states`.
   - `"fc1"`: Hook on `model.model.encoder.layers[layer_idx].fc1` output (8192-dim). This is what the existing ablation hooks use.
4. The concept vector dimension must match the target dimension (1024 for residual, 8192 for fc1). Raise a clear error if mismatched.

Then implement a calibration utility:

```python
def calibrate_intervention_strength(
    model, tokenizer, concept_vector, sentences, 
    lang_code, target_lang, concept_token_ids,
    alphas=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    layers=None, target="residual", device="cuda"
) -> Dict[float, Dict[str, float]]:
    """
    For each alpha value, run the intervention on all sentences and measure:
    - concept_deletion_rate (binary: was the concept token removed?)
    - concept_token_probability (continuous: P(concept_token) in output)
    - mean_bleu_degradation (how much did overall translation quality drop?)
    
    Returns dict mapping alpha -> {deletion_rate, mean_concept_prob, mean_bleu_delta}
    
    The goal is to find the alpha range where:
    - deletion_rate is between 0.2 and 0.8 (not saturated at 0 or 1)
    - There is meaningful variance across sentences
    - BLEU degradation is modest (intervention is targeted, not destructive)
    """
```

Generate a plot: x-axis = alpha, y-axis = deletion_rate, with a horizontal band highlighting the "diagnostic range" (0.2–0.8). Save as `calibration_curve.png`.

**Run calibration** on the sacred concept with English sentences using the Arabic concept vector, targeting the residual stream at layers [5, 6, 7, 8] (the peak circuit layers from Phase 1 results). Report the optimal alpha range.

**Important architectural note:** The existing `_make_ablation_hook` zeros out specific neuron indices (sparse intervention). The new vector subtraction hook is a *dense* intervention — it modifies the entire activation vector by subtracting a direction. These are fundamentally different operations. The vector subtraction is what the proposal specifies for concept vector experiments (Phase 2–4), while neuron ablation is what was used for the circuit discovery (Phase 1). Do not conflate them.

---

## Task 2: Re-run Pivot Language Diagnosis with Calibrated Interventions

**Problem:** The current pivot results (exp2_pivot_kinship.png) show 1.00 deletion across all conditions A–D for almost every translation pair, including the random control (Condition D). This is uninformative.

**What to do:**

Using the calibrated alpha from Task 1 (or if calibration hasn't been run for all language pairs yet, start with alpha=0.25 as a conservative default and adjust):

1. For each of the 6 non-English translation pairs (arb→zho, arb→spa, zho→arb, zho→spa, spa→arb, spa→zho), run 4 intervention conditions:
   - **A: Source language vector** (e.g., Arabic vector for arb→zho)
   - **B: Target language vector** (e.g., Chinese vector for arb→zho)
   - **C: English vector** (the pivot test — English is neither source nor target)
   - **D: Random vector** (same L2 norm as the concept vector, random direction)

2. **Use the continuous metric** (concept_token_probability), not just binary deletion_rate. The binary metric is what saturated at 1.00. Report both, but the continuous metric is the primary outcome.

3. For each translation pair, compute a **pivot index**:
   ```
   pivot_index = effect(English vector) / mean(effect(source vector), effect(target vector))
   ```
   where `effect(v) = baseline_concept_prob - intervened_concept_prob`.
   
   Interpretation:
   - pivot_index > 1.0 → English pivot behavior (English vector is more effective than source/target)
   - pivot_index ≈ 1.0 → True interlingua (all vectors equally effective)
   - pivot_index < 1.0 → Source/target privileged, no English pivot

4. Run this for **both sacred and kinship** concept domains.

5. Generate two visualizations:
   - Grouped bar chart: concept_token_probability reduction (Δ from baseline) by condition (A/B/C/D) for each translation pair. This replaces the current exp2_pivot_kinship.png.
   - Summary table/chart of pivot_index values across all pairs and both concept domains.

**Success criterion:** The random control (D) should show meaningfully less deletion than conditions A/B/C. If it doesn't, alpha is still too high — reduce and re-run.

---

## Task 3: Re-validate Sacred Circuit Necessity with Additional Stimuli

**Problem:** H1 (Necessity) showed d=0.888, p=0.0228 — a large effect size that fails the Bonferroni-corrected threshold of α=0.0167 (0.05/3). This is a power issue, not a null result.

**What to do:**

1. **Increase the stimulus set.** The current necessity test uses whatever sentences are in `stimuli["sacred"]`. Generate additional sacred and secular sentence pairs to approximately double the sample size. Use the existing `StimulusGenerator` class — it has template patterns and concept pools. Target at least 30 sacred and 30 secular sentences (if currently fewer).

2. **Re-run `test_circuit_necessity()`** with the expanded stimuli. The test uses paired t-test (`ttest_rel`) on baseline vs. ablated sacred_token_probability, with Cohen's d as effect size.

3. **Also consider:** The necessity test ablates neurons in `fc1` (8192-dim MLP intermediate). As a supplementary analysis, try the vector subtraction approach from Task 1 on the *residual stream* (1024-dim) — extract the sacred concept vector in residual-stream space and subtract it (at calibrated alpha). Compare the effect size to the sparse neuron ablation. This tests whether the causal pathway is better captured by the dense concept direction or the sparse neuron set.

4. Re-generate the statistical summary plot with updated H1 results. If H1 now passes at α=0.01 with Bonferroni correction, great. If it still doesn't, report the exact p-value and frame it as: "large effect size, borderline significance under conservative correction, suggesting partial necessity with redundant encoding pathways."

---

## Task 4: Scale Up t-SNE / UMAP Visualizations

**Problem:** The current t-SNE panels (exp3_tsne_panels.png) show ~4 points per language at 2 layers. This is far too few for t-SNE to produce meaningful structure — the visualization is essentially random scatter.

**What to do:**

1. Extract encoder hidden states for **at least 100 parallel sentences** from FLORES-200 (the dataset NLLB was evaluated on) for all 4 languages at **4 key layers**: 0 (input), 4 (early-mid), 8 (mid-late), 11 (final).

2. Use mean-pooling across the sequence dimension to get one vector per sentence per language per layer.

3. Generate t-SNE panels (perplexity=30, or tune if needed) showing all 4 languages at each of the 4 layers. Color by language. Each panel should have 400 points (100 sentences × 4 languages).

4. As a complementary visualization, also generate UMAP panels (n_neighbors=15, min_dist=0.1) at the same layers. UMAP tends to preserve global structure better than t-SNE.

5. Compute and overlay **silhouette scores** (language as the cluster label) at each layer. This gives a quantitative measure of how well-separated the language clusters are. Plot silhouette score vs. layer as a line chart.

6. Expected pattern based on the CKA results: language clusters should be somewhat mixed at early layers and either (a) merge into a single cluster by late layers (interlingua prediction) or (b) cluster around English with non-English languages as satellites (English-centricity prediction). The CKA and English-centricity index from Phase 1 suggest (b).

**Output:** 
- `tsne_panels_scaled.png` — 2×2 grid of t-SNE plots (layers 0, 4, 8, 11)
- `umap_panels_scaled.png` — same layout with UMAP
- `silhouette_trajectory.png` — silhouette score vs. layer

---

## Task 5: Re-run Transfer Matrix with Calibrated Interventions

**Problem:** The kinship transfer matrix (exp4_transfer_matrix_kinship.png) shows 1.00 in all 16 cells — same ceiling effect as the pivot diagnosis.

**What to do:**

1. Using calibrated alpha from Task 1, re-run the full 4×4 transfer matrix for **both sacred and kinship** domains. For each cell (source_lang, target_lang):
   - Extract concept vector from source_lang
   - Apply scaled vector subtraction (calibrated alpha) to sentences in target_lang during translation
   - Measure concept_token_probability reduction (continuous metric)

2. Generate two heatmaps:
   - Sacred transfer matrix (4×4, continuous deletion metric, values between 0 and 1)
   - Kinship transfer matrix (same format)

3. Analyze asymmetries:
   - Is EN→X transfer stronger than X→EN? (tests whether English representations are "upstream" in the causal chain)
   - Within-script vs. cross-script: spa↔eng (both Latin) vs. arb↔zho (different scripts) — does shared script help?
   - Does the diagonal (same-language) consistently outperform off-diagonal? By how much?

4. Compute a **cross-lingual transfer score** per cell:
   ```
   transfer_score = off_diagonal_effect / diagonal_effect
   ```
   This normalizes by same-language effectiveness. Your proposal's success criterion is >70% cross-lingual transfer for universal concepts.

5. Generate a comparison bar chart: mean cross-lingual transfer score for sacred vs. kinship, with error bars. This directly answers RQ1 (do universal semantic circuits generalize beyond sacred?).

**Output:**
- `transfer_matrix_sacred_calibrated.png`
- `transfer_matrix_kinship_calibrated.png`  
- `transfer_comparison_sacred_vs_kinship.png`

---

## General Instructions

- All new code should be added to `nllb_causal_intervention.py` or new files in the same directory. Follow the existing code style (dataclasses for results, type hints, docstrings).
- Save all plots to a `results/` directory.
- After each task, print a summary of key metrics to stdout.
- If any task reveals that the calibrated alpha needs adjustment (e.g., still seeing saturation, or no effect at all), report this and re-calibrate before proceeding.
- The concept vectors should already exist from Phase 1. If they need to be re-extracted for the residual stream target (1024-dim instead of 8192-dim fc1), do so using the existing `ActivationCapture` infrastructure — register hooks on the layer output instead of fc1.
- For FLORES-200 data in Task 4: if not already downloaded, use `datasets` library (`load_dataset("facebook/flores", "all")`). Extract the `sentence` field for each language using the FLORES language codes.