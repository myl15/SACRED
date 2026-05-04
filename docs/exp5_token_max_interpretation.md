# Interpreting Exp5 Token-Max Cosine Outputs

This guide explains how to read the new Experiment 5 artifacts produced by
`experiments/exp5_cosine_supplement.py` after the token-level max-cosine rewrite.

The new metric does **not** use sentence-mean pooling. For each output sentence, it:

1. embeds each generated token from `model.model.shared.weight`,
2. computes cosine similarity to the domain/language concept anchor,
3. takes the maximum similarity across positions (`max_sim`).

Concept presence and deletion are then defined from `b_max` (baseline) and `a_max`
(ablated) with thresholded gating.

---

## Output Files

Default outputs in `results/paper/`:

- `table_cosine_token_max_coherence.csv`
- `table_cosine_token_max_calibration.csv`
- `table_cosine_token_max.csv`
- `table_cosine_token_max_pivot_comparison.csv`

These files are complementary and should be interpreted together.

---

## 1) Vocabulary Coherence

File: `table_cosine_token_max_coherence.csv`

Columns:

- `domain`, `language`
- `n_terms`
- `mean_pairwise_cosine`
- `coherence_flag` (`tight`, `mid`, `loose`, `nan`)

What it means:

- Measures how internally consistent the anchor vocabulary is before full evaluation.
- `tight` (mean pairwise cosine > 0.5): anchor is well-defined.
- `loose` (< 0.3): vocabulary may be semantically heterogeneous; interpret downstream
  deletion rates with caution.

Recommendation:

- If a `(domain, language)` row is `loose`, review/prune terms before trusting
  cross-lingual conclusions for that anchor.

---

## 2) Calibration Table

File: `table_cosine_token_max_calibration.csv`

Columns:

- `dataset`, `domain`, `pair_label`, `condition`
- `gate_pass_rate`, `cosine_deletion_rate`
- `mean_b_max`, `mean_a_max`
- `token_deletion_rate`
- `directional_agreement`

What it means:

- This checks that thresholds are sensible on English same-language sacred cases.
- `gate_pass_rate` should generally be high (target: >= 0.8 aggregate sacred diagonal).
- `directional_agreement=true` means cosine and token metrics move in the same
  qualitative direction for that row.

How to use it:

- If sacred English gate rates are low, lower `--presence-threshold`.
- If directional agreement is broadly poor, tune `--deletion-threshold`.
- Run this check before relying on full Exp2/Exp4 outputs.

---

## 3) Main Token-Max Metric Table

File: `table_cosine_token_max.csv`

One row per:

- `(experiment, domain, vector_method, pair_label, condition)`

Columns:

- `gate_pass_rate`
- `cosine_deletion_rate`
- `mean_b_max`
- `mean_a_max`
- `low_gate`

Interpretation:

- `gate_pass_rate`: fraction of sentence pairs where baseline passes the presence gate
  (`b_max >= presence_threshold`).
- `cosine_deletion_rate`: among gate-passing pairs, fraction with deletion
  (`a_max < deletion_threshold`).
- `mean_b_max` vs `mean_a_max`: diagnostic shift in best token-level concept evidence.
- `low_gate=true`: row is unreliable; `cosine_deletion_rate` is set to `nan`.

Rule of thumb:

- `low_gate=false` + high deletion rate -> stronger evidence for concept suppression.
- `low_gate=true` -> do not over-interpret that row, regardless of other values.

---

## 4) Pivot Comparison Table

File: `table_cosine_token_max_pivot_comparison.csv`

Columns:

- `domain`, `vector_method`, `pair_label`
- `token_pi`, `cosine_pi`
- `token_underpowered`, `cosine_underpowered`

`cosine_pi` uses:

`PI_cosine = (del_C - del_D) / (0.5 * (del_A + del_B) - del_D)`

Underpowered rule:

- If denominator `< 0.05`, PI is treated as underpowered and reported as `nan`.

How to use it:

- Compare `token_pi` and `cosine_pi` where both are defined.
- Agreement strengthens confidence that findings are not an artifact of one metric.
- Divergence identifies language/domain pairs needing qualitative inspection.

---

## Recommended Interpretation Workflow

1. Check coherence table for `loose` anchors.
2. Check calibration table for sacred English gate pass and directional agreement.
3. Use main table for Exp2/Exp4 conclusions, ignoring `low_gate=true` rows.
4. Use pivot comparison for metric-level robustness of pivot claims.

This order prevents drawing conclusions from poorly calibrated or weakly grounded
anchor configurations.

