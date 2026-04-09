# SACRED — Claims, metrics, and pre-registration scope

This document locks **what the paper is allowed to claim** from the current pipeline. Update it when hypotheses or success criteria change.

## Model and data scope (fixed for v1 paper)

| Item | Value |
|------|--------|
| Model | `facebook/nllb-200-1.3B` ([config.py](../config.py) `MODEL_NAME`) |
| Encoder layers | 24; interventions default to `INTERVENTION_LAYERS` (e.g. 10–15) |
| Languages (transfer / pivot) | `EXPERIMENT_LANGUAGES`: eng, arb, zho, spa |
| Concept domains | `kinship`, `sacred` (contrastive pairs + vocabularies) |
| Stimuli | `outputs/stimuli/{domain}_pairs.json` from Experiment 1 |

## Primary hypotheses (confirmatory)

Pre-register **at most 2–3** primary claims; everything else is exploratory unless promoted in a revision.

1. **H_cross**: For a fixed concept domain, subtracting a language-specific residual-stream concept direction reduces concept presence in translation outputs **beyond** a Monte Carlo random-direction control (Experiment 2, conditions A/B/C vs D; Experiment 4 off-diagonal cells vs null).
2. **H_transfer**: Cross-lingual transfer scores (off-diagonal / reference diagonal per [analysis/transfer_matrix.py](../analysis/transfer_matrix.py)) exceed a **pre-specified** threshold on **pre-specified** language pairs (declare pairs in this doc before final runs).
3. **H_geom** (optional): Layer-wise geometric alignment (Experiment 3 / PCA–mean geometry) **correlates with** causal transfer at the same layer — report as secondary if pre-registered.

## Exploratory analyses (not confirmatory)

- Full N×N matrix “best cell” post-hoc mining without multiple-comparison correction.
- English-pivot index (Experiment 2) for all pairs without a primary subset.
- Any analysis using placeholder code paths (e.g. CV placeholder in [analysis/statistical.py](../analysis/statistical.py), sufficiency placeholder in [intervention/necessity.py](../intervention/necessity.py)).

## Metrics (which endpoint is primary)

| Experiment | Primary metric | Secondary |
|------------|----------------|-----------|
| Exp 2 (pivot) | Continuous: Δ mean P(concept) vs baseline; binary deletion rate | `pivot_index` (exploratory unless pair subset fixed) |
| Exp 4 (transfer) | `transfer_scores` + saturation checks | Raw heatmaps |
| Exp 3 (layers) | CKA / silhouette / alignment curves | t-SNE panels (qualitative) |

## Success criteria (fill before final runs)

- **Transfer**: e.g. ≥70% of **pre-listed** off-diagonal pairs above threshold T (default project note: 0.7 in code comments — justify T in paper).
- **Null control**: Random-direction condition D mean deletion rate **below** A/B/C by a margin; document `n_random_controls` and seed.
- **Saturation**: Mean deletion matrix < 0.9 (or alpha reduced per calibration); see Experiment 4 warnings.
- **Underpowered guard (Exp2)**: If `pivot_index` is undefined for a pair, treat as underpowered (not null).
  Final confirmatory operating point should be selected by a fixed grid rule (maximize defined pairs, then
  maximize English-vs-random margin, then smallest alpha) and logged in `results/json/exp2_sensitivity_*.json`.

## Non-goals

- Universal claims over all languages or all concepts.
- Causal claims about training data or human cognition beyond “model behavior under intervention.”

## Versioning

- Tie each submission to a **run manifest** JSON (see [journal/run_manifest.py](../journal/run_manifest.py)) and frozen stimuli/vector checksums ([journal/validate_claims.py](../journal/validate_claims.py)).
