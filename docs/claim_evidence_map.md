# SACRED Claim-to-Evidence Map

This file is the canonical bridge between manuscript claims and generated artifacts.
Use it to prevent narrative drift and to simplify reviewer/auditor verification.

## Frozen run basis (paper primary)

- **Primary Exp2 run:** `logs/sacred_exp2_pivot_11340243.out` (alpha=0.6)
- **Primary Exp4 run:** `logs/sacred_exp4_transfer_11340547.out` (alpha=0.6, `output_lang=eng_Latn`)
- **Historical context only (appendix/background):** alpha=0.25 runs

## Confirmatory claims

| Claim ID | Claim text | Primary metric/test | Multiple-comparison rule | Frozen artifact source |
|---|---|---|---|---|
| H_cross | Structured concept vectors outperform random controls in suppressing concept signal | Exp2 continuous suppression with Condition D separation; Exp4 transfer diagnostics | FDR where p-values are reported | `results/json/exp2_pivot_{sacred,kinship}_{mean,pca}.json` (from run 11340243), `results/eng_Latn/json/exp4_transfer_summary_{sacred,kinship}_{mean,pca}.json` (from run 11340547) |
| H_transfer | Cross-lingual transfer exceeds pre-registered threshold on pre-listed pairs | `transfer_scores` from Exp4, plus pass-rate and hub diagnostics | Threshold fixed in prereg | `results/eng_Latn/json/exp4_transfer_summary_{sacred,kinship}_{mean,pca}.json`, `results/paper/table_exp4_transfer_summary.csv` |
| H_geom | Geometric alignment tracks causal transfer in intervention layers (secondary) | Exp3 geometry summary + layer-wise transfer comparison | Exploratory unless promoted before final freeze | `results/json/exp3_concept_geometry_summary.json`, `results/json/exp3_layer_wise.json`, `results/figures/exp3_cka_curves.png`, `results/figures/cross_lingual_probe_transfer_sacred_layer12.png` |

## Fixed analysis conventions

- Main narrative uses **alpha=0.6** frozen runs; alpha=0.25 values are appendix/background.
- Primary Exp2 endpoint: continuous concept-probability suppression.
- `pivot_index` is diagnostic and flagged underpowered (NaN) when A/B deltas are too small.
- Exp4 primary interpretation is English-target only (`output_lang=eng_Latn`).
- Exp4 must report both relative hub score and absolute hub/ceiling diagnostics.
- All runs must include `run_manifest` in output JSON.

## Camera-ready checklist linkage

1. Build aggregate manuscript tables:
   - `python -m journal.build_paper_artifacts --results-dir results`
2. Validate checksums and JSON structure:
   - `python -m journal.validate_claims --light-only --stimuli outputs/stimuli/kinship_pairs.json --vectors-glob 'outputs/vectors/kinship_*.pt'`
3. Freeze claim-to-artifact mapping against primary runs:
   - verify cited Exp2/Exp4 JSON correspond to `11340243` / `11340547`
   - verify Exp4 citations use `results/eng_Latn/json/...` paths
4. Include generated index:
   - `results/paper/artifact_index.json`
