# SACRED Claim-to-Evidence Map

This file is the canonical bridge between manuscript claims and generated artifacts.
Use it to prevent narrative drift and to simplify reviewer/auditor verification.

## Confirmatory claims

| Claim ID | Claim text | Primary metric/test | Multiple-comparison rule | Artifact source |
|---|---|---|---|---|
| H_cross | Structured concept vectors outperform random controls in suppressing concept signal | Exp2 condition C vs D and Exp4 transfer statistics | FDR where p-values are reported | `results/json/exp2_pivot_*.json`, `results/*/json/exp4_transfer_summary_*.json` |
| H_transfer | Cross-lingual transfer exceeds pre-registered threshold on pre-listed pairs | `transfer_scores` from Exp4 | Threshold fixed in prereg | `results/*/json/exp4_transfer_summary_*.json` |
| H_geom | Geometric alignment tracks causal transfer in intervention layers (secondary) | Exp3 geometry summary + layer-wise transfer comparison | Exploratory unless promoted before final freeze | `results/json/exp3_concept_geometry_summary.json`, `results/json/exp3_layer_wise.json` |

## Fixed analysis conventions

- Primary Exp2 endpoint: continuous concept-probability suppression.
- `pivot_index` is diagnostic and flagged underpowered when A/B deltas are too small.
- Exp4 reports both relative hub score and absolute hub/ceiling diagnostics.
- All runs must include `run_manifest` in output JSON.

## Camera-ready checklist linkage

1. Build aggregate manuscript tables:
   - `python -m journal.build_paper_artifacts --results-dir results`
2. Validate checksums and JSON structure:
   - `python -m journal.validate_claims --light-only --stimuli outputs/stimuli/kinship_pairs.json --vectors-glob 'outputs/vectors/kinship_*.pt'`
3. Include generated index:
   - `results/paper/artifact_index.json`
