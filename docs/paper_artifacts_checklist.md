# Paper-facing figures, tables, and appendix

Align outputs with [journal_claims_prereg.md](journal_claims_prereg.md) (confirmatory vs exploratory). This checklist is for submission packaging.

## Main text — suggested figure set

| Figure | Source | Notes |
|--------|--------|--------|
| Pivot / causal (primary) | `results/figures/exp2_pivot_*_continuous.png` | Pair with `statistics.pivot_index_*` in `results/json/exp2_pivot_*.json` |
| Transfer matrix | `results/<output_lang>/figures/exp4_transfer_matrix_*_calibrated.png` | Include saturation warning from console or `summary` JSON |
| Transfer vs baseline | `results/figures/exp4_transfer_comparison_sacred_vs_kinship_*.png` | When both domains run |
| Geometry (secondary) | `results/figures/exp3_cka_curves.png`, alignment / probe figures from exp3 | Pre-register H_geom if used |

## Tables

| Table | Content |
|-------|---------|
| Model & setup | `MODEL_NAME`, `INTERVENTION_LAYERS`, `VECTOR_SCALING_POLICY`, alpha, matching mode |
| Languages | `EXPERIMENT_LANGUAGES`, FLORES mapping note if exp3 uses FLORES+ |
| Stimuli | Paths under `outputs/stimuli/`, `n_per_concept` from exp1 manifest |

## Appendix

- Full **run manifests** embedded in experiment JSON (`run_manifest`) or sidecar `outputs/manifests/exp1_*.json`.
- **Hyperparameter sweep plans**: `results/journal/hyperparam_sweep_plan.json` (from `python -m journal.hyperparam_sweep`).
- **Ablations**: commands logged via `python -m journal.ablation_runner` (wrong-domain vectors, layer subsets, matching modes).
- **Bootstrap / multiple comparison**: `statistics` blocks in exp2/exp4 JSON; FDR applies only when reporting p-values (see `analysis/journal_stats.py`).
- **External validation**: checksum report from `python -m journal.validate_claims` with `--stimuli`, `--vectors-glob`, optional `--config`.

## Frozen “paper run” bundle (recommended)

1. Git tag + `run_manifest.git_commit` from exp1–exp4 outputs.
2. SHA-256 of `outputs/stimuli/*_pairs.json` and vector `.pt` files (validator CLI).
3. Optional frozen `config.json` listing alpha, layers, domains, vector method, matching mode, seeds.
4. Deterministic manuscript tables and checksum index:
   - `python -m journal.build_paper_artifacts --results-dir results`
   - Include `results/paper/artifact_index.json` in submission bundle.
5. Claim provenance map:
   - `docs/claim_evidence_map.md` must be up to date before camera-ready freeze.
