"""
Post-hoc p-value computation for Experiment 2 (H_cross).

Reads existing exp2_pivot JSON files and computes:
  - Per-pair empirical p-value: fraction of random control trials that
    equal or exceed condition C's deletion_rate (one-sided, H_cross).
  - Per-pair Cohen's d: effect size of C relative to the D null distribution.
  - Same two metrics for mean_prob (continuous metric; direction reversed).
  - FDR correction across all per-pair tests via Benjamini-Hochberg.
  - Across-pairs aggregate permutation test on (C - D_mean) differences.
  - Bootstrap CI on the mean (C - D_mean) difference.

No model inference is performed — all inputs come from saved JSON.

Usage:
    python analysis/exp2_pvalues.py [--results-dir results] [--out results/json/exp2_pvalues.json]

Important:
    This script does not create random-control trials; it consumes whatever is in
    exp2_pivot_*.json. To use 100 random controls, re-run Exp2 with:
      --n-random-controls 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analysis.journal_stats import bootstrap_ci_mean, fdr_correct_pvalues
from analysis.statistical import permutation_test


def _format_group_label(domain: str | None, vector_method: str | None) -> str:
    d = domain if domain is not None else "all_domains"
    v = vector_method if vector_method is not None else "all_methods"
    return f"{d}__{v}"


def _print_aggregate_block(title: str, aggregate: dict) -> None:
    agg_dr = aggregate["deletion_rate_diff_C_minus_D"]
    agg_prob = aggregate["mean_prob_diff_D_minus_C"]
    eng_vs = aggregate["english_vs_source_target"]
    print(f"\n  [{title}]")
    print(f"    C−D deletion: mean={agg_dr['mean']:+.3f}  p={agg_dr['permutation_p']:.3f}  n={agg_dr['n']}")
    print(f"    D−C mean_prob: mean={agg_prob['mean']:+.3f}  p={agg_prob['permutation_p']:.3f}  n={agg_prob['n']}")
    print("    English vs structured vectors (positive => English more suppressive):")
    print(f"      DR  C−A:      mean={eng_vs['deletion_rate_C_minus_A']['mean']:+.3f}  p={eng_vs['deletion_rate_C_minus_A']['permutation_p']:.3f}")
    print(f"      DR  C−B:      mean={eng_vs['deletion_rate_C_minus_B']['mean']:+.3f}  p={eng_vs['deletion_rate_C_minus_B']['permutation_p']:.3f}")
    print(f"      DR  C−meanAB: mean={eng_vs['deletion_rate_C_minus_meanAB']['mean']:+.3f}  p={eng_vs['deletion_rate_C_minus_meanAB']['permutation_p']:.3f}")
    print(f"      Pr  A−C:      mean={eng_vs['mean_prob_A_minus_C']['mean']:+.3f}  p={eng_vs['mean_prob_A_minus_C']['permutation_p']:.3f}")
    print(f"      Pr  B−C:      mean={eng_vs['mean_prob_B_minus_C']['mean']:+.3f}  p={eng_vs['mean_prob_B_minus_C']['permutation_p']:.3f}")
    print(f"      Pr  meanAB−C: mean={eng_vs['mean_prob_meanAB_minus_C']['mean']:+.3f}  p={eng_vs['mean_prob_meanAB_minus_C']['permutation_p']:.3f}")


# --------------------------------------------------------------------------- #
# Per-pair computation                                                          #
# --------------------------------------------------------------------------- #

def _per_pair_stats(pair_result: dict) -> dict:
    """
    Compute per-pair statistics for condition C vs. D random trials.

    Empirical p-values:
      - deletion_rate: one-sided p = fraction of D trials with DR >= C_DR
        (tests "C suppresses the concept more than random")
      - mean_prob: one-sided p = fraction of D trials with prob <= C_prob
        (lower prob = more suppression, so same directional test)

    Cohen's d uses the saved D trials as the reference distribution.
    """
    trials = pair_result.get("condition_D_random_trials", [])
    if not trials:
        nan = float("nan")
        return {
            "deletion_rate": {"empirical_p": nan, "cohens_d": nan, "c_val": nan, "d_mean": nan, "d_std": nan},
            "mean_prob":     {"empirical_p": nan, "cohens_d": nan, "c_val": nan, "d_mean": nan, "d_std": nan},
            "n_random_trials": 0,
        }

    d_dr   = np.array([t["deletion_rate"] for t in trials], dtype=float)
    d_prob = np.array([t["mean_prob"]     for t in trials], dtype=float)

    c_dr   = float(pair_result["condition_C_english"]["deletion_rate"])
    c_prob = float(pair_result["condition_C_english"]["mean_prob"])
    a_dr   = float(pair_result["condition_A_source"]["deletion_rate"])
    b_dr   = float(pair_result["condition_B_target"]["deletion_rate"])
    a_prob = float(pair_result["condition_A_source"]["mean_prob"])
    b_prob = float(pair_result["condition_B_target"]["mean_prob"])

    # Empirical p-values (one-sided)
    p_dr   = float(np.mean(d_dr   >= c_dr))    # DR: higher = more suppression
    p_prob = float(np.mean(d_prob <= c_prob))   # prob: lower = more suppression

    # Cohen's d: standardized distance of C from the D null distribution.
    # d = (c_val - D_mean) / D_std  (positive = C more suppressive than average random)
    # For mean_prob the sign is flipped: lower prob = more suppression, so d is negated.
    d_dr_std   = float(d_dr.std(ddof=1))   if len(d_dr)   > 1 else 0.0
    d_prob_std = float(d_prob.std(ddof=1)) if len(d_prob) > 1 else 0.0
    d_dr_cohens   = (c_dr   - float(d_dr.mean()))   / d_dr_std   if d_dr_std   > 0 else float("nan")
    d_prob_cohens = (float(d_prob.mean()) - c_prob) / d_prob_std if d_prob_std > 0 else float("nan")

    return {
        "deletion_rate": {
            "empirical_p": p_dr,
            "cohens_d":    d_dr_cohens,
            "c_val":       c_dr,
            "a_val":       a_dr,
            "b_val":       b_dr,
            "d_mean":      float(d_dr.mean()),
            "d_std":       float(d_dr.std(ddof=1)) if len(d_dr) > 1 else 0.0,
        },
        "mean_prob": {
            "empirical_p": p_prob,
            "cohens_d":    d_prob_cohens,
            "c_val":       c_prob,
            "a_val":       a_prob,
            "b_val":       b_prob,
            "d_mean":      float(d_prob.mean()),
            "d_std":       float(d_prob.std(ddof=1)) if len(d_prob) > 1 else 0.0,
        },
        "n_random_trials": len(trials),
    }


# --------------------------------------------------------------------------- #
# Aggregate across-pairs test                                                   #
# --------------------------------------------------------------------------- #

def _aggregate_stats(per_pair_rows: list[dict], seed: int = 42) -> dict:
    """
    Test whether C is systematically better than random across all pairs.

    Uses C_deletion_rate - D_mean_deletion_rate as the per-pair effect.
    Permutation test: compare effects vs. a zero null (two-sided).
    Bootstrap CI on the mean effect.
    """
    diffs = np.array(
        [r["deletion_rate"]["c_val"] - r["deletion_rate"]["d_mean"]
         for r in per_pair_rows
         if not np.isnan(r["deletion_rate"]["c_val"])],
        dtype=float,
    )
    diffs_prob = np.array(
        [r["mean_prob"]["d_mean"] - r["mean_prob"]["c_val"]   # positive = C more suppressive
         for r in per_pair_rows
         if not np.isnan(r["mean_prob"]["c_val"])],
        dtype=float,
    )

    def _agg(arr: np.ndarray, label: str) -> dict[str, Any]:
        if arr.size == 0:
            nan = float("nan")
            return {"mean": nan, "ci_low": nan, "ci_high": nan, "permutation_p": nan, "n": 0}
        mean, lo, hi = bootstrap_ci_mean(arr, seed=seed)
        # permutation_test compares two groups; compare diffs vs. zeros null
        p = permutation_test(arr, np.zeros(len(arr)), n_permutations=10_000)
        return {"mean": mean, "ci_low": lo, "ci_high": hi, "permutation_p": p, "n": int(arr.size)}

    # English vector efficacy vs. structured language vectors.
    # Positive values mean English (C) is more suppressive.
    c_minus_a_dr = np.array(
        [r["deletion_rate"]["c_val"] - r["deletion_rate"]["a_val"] for r in per_pair_rows],
        dtype=float,
    )
    c_minus_b_dr = np.array(
        [r["deletion_rate"]["c_val"] - r["deletion_rate"]["b_val"] for r in per_pair_rows],
        dtype=float,
    )
    c_minus_ab_dr = np.array(
        [r["deletion_rate"]["c_val"] - (r["deletion_rate"]["a_val"] + r["deletion_rate"]["b_val"]) / 2.0
         for r in per_pair_rows],
        dtype=float,
    )

    # For mean_prob, lower value means more suppression, so flip signs to keep
    # "positive means C more suppressive" convention.
    a_minus_c_prob = np.array(
        [r["mean_prob"]["a_val"] - r["mean_prob"]["c_val"] for r in per_pair_rows],
        dtype=float,
    )
    b_minus_c_prob = np.array(
        [r["mean_prob"]["b_val"] - r["mean_prob"]["c_val"] for r in per_pair_rows],
        dtype=float,
    )
    ab_minus_c_prob = np.array(
        [((r["mean_prob"]["a_val"] + r["mean_prob"]["b_val"]) / 2.0) - r["mean_prob"]["c_val"]
         for r in per_pair_rows],
        dtype=float,
    )

    return {
        "deletion_rate_diff_C_minus_D": _agg(diffs, "dr"),
        "mean_prob_diff_D_minus_C":     _agg(diffs_prob, "prob"),
        "english_vs_source_target": {
            "deletion_rate_C_minus_A": _agg(c_minus_a_dr, "c_minus_a_dr"),
            "deletion_rate_C_minus_B": _agg(c_minus_b_dr, "c_minus_b_dr"),
            "deletion_rate_C_minus_meanAB": _agg(c_minus_ab_dr, "c_minus_ab_dr"),
            "mean_prob_A_minus_C": _agg(a_minus_c_prob, "a_minus_c_prob"),
            "mean_prob_B_minus_C": _agg(b_minus_c_prob, "b_minus_c_prob"),
            "mean_prob_meanAB_minus_C": _agg(ab_minus_c_prob, "ab_minus_c_prob"),
        },
        "note": (
            "deletion_rate_diff > 0 means English vector deletes concept more than random. "
            "mean_prob_diff > 0 means English vector lowers concept probability more than random."
        ),
    }


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def compute_exp2_pvalues(
    results_dir: str = "results",
    out_path: str | None = None,
    seed: int = 42,
    min_random_trials: int = 100,
    strict_min_trials: bool = True,
) -> dict:
    results_dir_path = Path(results_dir)
    json_dir = results_dir_path / "json"

    # Discover all exp2 pivot JSON files
    json_files = sorted(json_dir.glob("exp2_pivot_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No exp2_pivot_*.json files found in {json_dir}")

    all_rows: list[dict] = []          # flat list for FDR
    by_file: dict[str, Any] = {}

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        domain        = data.get("metadata", {}).get("domain", json_file.stem)
        vector_method = data.get("metadata", {}).get("vector_method", "unknown")
        file_key = json_file.stem

        pairs_out: dict[str, Any] = {}
        for pair_str, pair_result in data.get("results_by_pair", {}).items():
            stats = _per_pair_stats(pair_result)
            if strict_min_trials and stats["n_random_trials"] < min_random_trials:
                raise ValueError(
                    f"{json_file.name} pair {pair_str} has {stats['n_random_trials']} random trials; "
                    f"need >= {min_random_trials}. Re-run Exp2 with --n-random-controls {min_random_trials}."
                )
            stats["pair"] = pair_str
            stats["domain"] = domain
            stats["vector_method"] = vector_method
            stats["file"] = file_key
            stats["pivot_index"] = pair_result.get("pivot_index")
            stats["pivot_index_continuous"] = pair_result.get("pivot_index_continuous")
            all_rows.append(stats)
            pairs_out[pair_str] = stats

        by_file[file_key] = {
            "domain": domain,
            "vector_method": vector_method,
            "n_pairs": len(pairs_out),
            "pairs": pairs_out,
        }

    # FDR correction across all per-pair deletion_rate empirical p-values
    raw_p_dr   = np.array([r["deletion_rate"]["empirical_p"] for r in all_rows], dtype=float)
    raw_p_prob = np.array([r["mean_prob"]["empirical_p"]     for r in all_rows], dtype=float)

    valid_mask_dr   = ~np.isnan(raw_p_dr)
    valid_mask_prob = ~np.isnan(raw_p_prob)

    q_dr   = np.full_like(raw_p_dr,   float("nan"))
    q_prob = np.full_like(raw_p_prob, float("nan"))
    reject_dr   = np.zeros(len(raw_p_dr),   dtype=bool)
    reject_prob = np.zeros(len(raw_p_prob), dtype=bool)

    if valid_mask_dr.any():
        rej, pcorr = fdr_correct_pvalues(raw_p_dr[valid_mask_dr])
        q_dr[valid_mask_dr]      = pcorr
        reject_dr[valid_mask_dr] = rej

    if valid_mask_prob.any():
        rej, pcorr = fdr_correct_pvalues(raw_p_prob[valid_mask_prob])
        q_prob[valid_mask_prob]      = pcorr
        reject_prob[valid_mask_prob] = rej

    # Attach FDR results back to rows
    fdr_rows = []
    for i, row in enumerate(all_rows):
        fdr_rows.append({
            "file":           row["file"],
            "domain":         row["domain"],
            "vector_method":  row["vector_method"],
            "pair":           row["pair"],
            "pivot_index":    row["pivot_index"],
            "n_random_trials": row["n_random_trials"],
            "deletion_rate": {
                **row["deletion_rate"],
                "q_fdr":   float(q_dr[i]),
                "reject_fdr": bool(reject_dr[i]),
            },
            "mean_prob": {
                **row["mean_prob"],
                "q_fdr":   float(q_prob[i]),
                "reject_fdr": bool(reject_prob[i]),
            },
        })

    aggregate = _aggregate_stats(all_rows, seed=seed)

    # Grouped aggregates for:
    # - combined (all domains, all methods)
    # - per vector_method (mean, pca)
    # - per domain (sacred, kinship)
    # - per (domain, vector_method)
    grouped_aggregates: dict[str, Any] = {}
    grouped_aggregates[_format_group_label(None, None)] = aggregate

    domains = sorted(set(r["domain"] for r in all_rows))
    methods = sorted(set(r["vector_method"] for r in all_rows))

    for method in methods:
        rows = [r for r in all_rows if r["vector_method"] == method]
        grouped_aggregates[_format_group_label(None, method)] = _aggregate_stats(rows, seed=seed)
    for domain in domains:
        rows = [r for r in all_rows if r["domain"] == domain]
        grouped_aggregates[_format_group_label(domain, None)] = _aggregate_stats(rows, seed=seed)
    for domain in domains:
        for method in methods:
            rows = [r for r in all_rows if r["domain"] == domain and r["vector_method"] == method]
            grouped_aggregates[_format_group_label(domain, method)] = _aggregate_stats(rows, seed=seed)

    # Print summary table
    print(f"\n{'='*90}")
    print(f"  EXP2 H_CROSS: condition C (English) vs. D (random)  —  per-pair empirical p-values")
    print(f"{'='*90}")
    header = f"{'File/Pair':<42} {'p_DR':>6} {'q_DR':>6} {'d_DR':>6} {'p_pr':>6} {'q_pr':>6} {'PI':>6}"
    print(header)
    print("-" * 90)
    for row in fdr_rows:
        label = f"{row['file'].replace('exp2_pivot_','')[:20]} | {row['pair'][:18]}"
        p_dr  = row["deletion_rate"]["empirical_p"]
        q_dr_ = row["deletion_rate"]["q_fdr"]
        d_dr  = row["deletion_rate"]["cohens_d"]
        p_pr  = row["mean_prob"]["empirical_p"]
        q_pr  = row["mean_prob"]["q_fdr"]
        pi    = row["pivot_index"]
        pi_s  = f"{pi:.2f}" if isinstance(pi, float) and not np.isnan(pi) else "nan"
        print(f"  {label:<40} {p_dr:>6.3f} {q_dr_:>6.3f} {d_dr:>6.2f} {p_pr:>6.3f} {q_pr:>6.3f} {pi_s:>6}")

    _print_aggregate_block("combined (all domains, all methods)", grouped_aggregates[_format_group_label(None, None)])
    for method in methods:
        _print_aggregate_block(f"vector_method={method}", grouped_aggregates[_format_group_label(None, method)])
    for domain in domains:
        _print_aggregate_block(f"domain={domain}", grouped_aggregates[_format_group_label(domain, None)])
    for domain in domains:
        for method in methods:
            _print_aggregate_block(
                f"domain={domain}, vector_method={method}",
                grouped_aggregates[_format_group_label(domain, method)],
            )
    print(f"\n  Note: min achievable empirical p = 1/n_random_trials = "
          f"1/{all_rows[0]['n_random_trials'] if all_rows else '?'} "
          f"≈ {1/all_rows[0]['n_random_trials']:.3f}" if all_rows else "")
    print(f"{'='*90}\n")

    output = {
        "note": (
            "Empirical p-values are one-sided: fraction of random D trials >= C (deletion_rate) "
            "or <= C (mean_prob). Min achievable p = 1/n_random_trials. "
            "q_fdr is Benjamini-Hochberg corrected across all pairs and files."
        ),
        "n_tests_deletion_rate": int(valid_mask_dr.sum()),
        "n_tests_mean_prob":     int(valid_mask_prob.sum()),
        "per_file": by_file,
        "flat_rows_with_fdr": fdr_rows,
        "aggregate": aggregate,
        "grouped_aggregates": grouped_aggregates,
    }

    out = Path(out_path) if out_path else json_dir / "exp2_pvalues.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-hoc p-values for Exp2 H_cross claim")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default=None, help="Output JSON path (default: results/json/exp2_pvalues.json)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-random-trials", type=int, default=100,
                        help="Minimum required random-control trials per pair (default: 100)")
    parser.add_argument("--allow-fewer-trials", action="store_true",
                        help="Allow processing files with fewer than --min-random-trials")
    args = parser.parse_args()
    compute_exp2_pvalues(
        results_dir=args.results_dir,
        out_path=args.out,
        seed=args.seed,
        min_random_trials=args.min_random_trials,
        strict_min_trials=not args.allow_fewer_trials,
    )
