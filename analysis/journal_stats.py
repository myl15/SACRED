"""
Journal-oriented statistics: bootstrap CIs and multiple-comparison correction.

Use alongside [analysis/statistical.py](statistical.py) for paper-facing uncertainty.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from statsmodels.stats.multitest import multipletests


def bootstrap_ci_mean(
    values: np.ndarray,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for the mean of ``values``.

    Returns:
        (point_estimate_mean, ci_low, ci_high)
    """
    x = np.asarray(values, dtype=float).ravel()
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = float(np.mean(x))
    if x.size == 1:
        return point, point, point
    boots = []
    for _ in range(n_bootstrap):
        sample = rng.choice(x, size=x.size, replace=True)
        boots.append(float(np.mean(sample)))
    boots_arr = np.array(boots)
    lo = float(np.percentile(boots_arr, 100 * alpha / 2))
    hi = float(np.percentile(boots_arr, 100 * (1 - alpha / 2)))
    return point, lo, hi


def fdr_correct_pvalues(
    p_values: np.ndarray,
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini–Hochberg FDR or other ``statsmodels`` multipletests methods.

    Returns:
        (reject, p_corrected) same shapes as input
    """
    p = np.asarray(p_values, dtype=float).ravel()
    if p.size == 0:
        return np.array([]), np.array([])
    reject, p_corr, _, _ = multipletests(p, alpha=alpha, method=method)
    return reject, p_corr


def summarize_transfer_scores_with_fdr(
    transfer_scores: dict,
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> dict:
    """
    Attach FDR-corrected p-values when raw p-values exist (optional extension).

    For scores that are not p-values, returns a note-only wrapper for reporting.
    """
    labels = list(transfer_scores.keys())
    scores = np.array([transfer_scores[k] for k in labels], dtype=float)
    return {
        "labels": labels,
        "scores": scores.tolist(),
        "note": "transfer_scores are ratios, not p-values — use bootstrap CIs per pair or pre-register primary pairs; FDR applies to p-values only.",
        "fdr_not_applicable": True,
    }
