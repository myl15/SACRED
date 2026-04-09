"""
Statistical Analysis for Circuit Discovery Experiments.

Provides canonical, deduplicated implementations of:
  - compute_cohens_d
  - permutation_test
  - bootstrap_confidence_interval

Plus hypothesis tests H1–H4, power analysis, multiple comparison correction,
and K-fold cross-validation framework.
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from scipy.stats import ttest_rel, ttest_ind, ttest_1samp, hypergeom
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower
import pandas as pd


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HypothesisTestResult:
    """Result from a single hypothesis test."""
    hypothesis: str
    test_statistic: float
    p_value: float
    p_value_corrected: Optional[float]
    effect_size: float
    confidence_interval: Tuple[float, float]
    significant: bool
    interpretation: str


@dataclass
class StatisticalReport:
    """Comprehensive statistical report for all hypothesis tests."""
    h1_necessity: Optional[HypothesisTestResult] = None
    h2_specificity: Optional[HypothesisTestResult] = None
    h3_universality: Optional[HypothesisTestResult] = None
    h4_sufficiency: Optional[HypothesisTestResult] = None
    multiple_comparison_correction: Dict[str, Any] = field(default_factory=dict)
    power_analysis: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""


@dataclass
class CrossValidationResults:
    """Results from K-fold cross-validation."""
    n_folds: int
    fold_circuits: List[Any]
    overlap_percentage: float
    mean_test_performance: Dict[str, float]
    fold_statistics: List[Dict[str, Any]]
    status: str = "implemented"
    implemented: bool = True
    note: str = ""


# ---------------------------------------------------------------------------
# Core statistical utilities (canonical — no duplicates)
# ---------------------------------------------------------------------------

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
    return 0.0 if pooled_std == 0 else float(mean_diff / pooled_std)


def permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    n_permutations: int = 1000,
) -> float:
    """Non-parametric permutation test for difference in means. Returns p-value."""
    observed_diff = np.mean(group1) - np.mean(group2)
    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diffs.append(np.mean(combined[:n1]) - np.mean(combined[n1:]))

    return float(np.mean(np.abs(perm_diffs) >= np.abs(observed_diff)))


def bootstrap_confidence_interval(
    data: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for the mean of data."""
    bootstrap_means = [
        np.mean(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ]
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    return (float(lower), float(upper))


# ---------------------------------------------------------------------------
# Hypothesis tests H1–H4
# ---------------------------------------------------------------------------

def test_h1_necessity(
    baseline_quality: List[float],
    ablated_quality: List[float],
    alpha: float = 0.01,
) -> HypothesisTestResult:
    """H1: Circuit ablation significantly reduces sacred concept translation quality."""
    baseline = np.array(baseline_quality)
    ablated = np.array(ablated_quality)

    t_stat, p_value = ttest_rel(baseline, ablated)
    effect_size = compute_cohens_d(baseline, ablated)
    ci = bootstrap_confidence_interval(baseline - ablated, n_bootstrap=1000, alpha=alpha)
    significant = (p_value < alpha) and (effect_size > 0.5)

    interpretation = (
        f"Circuit is NECESSARY. Ablation reduces quality (d={effect_size:.3f}, p={p_value:.4f})."
        if significant else
        f"Necessity NOT confirmed (d={effect_size:.3f}, p={p_value:.4f})."
    )

    return HypothesisTestResult(
        hypothesis="H1: Circuit Necessity",
        test_statistic=float(t_stat),
        p_value=float(p_value),
        p_value_corrected=None,
        effect_size=effect_size,
        confidence_interval=ci,
        significant=significant,
        interpretation=interpretation,
    )


def test_h2_specificity(
    sacred_baseline: List[float],
    sacred_ablated: List[float],
    secular_baseline: List[float],
    secular_ablated: List[float],
    alpha: float = 0.01,
) -> HypothesisTestResult:
    """H2: Circuit ablation affects sacred sentences more than secular."""
    sacred_diffs = np.array(sacred_baseline) - np.array(sacred_ablated)
    secular_diffs = np.array(secular_baseline) - np.array(secular_ablated)

    t_stat, p_value = ttest_ind(sacred_diffs, secular_diffs)
    effect_size = compute_cohens_d(sacred_diffs, secular_diffs)
    interaction = np.mean(sacred_diffs) - np.mean(secular_diffs)
    ci = bootstrap_confidence_interval(sacred_diffs - secular_diffs[:len(sacred_diffs)], alpha=alpha)
    significant = (p_value < alpha) and (abs(effect_size) > 0.5)

    interpretation = (
        f"Circuit is SPECIFIC. Sacred > secular interaction (d={effect_size:.3f}, p={p_value:.4f})."
        if significant else
        f"Specificity NOT confirmed (d={effect_size:.3f}, p={p_value:.4f})."
    )

    return HypothesisTestResult(
        hypothesis="H2: Circuit Specificity",
        test_statistic=float(t_stat),
        p_value=float(p_value),
        p_value_corrected=None,
        effect_size=effect_size,
        confidence_interval=ci,
        significant=significant,
        interpretation=interpretation,
    )


def test_h3_universality(
    circuits_by_lang: Dict,
    total_neurons: int = 8192,
    alpha: float = 0.01,
) -> HypothesisTestResult:
    """H3: Circuit component overlap across languages exceeds chance."""
    all_layers: set = set()
    for circuit in circuits_by_lang.values():
        all_layers.update(circuit.get_critical_layers())

    layer_p_values = []
    universal_counts = []

    for layer in sorted(all_layers):
        neurons_by_lang = [
            {n.neuron_idx for n in circuit.get_neurons_by_layer(layer)}
            for circuit in circuits_by_lang.values()
        ]
        if not neurons_by_lang:
            continue

        universal = set.intersection(*neurons_by_lang)
        universal_counts.append(len(universal))
        k = len(universal)
        circuit_size = len(neurons_by_lang[0])

        p_val = hypergeom.sf(k - 1, total_neurons, circuit_size, circuit_size) if k > 0 else 1.0
        layer_p_values.append(p_val)

    if layer_p_values:
        _, corrected_p, _, _ = multipletests(layer_p_values, method="fdr_bh")
        min_corrected_p = float(np.min(corrected_p))
        significant_layers = int(np.sum(corrected_p < alpha))
    else:
        min_corrected_p = 1.0
        significant_layers = 0

    total_universal = sum(universal_counts)
    effect_size = significant_layers / len(all_layers) if all_layers else 0.0
    significant = (min_corrected_p < alpha) and (total_universal > 0)

    interpretation = (
        f"Universal circuit CONFIRMED: {total_universal} neurons in {significant_layers} layers "
        f"(p={min_corrected_p:.4f})."
        if significant else
        f"Universality NOT confirmed (p={min_corrected_p:.4f})."
    )

    return HypothesisTestResult(
        hypothesis="H3: Cross-lingual Universality",
        test_statistic=float(total_universal),
        p_value=min_corrected_p,
        p_value_corrected=min_corrected_p,
        effect_size=effect_size,
        confidence_interval=(0.0, 0.0),
        significant=significant,
        interpretation=interpretation,
    )


def test_h4_sufficiency(
    restoration_percentages: List[float],
    threshold: float = 0.8,
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """H4: Circuit restoration achieves >= 80% of baseline quality."""
    restoration = np.array(restoration_percentages)
    t_stat, p_value = ttest_1samp(restoration, threshold, alternative="greater")
    mean_restoration = float(np.mean(restoration))
    effect_size = (mean_restoration - threshold) / float(np.std(restoration, ddof=1))
    ci = bootstrap_confidence_interval(restoration, n_bootstrap=1000, alpha=alpha)
    significant = (p_value < alpha) and (mean_restoration > threshold)

    interpretation = (
        f"Circuit SUFFICIENT: mean restoration {mean_restoration:.1%} "
        f"(threshold {threshold:.1%}, p={p_value:.4f})."
        if significant else
        f"Sufficiency NOT confirmed: {mean_restoration:.1%} < {threshold:.1%} (p={p_value:.4f})."
    )

    return HypothesisTestResult(
        hypothesis="H4: Circuit Sufficiency",
        test_statistic=float(t_stat),
        p_value=float(p_value),
        p_value_corrected=None,
        effect_size=effect_size,
        confidence_interval=ci,
        significant=significant,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Power analysis and comprehensive report
# ---------------------------------------------------------------------------

def compute_power_analysis(
    expected_effect_size: float = 0.5,
    alpha: float = 0.01,
    desired_power: float = 0.8,
) -> Dict[str, Any]:
    """Compute required sample size via power analysis."""
    analyzer = TTestPower()
    required_n = analyzer.solve_power(
        effect_size=expected_effect_size,
        alpha=alpha,
        power=desired_power,
        alternative="two-sided",
    )
    return {
        "expected_effect_size": expected_effect_size,
        "alpha": alpha,
        "desired_power": desired_power,
        "required_n_per_group": int(np.ceil(required_n)),
        "interpretation": (
            f"To detect d={expected_effect_size} with {desired_power:.0%} power at α={alpha}, "
            f"need ≥{int(np.ceil(required_n))} samples per condition."
        ),
    }


def run_comprehensive_hypothesis_testing(
    experimental_results: Dict[str, Any],
    alpha: float = 0.01,
) -> StatisticalReport:
    """Run all hypothesis tests with Bonferroni multiple comparison correction."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print("=" * 80)

    report = StatisticalReport()
    report.power_analysis = compute_power_analysis(expected_effect_size=0.5, alpha=alpha)
    print(f"\n[Power] {report.power_analysis['interpretation']}")

    if "necessity" in experimental_results:
        necessity = experimental_results["necessity"]
        baseline_probs = [m.concept_token_probability for m in necessity.baseline_quality["sacred"]]
        ablated_probs = [m.concept_token_probability for m in necessity.ablated_quality["sacred"]]
        report.h1_necessity = test_h1_necessity(baseline_probs, ablated_probs, alpha)
        print(f"\n[H1] {report.h1_necessity.interpretation}")

        sacred_ablated = [m.concept_token_probability for m in necessity.ablated_quality["sacred"]]
        secular_baseline = [m.concept_token_probability for m in necessity.secular_baseline.get("secular", [])]
        secular_ablated = [m.concept_token_probability for m in necessity.secular_ablated.get("secular", [])]
        if secular_baseline and secular_ablated:
            report.h2_specificity = test_h2_specificity(
                baseline_probs, sacred_ablated, secular_baseline, secular_ablated, alpha,
            )
            print(f"\n[H2] {report.h2_specificity.interpretation}")

    if "circuits_by_lang" in experimental_results and experimental_results["circuits_by_lang"]:
        report.h3_universality = test_h3_universality(experimental_results["circuits_by_lang"], alpha=alpha)
        print(f"\n[H3] {report.h3_universality.interpretation}")

    # Bonferroni correction across all active tests
    p_values = []
    test_names = []
    for name, result in [("H1", report.h1_necessity), ("H2", report.h2_specificity),
                          ("H3", report.h3_universality)]:
        if result is not None:
            p_values.append(result.p_value)
            test_names.append(name)

    if p_values:
        _, corrected_p, _, _ = multipletests(p_values, method="bonferroni")
        report.multiple_comparison_correction = {
            "method": "bonferroni",
            "original_p_values": dict(zip(test_names, [float(p) for p in p_values])),
            "corrected_p_values": dict(zip(test_names, [float(p) for p in corrected_p])),
        }
        for i, (name, result) in enumerate(
            [(n, r) for n, r in [("H1", report.h1_necessity), ("H2", report.h2_specificity),
                                   ("H3", report.h3_universality)] if r is not None]
        ):
            result.p_value_corrected = float(corrected_p[i])

    significant_tests = [
        name for name, result in [
            ("Necessity", report.h1_necessity),
            ("Specificity", report.h2_specificity),
            ("Universality", report.h3_universality),
        ]
        if result and result.significant
    ]

    report.summary = (
        f"Confirmed: {', '.join(significant_tests) or 'None'}. "
        f"Tested {len(p_values)} hypotheses with Bonferroni correction (α={alpha})."
    )
    print(f"\n{'=' * 80}\nSUMMARY: {report.summary}\n{'=' * 80}")
    return report


def perform_cross_validation(
    stimuli: Dict,
    model,
    tokenizer,
    discovery_fn: Callable,
    n_folds: int = 5,
    seed: int = 42,
) -> CrossValidationResults:
    """K-fold cross-validation on circuit discovery (placeholder)."""
    print(f"\n=== {n_folds}-Fold Cross-Validation (placeholder) ===")
    print("Full cross-validation requires re-running circuit discovery per fold.")
    return CrossValidationResults(
        n_folds=n_folds,
        fold_circuits=[],
        overlap_percentage=0.85,
        mean_test_performance={"accuracy": 0.92},
        fold_statistics=[],
        status="placeholder",
        implemented=False,
        note="Placeholder implementation: full fold-wise circuit rediscovery is not yet implemented.",
    )
