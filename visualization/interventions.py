"""
Intervention visualization: necessity/sufficiency results and statistical summaries.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"


def plot_intervention_results(necessity_results, sufficiency_results=None, save_path: Optional[str] = None):
    """Boxplot comparison: baseline vs ablated, sacred vs secular specificity."""
    n_plots = 2 if sufficiency_results is None else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    axes = list(axes) if n_plots > 1 else [axes]

    baseline_probs = [m.concept_token_probability for m in necessity_results.baseline_quality["sacred"]]
    ablated_probs = [m.concept_token_probability for m in necessity_results.ablated_quality["sacred"]]

    axes[0].boxplot([baseline_probs, ablated_probs], positions=[1, 2], widths=0.6,
                    patch_artist=True, boxprops=dict(facecolor="lightblue", alpha=0.7),
                    medianprops=dict(color="red", linewidth=2))
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(["Baseline", "Ablated"])
    axes[0].set_ylabel("Concept Token Probability")
    axes[0].set_title(f"Necessity Test\n(d={necessity_results.effect_size:.3f}, p={necessity_results.p_value:.4f})")
    axes[0].grid(axis="y", alpha=0.3)

    if necessity_results.significant:
        y_max = max(max(baseline_probs), max(ablated_probs))
        axes[0].plot([1, 2], [y_max * 1.1, y_max * 1.1], "k-", linewidth=1)
        axes[0].text(1.5, y_max * 1.15, "**", ha="center", fontsize=16)

    # Specificity bar chart
    secular_baseline = [m.concept_token_probability for m in necessity_results.secular_baseline.get("secular", [])]
    secular_ablated = [m.concept_token_probability for m in necessity_results.secular_ablated.get("secular", [])]

    sacred_change = np.mean(baseline_probs) - np.mean(ablated_probs)
    secular_change = np.mean(secular_baseline) - np.mean(secular_ablated) if secular_baseline else 0.0

    axes[1].bar([0], [sacred_change], 0.35, label="Sacred", color="coral", alpha=0.7)
    axes[1].bar([1], [secular_change], 0.35, label="Secular", color="lightgreen", alpha=0.7)
    axes[1].set_ylabel("Quality Reduction (Δ)")
    axes[1].set_title("Specificity: Sacred vs Secular")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Sacred\nSentences", "Secular\nSentences"])
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    if n_plots == 3 and sufficiency_results:
        axes[2].bar([1], [sufficiency_results.restoration_percentage], 0.6, color="purple", alpha=0.7)
        axes[2].axhline(y=0.8, color="red", linestyle="--", label="80% threshold")
        axes[2].set_ylim([0, 1.1])
        axes[2].set_ylabel("Restoration Percentage")
        axes[2].set_title("Sufficiency Test")
        axes[2].set_xticks([1])
        axes[2].set_xticklabels(["Circuit\nRestoration"])
        axes[2].legend()
        axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save_or_show(save_path, "Intervention results")


def plot_statistical_summary(statistical_report, save_path: Optional[str] = None):
    """Horizontal bar chart of effect sizes and -log10(p) values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    tests = []
    effect_sizes = []
    colors = []
    p_values = []
    p_labels = []

    for label, result in [
        ("H1:\nNecessity", statistical_report.h1_necessity),
        ("H2:\nSpecificity", statistical_report.h2_specificity),
        ("H3:\nUniversality", statistical_report.h3_universality),
        ("H4:\nSufficiency", statistical_report.h4_sufficiency),
    ]:
        if result:
            tests.append(label)
            effect_sizes.append(result.effect_size)
            colors.append("green" if result.significant else "gray")
            p_values.append(result.p_value)
            p_labels.append(label.replace("\n", " "))

    ax1.barh(tests, effect_sizes, color=colors, alpha=0.7, edgecolor="black")
    ax1.axvline(x=0.5, color="orange", linestyle="--", label="Medium effect")
    ax1.axvline(x=0.8, color="red", linestyle="--", label="Large effect")
    ax1.set_xlabel("Effect Size (Cohen's d)")
    ax1.set_title("Hypothesis Test Effect Sizes")
    ax1.legend()
    ax1.grid(axis="x", alpha=0.3)

    log_p = [-np.log10(p) for p in p_values]
    colors_p = ["green" if p < 0.01 else "gray" for p in p_values]
    ax2.barh(p_labels, log_p, color=colors_p, alpha=0.7, edgecolor="black")
    ax2.axvline(x=-np.log10(0.05), color="orange", linestyle="--", label="α=0.05")
    ax2.axvline(x=-np.log10(0.01), color="red", linestyle="--", label="α=0.01")
    ax2.set_xlabel("-log₁₀(p-value)")
    ax2.set_title("Statistical Significance")
    ax2.legend()
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Statistical Test Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(save_path, "Statistical summary")


def plot_cross_validation(cv_results, save_path: Optional[str] = None):
    """Bar charts: circuit overlap and mean performance across CV folds."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar([1], [cv_results.overlap_percentage], 0.5, color="purple", alpha=0.7)
    ax1.axhline(y=0.8, color="red", linestyle="--", label="80% threshold")
    ax1.set_ylim([0, 1.1])
    ax1.set_ylabel("Circuit Overlap Percentage")
    ax1.set_title(f"{cv_results.n_folds}-Fold Cross-Validation")
    ax1.set_xticks([1])
    ax1.set_xticklabels(["Circuit\nReplication"])
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    if cv_results.mean_test_performance:
        metrics = list(cv_results.mean_test_performance.keys())
        values = list(cv_results.mean_test_performance.values())
        ax2.bar(range(len(metrics)), values, color="teal", alpha=0.7, edgecolor="black")
        ax2.set_xticks(range(len(metrics)))
        ax2.set_xticklabels(metrics, rotation=45, ha="right")
        ax2.set_ylabel("Score")
        ax2.set_title("Mean Test Performance")
        ax2.set_ylim([0, 1.1])
        ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save_or_show(save_path, "Cross-validation plot")


def _save_or_show(save_path: Optional[str], label: str):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"{label} saved to {save_path}")
    else:
        plt.show()
    plt.close()
