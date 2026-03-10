"""
Transfer matrix and pivot diagnosis visualizations (Experiments 2 & 4).

  - plot_transfer_heatmap: NxN concept transfer heatmap with asymmetry highlight
  - plot_pivot_diagnosis: grouped bar chart per pivot condition (binary metric)
  - plot_pivot_diagnosis_continuous: grouped bar chart using P(concept) continuous metric
  - plot_pivot_index_summary: pivot index across all pairs / domains
  - plot_transfer_comparison: sacred vs kinship cross-lingual transfer scores
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"


def plot_transfer_heatmap(
    matrix: np.ndarray,
    languages: List[str],
    save_path: Optional[str] = None,
    title: str = "Cross-Lingual Concept Transfer Matrix",
):
    """
    NxN heatmap of concept deletion rates.

    Rows = source vector language; columns = target sentence language.
    Off-diagonal asymmetry is highlighted by annotating asymmetric pairs.

    Args:
        matrix: [N, N] deletion rate array (from compute_transfer_matrix)
        languages: Language code labels
        save_path: Optional output path
        title: Plot title
    """
    lang_labels = [l.split("_")[0] for l in languages]
    N = len(languages)

    fig, ax = plt.subplots(figsize=(max(6, N * 1.5), max(5, N * 1.3)))
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(lang_labels, rotation=45, ha="right")
    ax.set_yticklabels(lang_labels)
    ax.set_xlabel("Target Sentence Language")
    ax.set_ylabel("Source Vector Language")
    ax.set_title(title, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Concept Deletion Rate", rotation=270, labelpad=20)

    # Annotate cells
    for i in range(N):
        for j in range(N):
            val = matrix[i, j]
            txt_color = "white" if val > 0.6 else "black"
            asymmetry = abs(val - matrix[j, i]) if i != j else 0.0
            annotation = f"{val:.2f}"
            if i != j and asymmetry > 0.15:
                annotation += "*"    # flag highly asymmetric pairs
            ax.text(j, i, annotation, ha="center", va="center",
                    color=txt_color, fontsize=max(7, 10 - N))

    plt.tight_layout()
    _save_or_show(save_path, "Transfer heatmap")


def plot_pivot_diagnosis(
    results_by_pair: Dict[tuple, Dict],
    save_path: Optional[str] = None,
):
    """
    Grouped bar chart showing binary deletion rates for each pivot condition.

    Note: this chart saturates when alpha is too high. Prefer
    plot_pivot_diagnosis_continuous() as the primary output.

    Args:
        results_by_pair: {(src_lang, tgt_lang): run_pivot_diagnosis() output}
        save_path: Optional output path
    """
    conditions = ["condition_A_source", "condition_B_target",
                  "condition_C_english", "condition_D_random"]
    condition_labels = ["A: Source\nvector", "B: Target\nvector",
                        "C: English\n(pivot test)", "D: Random\n(control)"]
    colors = ["coral", "lightgreen", "steelblue", "gray"]

    pairs = list(results_by_pair.keys())
    n_pairs = len(pairs)
    n_conditions = len(conditions)

    fig, axes = plt.subplots(1, max(1, n_pairs), figsize=(4 * max(1, n_pairs), 5))
    if n_pairs == 1:
        axes = [axes]

    for ax, pair in zip(axes, pairs):
        result = results_by_pair[pair]
        deletion_rates = [result[c]["deletion_rate"] for c in conditions]

        x = np.arange(n_conditions)
        bars = ax.bar(x, deletion_rates, color=colors, alpha=0.8, edgecolor="black")

        # Baseline line
        if "baseline" in result:
            base = 1.0 - result["baseline"].get("deletion_rate", 0.0)
            ax.axhline(y=base, color="black", linestyle=":", linewidth=1, label="Baseline")

        # Pivot index annotation
        pivot_idx = result.get("pivot_index", float("nan"))
        ax.set_title(
            f"{pair[0].split('_')[0]} → {pair[1].split('_')[0]}\npivot_index={pivot_idx:.2f}",
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(condition_labels, fontsize=8)
        ax.set_ylim([0, 1.15])
        ax.grid(axis="y", alpha=0.3)

        for bar, rate in zip(bars, deletion_rates):
            label_y = min(bar.get_height() + 0.02, 1.12)
            ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                    f"{rate:.2f}", ha="center", va="bottom", fontsize=8)

    axes[0].set_ylabel("Concept Deletion Rate")
    fig.suptitle("Pivot Language Diagnosis: Deletion Rates by Condition", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(save_path, "Pivot diagnosis plot")


def plot_pivot_diagnosis_continuous(
    results_by_pair: Dict[tuple, Dict],
    save_path: Optional[str] = None,
):
    """
    Grouped bar chart showing P(concept) for each pivot condition (continuous metric).

    The continuous metric avoids the saturation problem: even when binary
    deletion rate saturates at 1.0, P(concept) still varies meaningfully.
    This is the primary output for Experiment 2.

    Bars show raw P(concept) per condition; lower = more concept suppression.

    Args:
        results_by_pair: {(src_lang, tgt_lang): run_pivot_diagnosis() output}
        save_path: Optional output path
    """
    conditions = ["baseline", "condition_A_source", "condition_B_target",
                  "condition_C_english", "condition_D_random"]
    condition_labels = ["Baseline", "A: Source\nvector", "B: Target\nvector",
                        "C: English\n(pivot test)", "D: Random\n(control)"]
    colors = ["lightgray", "coral", "lightgreen", "steelblue", "gray"]

    pairs = list(results_by_pair.keys())
    n_pairs = len(pairs)

    # Compute a single y-limit from all pairs before creating any axes.
    # sharey=True is avoided because pairs can have very different scales;
    # iterating set_ylim with sharey would let the last subplot overwrite the
    # shared limit, clipping bars in other subplots and pushing annotations
    # out of bounds (which savefig bbox=tight then expands into a huge figure).
    all_probs = [results_by_pair[p][c]["mean_prob"]
                 for p in pairs for c in conditions]
    global_max = max(all_probs) if all_probs else 1.0
    global_ylim = global_max * 1.3 + 0.001
    label_offset = global_ylim * 0.02       # fixed offset relative to axis height

    fig, axes = plt.subplots(1, max(1, n_pairs), figsize=(4 * max(1, n_pairs), 5))
    if n_pairs == 1:
        axes = [axes]

    for ax, pair in zip(axes, pairs):
        result = results_by_pair[pair]
        probs = [result[c]["mean_prob"] for c in conditions]

        x = np.arange(len(conditions))
        bars = ax.bar(x, probs, color=colors, alpha=0.8, edgecolor="black")

        pivot_idx = result.get("pivot_index", float("nan"))
        ax.set_title(
            f"{pair[0].split('_')[0]} → {pair[1].split('_')[0]}\npivot_index={pivot_idx:.2f}",
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(condition_labels, fontsize=8)
        ax.set_ylim([0, global_ylim])
        ax.grid(axis="y", alpha=0.3)

        for bar, prob in zip(bars, probs):
            # Clamp label inside the axes to prevent tight-bbox expansion
            label_y = min(bar.get_height() + label_offset, global_ylim * 0.97)
            ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                    f"{prob:.3f}", ha="center", va="bottom", fontsize=8)

    axes[0].set_ylabel("Mean P(concept token) in output")
    fig.suptitle(
        "Pivot Language Diagnosis: Concept Probability by Condition (Continuous Metric)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save_or_show(save_path, "Pivot diagnosis continuous plot")


def plot_pivot_index_summary(
    results_by_domain: Dict[str, Dict[tuple, Dict]],
    save_path: Optional[str] = None,
):
    """
    Summary chart of pivot_index values across all translation pairs and domains.

    pivot_index > 1.0 → English vector as effective as source/target (pivot evidence)
    pivot_index ≈ 1.0 → True interlingua (all vectors equally effective)
    pivot_index < 1.0 → Source/target privileged, no English pivot

    Args:
        results_by_domain: {domain: {(src, tgt): run_pivot_diagnosis() output}}
        save_path: Optional output path
    """
    all_pairs = sorted({pair for domain_results in results_by_domain.values()
                        for pair in domain_results})
    pair_labels = [f"{p[0].split('_')[0]}→{p[1].split('_')[0]}" for p in all_pairs]

    domains = list(results_by_domain.keys())
    x = np.arange(len(all_pairs))
    width = 0.8 / max(len(domains), 1)
    domain_colors = ["steelblue", "coral", "mediumseagreen", "mediumpurple"]

    fig, ax = plt.subplots(figsize=(max(8, len(all_pairs) * 1.4), 5))

    for d_idx, domain in enumerate(domains):
        domain_results = results_by_domain[domain]
        pivot_indices = []
        for pair in all_pairs:
            if pair in domain_results:
                pi = domain_results[pair].get("pivot_index", float("nan"))
                # Replace NaN with 0 for plotting
                pivot_indices.append(pi if pi == pi else 0.0)
            else:
                pivot_indices.append(0.0)
        offset = (d_idx - len(domains) / 2 + 0.5) * width
        ax.bar(x + offset, pivot_indices, width, label=domain.title(),
               color=domain_colors[d_idx % len(domain_colors)], alpha=0.8, edgecolor="black")

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5,
               label="Pivot index = 1.0 (interlingua boundary)")
    ax.axhspan(0.8, 1.2, alpha=0.08, color="gray", label="Near-interlingua range (0.8–1.2)")

    ax.set_xlabel("Translation Pair")
    ax.set_ylabel("Pivot Index  [effect(English) / mean(src, tgt)]")
    ax.set_title("Pivot Language Hypothesis: Pivot Index Across Pairs and Domains",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=45, ha="right")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_or_show(save_path, "Pivot index summary")


def plot_transfer_comparison(
    sacred_matrix: np.ndarray,
    kinship_matrix: np.ndarray,
    languages: list,
    save_path: Optional[str] = None,
):
    """
    Comparison bar chart: cross-lingual transfer score for sacred vs kinship.

    transfer_score = off_diagonal_cell / diagonal_cell  (normalises by same-lang effect)

    Values above 0.7 meet the proposal's success criterion for universal concepts.

    Args:
        sacred_matrix: [N, N] deletion rate array for sacred domain
        kinship_matrix: [N, N] deletion rate array for kinship domain
        languages: Language labels (length N)
        save_path: Optional output path
    """
    def _transfer_scores(matrix):
        N = matrix.shape[0]
        scores = {}
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diag = matrix[i, i]
                off = matrix[i, j]
                score = off / diag if diag > 1e-6 else 0.0
                label = f"{languages[i].split('_')[0]}→{languages[j].split('_')[0]}"
                scores[label] = score
        return scores

    sacred_scores = _transfer_scores(sacred_matrix)
    kinship_scores = _transfer_scores(kinship_matrix)
    all_labels = sorted(set(sacred_scores) | set(kinship_scores))

    x = np.arange(len(all_labels))
    width = 0.35
    sacred_vals = [sacred_scores.get(l, 0.0) for l in all_labels]
    kinship_vals = [kinship_scores.get(l, 0.0) for l in all_labels]

    fig, ax = plt.subplots(figsize=(max(10, len(all_labels) * 1.2), 5))
    ax.bar(x - width / 2, sacred_vals, width, label="Sacred", color="coral",
           alpha=0.8, edgecolor="black")
    ax.bar(x + width / 2, kinship_vals, width, label="Kinship", color="steelblue",
           alpha=0.8, edgecolor="black")

    ax.axhline(y=0.7, color="green", linestyle="--", linewidth=1.5, alpha=0.7,
               label="70% cross-lingual transfer threshold (success criterion)")
    ax.set_xlabel("Language Pair (Source Vector → Target Sentences)")
    ax.set_ylabel("Cross-Lingual Transfer Score  (off-diag / diagonal)")
    ax.set_title("Cross-Lingual Transfer: Sacred vs Kinship Concept Domains",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha="right")
    ax.set_ylim([0, max(max(sacred_vals, default=0), max(kinship_vals, default=0)) * 1.25 + 0.05])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_or_show(save_path, "Transfer comparison")


def _save_or_show(save_path: Optional[str], label: str):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"{label} saved to {save_path}")
    else:
        plt.show()
    plt.close()
