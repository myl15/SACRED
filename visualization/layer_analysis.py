"""
Layer-wise analysis visualizations (Experiment 3).

  - plot_cka_curves: CKA similarity across encoder layers for language pairs
  - plot_tsne_panels: t-SNE panels at selected layers
  - plot_umap_panels: UMAP panels at selected layers (complements t-SNE)
  - plot_english_centricity: English-centricity index across layers
  - plot_silhouette_trajectory: silhouette score vs layer
  - plot_concept_domain_comparison: side-by-side sacred vs kinship transfer bars
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


def plot_cka_curves(
    cka_by_pair_by_layer: Dict[tuple, Dict[int, float]],
    save_path: Optional[str] = None,
):
    """
    Line plot of CKA similarity across encoder layers for each language pair.

    Args:
        cka_by_pair_by_layer: {(lang_a, lang_b): {layer: cka_value}}
        save_path: Optional output path
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for pair, layer_cka in sorted(cka_by_pair_by_layer.items()):
        layers = sorted(layer_cka.keys())
        values = [layer_cka[l] for l in layers]
        label = f"{pair[0].split('_')[0]} ↔ {pair[1].split('_')[0]}"
        ax.plot(layers, values, marker="o", markersize=3, linewidth=1.5, label=label)

    ax.set_xlabel("Encoder Layer")
    ax.set_ylabel("CKA Similarity")
    ax.set_title("Cross-Lingual Representation Convergence (CKA)", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_or_show(save_path, "CKA curves")


def plot_tsne_panels(
    reps_by_lang: Dict[str, Dict[int, "torch.Tensor"]],
    panel_layers: List[int] = None,
    save_path: Optional[str] = None,
    seed: int = 42,
):
    """
    2×2 t-SNE panel at selected encoder layers.

    For meaningful t-SNE structure, each panel should have ≥100 points per
    language (use exp3 with FLORES-200 data, not the 4-sentence sample).

    Args:
        reps_by_lang: {lang: {layer: tensor[n_sentences, hidden_dim]}}
        panel_layers: Layers to show (default: 0, 8, 16, 23)
        save_path: Optional output path
        seed: Random seed for reproducibility
    """
    from sklearn.manifold import TSNE
    import torch

    if panel_layers is None:
        panel_layers = [0, 8, 16, 23]

    n = len(panel_layers)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    languages = sorted(reps_by_lang.keys())
    palette = sns.color_palette("tab10", len(languages))
    lang_color = dict(zip(languages, palette))

    for ax_idx, layer in enumerate(panel_layers):
        ax = axes[ax_idx]
        X_list, labels = [], []

        for lang in languages:
            if layer not in reps_by_lang.get(lang, {}):
                continue
            reps = reps_by_lang[lang][layer].float()
            X_list.append(reps.numpy())
            labels.extend([lang] * reps.shape[0])

        if not X_list:
            ax.axis("off")
            continue

        X = np.concatenate(X_list, axis=0)
        perplexity = min(30, max(5, len(labels) // 4 - 1))
        tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity)
        X_2d = tsne.fit_transform(X)

        for lang in languages:
            mask = [i for i, l in enumerate(labels) if l == lang]
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[lang_color[lang]], label=lang.split("_")[0],
                       s=20, alpha=0.7, edgecolors="none")

        ax.set_title(f"Layer {layer}", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        if ax_idx == 0:
            ax.legend(fontsize=7, markerscale=1.5, bbox_to_anchor=(0, 1.02), loc="lower left",
                      ncol=len(languages))

    # Hide unused axes
    for ax_idx in range(len(panel_layers), len(axes)):
        axes[ax_idx].axis("off")

    fig.suptitle("t-SNE Representation Panels by Layer", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(save_path, "t-SNE panels")


def plot_umap_panels(
    reps_by_lang: Dict[str, Dict[int, "torch.Tensor"]],
    panel_layers: List[int] = None,
    save_path: Optional[str] = None,
    seed: int = 42,
):
    """
    2×2 UMAP panel at selected encoder layers.

    UMAP preserves global structure better than t-SNE, making it easier to see
    whether language clusters merge (interlingua) or stay separated at late layers.

    Args:
        reps_by_lang: {lang: {layer: tensor[n_sentences, hidden_dim]}}
        panel_layers: Layers to show (default: 0, 8, 16, 23)
        save_path: Optional output path
        seed: Random seed for reproducibility
    """
    try:
        import umap as umap_module
    except ImportError:
        print("umap-learn not installed. Install with: pip install umap-learn")
        return

    if panel_layers is None:
        panel_layers = [0, 8, 16, 23]

    n = len(panel_layers)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    languages = sorted(reps_by_lang.keys())
    palette = sns.color_palette("tab10", len(languages))
    lang_color = dict(zip(languages, palette))

    for ax_idx, layer in enumerate(panel_layers):
        ax = axes[ax_idx]
        X_list, labels = [], []

        for lang in languages:
            if layer not in reps_by_lang.get(lang, {}):
                continue
            reps = reps_by_lang[lang][layer].float()
            X_list.append(reps.numpy())
            labels.extend([lang] * reps.shape[0])

        if not X_list:
            ax.axis("off")
            continue

        X = np.concatenate(X_list, axis=0)
        reducer = umap_module.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                                   random_state=seed)
        X_2d = reducer.fit_transform(X)

        for lang in languages:
            mask = [i for i, l in enumerate(labels) if l == lang]
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[lang_color[lang]], label=lang.split("_")[0],
                       s=20, alpha=0.7, edgecolors="none")

        ax.set_title(f"Layer {layer}", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        if ax_idx == 0:
            ax.legend(fontsize=7, markerscale=1.5, bbox_to_anchor=(0, 1.02), loc="lower left",
                      ncol=len(languages))

    for ax_idx in range(len(panel_layers), len(axes)):
        axes[ax_idx].axis("off")

    fig.suptitle("UMAP Representation Panels by Layer", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(save_path, "UMAP panels")


def plot_english_centricity(
    centricity_by_layer: Dict[int, float],
    save_path: Optional[str] = None,
):
    """
    Line plot of English-centricity index across encoder layers.

    Values < 1 indicate representations are closer to English than global mean.
    """
    layers = sorted(centricity_by_layer.keys())
    values = [centricity_by_layer[l] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, values, marker="o", markersize=5, linewidth=2, color="steelblue")
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label="Neutral (index=1.0)")
    ax.fill_between(layers, values, 1.0,
                    where=[v < 1.0 for v in values],
                    alpha=0.2, color="orange", label="English-centric region")
    ax.fill_between(layers, values, 1.0,
                    where=[v > 1.0 for v in values],
                    alpha=0.2, color="blue", label="Language-neutral region")
    ax.set_xlabel("Encoder Layer")
    ax.set_ylabel("English-Centricity Index")
    ax.set_title("English-Centricity of Cross-Lingual Representations", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_or_show(save_path, "English-centricity plot")


def plot_silhouette_trajectory(
    silhouette_by_layer: Dict[int, float],
    save_path: Optional[str] = None,
):
    """
    Line chart of silhouette score (language-cluster separation) across encoder layers.

    High score (→ 1.0) → representations strongly cluster by language.
    Low/negative → language boundaries blurred (consistent with interlingua).

    Args:
        silhouette_by_layer: {layer: silhouette_score}
        save_path: Optional output path
    """
    layers = sorted(silhouette_by_layer.keys())
    scores = [silhouette_by_layer[l] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, scores, marker="o", markersize=5, linewidth=2, color="darkorchid")
    ax.axhline(y=0.0, color="red", linestyle="--", linewidth=1.5,
               label="Score=0 (random clustering)")

    ax.fill_between(layers, scores, 0.0,
                    where=[s > 0 for s in scores],
                    alpha=0.15, color="darkorchid", label="Language-clustered (positive)")
    ax.fill_between(layers, scores, 0.0,
                    where=[s <= 0 for s in scores],
                    alpha=0.15, color="salmon", label="Language-mixed (negative)")

    ax.set_xlabel("Encoder Layer")
    ax.set_ylabel("Silhouette Score (cosine, language labels)")
    ax.set_title("Language Cluster Separation Across Encoder Layers", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    if scores:
        max_layer = layers[int(np.argmax(scores))]
        min_layer = layers[int(np.argmin(scores))]
        ax.annotate(f"max={max(scores):.3f}", (max_layer, max(scores)),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
        ax.annotate(f"min={min(scores):.3f}", (min_layer, min(scores)),
                    textcoords="offset points", xytext=(5, -12), fontsize=9)

    plt.tight_layout()
    _save_or_show(save_path, "Silhouette trajectory")


def plot_concept_domain_comparison(
    sacred_transfer: Dict[str, float],
    kinship_transfer: Dict[str, float],
    save_path: Optional[str] = None,
):
    """
    Side-by-side bar chart comparing sacred vs kinship cross-lingual transfer rates.

    Args:
        sacred_transfer: {lang_pair_label: deletion_rate}
        kinship_transfer: {lang_pair_label: deletion_rate}
        save_path: Optional output path
    """
    pairs = sorted(set(sacred_transfer.keys()) | set(kinship_transfer.keys()))
    x = np.arange(len(pairs))
    width = 0.35

    sacred_vals = [sacred_transfer.get(p, 0.0) for p in pairs]
    kinship_vals = [kinship_transfer.get(p, 0.0) for p in pairs]

    fig, ax = plt.subplots(figsize=(max(8, len(pairs) * 1.2), 5))
    ax.bar(x - width / 2, sacred_vals, width, label="Sacred", color="coral", alpha=0.8, edgecolor="black")
    ax.bar(x + width / 2, kinship_vals, width, label="Kinship", color="steelblue", alpha=0.8, edgecolor="black")

    ax.set_xlabel("Language Pair (Source → Target)")
    ax.set_ylabel("Concept Deletion Rate")
    ax.set_title("Concept Domain Transfer Comparison: Sacred vs Kinship", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=45, ha="right")
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_or_show(save_path, "Concept domain comparison")


def _save_or_show(save_path: Optional[str], label: str):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"{label} saved to {save_path}")
    else:
        plt.show()
    plt.close()
