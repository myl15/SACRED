"""
Visualization: PCA reading vectors vs. mean-difference concept vectors.

Produces three plots (saved to results/pca_vs_mean/):

  1. layer_cosine_similarity.png
       Cosine similarity between PCA and mean vectors at each layer,
       one curve per language × concept combination.

  2. pca_explained_variance.png
       PC1 explained variance ratio across layers — shows how much of the
       pair-level variance is captured by the dominant direction. Higher values
       indicate a cleaner, more linearly-separable concept signal.

  3. per_pair_projections.png
       Scatter of per-pair projection scores onto PCA vs. mean vectors for a
       single (lang, concept, layer) slice. Reveals whether the two directions
       rank pairs consistently.

Usage (requires saved vectors from exp1, or pass precomputed diffs):

    python visualization/pca_vs_mean.py \
        --domain kinship \
        --layer 12 \
        --lang eng_Latn

Or call compare_pca_vs_mean() programmatically after running Experiment 1.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from config import DEFAULT_DEVICE, DEFAULT_LAYERS, EXPERIMENT_LANGUAGES, HF_CACHE_DIR, MODEL_NAME, VECTORS_DIR
from extraction.concept_vectors import _pca_reading_vector


RESULTS_DIR = Path("results/pca_vs_mean")


# ---------------------------------------------------------------------------
# Core comparison logic (no model required — works on saved diff matrices)
# ---------------------------------------------------------------------------

def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0).float(), b.unsqueeze(0).float()
    ).item()


def compute_layer_stats(
    diff_matrices: Dict[int, torch.Tensor],  # {layer: [n_pairs, hidden_dim]}
) -> Dict[str, Dict[int, float]]:
    """
    For each layer, compute:
      - cosine similarity between PCA and mean vectors
      - PC1 explained variance ratio
      - mean vector norm
      - PCA vector norm (should always be ~1.0)
    """
    stats: Dict[str, Dict[int, float]] = {
        "cosine_sim": {},
        "pc1_var_ratio": {},
        "mean_norm": {},
    }
    for layer, diff in diff_matrices.items():
        mean_vec = diff.float().mean(dim=0)
        pca_vec  = _pca_reading_vector(diff)

        stats["cosine_sim"][layer] = _cosine(pca_vec, mean_vec / (mean_vec.norm() + 1e-8))

        # PC1 explained variance ratio
        X = diff.float().numpy()
        if X.shape[0] >= 2:
            pca = PCA(n_components=min(X.shape[0], X.shape[1]))
            pca.fit(X)
            stats["pc1_var_ratio"][layer] = float(pca.explained_variance_ratio_[0])
        else:
            stats["pc1_var_ratio"][layer] = float("nan")

        stats["mean_norm"][layer] = mean_vec.norm().item()

    return stats


# ---------------------------------------------------------------------------
# Plot 1: Layer-wise cosine similarity
# ---------------------------------------------------------------------------

def plot_layer_cosine_similarity(
    all_stats: Dict[str, Dict[str, Dict[int, float]]],  # {label: stats}
    layers: List[int],
    output_path: Path,
):
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, stats in all_stats.items():
        cos_by_layer = [stats["cosine_sim"].get(l, float("nan")) for l in layers]
        ax.plot(layers, cos_by_layer, marker="o", markersize=4, label=label)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="Perfect agreement")
    ax.set_xlabel("Encoder Layer")
    ax.set_ylabel("Cosine Similarity (PCA vs. Mean)")
    ax.set_title("Layer-wise Agreement: PCA Reading Vector vs. Mean-Difference Vector")
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: PC1 explained variance ratio
# ---------------------------------------------------------------------------

def plot_pca_explained_variance(
    all_stats: Dict[str, Dict[str, Dict[int, float]]],
    layers: List[int],
    output_path: Path,
):
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, stats in all_stats.items():
        var_by_layer = [stats["pc1_var_ratio"].get(l, float("nan")) for l in layers]
        ax.plot(layers, var_by_layer, marker="s", markersize=4, label=label)

    ax.set_xlabel("Encoder Layer")
    ax.set_ylabel("PC1 Explained Variance Ratio")
    ax.set_title("Concept Signal Linearity: PC1 Explained Variance Ratio per Layer")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 3: Per-pair projection scatter for a single slice
# ---------------------------------------------------------------------------

def plot_per_pair_projections(
    diff: torch.Tensor,  # [n_pairs, hidden_dim]
    layer: int,
    label: str,
    output_path: Path,
):
    """Scatter per-pair scores on PCA vs. mean direction."""
    mean_vec = diff.float().mean(dim=0)
    mean_vec_n = mean_vec / (mean_vec.norm() + 1e-8)
    pca_vec = _pca_reading_vector(diff)

    pca_scores  = diff.float().mv(pca_vec).numpy()
    mean_scores = diff.float().mv(mean_vec_n).numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(mean_scores, pca_scores, alpha=0.7, s=40)

    # Regression line
    m, b = np.polyfit(mean_scores, pca_scores, 1)
    xs = np.linspace(mean_scores.min(), mean_scores.max(), 100)
    ax.plot(xs, m * xs + b, "r--", linewidth=1, label=f"slope={m:.2f}")

    corr = np.corrcoef(mean_scores, pca_scores)[0, 1]
    ax.set_xlabel("Projection onto Mean Vector")
    ax.set_ylabel("Projection onto PCA Vector")
    ax.set_title(f"Per-Pair Projections: {label} (layer {layer})\nr={corr:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# High-level entry point: compare from saved vectors
# ---------------------------------------------------------------------------

def compare_pca_vs_mean(
    domain: str = "kinship",
    languages: Optional[List[str]] = None,
    layers: Optional[List[int]] = None,
    scatter_layer: int = 12,
    scatter_lang: str = "eng_Latn",
    vectors_dir: str = VECTORS_DIR,
    output_dir: Optional[Path] = None,
):
    """
    Load diff matrices saved by exp1 and produce all three comparison plots.

    Requires outputs/vectors/{domain}_{lang}_diffs.pt (written by exp1 when
    return_diffs=True is passed to extract_concept_vectors).
    """
    if languages is None:
        languages = EXPERIMENT_LANGUAGES
    if layers is None:
        layers = DEFAULT_LAYERS
    if output_dir is None:
        output_dir = RESULTS_DIR

    all_stats: Dict[str, Dict[str, Dict[int, float]]] = {}
    # Collect diff matrices for the scatter plot
    scatter_diffs: Dict[str, Dict[int, torch.Tensor]] = {}  # {concept: {layer: diff}}

    for lang in languages:
        diff_path = Path(vectors_dir) / f"{domain}_{lang}_diffs.pt"
        if not diff_path.exists():
            print(f"  Skipping {lang}: {diff_path} not found (run exp1 first)")
            continue

        # {concept: {str(layer): tensor[n_pairs, hidden_dim]}}
        diff_data = torch.load(diff_path, map_location="cpu")

        for concept, layer_str_diffs in diff_data.items():
            layer_diffs = {int(l): d for l, d in layer_str_diffs.items()}
            label = f"{lang[:3]}/{concept}"
            all_stats[label] = compute_layer_stats(layer_diffs)
            if lang == scatter_lang:
                scatter_diffs[concept] = layer_diffs

    if not all_stats:
        print("No diff data found. Run exp1_kinship.py first.")
        return

    plot_layer_cosine_similarity(all_stats, layers, output_dir / "layer_cosine_similarity.png")
    plot_pca_explained_variance(all_stats,  layers, output_dir / "pca_explained_variance.png")

    # Per-pair scatter: first concept available for scatter_lang at scatter_layer
    for concept, layer_diffs in scatter_diffs.items():
        if scatter_layer in layer_diffs:
            plot_per_pair_projections(
                layer_diffs[scatter_layer],
                layer=scatter_layer,
                label=f"{scatter_lang}/{concept}",
                output_path=output_dir / f"per_pair_projections_{scatter_lang}_{scatter_layer}.png",
            )
            break
    else:
        print(f"  No scatter data for {scatter_lang} layer {scatter_layer}")


# ---------------------------------------------------------------------------
# In-experiment comparison (call from within exp1 with raw diffs)
# ---------------------------------------------------------------------------

def compare_from_diff_matrices(
    diff_matrices_by_lang: Dict[str, Dict[str, Dict[int, torch.Tensor]]],
    # {lang: {concept: {layer: diff [n_pairs, hidden_dim]}}}
    layers: List[int],
    scatter_layer: int = 12,
    scatter_lang: str = "eng_Latn",
    scatter_concept: Optional[str] = None,
    output_dir: Optional[Path] = None,
):
    """
    Produce all three plots directly from raw difference matrices.

    This gives accurate per-pair scatter and PC1 variance ratios.
    Call this from exp1_kinship.py before saving vectors.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR

    all_stats: Dict[str, Dict[str, Dict[int, float]]] = {}

    for lang, concepts in diff_matrices_by_lang.items():
        for concept, layer_diffs in concepts.items():
            label = f"{lang[:3]}/{concept}"
            all_stats[label] = compute_layer_stats(layer_diffs)

    plot_layer_cosine_similarity(all_stats, layers, output_dir / "layer_cosine_similarity.png")
    plot_pca_explained_variance(all_stats,  layers, output_dir / "pca_explained_variance.png")

    # Per-pair scatter
    lang_diffs = diff_matrices_by_lang.get(scatter_lang, {})
    if scatter_concept is None:
        scatter_concept = next(iter(lang_diffs), None)
    if scatter_concept and scatter_layer in lang_diffs.get(scatter_concept, {}):
        plot_per_pair_projections(
            lang_diffs[scatter_concept][scatter_layer],
            layer=scatter_layer,
            label=f"{scatter_lang}/{scatter_concept}",
            output_path=output_dir / f"per_pair_projections_{scatter_lang}_{scatter_layer}.png",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare PCA vs. mean concept vectors")
    parser.add_argument("--domain",        default="kinship", help="Concept domain")
    parser.add_argument("--layer",   type=int, default=12,    help="Layer for per-pair scatter")
    parser.add_argument("--lang",          default="eng_Latn", help="Language for per-pair scatter")
    parser.add_argument("--vectors-dir",   default=VECTORS_DIR)
    parser.add_argument("--output-dir",    default=str(RESULTS_DIR))
    args = parser.parse_args()

    compare_pca_vs_mean(
        domain=args.domain,
        scatter_layer=args.layer,
        scatter_lang=args.lang,
        vectors_dir=args.vectors_dir,
        output_dir=Path(args.output_dir),
    )
