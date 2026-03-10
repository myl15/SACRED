"""
Circuit visualization: circuit maps and universal circuit heatmaps.
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"


def plot_circuit_map(circuit, save_path: Optional[str] = None, title: str = "Sacred Concept Circuit Map"):
    """Bar + violin chart of circuit neuron counts and effect sizes per layer."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    neurons_by_layer: Dict = {}
    for neuron in circuit.neurons:
        neurons_by_layer.setdefault(neuron.layer, []).append(neuron)

    layers = sorted(neurons_by_layer.keys())
    counts = [len(neurons_by_layer[l]) for l in layers]

    ax1.bar(layers, counts, color="steelblue", alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Number of Significant Neurons")
    ax1.set_title("Circuit Neuron Distribution")
    ax1.grid(axis="y", alpha=0.3)

    layer_effect_sizes = []
    layer_labels = []
    for layer in layers:
        for neuron in neurons_by_layer[layer]:
            layer_effect_sizes.append(abs(neuron.effect_size))
            layer_labels.append(layer)

    df = pd.DataFrame({"Layer": layer_labels, "Effect Size (|d|)": layer_effect_sizes})
    sns.violinplot(data=df, x="Layer", y="Effect Size (|d|)", ax=ax2, color="coral", inner="box")
    ax2.set_title("Effect Size Distribution by Layer")
    ax2.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Medium effect")
    ax2.axhline(y=0.8, color="darkred", linestyle="--", alpha=0.5, label="Large effect")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(save_path, "Circuit map")


def plot_universal_circuit_heatmap(circuits_by_lang: Dict, save_path: Optional[str] = None):
    """Heatmap: layers × languages, cell = # significant neurons."""
    all_layers: set = set()
    for circuit in circuits_by_lang.values():
        all_layers.update(circuit.get_critical_layers())

    layers = sorted(all_layers)
    languages = sorted(circuits_by_lang.keys())

    matrix = np.zeros((len(layers), len(languages)))
    for lang_idx, lang in enumerate(languages):
        for layer_idx, layer in enumerate(layers):
            matrix[layer_idx, lang_idx] = len(circuits_by_lang[lang].get_neurons_by_layer(layer))

    fig, ax = plt.subplots(figsize=(max(8, len(languages) * 2), max(6, len(layers) // 2)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(languages)))
    ax.set_yticks(np.arange(len(layers)))
    ax.set_xticklabels(languages, rotation=45, ha="right")
    ax.set_yticklabels(layers)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Number of Significant Neurons", rotation=270, labelpad=20)

    for i in range(len(layers)):
        for j in range(len(languages)):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black", fontsize=8)

    ax.set_title("Circuit Components Across Languages and Layers", fontweight="bold")
    ax.set_xlabel("Language")
    ax.set_ylabel("Layer")
    plt.tight_layout()
    _save_or_show(save_path, "Universal circuit heatmap")


def create_comprehensive_report_figure(
    circuit,
    necessity_results,
    statistical_report,
    save_path: str = "outputs/figures/comprehensive_report.png",
):
    """Multi-panel publication figure: circuit + necessity + statistical summary."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    neurons_by_layer: Dict = {}
    for neuron in circuit.neurons:
        neurons_by_layer.setdefault(neuron.layer, []).append(neuron)
    layers = sorted(neurons_by_layer.keys())
    counts = [len(neurons_by_layer[l]) for l in layers]

    # Panel A
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(layers, counts, color="steelblue", alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("N Neurons")
    ax1.set_title("A. Circuit Distribution", fontweight="bold", loc="left")
    ax1.grid(axis="y", alpha=0.3)

    # Panel B
    ax2 = fig.add_subplot(gs[0, 1])
    sizes = [abs(n.effect_size) for l in layers for n in neurons_by_layer[l]]
    lbls = [l for l in layers for _ in neurons_by_layer[l]]
    df = pd.DataFrame({"Layer": lbls, "Effect Size": sizes})
    sns.violinplot(data=df, x="Layer", y="Effect Size", ax=ax2, color="coral")
    ax2.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
    ax2.set_title("B. Effect Size Distribution", fontweight="bold", loc="left")

    # Panel C
    ax3 = fig.add_subplot(gs[0, 2])
    baseline_probs = [m.concept_token_probability for m in necessity_results.baseline_quality["sacred"]]
    ablated_probs = [m.concept_token_probability for m in necessity_results.ablated_quality["sacred"]]
    ax3.boxplot([baseline_probs, ablated_probs], positions=[1, 2],
                patch_artist=True, boxprops=dict(facecolor="lightblue", alpha=0.7))
    ax3.set_xticks([1, 2])
    ax3.set_xticklabels(["Baseline", "Ablated"])
    ax3.set_ylabel("Concept Token Prob")
    ax3.set_title(f"C. Necessity (d={necessity_results.effect_size:.2f})", fontweight="bold", loc="left")

    # Panel D: summary text
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis("off")
    lines = [f"D. Statistical Summary", ""]
    for name, result in [("H1: Necessity", statistical_report.h1_necessity),
                          ("H2: Specificity", statistical_report.h2_specificity),
                          ("H3: Universality", statistical_report.h3_universality)]:
        if result:
            sig = "PASS" if result.significant else "FAIL"
            lines.append(f"{sig}  {name}: d={result.effect_size:.3f}, p={result.p_value:.4f}")
    lines += ["", f"Summary: {statistical_report.summary}"]
    ax4.text(0.05, 0.5, "\n".join(lines), transform=ax4.transAxes, fontsize=11,
             verticalalignment="center", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    fig.suptitle("Sacred Concept Circuit Discovery: Results", fontsize=16, fontweight="bold")
    _save_or_show(save_path, "Comprehensive report figure")


def _save_or_show(save_path: Optional[str], label: str):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"{label} saved to {save_path}")
    else:
        plt.show()
    plt.close()
