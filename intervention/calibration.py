"""
Intervention strength calibration for concept vector experiments.

The ceiling-effect problem: alpha=1.0 subtracts the full concept vector and
saturates all conditions (including random controls) at 1.00 deletion rate,
destroying diagnostic value.

calibrate_intervention_strength() sweeps alpha values and finds the range
where deletion_rate is between 0.2 and 0.8 (the "diagnostic range").

Usage:
    from intervention.calibration import calibrate_intervention_strength, plot_calibration_curve

    results = calibrate_intervention_strength(
        model, tokenizer, concept_vector, sentences,
        lang_code="eng_Latn", target_lang="spa_Latn",
        concept_token_ids=token_ids, layers=[10, 11, 12, 13],
    )
    plot_calibration_curve(results, save_path="results/calibration_curve.png")
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union

from intervention.hooks import InterventionHook
from intervention.necessity import measure_concept_deletion


def calibrate_intervention_strength(
    model,
    tokenizer,
    concept_vector: torch.Tensor,
    sentences: List[str],
    lang_code: str,
    target_lang: str,
    concept_token_ids: List[int],
    alphas: Optional[List[float]] = None,
    layers: Optional[List[int]] = None,
    target: str = "residual",
    device: str = "cuda",
    concept_words: Optional[List[str]] = None,
) -> Dict[float, Dict[str, float]]:
    """
    Sweep alpha values and measure intervention effects to find the diagnostic range.

    For each alpha, measures:
      - concept_deletion_rate: binary (fraction of outputs without the concept)
      - mean_concept_prob: continuous P(concept_token) averaged over outputs
      - mean_prob_delta: baseline_concept_prob - intervened_concept_prob

    The diagnostic alpha range is where deletion_rate is between 0.2 and 0.8,
    ensuring meaningful variance across sentences (not saturated at 0 or 1).

    Args:
        model: NLLB model
        tokenizer: NLLB tokenizer
        concept_vector: [hidden_dim] tensor for the concept direction to subtract
        sentences: Source-language sentences containing the concept
        lang_code: Source language code
        target_lang: Translation target language code
        concept_token_ids: Token IDs to check for in outputs
        alphas: Scaling factors to sweep (default: [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
        layers: Encoder layers for the intervention (default: from config)
        target: "residual" (1024-dim) or "fc1" (MLP intermediate)
        device: Compute device
        concept_words: Optional word strings for string-based presence checking

    Returns:
        {alpha: {"deletion_rate": float, "mean_concept_prob": float, "mean_prob_delta": float}}
    """
    if alphas is None:
        alphas = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    if layers is None:
        from config import DEFAULT_LAYERS
        layers = DEFAULT_LAYERS

    print(f"\n=== Calibrating intervention strength ===")
    print(f"Target: {target}, Layers: {layers}, n_sentences: {len(sentences)}")

    # Baseline (no intervention)
    baseline = measure_concept_deletion(
        sentences, model, tokenizer, lang_code, target_lang,
        concept_token_ids, None, device, concept_words=concept_words,
    )
    baseline_prob = baseline["mean_concept_probability"]
    baseline_rate = baseline["concept_present_rate"]
    print(f"  [Baseline] present_rate={baseline_rate:.3f}, mean_prob={baseline_prob:.4f}")

    results: Dict[float, Dict[str, float]] = {}

    for alpha in alphas:
        hook = InterventionHook()
        hook.register_scaled_vector_subtraction_hook(
            model, concept_vector, layers, alpha=alpha, target=target,
        )
        result = measure_concept_deletion(
            sentences, model, tokenizer, lang_code, target_lang,
            concept_token_ids, hook, device, concept_words=concept_words,
        )
        hook.cleanup()

        deletion_rate = 1.0 - result["concept_present_rate"]
        mean_prob = result["mean_concept_probability"]
        prob_delta = baseline_prob - mean_prob

        results[alpha] = {
            "deletion_rate": deletion_rate,
            "mean_concept_prob": mean_prob,
            "mean_prob_delta": prob_delta,
        }
        in_range = "✓ DIAGNOSTIC" if 0.2 <= deletion_rate <= 0.8 else ""
        print(f"  alpha={alpha:.2f}: deletion_rate={deletion_rate:.3f}, "
              f"mean_prob={mean_prob:.4f}, prob_delta={prob_delta:.4f} {in_range}")

    # Report optimal range
    diagnostic = [a for a, v in results.items() if 0.2 <= v["deletion_rate"] <= 0.8]
    if diagnostic:
        print(f"\n  Optimal alpha range: {min(diagnostic):.2f} – {max(diagnostic):.2f}")
        print(f"  Recommended alpha: {min(diagnostic):.2f} (lowest in diagnostic range)")
    else:
        print("\n  WARNING: No alpha in [0.05, 2.0] achieved 0.2–0.8 deletion rate.")
        print("  Try a finer sweep or check that sentences contain the concept at baseline.")

    return results


def find_optimal_alpha(
    calibration_results: Dict[float, Dict[str, float]],
    target_deletion_rate: float = 0.5,
) -> float:
    """
    Find the alpha closest to target_deletion_rate within the diagnostic range (0.2–0.8).

    Falls back to the alpha with deletion rate closest to target if none are in range.
    """
    diagnostic = {a: v for a, v in calibration_results.items()
                  if 0.2 <= v["deletion_rate"] <= 0.8}
    pool = diagnostic if diagnostic else calibration_results

    return min(pool, key=lambda a: abs(pool[a]["deletion_rate"] - target_deletion_rate))


def plot_calibration_curve(
    calibration_results: Dict[float, Dict[str, float]],
    save_path: str = "results/calibration_curve.png",
    title: str = "Intervention Strength Calibration",
):
    """
    Plot deletion_rate vs alpha with the diagnostic range (0.2–0.8) highlighted.

    Also plots mean_prob_delta on a secondary y-axis to show the continuous signal.

    Args:
        calibration_results: Output of calibrate_intervention_strength()
        save_path: Output file path
        title: Plot title
    """
    alphas = sorted(calibration_results.keys())
    deletion_rates = [calibration_results[a]["deletion_rate"] for a in alphas]
    prob_deltas = [calibration_results[a]["mean_prob_delta"] for a in alphas]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_primary = "steelblue"
    color_secondary = "darkorange"

    ax1.plot(alphas, deletion_rates, marker="o", markersize=6, linewidth=2,
             color=color_primary, label="Binary deletion rate")
    ax1.axhspan(0.2, 0.8, alpha=0.12, color="green", label="Diagnostic range (0.2–0.8)")
    ax1.axhline(y=0.2, color="green", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.axhline(y=0.8, color="green", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_xlabel("Alpha (scaling factor)", fontsize=12)
    ax1.set_ylabel("Concept Deletion Rate", color=color_primary, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_primary)
    ax1.set_ylim([0, 1.15])
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(alphas, prob_deltas, marker="s", markersize=5, linewidth=1.5,
             color=color_secondary, linestyle="--", label="P(concept) reduction (continuous)")
    ax2.set_ylabel("Mean P(concept) Reduction", color=color_secondary, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color_secondary)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    # Annotate each point
    for alpha, rate in zip(alphas, deletion_rates):
        ax1.annotate(f"{rate:.2f}", (alpha, rate), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color=color_primary)

    ax1.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Calibration curve saved to {save_path}")
