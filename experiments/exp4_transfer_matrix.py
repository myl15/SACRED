"""
Experiment 4: Full NxN Cross-Lingual Transfer Matrix (calibrated).

Key changes from original:
  - alpha=0.25 default (was 1.0) to avoid ceiling effects
  - Returns both deletion_matrix (binary) and prob_matrix (continuous metric)
  - Computes cross-lingual transfer scores (off-diagonal / diagonal)
  - Runs for both sacred and kinship domains when called with --both-domains
  - Generates: sacred heatmap, kinship heatmap, sacred vs kinship comparison chart
  - All outputs saved to results/

Run after exp1_kinship.py (needs saved concept vectors in outputs/vectors/).
"""

import json
from pathlib import Path
import os

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import DEFAULT_DEVICE, EXPERIMENT_LANGUAGES, HF_CACHE_DIR, INTERVENTION_LAYERS, MODEL_NAME
from data.contrastive_pairs import get_concept_words, load_independent_sacred_tokens
from analysis.transfer_matrix import (
    compute_transfer_matrix,
    compute_cross_lingual_transfer_scores,
    interpret_transfer_matrix,
)
from visualization.transfer_heatmap import plot_transfer_heatmap, plot_transfer_comparison

# Use calibrated alpha to avoid ceiling effects.
# Run experiments/run_calibration.py to find the optimal value for your data.
DEFAULT_ALPHA = 0.25


def run_exp4(
    domain: str = "sacred",
    vectors_dir: str = "outputs/vectors",
    test_sentences_path: str = None,
    layers: list = None,
    alpha: float = DEFAULT_ALPHA,
    device: str = DEFAULT_DEVICE,
    results_dir: str = "results",
):
    """
    Run Experiment 4: full NxN transfer matrix.

    Args:
        domain: Concept domain ("sacred" or "kinship")
        vectors_dir: Directory with concept vector .pt files
        test_sentences_path: JSON with test sentences per language; defaults to
            outputs/stimuli/<domain>_pairs.json so sentences always match the domain
        layers: Encoder layers for intervention
        alpha: Vector subtraction scaling factor (default: 0.25 to avoid ceiling)
        device: Compute device
        results_dir: Where to save output figures and JSON
    """
    if test_sentences_path is None:
        test_sentences_path = f"outputs/stimuli/{domain}_pairs.json"
    if layers is None:
        layers = INTERVENTION_LAYERS

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(results_dir, "json")).mkdir(exist_ok=True)
    Path(os.path.join(results_dir, "figures")).mkdir(exist_ok=True)
    Path(os.path.join(results_dir, "vectors")).mkdir(exist_ok=True)

    print("=" * 70)
    print(f"EXPERIMENT 4: N×N Transfer Matrix  [domain={domain}, alpha={alpha}]")
    print("=" * 70)
    print(f"NOTE: Using alpha={alpha} (conservative) to avoid ceiling effects.")

    # --- Load model ---
    print(f"\nLoading {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR).to(device)
    model.eval()

    # --- Load concept vectors ---
    print(f"\nLoading {domain} concept vectors...")
    concept_vectors = {}
    for lang in EXPERIMENT_LANGUAGES:
        vec_path = f"{vectors_dir}/{domain}_{lang}.pt"
        if Path(vec_path).exists():
            raw = torch.load(vec_path, map_location=device)
            concept_vectors[lang] = _average_concept_vectors(raw)
        else:
            print(f"  WARNING: {vec_path} not found — run exp1 first")

    if not concept_vectors:
        print("No concept vectors found. Run exp1_kinship.py first.")
        return None, None, None

    # --- Load test sentences ---
    if not Path(test_sentences_path).exists():
        alt_path = f"outputs/stimuli/sacred_pairs.json"
        if Path(alt_path).exists():
            test_sentences_path = alt_path
            print(f"  Falling back to {test_sentences_path}")
        else:
            print(f"  ERROR: test sentences not found at {test_sentences_path}")
            return None, None, None

    print(f"\nLoading test sentences from {test_sentences_path}...")
    with open(test_sentences_path, "r") as f:
        pairs_data = json.load(f)

    test_sentences = {}
    for lang in EXPERIMENT_LANGUAGES:
        lang_data = pairs_data.get(lang, {})
        sents = [p["positive"] for concept_pairs in lang_data.values() for p in concept_pairs]
        test_sentences[lang] = sents[:15]    # cap for speed

    # --- Concept token IDs and word lists ---
    concept_token_ids = {
        lang: load_independent_sacred_tokens(lang, tokenizer, domain=domain)
        for lang in EXPERIMENT_LANGUAGES
    }
    concept_words_by_lang = {
        lang: get_concept_words(lang, domain=domain)
        for lang in EXPERIMENT_LANGUAGES
    }

    # --- Compute transfer matrix ---
    print("\nComputing transfer matrix...")
    deletion_matrix, prob_matrix, languages = compute_transfer_matrix(
        concept_vectors=concept_vectors,
        test_sentences=test_sentences,
        concept_token_ids=concept_token_ids,
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        alpha=alpha,
        device=device,
        concept_words_by_lang=concept_words_by_lang,
    )

    # --- Cross-lingual transfer scores ---
    transfer_scores = compute_cross_lingual_transfer_scores(deletion_matrix, languages)

    # --- Interpret ---
    summary = interpret_transfer_matrix(deletion_matrix, languages)

    print(f"\nTransfer Matrix Summary [{domain}]:")
    print(f"  Mean off-diagonal deletion: {summary['mean_off_diagonal_deletion']:.3f}")
    print(f"  Best transfer: {summary['best_transfer_pair']} ({summary['best_transfer_rate']:.3f})")
    print(f"  Mean asymmetry: {summary['mean_asymmetry']:.3f}")
    print(f"  English hub score: {summary['english_hub_score']:.3f}")

    print(f"\n  Cross-lingual transfer scores (>0.7 meets success criterion):")
    passing = 0
    for label, score in sorted(transfer_scores.items()):
        flag = "✓" if score >= 0.7 else " "
        print(f"    {flag} {label}: {score:.3f}")
        if score >= 0.7:
            passing += 1
    print(f"  {passing}/{len(transfer_scores)} pairs pass 70% threshold")

    # --- Check for saturation ---
    if np.mean(deletion_matrix) > 0.9:
        print(f"\n  WARNING: Mean deletion rate {np.mean(deletion_matrix):.3f} is very high.")
        print(f"  Try alpha={alpha/2:.3f}. Run experiments/run_calibration.py for guidance.")

    # --- Save ---
    np.save(f"{results_dir}/vectors/exp4_deletion_matrix_{domain}.npy", deletion_matrix)
    np.save(f"{results_dir}/vectors/exp4_prob_matrix_{domain}.npy", prob_matrix)

    with open(f"{results_dir}/json/exp4_transfer_summary_{domain}.json", "w") as f:
        json.dump({
            "domain": domain,
            "alpha": alpha,
            "languages": languages,
            "summary": summary,
            "transfer_scores": transfer_scores,
            "n_passing_70pct": passing,
            "n_total_pairs": len(transfer_scores),
        }, f, indent=2)

    # --- Visualize deletion matrix ---
    plot_transfer_heatmap(
        deletion_matrix, languages,
        save_path=f"{results_dir}/figures/exp4_transfer_matrix_{domain}_calibrated.png",
        title=f"Cross-Lingual Transfer Matrix — {domain.title()} (alpha={alpha})",
    )

    # --- Visualize probability-reduction matrix (continuous) ---
    plot_transfer_heatmap(
        prob_matrix, languages,
        save_path=f"{results_dir}/figures/exp4_transfer_matrix_{domain}_prob_reduction.png",
        title=f"Concept Prob Reduction Matrix — {domain.title()} (alpha={alpha})",
    )

    return deletion_matrix, prob_matrix, languages


def run_both_domains(
    vectors_dir: str = "outputs/vectors",
    layers: list = None,
    alpha: float = DEFAULT_ALPHA,
    device: str = DEFAULT_DEVICE,
    results_dir: str = "results",
):
    """Run exp4 for both sacred and kinship, then generate comparison chart."""
    # Pass test_sentences_path=None so each call derives the correct domain-specific path
    sacred_result = run_exp4(
        domain="sacred", vectors_dir=vectors_dir,
        layers=layers, alpha=alpha, device=device, results_dir=results_dir,
    )
    kinship_result = run_exp4(
        domain="kinship", vectors_dir=vectors_dir,
        layers=layers, alpha=alpha, device=device, results_dir=results_dir,
    )

    sacred_matrix, _, sacred_langs = sacred_result
    kinship_matrix, _, kinship_langs = kinship_result

    if sacred_matrix is not None and kinship_matrix is not None:
        # Use the common language set for comparison
        common_langs = [l for l in sacred_langs if l in kinship_langs]
        s_idx = [sacred_langs.index(l) for l in common_langs]
        k_idx = [kinship_langs.index(l) for l in common_langs]
        sacred_sub = sacred_matrix[np.ix_(s_idx, s_idx)]
        kinship_sub = kinship_matrix[np.ix_(k_idx, k_idx)]

        plot_transfer_comparison(
            sacred_sub, kinship_sub, common_langs,
            save_path=f"{results_dir}/figures/exp4_transfer_comparison_sacred_vs_kinship.png",
        )


def _average_concept_vectors(raw: dict) -> dict:
    """Average all concept vectors to get one vector per layer."""
    layer_vecs = {}
    for concept, layer_dict in raw.items():
        for layer_str, vec in layer_dict.items():
            layer = int(layer_str)
            layer_vecs.setdefault(layer, []).append(vec)
    return {layer: torch.stack(vecs).mean(0) for layer, vecs in layer_vecs.items()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 4: N×N transfer matrix")
    parser.add_argument("--domain", default="kinship", choices=["kinship", "sacred"])
    parser.add_argument("--both-domains", action="store_true",
                        help="Run for both sacred and kinship and generate comparison chart")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Vector subtraction scaling factor (default: 0.25)")
    parser.add_argument("--vectors-dir", default="outputs/vectors")
    parser.add_argument("--test-sentences", default=None,
                        help="Path to stimuli JSON; defaults to outputs/stimuli/<domain>_pairs.json. "
                             "Ignored when --both-domains is set.")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    if args.both_domains:
        run_both_domains(
            vectors_dir=args.vectors_dir,
            alpha=args.alpha,
            results_dir=args.results_dir,
        )
    else:
        run_exp4(
            domain=args.domain,
            vectors_dir=args.vectors_dir,
            test_sentences_path=args.test_sentences,
            alpha=args.alpha,
            results_dir=args.results_dir,
        )
