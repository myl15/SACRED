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
from extraction.concept_vectors import VECTOR_SCALING_POLICY
from analysis.transfer_matrix import (
    compute_transfer_matrix,
    compute_cross_lingual_transfer_scores,
    interpret_transfer_matrix,
)
from visualization.transfer_heatmap import plot_transfer_heatmap, plot_transfer_comparison

from journal.run_manifest import build_manifest
from analysis.journal_stats import bootstrap_ci_mean, summarize_transfer_scores_with_fdr

# Use calibrated alpha to avoid ceiling effects.
# Run experiments/run_calibration.py to find the optimal value for your data.
DEFAULT_ALPHA = 0.25


def run_exp4(
    domain: str = "sacred",
    vectors_dir: str = "outputs/vectors",
    test_sentences_path: str = None,
    layers: list = None,
    alpha: float = DEFAULT_ALPHA,
    alpha_mean: float = None,
    alpha_pca: float = None,
    device: str = DEFAULT_DEVICE,
    results_dir: str = "results",
    vector_method: str = "both",
    output_lang: str = "eng_Latn",
    vector_source_domain: str = None,
    matching_mode: str = "hybrid",
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
        vector_method: "mean", "pca", or "both". When "both", runs the experiment
            twice and produces separate output files for each method.
    """
    if vector_method == "both":
        alpha_by_method = {
            "mean": alpha_mean if alpha_mean is not None else alpha,
            "pca": alpha_pca if alpha_pca is not None else alpha,
        }
        results = {}
        for method in ("mean", "pca"):
            results[method] = run_exp4(
                domain=domain, vectors_dir=vectors_dir,
                test_sentences_path=test_sentences_path, layers=layers,
                alpha=alpha_by_method[method], alpha_mean=alpha_mean, alpha_pca=alpha_pca,
                device=device, results_dir=results_dir,
                vector_method=method,
                output_lang=output_lang,
                vector_source_domain=vector_source_domain,
                matching_mode=matching_mode,
            )
        return results

    if test_sentences_path is None:
        test_sentences_path = f"outputs/stimuli/{domain}_pairs.json"
    if layers is None:
        layers = INTERVENTION_LAYERS

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(results_dir, "json")).mkdir(exist_ok=True)
    Path(os.path.join(results_dir, "figures")).mkdir(exist_ok=True)
    Path(os.path.join(results_dir, "vectors")).mkdir(exist_ok=True)

    print("=" * 70)
    print(f"EXPERIMENT 4: N×N Transfer Matrix  [domain={domain}, alpha={alpha}, vectors={vector_method}], output_lang={output_lang}]")
    print("=" * 70)
    print(f"NOTE: Using alpha={alpha} (conservative) to avoid ceiling effects.")

    # --- Load model ---
    print(f"\nLoading {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR).to(device)
    model.eval()

    # --- Load concept vectors ---
    print(f"\nLoading {domain} concept vectors ({vector_method})...")
    concept_vectors = _load_concept_vectors(
        domain, vectors_dir, device, vector_method,
        vector_source_domain=vector_source_domain,
    )

    if not concept_vectors:
        print(f"No {vector_method} concept vectors found. Run exp1_kinship.py first.")
        return None, None, None

    # --- Load test sentences ---
    if not Path(test_sentences_path).exists():
        # Do NOT fall back to a different domain's stimuli — using the wrong sentences
        # gives near-zero baseline concept presence → deletion_rate ≈ 1.0 for all cells,
        # making the entire transfer matrix artifactual.
        print(f"  ERROR: test sentences not found at {test_sentences_path}")
        print(f"  Run: python experiments/exp1_kinship.py --domain {domain}")
        return None, None, None

    print(f"\nLoading test sentences from {test_sentences_path}...")
    with open(test_sentences_path, "r") as f:
        pairs_data = json.load(f)

    test_sentences = {}
    for lang in EXPERIMENT_LANGUAGES:
        lang_data = pairs_data.get(lang, {})
        sents = [p["positive"] for concept_pairs in lang_data.values() for p in concept_pairs]
        test_sentences[lang] = sents # Remove cap for comprehensive transfer matrix

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
    deletion_matrix, prob_matrix, languages, transfer_diagnostics = compute_transfer_matrix(
        concept_vectors=concept_vectors,
        test_sentences=test_sentences,
        concept_token_ids=concept_token_ids,
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        alpha=alpha,
        device=device,
        concept_words_by_lang=concept_words_by_lang,
        output_lang=output_lang,
        matching_mode=matching_mode,
    )

    # --- Cross-lingual transfer scores ---
    transfer_scores = compute_cross_lingual_transfer_scores(deletion_matrix, languages)

    # --- Interpret ---
    summary = interpret_transfer_matrix(deletion_matrix, languages)

    print(f"\nTransfer Matrix Summary [{domain}, {vector_method}]:")
    print(f"  Mean off-diagonal deletion: {summary['mean_off_diagonal_deletion']:.3f}")
    print(f"  Best transfer: {summary['best_transfer_pair']} ({summary['best_transfer_rate']:.3f})")
    print(f"  Mean asymmetry: {summary['mean_asymmetry']:.3f}")
    print(f"  English hub score (relative): {summary['english_hub_score']:.3f}")
    print(f"  English absolute mean deletion: {summary['english_hub_absolute_mean']:.3f}")
    print(f"  Non-English absolute mean deletion: {summary['non_english_absolute_mean']:.3f}")

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
    
    np.save(f"{results_dir}/vectors/exp4_deletion_matrix_{domain}_{vector_method}_output_lang_{output_lang}.npy", deletion_matrix)
    np.save(f"{results_dir}/vectors/exp4_prob_matrix_{domain}_{vector_method}_output_lang_{output_lang}.npy", prob_matrix)

    os.makedirs(os.path.join(results_dir, f"{output_lang}", "json"), exist_ok=True)
    summary_path = f"{results_dir}/{output_lang}/json/exp4_transfer_summary_{domain}_{vector_method}.json"
    run_manifest = build_manifest(
        f"exp4_{domain}_{vector_method}_{output_lang}",
        extra={
            "domain": domain,
            "vector_source_domain": vector_source_domain,
            "vectors_dir": vectors_dir,
            "test_sentences_path": test_sentences_path,
            "layers": layers,
            "alpha": alpha,
            "vector_method": vector_method,
            "output_lang": output_lang,
            "matching_mode": matching_mode,
            "output_json": summary_path,
        },
    )
    score_vals = np.array(list(transfer_scores.values()), dtype=float)
    ts_mean, ts_lo, ts_hi = bootstrap_ci_mean(score_vals, seed=42)
    mc_note = summarize_transfer_scores_with_fdr(transfer_scores)
    with open(summary_path, "w") as f:
        json.dump({
            "run_manifest": run_manifest,
            "domain": domain,
            "alpha": alpha,
            "alpha_mean": alpha_mean if alpha_mean is not None else alpha,
            "alpha_pca": alpha_pca if alpha_pca is not None else alpha,
            "vector_method": vector_method,
            "vector_scaling_policy": VECTOR_SCALING_POLICY,
            "vector_source_domain": vector_source_domain,
            "matching_mode": matching_mode,
            "output_lang": output_lang,
            "languages": languages,
            "summary": summary,
            "diagnostics": transfer_diagnostics,
            "transfer_scores": transfer_scores,
            "n_passing_70pct": passing,
            "n_total_pairs": len(transfer_scores),
            "statistics": {
                "transfer_score_bootstrap_mean": ts_mean,
                "transfer_score_ci95_low": ts_lo,
                "transfer_score_ci95_high": ts_hi,
                "multiple_comparison": mc_note,
            },
        }, f, indent=2)

    os.makedirs(os.path.join(results_dir, f"{output_lang}", "figures"), exist_ok=True)
    # --- Visualize deletion matrix ---
    plot_transfer_heatmap(
        deletion_matrix, languages,
        save_path=f"{results_dir}/{output_lang}/figures/exp4_transfer_matrix_{domain}_{vector_method}_calibrated.png",
        title=f"Cross-Lingual Transfer Matrix — {domain.title()} ({vector_method}, alpha={alpha})",
    )

    # --- Visualize probability-reduction matrix (continuous) ---
    plot_transfer_heatmap(
        prob_matrix, languages,
        save_path=f"{results_dir}/{output_lang}/figures/exp4_transfer_matrix_{domain}_{vector_method}_prob_reduction.png",
        title=f"Concept Prob Reduction Matrix — {domain.title()} ({vector_method}, alpha={alpha})",
    )

    return deletion_matrix, prob_matrix, languages


def run_both_domains(
    vectors_dir: str = "outputs/vectors",
    layers: list = None,
    alpha: float = DEFAULT_ALPHA,
    alpha_mean: float = None,
    alpha_pca: float = None,
    device: str = DEFAULT_DEVICE,
    results_dir: str = "results",
    vector_method: str = "both",
    output_lang: str = "eng_Latn",
    vector_source_domain: str = None,
    matching_mode: str = "hybrid",
):
    """Run exp4 for both sacred and kinship, then generate comparison chart(s)."""
    # Pass test_sentences_path=None so each call derives the correct domain-specific path
    sacred_result = run_exp4(
        domain="sacred", vectors_dir=vectors_dir,
        layers=layers, alpha=alpha, device=device, results_dir=results_dir,
        alpha_mean=alpha_mean, alpha_pca=alpha_pca,
        vector_method=vector_method, output_lang=output_lang,
        vector_source_domain=vector_source_domain,
        matching_mode=matching_mode,
    )
    kinship_result = run_exp4(
        domain="kinship", vectors_dir=vectors_dir,
        layers=layers, alpha=alpha, device=device, results_dir=results_dir,
        alpha_mean=alpha_mean, alpha_pca=alpha_pca,
        vector_method=vector_method, output_lang=output_lang,  # Ensure same output_lang for direct comparison (e.g., English as pivot for both domains
        vector_source_domain=vector_source_domain,
        matching_mode=matching_mode,
    )

    # Generate a comparison chart per method
    methods = ("mean", "pca") if vector_method == "both" else (vector_method,)
    for method in methods:
        # When vector_method="both", run_exp4 returns {method: (matrix, prob, langs)}
        s = sacred_result.get(method, sacred_result) if vector_method == "both" else sacred_result
        k = kinship_result.get(method, kinship_result) if vector_method == "both" else kinship_result
        if s is None or k is None:
            continue
        sacred_matrix, _, sacred_langs = s
        kinship_matrix, _, kinship_langs = k
        if sacred_matrix is None or kinship_matrix is None:
            continue
        common_langs = [l for l in sacred_langs if l in kinship_langs]
        s_idx = [sacred_langs.index(l) for l in common_langs]
        k_idx = [kinship_langs.index(l) for l in common_langs]
        plot_transfer_comparison(
            sacred_matrix[np.ix_(s_idx, s_idx)],
            kinship_matrix[np.ix_(k_idx, k_idx)],
            common_langs,
            save_path=f"{results_dir}/figures/exp4_transfer_comparison_sacred_vs_kinship_{method}.png",
        )


def _load_concept_vectors(
    domain: str,
    vectors_dir: str,
    device: str,
    method: str,
    vector_source_domain: str = None,
) -> dict:
    """Load per-language concept vectors from the appropriate .pt file for `method`."""
    file_domain = vector_source_domain if vector_source_domain is not None else domain
    suffix = "_pca" if method == "pca" else ""
    concept_vectors = {}
    for lang in EXPERIMENT_LANGUAGES:
        vec_path = f"{vectors_dir}/{file_domain}_{lang}{suffix}.pt"
        if Path(vec_path).exists():
            raw = torch.load(vec_path, map_location=device)
            concept_vectors[lang] = _average_concept_vectors(raw)
        else:
            print(f"  WARNING: {vec_path} not found — run exp1 first")
    return concept_vectors


def _average_concept_vectors(raw: dict) -> dict:
    """Average all concept vectors to get one vector per layer."""
    layer_vecs = {}
    for _, layer_dict in raw.items():
        for layer_str, vec in layer_dict.items():
            layer = int(layer_str)
            layer_vecs.setdefault(layer, []).append(vec)
    return {layer: torch.stack(vecs).mean(0) for layer, vecs in layer_vecs.items()}


def _parse_layers_arg(s: str):
    if not s or not str(s).strip():
        return None
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 4: N×N transfer matrix")
    parser.add_argument("--domain", default="kinship", choices=["kinship", "sacred"])
    parser.add_argument("--both-domains", action="store_true",
                        help="Run for both sacred and kinship and generate comparison chart")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Vector subtraction scaling factor (default: 0.25)")
    parser.add_argument("--alpha-mean", type=float, default=None,
                        help="Optional alpha override for mean vectors (defaults to --alpha)")
    parser.add_argument("--alpha-pca", type=float, default=None,
                        help="Optional alpha override for PCA vectors (defaults to --alpha)")
    parser.add_argument("--vectors-dir", default="outputs/vectors")
    parser.add_argument("--test-sentences", default=None,
                        help="Path to stimuli JSON; defaults to outputs/stimuli/<domain>_pairs.json. "
                             "Ignored when --both-domains is set.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--vector-method", default="both", choices=["mean", "pca", "both"],
                        help="Which concept vectors to use: mean-difference, PCA reading vector, or both (default)")
    parser.add_argument("--output-lang", default="eng_Latn",
                        help="Fixed translation target for all cells (default: eng_Latn). "
                             "Using English ensures consistent concept_words matching across all source languages.")
    parser.add_argument("--vector-source-domain", default=None,
                        help="Load vectors from this domain's files (wrong-domain ablation)")
    parser.add_argument("--matching-mode", default="hybrid",
                        choices=["substring", "word_boundary", "token_only", "hybrid"],
                        help="Concept presence matching in outputs")
    parser.add_argument("--layers", default=None,
                        help="Comma-separated encoder layer indices (default: INTERVENTION_LAYERS from config)")
    args = parser.parse_args()
    layers_cli = _parse_layers_arg(args.layers)

    if args.both_domains:
        run_both_domains(
            vectors_dir=args.vectors_dir,
            layers=layers_cli,
            alpha=args.alpha,
            alpha_mean=args.alpha_mean,
            alpha_pca=args.alpha_pca,
            results_dir=args.results_dir,
            vector_method=args.vector_method,
            output_lang=args.output_lang,
            vector_source_domain=args.vector_source_domain,
            matching_mode=args.matching_mode,
        )
    else:
        run_exp4(
            domain=args.domain,
            vectors_dir=args.vectors_dir,
            test_sentences_path=args.test_sentences,
            layers=layers_cli,
            alpha=args.alpha,
            alpha_mean=args.alpha_mean,
            alpha_pca=args.alpha_pca,
            results_dir=args.results_dir,
            vector_method=args.vector_method,
            output_lang=args.output_lang,
            vector_source_domain=args.vector_source_domain,
            matching_mode=args.matching_mode,
        )
