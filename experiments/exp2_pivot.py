"""
Experiment 2: Pivot Language Diagnosis (re-run with calibrated interventions).

Key changes from original:
  - alpha=0.25 default (was 1.0) to avoid ceiling effects
  - Continuous metric (P(concept) reduction) is the primary outcome
  - Runs for one or both domains (--both-domains flag)
  - Generates: grouped bar chart of prob reduction + pivot index summary

Run after exp1_kinship.py (needs saved concept vectors in outputs/vectors/).
"""

import json
from itertools import permutations
from pathlib import Path
import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import DEFAULT_DEVICE, EXPERIMENT_LANGUAGES, HF_CACHE_DIR, INTERVENTION_LAYERS, MODEL_NAME
from data.contrastive_pairs import get_concept_words, load_independent_sacred_tokens
from extraction.concept_vectors import load_concept_vectors
from intervention.pivot_diagnosis import run_pivot_diagnosis
from visualization.transfer_heatmap import plot_pivot_diagnosis, plot_pivot_diagnosis_continuous, plot_pivot_index_summary

# Use a conservative alpha to avoid ceiling effects.
# Run experiments/run_calibration.py to get the optimal value for your data.
DEFAULT_ALPHA = 0.25


def run_exp2(
    domain: str = "sacred",
    vectors_dir: str = "outputs/vectors",
    test_sentences_path: str = None,
    layers: list = None,
    alpha: float = DEFAULT_ALPHA,
    device: str = DEFAULT_DEVICE,
    results_dir: str = "results",
    vector_method: str = "both",
):
    """
    Run Experiment 2: pivot language diagnosis for all non-trivial lang pairs.

    Args:
        domain: Concept domain ("sacred" or "kinship")
        vectors_dir: Directory containing .pt concept vector files
        test_sentences_path: JSON file with test sentences per language; defaults to
            outputs/stimuli/<domain>_pairs.json so sentences always match the domain
        layers: Encoder layers for intervention (default: INTERVENTION_LAYERS)
        alpha: Vector subtraction scaling factor (default: 0.25 to avoid ceiling)
        device: Compute device
        results_dir: Where to save output figures and JSON
        vector_method: "mean", "pca", or "both". When "both", runs the experiment
            twice and produces separate output files for each method.
    """
    if vector_method == "both":
        results = {}
        for method in ("mean", "pca"):
            results[method] = run_exp2(
                domain=domain, vectors_dir=vectors_dir,
                test_sentences_path=test_sentences_path, layers=layers,
                alpha=alpha, device=device, results_dir=results_dir,
                vector_method=method,
            )
        return results

    if layers is None:
        layers = INTERVENTION_LAYERS
    if test_sentences_path is None:
        test_sentences_path = f"outputs/stimuli/{domain}_pairs.json"

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(results_dir, "json")).mkdir(exist_ok=True)
    Path(os.path.join(results_dir, "figures")).mkdir(exist_ok=True)

    print("=" * 70)
    print(f"EXPERIMENT 2: Pivot Language Diagnosis  [domain={domain}, alpha={alpha}, vectors={vector_method}]")
    print("=" * 70)
    print(f"NOTE: Using alpha={alpha} (conservative) to avoid ceiling effects.")
    print("      Run experiments/run_calibration.py to find the optimal alpha.")

    # --- Load model ---
    print(f"\nLoading {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR).to(device)
    model.eval()

    # --- Load concept vectors ---
    print(f"\nLoading {domain} concept vectors ({vector_method}) from {vectors_dir}...")
    concept_vectors = _load_concept_vectors(domain, vectors_dir, device, vector_method)
    for lang in concept_vectors:
        print(f"  Loaded {lang}: {len(concept_vectors[lang])} layers")

    if not concept_vectors:
        print(f"No {vector_method} concept vectors found. Run exp1_kinship.py first.")
        return {}

    # --- Load test sentences ---
    if not Path(test_sentences_path).exists():
        print(f"  ERROR: test sentences not found at {test_sentences_path}")
        print(f"  Run: python experiments/exp1_kinship.py --domain {domain}")
        return {}

    print(f"\nLoading test sentences from {test_sentences_path}...")
    with open(test_sentences_path, "r") as f:
        pairs_data = json.load(f)

    test_sentences = {}
    for lang in EXPERIMENT_LANGUAGES:
        lang_data = pairs_data.get(lang, {})
        sents = []
        for concept_pairs in lang_data.values():
            sents.extend([p["positive"] for p in concept_pairs])
        test_sentences[lang] = sents[:20]    # cap at 20 for speed

    # --- Concept token IDs and word lists ---
    concept_token_ids = {
        lang: load_independent_sacred_tokens(lang, tokenizer, domain=domain)
        for lang in EXPERIMENT_LANGUAGES
    }
    concept_words_by_lang = {
        lang: get_concept_words(lang, domain=domain)
        for lang in EXPERIMENT_LANGUAGES
    }

    # --- Run pivot diagnosis for all non-English pairs ---
    non_english = [l for l in EXPERIMENT_LANGUAGES if l != "eng_Latn"]
    pairs_to_test = [(a, b) for a, b in permutations(non_english, 2)]

    results_by_pair = {}
    for pair in pairs_to_test:
        if pair[0] not in concept_vectors or pair[1] not in concept_vectors:
            print(f"  Skipping {pair}: missing vectors")
            continue
        if not test_sentences.get(pair[0]):
            print(f"  Skipping {pair}: no test sentences for {pair[0]}")
            continue
        results_by_pair[pair] = run_pivot_diagnosis(
            translation_pair=pair,
            concept_vectors=concept_vectors,
            test_sentences=test_sentences[pair[0]],
            concept_token_ids=concept_token_ids,
            model=model,
            tokenizer=tokenizer,
            layers=layers,
            alpha=alpha,
            device=device,
            concept_words_by_lang=concept_words_by_lang,
        )

    # --- Print summary ---
    print(f"\n{'='*60}")
    print(f"PIVOT DIAGNOSIS SUMMARY  [{domain}, alpha={alpha}]")
    print(f"{'='*60}")
    print(f"{'Pair':<22} {'A:src':>6} {'B:tgt':>6} {'C:eng':>6} {'D:rnd':>6} {'pivot_idx':>10}")
    print("-" * 60)
    for pair, res in results_by_pair.items():
        label = f"{pair[0].split('_')[0]}→{pair[1].split('_')[0]}"
        a = res["condition_A_source"]["deletion_rate"]
        b = res["condition_B_target"]["deletion_rate"]
        c = res["condition_C_english"]["deletion_rate"]
        d = res["condition_D_random"]["deletion_rate"]
        pi = res["pivot_index"]
        flag = "" if not _is_saturated(res) else "(saturated)"
        print(f"  {label:<20} {a:>6.3f} {b:>6.3f} {c:>6.3f} {d:>6.3f} {pi:>10.3f} {flag}")
    print()

    _check_for_saturation(results_by_pair, alpha)

    # --- Save results ---
    out_path = f"{results_dir}/json/exp2_pivot_{domain}_{vector_method}.json"
    with open(out_path, "w") as f:
        json.dump(
            {str(k): v for k, v in results_by_pair.items()}, f, indent=2
        )
    print(f"\nResults saved to {out_path}")

    # --- Visualize ---
    # Original binary deletion rate chart
    plot_pivot_diagnosis(
        results_by_pair,
        save_path=f"{results_dir}/figures/exp2_pivot_{domain}_{vector_method}_binary.png",
    )
    # New continuous metric chart (primary)
    plot_pivot_diagnosis_continuous(
        results_by_pair,
        save_path=f"{results_dir}/figures/exp2_pivot_{domain}_{vector_method}_continuous.png",
    )
    # Pivot index summary
    plot_pivot_index_summary(
        {domain: results_by_pair},
        save_path=f"{results_dir}/figures/exp2_pivot_index_summary_{domain}_{vector_method}.png",
    )

    return results_by_pair


def run_both_domains(
    vectors_dir: str = "outputs/vectors",
    alpha: float = DEFAULT_ALPHA,
    device: str = DEFAULT_DEVICE,
    results_dir: str = "results",
    vector_method: str = "both",
) -> dict:
    """Run pivot diagnosis for both sacred and kinship domains and save a combined summary plot."""
    all_results = {}
    for domain in ["sacred", "kinship"]:
        stimuli_path = f"outputs/stimuli/{domain}_pairs.json"
        print(f"\n{'#'*70}")
        print(f"# DOMAIN: {domain.upper()}")
        print(f"{'#'*70}")
        results = run_exp2(
            domain=domain,
            vectors_dir=vectors_dir,
            test_sentences_path=stimuli_path,
            alpha=alpha,
            device=device,
            results_dir=results_dir,
            vector_method=vector_method,
        )
        if results:
            all_results[domain] = results

    # Combined pivot index summary — one per vector_method
    methods = ("mean", "pca") if vector_method == "both" else (vector_method,)
    for method in methods:
        # For "both", all_results[domain] is a {method: results} dict
        combined = {}
        for domain, res in all_results.items():
            domain_res = res.get(method, res) if vector_method == "both" else res
            if domain_res:
                combined[domain] = domain_res
        if len(combined) > 1:
            save_path = f"{results_dir}/figures/exp2_pivot_index_summary_both_{method}.png"
            plot_pivot_index_summary(combined, save_path=save_path)
            print(f"\nCombined pivot index summary ({method}) saved to {save_path}")

    return all_results


def _load_concept_vectors(domain: str, vectors_dir: str, device: str, method: str) -> dict:
    """Load per-language concept vectors from the appropriate .pt file for `method`."""
    suffix = "_pca" if method == "pca" else ""
    concept_vectors = {}
    for lang in EXPERIMENT_LANGUAGES:
        vec_path = f"{vectors_dir}/{domain}_{lang}{suffix}.pt"
        if Path(vec_path).exists():
            raw = torch.load(vec_path, map_location=device)
            concept_vectors[lang] = _average_concept_vectors(raw)
        else:
            print(f"  WARNING: {vec_path} not found — run exp1 first")
    return concept_vectors


def _average_concept_vectors(raw: dict) -> dict:
    """Average all concept vectors to get one vector per layer."""
    layer_vecs = {}
    for concept, layer_dict in raw.items():
        for layer_str, vec in layer_dict.items():
            layer = int(layer_str)
            if layer not in layer_vecs:
                layer_vecs[layer] = []
            layer_vecs[layer].append(vec)
    return {layer: torch.stack(vecs).mean(0) for layer, vecs in layer_vecs.items()}


def _is_saturated(result: dict) -> bool:
    """Check if a result is saturated (all conditions close to 1.0)."""
    rates = [result[c]["deletion_rate"] for c in
             ["condition_A_source", "condition_B_target",
              "condition_C_english", "condition_D_random"]]
    return all(r >= 0.95 for r in rates)


def _check_for_saturation(results_by_pair: dict, alpha: float):
    """Warn if random control is still saturated — alpha needs reducing."""
    saturated = [pair for pair, res in results_by_pair.items()
                 if res["condition_D_random"]["deletion_rate"] >= 0.9]
    if saturated:
        print(f"  WARNING: Random control still near-saturated for {len(saturated)} pairs.")
        print(f"  Current alpha={alpha}. Try alpha={alpha/2:.3f}.")
        print("  Random control must show meaningfully less deletion than A/B/C for")
        print("  the pivot test to be diagnostic.")
    else:
        print("  Random control separated from concept conditions — intervention is diagnostic.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 2: Pivot Language Diagnosis")
    parser.add_argument("--domain", default="sacred", choices=["sacred", "kinship"],
                        help="Concept domain (ignored when --both-domains is set)")
    parser.add_argument("--both-domains", action="store_true",
                        help="Run both sacred and kinship domains and save a combined plot")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Vector subtraction scaling factor (default: 0.25)")
    parser.add_argument("--vectors-dir", default="outputs/vectors")
    parser.add_argument("--test-sentences", default=None,
                        help="Path to stimuli JSON; defaults to outputs/stimuli/<domain>_pairs.json. "
                             "Ignored when --both-domains is set.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--vector-method", default="both", choices=["mean", "pca", "both"],
                        help="Which concept vectors to use: mean-difference, PCA reading vector, or both (default)")
    args = parser.parse_args()

    if args.both_domains:
        run_both_domains(
            vectors_dir=args.vectors_dir,
            alpha=args.alpha,
            results_dir=args.results_dir,
            vector_method=args.vector_method,
        )
    else:
        run_exp2(
            domain=args.domain,
            alpha=args.alpha,
            vectors_dir=args.vectors_dir,
            test_sentences_path=args.test_sentences,
            results_dir=args.results_dir,
            vector_method=args.vector_method,
        )
