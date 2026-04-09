"""
Experiment 1: Concept Vector Extraction and Same-Language Deletion.

Supports "kinship", "sacred", or both domains (--both-domains flag).

Pipeline:
  1. Load contrastive pairs for the domain
  2. Extract concept vectors across all encoder layers
  3. Run same-language concept deletion tests
  4. Save vectors to outputs/vectors/ and stimuli to outputs/stimuli/

Run with --both-domains to produce outputs/stimuli/sacred_pairs.json
and outputs/stimuli/kinship_pairs.json in one go (required for exp2 --both-domains).
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import DEFAULT_DEVICE, DEFAULT_LAYERS, EXPERIMENT_LANGUAGES, HF_CACHE_DIR, INTERVENTION_LAYERS, MODEL_NAME, VECTORS_DIR
from data.contrastive_pairs import ContrastivePairGenerator, get_concept_words, load_independent_sacred_tokens
from extraction.concept_vectors import extract_concept_vectors, save_concept_vectors
from intervention.hooks import InterventionHook
from intervention.necessity import measure_concept_deletion

from journal.run_manifest import build_manifest


def run_exp1(
    domain: str = "kinship",
    n_per_concept: int = 15,
    layers: list = None,
    device: str = DEFAULT_DEVICE,
    model=None,
    tokenizer=None,
):
    """
    Run Experiment 1: concept vector extraction + deletion test for one domain.

    Args:
        domain: Concept domain ("kinship" or "sacred")
        n_per_concept: Contrastive pairs per concept per language
        layers: Encoder layers to extract (default: all 24)
        device: Compute device
        model: Pre-loaded model (loaded from HF if None)
        tokenizer: Pre-loaded tokenizer (loaded from HF if None)
    """
    if layers is None:
        layers = DEFAULT_LAYERS

    print("=" * 70)
    print(f"EXPERIMENT 1: {domain.capitalize()} Concept Vector Extraction")
    print("=" * 70)

    # --- Load model (shared across domains when run_both_domains passes it in) ---
    if model is None or tokenizer is None:
        print(f"\nLoading {MODEL_NAME} on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR).to(device)
        model.eval()

    # --- Generate contrastive pairs ---
    # Non-English languages are translated from the English templates using the
    # loaded NLLB model, so the stimuli are in the correct source language for
    # activation extraction. Without this translation, concept vectors for
    # arb_Arab/zho_Hant/spa_Latn would be extracted from mis-labeled English text.
    print(f"\nGenerating {domain} contrastive pairs (with NLLB translation for non-English)...")
    generator = ContrastivePairGenerator(seed=42)
    pairs_by_lang = generator.generate_pairs(
        domain=domain,
        n_per_concept=n_per_concept,
        languages=EXPERIMENT_LANGUAGES,
        output_path=f"outputs/stimuli/{domain}_pairs.json",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    # --- Extract concept vectors ---
    print(f"\nExtracting {domain} concept vectors...")
    all_vectors = {}

    for lang in EXPERIMENT_LANGUAGES:
        print(f"\n  Language: {lang}")
        all_vectors[lang] = {}

        lang_pairs = pairs_by_lang.get(lang, {})
        lang_diffs = {}
        for concept, pairs in lang_pairs.items():
            print(f"    Concept: {concept} ({len(pairs)} pairs)")
            both = extract_concept_vectors(
                contrastive_pairs=pairs,
                model=model,
                tokenizer=tokenizer,
                lang_code=lang,
                layers=layers,
                component="encoder_hidden",
                pooling="mean",
                device=device,
                method="both",
                return_diffs=True,
            )
            # Store mean vectors for interventions (backward-compatible)
            all_vectors[lang][concept] = both["mean"]
            all_vectors[lang][f"{concept}_pca"] = both["pca"]
            lang_diffs[concept] = both["diffs"]  # {layer: [n_pairs, hidden_dim]}

        # Save per language — mean vectors in the primary file (backward-compat),
        # PCA vectors and raw diffs in sidecar files for visualization.
        save_path      = f"{VECTORS_DIR}/{domain}_{lang}.pt"
        pca_save_path  = f"{VECTORS_DIR}/{domain}_{lang}_pca.pt"
        diff_save_path = f"{VECTORS_DIR}/{domain}_{lang}_diffs.pt"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        mean_only = {c: lvecs for c, lvecs in all_vectors[lang].items() if not c.endswith("_pca")}
        pca_only  = {c.removesuffix("_pca"): lvecs for c, lvecs in all_vectors[lang].items() if c.endswith("_pca")}
        torch.save(
            {concept: {str(l): v.cpu() for l, v in lvecs.items()}
             for concept, lvecs in mean_only.items()},
            save_path,
        )
        torch.save(
            {concept: {str(l): v.cpu() for l, v in lvecs.items()}
             for concept, lvecs in pca_only.items()},
            pca_save_path,
        )
        torch.save(
            {concept: {str(l): d.cpu() for l, d in layer_diffs.items()}
             for concept, layer_diffs in lang_diffs.items()},
            diff_save_path,
        )
        print(f"    Saved mean vectors to {save_path}")
        print(f"    Saved PCA  vectors to {pca_save_path}")
        print(f"    Saved diff matrices to {diff_save_path}")

    # --- Run same-language deletion tests ---
    # Design: translate source_lang → English, measure whether English concept
    # words survive in the output with vs. without concept vector subtraction.
    # Using English as the fixed target ensures concept_words matching is
    # consistent across all source languages.
    print("\nRunning concept deletion tests (source → English)...")

    deletion_results = {}
    target_lang = "eng_Latn"

    # English concept words and token IDs for checking the output
    eng_concept_words = get_concept_words("eng_Latn", domain=domain)
    eng_token_ids = load_independent_sacred_tokens("eng_Latn", tokenizer, domain=domain)

    for lang in EXPERIMENT_LANGUAGES:
        print(f"\n  Source language: {lang}")
        deletion_results[lang] = {}

        # Build a mean concept vector for this language (averaged over all
        # concepts and layers) to use as the intervention vector
        lang_vecs = all_vectors.get(lang, {})

        for concept, lang_concept_pairs in pairs_by_lang.get(lang, {}).items():
            sentences = [p["positive"] for p in lang_concept_pairs]

            # Baseline: no intervention
            baseline = measure_concept_deletion(
                sentences, model, tokenizer,
                source_lang=lang, target_lang=target_lang,
                concept_token_ids=eng_token_ids,
                concept_words=eng_concept_words,
                device=device,
            )

            # Intervention: subtract this concept's mean vector from encoder
            concept_layer_vecs = lang_vecs.get(concept, {})
            if concept_layer_vecs:
                # Average across layers to get a single intervention vector
                stacked = torch.stack(list(concept_layer_vecs.values()), dim=0)
                mean_vec = stacked.mean(dim=0).to(device)

                hook = InterventionHook()
                hook.register_vector_subtraction_hook(model, mean_vec, INTERVENTION_LAYERS, alpha=1.0)
                ablated = measure_concept_deletion(
                    sentences, model, tokenizer,
                    source_lang=lang, target_lang=target_lang,
                    concept_token_ids=eng_token_ids,
                    concept_words=eng_concept_words,
                    intervention=hook,
                    device=device,
                )
                hook.cleanup()
                deletion_rate = baseline["concept_present_rate"] - ablated["concept_present_rate"]
            else:
                ablated = baseline
                deletion_rate = 0.0

            deletion_results[lang][concept] = {
                "baseline_present_rate": baseline["concept_present_rate"],
                "ablated_present_rate": ablated["concept_present_rate"],
                "deletion_rate": max(0.0, deletion_rate),
                "baseline_mean_prob": baseline["mean_concept_probability"],
                "ablated_mean_prob": ablated["mean_concept_probability"],
            }
            print(f"    {concept}: baseline={baseline['concept_present_rate']:.2f} "
                  f"→ ablated={ablated['concept_present_rate']:.2f} "
                  f"(deletion={max(0.0, deletion_rate):.2f})")

    # Save results
    out_path = f"outputs/exp1_{domain}_deletion.json"
    manifest_path = f"outputs/manifests/exp1_{domain}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    run_manifest = build_manifest(
        f"exp1_{domain}",
        extra={
            "domain": domain,
            "n_per_concept": n_per_concept,
            "layers": layers,
            "pooling": "mean",
            "vector_extraction_method": "both",
            "stimuli_path": f"outputs/stimuli/{domain}_pairs.json",
            "vectors_glob": f"{VECTORS_DIR}/{domain}_*.pt",
            "output_json": out_path,
        },
    )
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(run_manifest, mf, indent=2)
    with open(out_path, "w") as f:
        json.dump({
            "run_manifest": run_manifest,
            "deletion_results": deletion_results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Manifest saved to {manifest_path}")

    return all_vectors, deletion_results


def run_both_domains(
    n_per_concept: int = 15,
    layers: list = None,
    device: str = DEFAULT_DEVICE,
) -> dict:
    """
    Run Experiment 1 for both sacred and kinship domains, sharing one model load.

    Produces:
      outputs/stimuli/sacred_pairs.json
      outputs/stimuli/kinship_pairs.json
      outputs/vectors/sacred_<lang>.pt  (for each language)
      outputs/vectors/kinship_<lang>.pt (for each language)
    """
    print(f"\nLoading {MODEL_NAME} on {device} (shared for both domains)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR).to(device)
    model.eval()

    all_results = {}
    for domain in ["sacred", "kinship"]:
        print(f"\n{'#'*70}")
        print(f"# DOMAIN: {domain.upper()}")
        print(f"{'#'*70}")
        vectors, deletion = run_exp1(
            domain=domain,
            n_per_concept=n_per_concept,
            layers=layers,
            device=device,
            model=model,
            tokenizer=tokenizer,
        )
        all_results[domain] = {"vectors": vectors, "deletion": deletion}

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 1: Concept Vector Extraction")
    parser.add_argument("--domain", default="kinship", choices=["kinship", "sacred"],
                        help="Concept domain (ignored when --both-domains is set)")
    parser.add_argument("--both-domains", action="store_true",
                        help="Run both kinship and sacred domains (shares model load)")
    parser.add_argument("--n-per-concept", type=int, default=15,
                        help="Contrastive pairs per concept per language")
    args = parser.parse_args()

    if args.both_domains:
        run_both_domains(n_per_concept=args.n_per_concept)
    else:
        run_exp1(domain=args.domain, n_per_concept=args.n_per_concept)
