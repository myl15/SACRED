"""
Calibration script: find the optimal alpha for scaled vector subtraction.

Runs calibration on the sacred concept using English sentences as test input
and the Arabic concept vector as the intervention, targeting residual stream
layers [10, 11, 12, 13] (peak circuit layers, proportionally scaled for 24-layer 1.3B).

If sacred concept vectors don't yet exist in outputs/vectors/, they are
generated automatically using ContrastivePairGenerator + extract_concept_vectors.

Outputs:
  results/calibration_curve.png   — deletion_rate vs alpha
  results/calibration_results.json — raw calibration metrics
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import DEFAULT_DEVICE, EXPERIMENT_LANGUAGES, HF_CACHE_DIR, INTERVENTION_LAYERS, MODEL_NAME, VECTORS_DIR, DEFAULT_LAYERS
from data.contrastive_pairs import ContrastivePairGenerator, get_concept_words, load_independent_sacred_tokens
from extraction.concept_vectors import extract_concept_vectors, save_concept_vectors
from intervention.calibration import calibrate_intervention_strength, find_optimal_alpha, plot_calibration_curve

# Calibration parameters
CALIBRATION_LAYERS = INTERVENTION_LAYERS  # must match the layers used in exp1/exp2/exp4
DOMAIN = "sacred"
SOURCE_LANG = "eng_Latn"             # test sentences are in English
VECTOR_LANG = "arb_Arab"             # use Arabic concept vector (cross-lingual test)
TARGET_LANG = "spa_Latn"             # translate English → Spanish, check concept in output
N_CALIBRATION_SENTENCES = 20        # number of test sentences

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def _ensure_sacred_vectors(model, tokenizer, device):
    """
    Load existing sacred concept vectors, or generate + save them if missing.

    Returns {lang: {layer: tensor[1024]}} averaged over all sacred concepts.
    """
    concept_vectors = {}

    for lang in EXPERIMENT_LANGUAGES:
        vec_path = f"{VECTORS_DIR}/{DOMAIN}_{lang}.pt"
        if Path(vec_path).exists():
            raw = torch.load(vec_path, map_location=device)
            concept_vectors[lang] = _average_vectors(raw)
            print(f"  Loaded {lang}: {len(concept_vectors[lang])} layers")
        else:
            print(f"  Generating sacred concept vectors for {lang}...")
            generator = ContrastivePairGenerator(seed=42)
            pairs_by_lang = generator.generate_pairs(
                domain=DOMAIN,
                n_per_concept=15,
                languages=[lang],
            )
            lang_vecs = {}
            for concept, pairs in pairs_by_lang.get(lang, {}).items():
                vecs = extract_concept_vectors(
                    contrastive_pairs=pairs,
                    model=model,
                    tokenizer=tokenizer,
                    lang_code=lang,
                    layers=DEFAULT_LAYERS,
                    component="encoder_hidden",
                    pooling="mean",
                    device=device,
                    method="mean",
                )
                for layer, vec in vecs.items():
                    lang_vecs.setdefault(layer, []).append(vec)

            # Average over concepts
            avg_vecs = {layer: torch.stack(vecs).mean(0) for layer, vecs in lang_vecs.items()}
            concept_vectors[lang] = avg_vecs

            # Save in the same format as exp1_kinship
            Path(vec_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"_averaged": {str(l): v.cpu() for l, v in avg_vecs.items()}},
                vec_path,
            )
            print(f"  Saved to {vec_path}")

    return concept_vectors


def _average_vectors(raw: dict) -> dict:
    """Average all concept vectors to get one vector per layer."""
    layer_vecs = {}
    for concept, layer_dict in raw.items():
        for layer_str, vec in layer_dict.items():
            layer = int(layer_str)
            layer_vecs.setdefault(layer, []).append(vec)
    return {layer: torch.stack(vecs).mean(0) for layer, vecs in layer_vecs.items()}


def _get_layer_mean_vector(vecs_by_layer: dict, layers: list) -> torch.Tensor:
    available = [vecs_by_layer[l] for l in layers if l in vecs_by_layer]
    if not available:
        raise KeyError(f"No vectors for layers {layers}")
    return torch.stack(available).mean(0)


def main():
    print("=" * 70)
    print("CALIBRATION: Scaled Vector Subtraction Strength")
    print("=" * 70)
    print(f"  Domain: {DOMAIN}")
    print(f"  Source lang (test sentences): {SOURCE_LANG}")
    print(f"  Vector lang: {VECTOR_LANG}")
    print(f"  Target lang (translation output): {TARGET_LANG}")
    print(f"  Layers: {CALIBRATION_LAYERS}")

    # --- Load model ---
    print(f"\nLoading {MODEL_NAME} on {DEFAULT_DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR).to(DEFAULT_DEVICE)
    model.eval()

    # --- Load / generate sacred concept vectors ---
    print(f"\nLoading {DOMAIN} concept vectors...")
    concept_vectors = _ensure_sacred_vectors(model, tokenizer, DEFAULT_DEVICE)

    # Get the Arabic concept vector (cross-lingual test)
    vec_arb = _get_layer_mean_vector(concept_vectors[VECTOR_LANG], CALIBRATION_LAYERS)
    vec_arb = vec_arb.to(DEFAULT_DEVICE)
    print(f"  Arabic concept vector shape: {vec_arb.shape}, norm: {vec_arb.norm():.4f}")

    # --- Generate English sacred test sentences ---
    print(f"\nGenerating English sacred test sentences...")
    generator = ContrastivePairGenerator(seed=42)
    pairs = generator.generate_pairs(
        domain=DOMAIN, n_per_concept=5, languages=[SOURCE_LANG],
    )
    test_sentences = []
    for concept_pairs in pairs.get(SOURCE_LANG, {}).values():
        test_sentences.extend([p["positive"] for p in concept_pairs])
    test_sentences = test_sentences[:N_CALIBRATION_SENTENCES]
    print(f"  Using {len(test_sentences)} test sentences")

    # Concept token IDs and words for checking output
    concept_token_ids = load_independent_sacred_tokens(TARGET_LANG, tokenizer, domain=DOMAIN)
    concept_words = get_concept_words(TARGET_LANG, domain=DOMAIN)

    # --- Run calibration ---
    cal_results = calibrate_intervention_strength(
        model=model,
        tokenizer=tokenizer,
        concept_vector=vec_arb,
        sentences=test_sentences,
        lang_code=SOURCE_LANG,
        target_lang=TARGET_LANG,
        concept_token_ids=concept_token_ids,
        alphas=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
        layers=CALIBRATION_LAYERS,
        target="residual",
        device=DEFAULT_DEVICE,
        concept_words=concept_words,
    )

    # --- Report optimal alpha ---
    optimal = find_optimal_alpha(cal_results)
    print(f"\n  Recommended alpha: {optimal:.2f}")

    # --- Plot ---
    plot_calibration_curve(
        cal_results,
        save_path=str(RESULTS_DIR / "calibration_curve.png"),
        title=f"Calibration: {DOMAIN} concept, {VECTOR_LANG}→{SOURCE_LANG} sentences, "
              f"layers {CALIBRATION_LAYERS}",
    )

    # --- Save results ---
    out = {
        "domain": DOMAIN,
        "source_lang": SOURCE_LANG,
        "vector_lang": VECTOR_LANG,
        "target_lang": TARGET_LANG,
        "layers": CALIBRATION_LAYERS,
        "recommended_alpha": float(optimal),
        "calibration": {str(a): v for a, v in cal_results.items()},
    }
    out_path = RESULTS_DIR / "calibration_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Calibration results saved to {out_path}")

    print(f"\nSummary: use alpha={optimal:.2f} for experiments 2, 4, and 5.")
    return cal_results, optimal


if __name__ == "__main__":
    main()
