"""
Experiment 3: Layer-Wise Representation Convergence (scaled up).

Key changes from original:
  - Uses FLORES-200 parallel sentences (100 per language) instead of 4 sentences.
    t-SNE/UMAP need ≥100 points per language to produce meaningful structure.
  - Panel layers fixed to [0, 8, 16, 23] (valid for NLLB-1.3B's 24 encoder layers).
  - Generates both t-SNE and UMAP panels.
  - Silhouette trajectory saved as a separate figure.
  - All outputs saved to results/.

FLORES-200 loading requires the `datasets` library and network access.
If FLORES is unavailable, falls back to the built-in 4-sentence sample with a
clear warning that the resulting t-SNE plots are not interpretable.
"""

import json
import os
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import DEFAULT_DEVICE, DEFAULT_LAYERS, EXPERIMENT_LANGUAGES, HF_CACHE_DIR, MODEL_NAME
from analysis.layer_wise import (
    extract_parallel_representations,
    compute_cka_similarity,
    compute_english_centricity_by_layer,
    compute_silhouette_by_language,
)
from visualization.layer_analysis import (
    plot_cka_curves,
    plot_tsne_panels,
    plot_umap_panels,
    plot_english_centricity,
    plot_silhouette_trajectory,
)

# Panel layers — must be valid indices for 24-layer NLLB-1.3B (0–23)
PANEL_LAYERS = [0, 8, 16, 23]
N_FLORES_SENTENCES = 100    # per language; enough for interpretable t-SNE/UMAP

# Minimal fallback (used when FLORES-200 is unavailable)
PARALLEL_SENTENCES_SAMPLE = {
    "eng_Latn": [
        "God guides humanity with love.",
        "The family is the cornerstone of society.",
        "The stone stands in the field.",
        "The president leads the nation.",
    ],
    "arb_Arab": [
        "يرشد الله البشرية بالمحبة.",
        "الأسرة هي ركيزة المجتمع.",
        "يقف الحجر في الحقل.",
        "يقود الرئيس الأمة.",
    ],
    "zho_Hant": [
        "神用愛引導人類。",
        "家庭是社會的基石。",
        "石頭站在田野裡。",
        "總統領導國家。",
    ],
    "spa_Latn": [
        "Dios guía a la humanidad con amor.",
        "La familia es la piedra angular de la sociedad.",
        "La piedra se encuentra en el campo.",
        "El presidente lidera la nación.",
    ],
}


def load_flores200(languages: list, n_sentences: int = 100) -> dict:
    """
    Load parallel sentences from the cached FLORES+ dataset for the specified languages.

    The dataset ID is read from the FLORES_DATASET environment variable (set by
    scripts/config.sh to "openlanguagedata/flores_plus"). This must match whatever
    was downloaded by scripts/download_models.py so that offline HF cache lookups
    succeed on compute nodes.

    Uses the 'devtest' split (~1012 sentences per language). Returns the first
    n_sentences for reproducibility. Falls back to PARALLEL_SENTENCES_SAMPLE
    if the datasets library is unavailable or the dataset cannot be loaded.

    Args:
        languages: NLLB-style language codes (e.g. "eng_Latn")
        n_sentences: Number of sentences to load per language

    Returns:
        {lang_code: [sentence_strings]}
    """
    import os

    try:
        from datasets import load_dataset
    except ImportError:
        print("WARNING: `datasets` library not installed. Falling back to 4-sentence sample.")
        print("         t-SNE / UMAP results will not be interpretable with only 4 points.")
        return {l: PARALLEL_SENTENCES_SAMPLE.get(l, []) for l in languages}

    # Match the dataset ID used in download_models.py so the HF cache hit works.
    dataset_id = os.environ.get("FLORES_DATASET", "openlanguagedata/flores_plus")
    print(f"  Dataset: {dataset_id}")

    # FLORES+ uses ISO 639-3 codes that differ from NLLB's BCP-47 tags in some cases.
    # Map NLLB codes → FLORES+ config names for the load_dataset() call only;
    # the returned dict still uses NLLB codes so the rest of the pipeline is unchanged.
    NLLB_TO_FLORES = {
        "zho_Hant": "cmn_Hant",   # NLLB uses zho (macrolanguage), FLORES+ uses cmn
    }

    parallel = {}
    loaded_any = False

    for lang in languages:
        flores_lang = NLLB_TO_FLORES.get(lang, lang)
        try:
            print(f"  Loading FLORES+ for {lang} (config: {flores_lang})...")
            ds = load_dataset(
                dataset_id,
                flores_lang,
                split="devtest",
                cache_dir=HF_CACHE_DIR,
            )
            # openlanguagedata/flores_plus uses "text"; older facebook/flores uses "sentence"
            text_col = "text" if "text" in ds.column_names else "sentence"
            sentences = [item[text_col] for item in ds][:n_sentences]
            parallel[lang] = sentences
            print(f"    Loaded {len(sentences)} sentences")
            loaded_any = True
        except Exception as e:
            print(f"  WARNING: Could not load FLORES+ for {lang}: {e}")
            parallel[lang] = PARALLEL_SENTENCES_SAMPLE.get(lang, [])

    if not loaded_any:
        print("WARNING: All FLORES+ loads failed. Falling back to 4-sentence sample.")
        print("         Run: bash scripts/download_cache.sh  (on a login node)")

    return parallel


def run_exp3(
    layers: list = None,
    panel_layers: list = None,
    parallel_sentences: dict = None,
    device: str = DEFAULT_DEVICE,
    results_dir: str = "results",
    use_flores: bool = True,
):
    """
    Run Experiment 3: layer-wise convergence analysis.

    Args:
        layers: Encoder layers to analyze (default: all 12)
        panel_layers: Layers for t-SNE / UMAP panels
        parallel_sentences: {lang: [sentences]} — overrides FLORES loading if provided
        device: Compute device
        results_dir: Output directory for figures and JSON
        use_flores: Whether to attempt FLORES-200 loading (default: True)
    """
    if layers is None:
        layers = DEFAULT_LAYERS
    if panel_layers is None:
        panel_layers = PANEL_LAYERS

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(results_dir, "json")).mkdir(exist_ok=True)
    Path(os.path.join(results_dir, "figures")).mkdir(exist_ok=True)
    Path(os.path.join(results_dir, "vectors")).mkdir(exist_ok=True)
    
    print("=" * 70)
    print("EXPERIMENT 3: Layer-Wise Representation Convergence (Scaled)")
    print("=" * 70)

    # --- Load model ---
    print(f"\nLoading {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR).to(device)
    model.eval()

    # --- Load parallel sentences ---
    if parallel_sentences is None:
        if use_flores:
            print(f"\nLoading FLORES-200 ({N_FLORES_SENTENCES} sentences per language)...")
            parallel_sentences = load_flores200(EXPERIMENT_LANGUAGES, N_FLORES_SENTENCES)
        else:
            print("\nUsing built-in sample sentences (t-SNE will not be interpretable).")
            parallel_sentences = PARALLEL_SENTENCES_SAMPLE

    langs_available = [l for l in EXPERIMENT_LANGUAGES
                       if l in parallel_sentences and parallel_sentences[l]]
    parallel_sentences = {l: parallel_sentences[l] for l in langs_available}
    n_per_lang = {l: len(s) for l, s in parallel_sentences.items()}
    print(f"  Sentences per language: {n_per_lang}")

    if any(n < 20 for n in n_per_lang.values()):
        print("  WARNING: < 20 sentences for some languages. "
              "t-SNE / UMAP results may not be meaningful.")

    # --- Extract representations ---
    print("\nExtracting parallel representations...")
    reps = extract_parallel_representations(parallel_sentences, model, tokenizer, layers, device)

    # --- CKA analysis ---
    print("\nComputing CKA similarity...")
    cka_by_pair_by_layer = {}
    lang_list = list(reps.keys())

    for i, la in enumerate(lang_list):
        for j, lb in enumerate(lang_list):
            if i >= j:
                continue
            pair = (la, lb)
            cka_by_pair_by_layer[pair] = {}
            for layer in layers:
                if layer in reps.get(la, {}) and layer in reps.get(lb, {}):
                    cka_by_pair_by_layer[pair][layer] = compute_cka_similarity(
                        reps[la][layer], reps[lb][layer]
                    )

    # --- English-centricity ---
    print("\nComputing English-centricity...")
    centricity = compute_english_centricity_by_layer(reps, layers)

    # --- Silhouette scores ---
    print("\nComputing silhouette scores per layer...")
    silhouette_by_layer = {}
    for layer in layers:
        layer_reps = {lang: reps[lang][layer] for lang in reps if layer in reps[lang]}
        if len(layer_reps) >= 2:
            silhouette_by_layer[layer] = compute_silhouette_by_language(layer_reps)

    # --- Print summary ---
    print(f"\n  Silhouette scores at panel layers:")
    for layer in panel_layers:
        score = silhouette_by_layer.get(layer, float("nan"))
        print(f"    Layer {layer:2d}: {score:.4f}")

    # --- Save results ---
    results = {
        "n_sentences_per_lang": n_per_lang,
        "panel_layers": panel_layers,
        "cka_by_pair_by_layer": {str(k): {str(l): v for l, v in ldict.items()}
                                  for k, ldict in cka_by_pair_by_layer.items()},
        "english_centricity": {str(l): v for l, v in centricity.items()},
        "silhouette_by_layer": {str(l): v for l, v in silhouette_by_layer.items()},
    }
    out_path = f"{results_dir}/json/exp3_layer_wise.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # --- Visualize ---
    plot_cka_curves(cka_by_pair_by_layer,
                    save_path=f"{results_dir}/figures/exp3_cka_curves.png")
    plot_tsne_panels(reps, panel_layers=panel_layers,
                     save_path=f"{results_dir}/figures/tsne_panels_scaled.png")
    plot_umap_panels(reps, panel_layers=panel_layers,
                     save_path=f"{results_dir}/figures/umap_panels_scaled.png")
    plot_english_centricity(centricity,
                            save_path=f"{results_dir}/figures/exp3_english_centricity.png")
    plot_silhouette_trajectory(silhouette_by_layer,
                               save_path=f"{results_dir}/figures/silhouette_trajectory.png")

    return reps, cka_by_pair_by_layer, centricity, silhouette_by_layer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 3: Layer-wise convergence")
    parser.add_argument("--no-flores", action="store_true",
                        help="Skip FLORES loading; use built-in sample sentences")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    run_exp3(use_flores=not args.no_flores, results_dir=args.results_dir)
