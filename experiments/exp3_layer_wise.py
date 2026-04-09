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

from config import DEFAULT_DEVICE, DEFAULT_LAYERS, EXPERIMENT_LANGUAGES, HF_CACHE_DIR, MODEL_NAME, VECTORS_DIR
from journal.run_manifest import build_manifest
from analysis.layer_wise import (
    extract_parallel_representations,
    compute_cka_similarity,
    compute_english_centricity_by_layer,
    compute_silhouette_by_language,
    load_diff_matrices,
    compute_concept_direction_alignment,
    compute_projection_consistency,
    compute_cross_lingual_projection_transfer,
    compute_linear_probe_cv_accuracy,
    compute_cross_lingual_probe_transfer,
)
from visualization.layer_analysis import (
    plot_cka_curves,
    plot_tsne_panels,
    plot_umap_panels,
    plot_english_centricity,
    plot_silhouette_trajectory,
)
from visualization.pca_vs_mean import (
    compute_layer_stats,
    plot_concept_direction_alignment,
    plot_projection_consistency,
    plot_cross_lingual_projection_transfer,
    plot_pc1_explained_variance,
    plot_linear_probe_accuracy,
    plot_cross_lingual_probe_heatmap,
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
    domains: list = None,
    vectors_dir: str = None,
    skip_concept_geometry: bool = False,
):
    """
    Run Experiment 3: layer-wise convergence analysis + concept direction geometry.

    Args:
        layers: Encoder layers to analyze (default: all 24)
        panel_layers: Layers for t-SNE / UMAP panels
        parallel_sentences: {lang: [sentences]} — overrides FLORES loading if provided
        device: Compute device
        results_dir: Output directory for figures and JSON
        use_flores: Whether to attempt FLORES-200 loading (default: True)
        domains: Concept domains to analyze geometry for, e.g. ["sacred", "kinship"].
            Defaults to both if diff files are present, skipped if none found.
        vectors_dir: Directory containing exp1 diff files (defaults to VECTORS_DIR).
        skip_concept_geometry: If True, skip the concept direction geometry section
            entirely (useful when running exp3 before exp1).
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
    out_path = f"{results_dir}/json/exp3_layer_wise.json"
    run_manifest = build_manifest(
        "exp3_layer_wise",
        extra={
            "layers": layers,
            "panel_layers": panel_layers,
            "use_flores": use_flores,
            "flores_dataset": os.environ.get("FLORES_DATASET", "openlanguagedata/flores_plus"),
            "n_sentences_per_lang": n_per_lang,
            "output_json": out_path,
        },
    )
    results = {
        "run_manifest": run_manifest,
        "n_sentences_per_lang": n_per_lang,
        "panel_layers": panel_layers,
        "cka_by_pair_by_layer": {str(k): {str(l): v for l, v in ldict.items()}
                                  for k, ldict in cka_by_pair_by_layer.items()},
        "english_centricity": {str(l): v for l, v in centricity.items()},
        "silhouette_by_layer": {str(l): v for l, v in silhouette_by_layer.items()},
    }
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

    # -----------------------------------------------------------------------
    # Concept direction geometry (requires exp1 diff matrices)
    # -----------------------------------------------------------------------
    if not skip_concept_geometry:
        _run_concept_geometry(
            layers=layers,
            domains=domains,
            vectors_dir=vectors_dir or VECTORS_DIR,
            results_dir=results_dir,
        )

    return reps, cka_by_pair_by_layer, centricity, silhouette_by_layer


def _run_concept_geometry(
    layers: list,
    domains: list,
    vectors_dir: str,
    results_dir: str,
):
    """
    Concept Direction Geometry section of Experiment 3.

    Loads per-pair activation difference matrices saved by exp1 and produces
    four figures per domain:

      1. concept_direction_alignment_{domain}.png
           Layer-wise |cosine| between PCA concept directions extracted from
           different languages.  Peak alignment at intervention layers (10-15)
           supports a shared cross-lingual concept subspace.

      2. projection_consistency_{domain}.png
           Fraction of contrastive pairs whose difference vector projects
           positively onto the PCA direction.  Near 1.0 = reliable probe;
           near 0.5 = noisy direction.

      3. pc1_explained_variance_{domain}.png
           PC1 explained variance ratio per language × concept across layers.
           Shows where the concept signal is most linearly concentrated.

      4. cross_lingual_projection_transfer_{domain}_layer{k}.png
           NxN heatmap at the peak intervention layer.  Entry [i,j] = fraction
           of language_i's pair differences that project positively onto
           language_j's PCA direction — the representation-space analogue of
           the causal transfer matrix (exp4).
    """
    from pathlib import Path
    from visualization.pca_vs_mean import compute_layer_stats

    if domains is None:
        # Auto-detect from available diff files
        domains = [
            d for d in ("sacred", "kinship")
            if any(
                (Path(vectors_dir) / f"{d}_{lang}_diffs.pt").exists()
                for lang in EXPERIMENT_LANGUAGES
            )
        ]

    if not domains:
        print("\n  [Concept geometry] No diff matrices found — run exp1 first.")
        return

    geom_dir = Path(results_dir) / "figures"
    geom_dir.mkdir(parents=True, exist_ok=True)

    # Layer used for the transfer heatmap — middle of intervention window
    heatmap_layer = 12

    print("\n" + "=" * 70)
    print("CONCEPT DIRECTION GEOMETRY  (PCA-based representation analysis)")
    print("=" * 70)

    summaries = {}
    for domain in domains:
        print(f"\n  Domain: {domain}")

        diff_by_lang = load_diff_matrices(domain, EXPERIMENT_LANGUAGES, vectors_dir)
        if not diff_by_lang:
            print(f"    No diff matrices loaded for {domain} — skipping.")
            continue

        # 1. Cross-language alignment
        alignment = compute_concept_direction_alignment(diff_by_lang, layers)
        if alignment:
            plot_concept_direction_alignment(
                alignment, layers, domain=domain,
                save_path=geom_dir / f"concept_direction_alignment_{domain}.png",
            )
        else:
            print(f"    Skipping alignment plot (need ≥2 languages with diff data).")

        # 2. Projection consistency
        consistency = compute_projection_consistency(diff_by_lang, layers)
        if consistency:
            plot_projection_consistency(
                consistency, layers, domain=domain,
                save_path=geom_dir / f"projection_consistency_{domain}.png",
            )

        # 3. PC1 explained variance (reuse compute_layer_stats from pca_vs_mean)
        all_stats = {}
        for lang, concepts in diff_by_lang.items():
            for concept, layer_diffs in concepts.items():
                label = f"{lang.split('_')[0]}/{concept}"
                all_stats[label] = compute_layer_stats(layer_diffs)
        if all_stats:
            plot_pc1_explained_variance(
                all_stats, layers, domain=domain,
                save_path=geom_dir / f"pc1_explained_variance_{domain}.png",
            )

        # 4. Cross-lingual projection transfer heatmap at heatmap_layer
        matrix, matrix_langs = compute_cross_lingual_projection_transfer(
            diff_by_lang, layer=heatmap_layer,
        )
        if len(matrix_langs) >= 2:
            plot_cross_lingual_projection_transfer(
                matrix, matrix_langs, layer=heatmap_layer, domain=domain,
                save_path=geom_dir / f"cross_lingual_projection_transfer_{domain}_layer{heatmap_layer}.png",
            )
        else:
            print(f"    Skipping transfer heatmap (need ≥2 languages at layer {heatmap_layer}).")

        # 5. Linear probe accuracy (k-fold CV, circularity-free replacement for
        #    projection consistency)
        cv_accuracy = compute_linear_probe_cv_accuracy(diff_by_lang, layers)
        if cv_accuracy:
            plot_linear_probe_accuracy(
                cv_accuracy, layers, domain=domain,
                save_path=geom_dir / f"linear_probe_accuracy_{domain}.png",
            )
        else:
            print("    Skipping linear probe accuracy (insufficient pairs for CV).")

        # 6. Cross-lingual linear probe transfer heatmap — train on lang_i, test on lang_j
        probe_matrix, probe_langs = compute_cross_lingual_probe_transfer(
            diff_by_lang, layer=heatmap_layer,
        )
        if len(probe_langs) >= 2:
            plot_cross_lingual_probe_heatmap(
                probe_matrix, probe_langs, layer=heatmap_layer, domain=domain,
                save_path=geom_dir / f"cross_lingual_probe_transfer_{domain}_layer{heatmap_layer}.png",
            )
        else:
            print(f"    Skipping probe transfer heatmap (need ≥2 languages at layer {heatmap_layer}).")

        summaries[domain] = {
            "heatmap_layer": heatmap_layer,
            "alignment_by_pair_by_layer": {
                str(k): {str(layer): float(v) for layer, v in layer_map.items()}
                for k, layer_map in alignment.items()
            },
            "projection_consistency_by_label_by_layer": {
                label: {str(layer): float(v) for layer, v in layer_map.items()}
                for label, layer_map in consistency.items()
            },
            "linear_probe_cv_accuracy_by_label_by_layer": {
                label: {str(layer): float(v) for layer, v in layer_map.items()}
                for label, layer_map in cv_accuracy.items()
            },
            "cross_lingual_projection_transfer": {
                "languages": matrix_langs,
                "matrix": matrix.tolist(),
            },
            "cross_lingual_probe_transfer": {
                "languages": probe_langs,
                "matrix": probe_matrix.tolist(),
            },
        }

    print("\n  Concept direction geometry complete.")
    json_dir = Path(results_dir) / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    out_path = json_dir / "exp3_concept_geometry_summary.json"
    with open(out_path, "w") as f:
        json.dump({"domains": summaries}, f, indent=2)
    print(f"  Concept geometry summary saved to {out_path}")
    return summaries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 3: Layer-wise convergence")
    parser.add_argument("--no-flores", action="store_true",
                        help="Skip FLORES loading; use built-in sample sentences")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--vectors-dir", default=VECTORS_DIR,
                        help="Directory with exp1 diff matrices (default: outputs/vectors)")
    parser.add_argument("--domain", default=None,
                        choices=["sacred", "kinship"],
                        help="Single concept domain for geometry analysis. "
                             "Defaults to both if diff files are found.")
    parser.add_argument("--skip-concept-geometry", action="store_true",
                        help="Skip concept direction geometry section "
                             "(use when exp1 has not been run yet)")
    args = parser.parse_args()

    domains = [args.domain] if args.domain else None

    run_exp3(
        use_flores=not args.no_flores,
        results_dir=args.results_dir,
        domains=domains,
        vectors_dir=args.vectors_dir,
        skip_concept_geometry=args.skip_concept_geometry,
    )
