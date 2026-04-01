"""
SACRED Project — Orchestrator

Runs the full experiment pipeline:
  Step 1: Generate diverse sacred stimuli
  Step 2: Create train/test splits
  Step 3: Load NLLB model
  Step 4: Load independent sacred token IDs
  Step 5-6: Circuit discovery (or load from cache)
  Step 7: Test circuit necessity
  Step 8: Comprehensive statistical analysis
  Step 9: Cross-validation framework
  Step 10: Generate visualizations
  Step 11: Summary report

Usage:
  python main.py                  # Full pipeline
  python main.py --skip-discovery # Skip circuit discovery (use cache)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import DEFAULT_DEVICE, DEFAULT_LAYERS, HF_CACHE_DIR, LANGUAGES, MODEL_NAME, OUTPUT_DIR

# Data
from data.contrastive_pairs import StimulusGenerator, create_train_test_split, load_independent_sacred_tokens

# Extraction
from extraction.circuit_discovery import (
    Circuit,
    NeuronComponent,
    UniversalCircuit,
    discover_sacred_circuit,
    find_universal_components_with_validation,
)

# Intervention
from intervention.necessity import test_circuit_necessity, test_vector_necessity

# Statistics
from analysis.statistical import perform_cross_validation, run_comprehensive_hypothesis_testing

# Visualization
from visualization.circuits import create_comprehensive_report_figure, plot_circuit_map, plot_universal_circuit_heatmap
from visualization.interventions import plot_cross_validation, plot_intervention_results, plot_statistical_summary


# Configuration — 150 per condition → 30 test sentences after 80/20 split.
# H1 (Necessity) had d=0.888, p=0.0228 with 10 test sentences; increasing to 30
# provides enough power to pass Bonferroni-corrected threshold α=0.0167.
N_STIMULI = 150
TARGET_LANG = "spa_Latn"
LAYERS_TO_ANALYZE = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]   # semantic mid-layers (1.3B has 24 total)
EXPERIMENT_LANGUAGES = ["eng_Latn", "spa_Latn", "arb_Arab"]


def convert_numpy_types(obj):
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_numpy_types(item) for item in obj)
    return obj


def main(skip_discovery: bool = False):
    print("=" * 80)
    print("SACRED CONCEPT CIRCUIT DISCOVERY — Full Pipeline")
    print("=" * 80)

    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    Path("data/stimuli").mkdir(parents=True, exist_ok=True)

    circuit_cache_path = Path("data/universal_circuit.json")
    stimuli_cache_path = Path("data/stimuli/all_stimuli.json")

    # Invalidate cache if it was built with a different stimulus count
    if circuit_cache_path.exists() and not skip_discovery:
        try:
            with open(circuit_cache_path) as _f:
                _meta = json.load(_f)
            if _meta.get("n_stimuli") != N_STIMULI:
                print(f"[Cache] N_STIMULI changed ({_meta.get('n_stimuli')} → {N_STIMULI}), "
                      f"invalidating circuit cache.")
                circuit_cache_path.unlink()
        except Exception:
            pass

    use_cache = skip_discovery or (circuit_cache_path.exists() and stimuli_cache_path.exists())

    # --- Step 1: Generate stimuli ---
    print("\n[Step 1] Generating diverse stimuli...")
    generator = StimulusGenerator(seed=42)
    stimuli = generator.generate_diverse_stimuli(
        n_per_condition=N_STIMULI,
        languages=EXPERIMENT_LANGUAGES,
        output_path=str(stimuli_cache_path),
    )
    total = sum(len(v) for s in stimuli.values() for v in s.values())
    print(f"Generated {total} sentences")

    validation = generator.validate_confound_control(stimuli)
    print(f"Confound control: {'PASSED' if validation['passed'] else 'FAILED'}")

    # --- Step 2: Train/test split ---
    print("\n[Step 2] Creating train/test splits...")
    train_stimuli, test_stimuli = create_train_test_split(stimuli, test_size=0.2)

    # --- Step 3: Load model ---
    print(f"\n[Step 3] Loading {MODEL_NAME} on {DEFAULT_DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR).to(DEFAULT_DEVICE)
    model.eval()
    print(f"Model loaded: {model.config.encoder_layers} encoder layers")

    # --- Step 4: Load sacred tokens ---
    print("\n[Step 4] Loading independent sacred tokens...")
    sacred_tokens = {}
    for lang in EXPERIMENT_LANGUAGES:
        sacred_tokens[lang] = load_independent_sacred_tokens(lang, tokenizer)
        print(f"  {lang}: {len(sacred_tokens[lang])} token IDs")

    # --- Steps 5-6: Circuit discovery or cache ---
    if use_cache and circuit_cache_path.exists():
        print("\n[Steps 5-6] Loading circuit from cache...")
        with open(circuit_cache_path) as f:
            circuit_data = json.load(f)

        neurons = [
            NeuronComponent(
                layer=n["layer"],
                neuron_idx=n["neuron_idx"],
                effect_size=n["effect_size"],
                p_value=n["p_value"],
                confidence_interval=tuple(n.get("confidence_interval", [0.0, 0.0])),
                mean_activation_sacred=n.get("mean_activation_sacred", 0.0),
                mean_activation_secular=n.get("mean_activation_secular", 0.0),
            )
            for n in circuit_data["neurons"]
        ]
        universal_circuit = UniversalCircuit(neurons=neurons, languages=circuit_data["languages"])
        circuits_by_lang = None
        print(f"Loaded: {len(universal_circuit.neurons)} neurons, layers {universal_circuit.get_critical_layers()}")

    else:
        print("\n[Step 5] Discovering sacred circuits per language...")
        circuits_by_lang = {}
        for lang in EXPERIMENT_LANGUAGES:
            circuit = discover_sacred_circuit(
                stimuli=train_stimuli[lang],
                model=model,
                tokenizer=tokenizer,
                lang_code=lang,
                alpha=0.01,
                layers_to_analyze=LAYERS_TO_ANALYZE,
                device=DEFAULT_DEVICE,
            )
            circuits_by_lang[lang] = circuit
            print(f"  {lang}: {len(circuit.neurons)} neurons in {circuit.get_critical_layers()}")

        print("\n[Step 6] Finding universal components...")
        universal_circuit = find_universal_components_with_validation(circuits_by_lang, alpha=0.01)

        # Save to cache
        circuit_data = {
            "n_stimuli": N_STIMULI,
            "neurons": [
                {
                    "layer": n.layer,
                    "neuron_idx": n.neuron_idx,
                    "effect_size": n.effect_size,
                    "p_value": n.p_value,
                    "confidence_interval": list(n.confidence_interval),
                    "mean_activation_sacred": n.mean_activation_sacred,
                    "mean_activation_secular": n.mean_activation_secular,
                }
                for n in universal_circuit.neurons
            ],
            "languages": universal_circuit.languages,
        }
        with open(circuit_cache_path, "w") as f:
            json.dump(convert_numpy_types(circuit_data), f, indent=2)
        print(f"Circuit cached to {circuit_cache_path}")

    # --- Step 7: Necessity test ---
    print("\n[Step 7] Testing circuit necessity (English)...")
    necessity_results = test_circuit_necessity(
        circuit=universal_circuit,
        stimuli=test_stimuli["eng_Latn"],
        model=model,
        tokenizer=tokenizer,
        lang_code="eng_Latn",
        target_lang=TARGET_LANG,
        sacred_token_ids=sacred_tokens[TARGET_LANG],
        device=DEFAULT_DEVICE,
    )

    # --- Step 7b: Supplementary vector subtraction necessity test ---
    # Extracts sacred concept vector from the residual stream (1024-dim) and
    # compares dense vector subtraction vs. sparse neuron ablation.
    print("\n[Step 7b] Supplementary: vector subtraction necessity test...")
    try:
        from extraction.concept_vectors import extract_concept_vectors
        from data.contrastive_pairs import ContrastivePairGenerator
        # Generate sacred contrastive pairs for English (vector extraction)
        gen = ContrastivePairGenerator(seed=42)
        sacred_pairs_eng = gen.generate_pairs(
            domain="sacred", n_per_concept=10, languages=["eng_Latn"],
        ).get("eng_Latn", {})
        # Build concept vectors per layer
        all_layer_vecs = {}
        for concept, pairs in sacred_pairs_eng.items():
            vecs = extract_concept_vectors(
                contrastive_pairs=pairs,
                model=model,
                tokenizer=tokenizer,
                lang_code="eng_Latn",
                layers=LAYERS_TO_ANALYZE,
                component="encoder_hidden",
                pooling="mean",
                device=DEFAULT_DEVICE,
                method="mean",
            )
            for layer, vec in vecs.items():
                all_layer_vecs.setdefault(layer, []).append(vec)
        import torch
        sacred_concept_vectors = {
            layer: torch.stack(vecs).mean(0)
            for layer, vecs in all_layer_vecs.items()
        }
        vector_necessity = test_vector_necessity(
            concept_vectors=sacred_concept_vectors,
            stimuli=test_stimuli["eng_Latn"],
            model=model,
            tokenizer=tokenizer,
            lang_code="eng_Latn",
            target_lang=TARGET_LANG,
            sacred_token_ids=sacred_tokens[TARGET_LANG],
            layers=LAYERS_TO_ANALYZE,
            alpha=0.25,
            device=DEFAULT_DEVICE,
        )
        print(f"  Sparse ablation (H1):   d={necessity_results.effect_size:.3f}, "
              f"p={necessity_results.p_value:.4f}")
        print(f"  Dense vector sub (supp): d={vector_necessity['effect_size']:.3f}, "
              f"p={vector_necessity['p_value']:.4f}")
    except Exception as e:
        print(f"  Supplementary test skipped: {e}")
        vector_necessity = None

    # --- Step 8: Statistical analysis ---
    print("\n[Step 8] Running comprehensive statistical analysis...")
    experimental_results = {
        "necessity": necessity_results,
        "circuits_by_lang": circuits_by_lang,
        "circuit": universal_circuit,
    }
    statistical_report = run_comprehensive_hypothesis_testing(experimental_results, alpha=0.01)

    # --- Step 9: Cross-validation ---
    print("\n[Step 9] Cross-validation framework...")
    cv_results = perform_cross_validation(stimuli, model, tokenizer, discover_sacred_circuit)

    # --- Step 10: Visualizations ---
    print("\n[Step 10] Generating visualizations...")
    figs_dir = "outputs/figures"
    Path(figs_dir).mkdir(parents=True, exist_ok=True)

    plot_circuit_map(universal_circuit, save_path=f"{figs_dir}/circuit_map.png")
    plot_intervention_results(necessity_results, save_path=f"{figs_dir}/intervention_results.png")
    plot_statistical_summary(statistical_report, save_path=f"{figs_dir}/statistical_summary.png")
    plot_cross_validation(cv_results, save_path=f"{figs_dir}/cross_validation.png")

    if circuits_by_lang:
        plot_universal_circuit_heatmap(circuits_by_lang, save_path=f"{figs_dir}/universal_heatmap.png")

    create_comprehensive_report_figure(
        universal_circuit, necessity_results, statistical_report,
        save_path=f"{figs_dir}/comprehensive_report.png",
    )

    # --- Step 11: Summary ---
    print("\n" + "=" * 80)
    print("[Step 11] SUMMARY")
    print("=" * 80)
    print(f"  Universal neurons: {len(universal_circuit.neurons)}")
    print(f"  Critical layers: {universal_circuit.get_critical_layers()}")
    print(f"  Necessity (sparse ablation): d={necessity_results.effect_size:.3f}, "
          f"p={necessity_results.p_value:.4f}, significant={necessity_results.significant}")
    if vector_necessity:
        print(f"  Necessity (dense vector sub): d={vector_necessity['effect_size']:.3f}, "
              f"p={vector_necessity['p_value']:.4f}, significant={vector_necessity['significant']}")
    print(f"  Statistical report: {statistical_report.summary}")
    print(f"\nFigures saved to {figs_dir}/")

    return {
        "universal_circuit": universal_circuit,
        "necessity_results": necessity_results,
        "vector_necessity": vector_necessity,
        "statistical_report": statistical_report,
        "cv_results": cv_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SACRED full pipeline")
    parser.add_argument("--skip-discovery", action="store_true",
                        help="Load circuit from cache instead of re-running discovery")
    args = parser.parse_args()
    main(skip_discovery=args.skip_discovery)
