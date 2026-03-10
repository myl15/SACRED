"""
Pivot Language Diagnosis Framework (Experiment 2).

Tests whether cross-lingual concept transfer routes through English as an
implicit pivot by comparing four intervention conditions:
  A: Subtract source language concept vector
  B: Subtract target language concept vector
  C: Subtract English concept vector (the pivot test)
  D: Subtract random vector (noise control)

A high pivot_index (effect(C) >> effect(D) and effect(C) ≈ effect(A,B))
supports the English-pivot hypothesis.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from intervention.hooks import InterventionHook
from intervention.necessity import measure_concept_deletion


def run_pivot_diagnosis(
    translation_pair: Tuple[str, str],
    concept_vectors: Dict[str, Dict[int, torch.Tensor]],
    test_sentences: List[str],
    concept_token_ids: Dict[str, List[int]],
    model,
    tokenizer,
    layers: List[int],
    alpha: float = 1.0,
    device: str = "cuda",
    concept_words_by_lang: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    Run the four-condition pivot diagnosis experiment.

    Args:
        translation_pair: (source_lang, target_lang), e.g. ("arb_Arab", "zho_Hant")
        concept_vectors: {lang: {layer: concept_vector_tensor}}
        test_sentences: Source language sentences containing the concept
        concept_token_ids: {lang: [token_ids]} for checking concept in output
        model: NLLB model
        tokenizer: NLLB tokenizer
        layers: Encoder layers to apply vector subtraction
        alpha: Vector subtraction scaling factor
        device: Compute device
        concept_words_by_lang: Optional {lang: [word_strings]}. When provided,
            concept presence is checked via string matching on the decoded
            translation rather than token ID lookup (more reliable for
            non-English target languages).

    Returns:
        {
            "condition_A_source": {"deletion_rate": float, "mean_prob": float},
            "condition_B_target": {"deletion_rate": float, "mean_prob": float},
            "condition_C_english": {"deletion_rate": float, "mean_prob": float},
            "condition_D_random":  {"deletion_rate": float, "mean_prob": float},
            "baseline":            {"deletion_rate": float, "mean_prob": float},
            "pivot_index": float,
            "interpretation": str,
        }
    """
    source_lang, target_lang = translation_pair
    target_concept_ids = concept_tokens_for(concept_token_ids, target_lang)
    target_concept_words = (
        concept_words_by_lang.get(target_lang) if concept_words_by_lang else None
    )

    def run_condition(vector: Optional[torch.Tensor], label: str) -> Dict[str, float]:
        hook = InterventionHook()
        if vector is not None:
            hook.register_vector_subtraction_hook(model, vector, layers, alpha=alpha)
        result = measure_concept_deletion(
            test_sentences, model, tokenizer, source_lang, target_lang,
            target_concept_ids, hook if vector is not None else None, device,
            concept_words=target_concept_words,
        )
        hook.cleanup()
        print(f"  [{label}] deletion_rate={result['concept_present_rate']:.3f}, "
              f"mean_prob={result['mean_concept_probability']:.4f}")
        return {
            "deletion_rate": 1.0 - result["concept_present_rate"],
            "mean_prob": result["mean_concept_probability"],
        }

    print(f"\n=== Pivot Diagnosis: {source_lang} → {target_lang} ===")

    # Baseline (no intervention)
    baseline = run_condition(None, "Baseline")

    # Condition A: source language vector
    vec_A = _get_layer_mean_vector(concept_vectors, source_lang, layers)
    cond_A = run_condition(vec_A, "A: source vector")

    # Condition B: target language vector
    vec_B = _get_layer_mean_vector(concept_vectors, target_lang, layers)
    cond_B = run_condition(vec_B, "B: target vector")

    # Condition C: English vector (pivot test)
    vec_C = _get_layer_mean_vector(concept_vectors, "eng_Latn", layers)
    cond_C = run_condition(vec_C, "C: English (pivot)")

    # Condition D: random vector (noise control)
    hidden_dim = next(iter(concept_vectors.values()))[layers[0]].shape[0]
    vec_D = torch.randn(hidden_dim, device=vec_C.device)
    vec_D = vec_D / vec_D.norm() * vec_C.norm()    # normalize to same magnitude
    cond_D = run_condition(vec_D, "D: random vector")

    # Pivot index (binary): how effective is the English vector relative to source/target?
    mean_AB = (cond_A["deletion_rate"] + cond_B["deletion_rate"]) / 2
    pivot_index = cond_C["deletion_rate"] / mean_AB if mean_AB > 1e-6 else float("nan")

    # Pivot index (continuous): uses P(concept) reduction as the effect measure.
    # effect(v) = baseline_prob - intervened_prob  (higher = more suppression)
    baseline_prob = baseline["mean_prob"]
    effect_A = baseline_prob - cond_A["mean_prob"]
    effect_B = baseline_prob - cond_B["mean_prob"]
    effect_C = baseline_prob - cond_C["mean_prob"]
    mean_effect_AB = (effect_A + effect_B) / 2
    pivot_index_continuous = (
        effect_C / mean_effect_AB if abs(mean_effect_AB) > 1e-7 else float("nan")
    )

    if np.isnan(pivot_index):
        interpretation = "Pivot index undefined (no ablation effect in conditions A/B)."
    elif pivot_index > 0.8:
        interpretation = (
            f"STRONG pivot evidence: English vector deletion rate ({cond_C['deletion_rate']:.2f}) "
            f"is {pivot_index:.2f}x the mean of source/target vectors. "
            "Concept appears to route through English."
        )
    elif pivot_index > 0.4:
        interpretation = (
            f"MODERATE pivot evidence: pivot_index={pivot_index:.2f}. "
            "English vector partially effective, concept representation overlaps with English."
        )
    else:
        interpretation = (
            f"WEAK pivot evidence: pivot_index={pivot_index:.2f}. "
            "English vector much less effective than source/target; no strong English pivot."
        )

    print(f"\n  Pivot index (binary): {pivot_index:.3f}")
    print(f"  Pivot index (continuous): {pivot_index_continuous:.3f}")
    print(f"  Interpretation: {interpretation}")

    return {
        "baseline": baseline,
        "condition_A_source": cond_A,
        "condition_B_target": cond_B,
        "condition_C_english": cond_C,
        "condition_D_random": cond_D,
        "pivot_index": float(pivot_index),
        "pivot_index_continuous": float(pivot_index_continuous),
        "interpretation": interpretation,
    }


def _get_layer_mean_vector(
    concept_vectors: Dict[str, Dict[int, torch.Tensor]],
    lang: str,
    layers: List[int],
) -> torch.Tensor:
    """Average concept vectors across specified layers for a language."""
    lang_vecs = concept_vectors.get(lang, {})
    available = [lang_vecs[l] for l in layers if l in lang_vecs]
    if not available:
        raise KeyError(f"No concept vector found for language '{lang}' in layers {layers}")
    stacked = torch.stack(available, dim=0)   # [n_layers, hidden_dim]
    return stacked.mean(dim=0)                # [hidden_dim]


def concept_tokens_for(
    concept_token_ids: Dict[str, List[int]],
    lang: str,
) -> List[int]:
    """Return concept token IDs for a language, falling back to English."""
    return concept_token_ids.get(lang, concept_token_ids.get("eng_Latn", []))
