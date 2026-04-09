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

import math
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from intervention.hooks import InterventionHook
from intervention.necessity import measure_concept_deletion


# #region agent log
_DEBUG_RUN_ID = "pre-fix"

def _dbg(hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    """Append NDJSON debug log for session 398199."""
    try:
        import json, time
        payload = {
            "sessionId": "398199",
            "runId": _DEBUG_RUN_ID,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open("/home/myl15/CS601R-interpretability/finalproject/SACRED/.cursor/debug-398199.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
# #endregion agent log


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
    n_random_controls: int = 20,
    random_seed: int = 42,
    matching_mode: str = "hybrid",
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
    global _DEBUG_RUN_ID
    # If the caller sets SACRED_DEBUG_RUN_ID, use it to tag logs.
    _DEBUG_RUN_ID = __import__("os").environ.get("SACRED_DEBUG_RUN_ID", _DEBUG_RUN_ID)
    _dbg(
        "H0",
        "intervention/pivot_diagnosis.py:run_pivot_diagnosis",
        "enter",
        {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "device_arg": device,
            "alpha": alpha,
            "layers": list(layers),
            "n_random_controls": n_random_controls,
            "random_seed": random_seed,
            "torch_version": getattr(torch, "__version__", "unknown"),
            "cuda_is_available": bool(torch.cuda.is_available()),
        },
    )
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
            matching_mode=matching_mode,
        )
        hook.cleanup()
        print(f"  [{label}] concept_present={result['concept_present_rate']:.3f}  "
              f"deletion_rate={1.0 - result['concept_present_rate']:.3f}  "
              f"mean_prob={result['mean_concept_probability']:.4f}")
        return {
            "deletion_rate": 1.0 - result["concept_present_rate"],
            "mean_prob": result["mean_concept_probability"],
        }

    def _summarize(values: List[float]) -> Dict[str, float]:
        arr = np.array(values, dtype=float)
        if arr.size == 0:
            return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        sem = std / math.sqrt(arr.size) if arr.size > 1 else 0.0
        ci_half = 1.96 * sem
        mean = float(arr.mean())
        return {
            "mean": mean,
            "std": std,
            "ci_low": mean - ci_half,
            "ci_high": mean + ci_half,
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
    # Respect the requested device (do not hardcode CUDA).
    vec_C = vec_C.to(device)
    _dbg(
        "H1",
        "intervention/pivot_diagnosis.py:vec_C",
        "english vector device",
        {
            "vec_C_device": str(getattr(vec_C, "device", "unknown")),
            "vec_C_dtype": str(getattr(vec_C, "dtype", "unknown")),
        },
    )
    cond_C = run_condition(vec_C, "C: English (pivot)")

    # Condition D: random vectors (noise control, Monte Carlo null)
    hidden_dim = next(iter(concept_vectors.values()))[layers[0]].shape[0]
    # Create a generator that matches the sampling device when possible.
    # Some torch builds require generator device == output device for torch.randn.
    try:
        generator = torch.Generator(device=vec_C.device)
        gen_device = str(vec_C.device)
    except TypeError:
        generator = torch.Generator()
        gen_device = "cpu"
    generator.manual_seed(random_seed)
    _dbg(
        "H2",
        "intervention/pivot_diagnosis.py:random_control_setup",
        "generator + hidden_dim",
        {
            "hidden_dim": int(hidden_dim),
            "generator_type": str(type(generator)),
            "generator_device": gen_device,
            "vec_C_device": str(getattr(vec_C, "device", "unknown")),
        },
    )
    random_trials = []
    for idx in range(max(1, n_random_controls)):
        _dbg(
            "H3",
            "intervention/pivot_diagnosis.py:randn",
            "about to sample random vector",
            {
                "idx": int(idx),
                "requested_device": str(getattr(vec_C, "device", "unknown")),
            },
        )
        try:
            if gen_device == "cpu":
                # CPU generator cannot be used to directly sample CUDA tensors.
                vec_D = torch.randn(hidden_dim, generator=generator, device="cpu").to(vec_C.device)
            else:
                vec_D = torch.randn(hidden_dim, generator=generator, device=vec_C.device)
        except Exception as e:
            _dbg(
                "H3",
                "intervention/pivot_diagnosis.py:randn",
                "randn failed",
                {
                    "idx": int(idx),
                    "requested_device": str(getattr(vec_C, "device", "unknown")),
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
            )
            raise
        vec_D = vec_D / vec_D.norm() * vec_C.norm()    # normalize to same magnitude
        trial = run_condition(vec_D, f"D: random vector [{idx + 1}/{max(1, n_random_controls)}]")
        random_trials.append(trial)

    random_deletions = [t["deletion_rate"] for t in random_trials]
    random_probs = [t["mean_prob"] for t in random_trials]
    d_del = _summarize(random_deletions)
    d_prob = _summarize(random_probs)
    cond_D = {
        # Backward-compatible scalar fields used by existing summaries/plots.
        "deletion_rate": d_del["mean"],
        "mean_prob": d_prob["mean"],
        # New Monte Carlo summary fields.
        "n_random_controls": max(1, n_random_controls),
        "random_seed": random_seed,
        "deletion_rate_std": d_del["std"],
        "deletion_rate_ci_low": d_del["ci_low"],
        "deletion_rate_ci_high": d_del["ci_high"],
        "mean_prob_std": d_prob["std"],
        "mean_prob_ci_low": d_prob["ci_low"],
        "mean_prob_ci_high": d_prob["ci_high"],
    }

    # Pivot index (binary): how effective is the English vector relative to source/target?
    # Uses DELTA from baseline (not absolute rates) so that natural baseline absence
    # does not produce spurious pivot_index=1.0 when the intervention has no effect.
    #
    # Underpowered guard: if |mean_delta_AB| < MIN_EFFECT the A/B interventions had
    # negligible effect; dividing by a near-zero denominator would yield a meaningless
    # large number. Report NaN ("underpowered") instead.
    MIN_EFFECT = 0.05   # ~1 sentence with n=20; below this the test is not diagnostic

    baseline_del = baseline["deletion_rate"]
    delta_A = cond_A["deletion_rate"] - baseline_del
    delta_B = cond_B["deletion_rate"] - baseline_del
    delta_C = cond_C["deletion_rate"] - baseline_del
    mean_delta_AB = (delta_A + delta_B) / 2
    pivot_index = delta_C / mean_delta_AB if abs(mean_delta_AB) >= MIN_EFFECT else float("nan")

    # Pivot index (continuous): uses peak P(concept) reduction as the effect measure.
    # effect(v) = baseline_prob - intervened_prob  (higher = more suppression)
    baseline_prob = baseline["mean_prob"]
    effect_A = baseline_prob - cond_A["mean_prob"]
    effect_B = baseline_prob - cond_B["mean_prob"]
    effect_C = baseline_prob - cond_C["mean_prob"]
    mean_effect_AB = (effect_A + effect_B) / 2
    # Same underpowered guard scaled to peak-prob units (threshold: 1% absolute reduction)
    PROB_MIN_EFFECT = 0.01
    pivot_index_continuous = (
        effect_C / mean_effect_AB if abs(mean_effect_AB) >= PROB_MIN_EFFECT else float("nan")
    )

    if np.isnan(pivot_index):
        interpretation = (
            "Pivot index undefined: A/B interventions had negligible effect "
            f"(mean_delta_AB={mean_delta_AB:.3f} < {MIN_EFFECT} threshold). "
            "Test is underpowered for this pair."
        )
    elif pivot_index > 0.8:
        interpretation = (
            f"STRONG pivot evidence: English vector deletion rate ({cond_C['deletion_rate']:.2f}) "
            f"is {pivot_index:.2f}x the mean of source/target vectors. "
            f"Random-control mean deletion={cond_D['deletion_rate']:.2f} "
            f"(95% CI [{cond_D['deletion_rate_ci_low']:.2f}, {cond_D['deletion_rate_ci_high']:.2f}]). "
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
        "condition_D_random_trials": random_trials,
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
