"""
Necessity and Sufficiency Testing for Circuit Analysis.

Provides:
  - measure_concept_deletion(): domain-agnostic translation quality measurement
    (generalizes the original measure_translation_quality / sacred_token_ids)
  - test_circuit_necessity(): ablation + random control
  - test_circuit_sufficiency(): activation patching (placeholder)
"""

import torch
import torch.nn.functional as F
import numpy as np
import re
import unicodedata
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

try:
    from sacrebleu import corpus_bleu
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False

from intervention.hooks import InterventionHook
from analysis.statistical import compute_cohens_d
from scipy.stats import ttest_rel


@dataclass
class QualityMetrics:
    """Translation quality metrics for a single sentence."""
    concept_token_present: bool
    concept_token_probability: float
    bleu_score: Optional[float]
    avg_token_prob: float
    perplexity: float
    translation: str = ""

    # Backward-compatible aliases
    @property
    def sacred_token_present(self) -> bool:
        return self.concept_token_present

    @property
    def sacred_token_probability(self) -> float:
        return self.concept_token_probability


@dataclass
class NecessityResults:
    """Results from necessity testing (ablation)."""
    baseline_quality: Dict[str, List[QualityMetrics]]
    ablated_quality: Dict[str, List[QualityMetrics]]
    secular_baseline: Dict[str, List[QualityMetrics]]
    secular_ablated: Dict[str, List[QualityMetrics]]
    random_ablated: Dict[str, List[QualityMetrics]]
    effect_size: float
    p_value: float
    significant: bool
    specificity_test: Dict[str, Any] = field(default_factory=dict)
    validation_report: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SufficiencyResults:
    """Results from sufficiency testing (activation patching)."""
    full_ablation_quality: Dict[str, List[QualityMetrics]]
    restored_quality: Dict[str, List[QualityMetrics]]
    baseline_quality: Dict[str, List[QualityMetrics]]
    restoration_percentage: float
    sufficient: bool
    status: str = "implemented"
    implemented: bool = True
    note: str = ""


# ---------------------------------------------------------------------------
# Core measurement function
# ---------------------------------------------------------------------------

def measure_concept_deletion(
    sentences: List[str],
    model,
    tokenizer,
    source_lang: str,
    target_lang: str,
    concept_token_ids: List[int],
    intervention: Optional[InterventionHook] = None,
    device: str = "cuda",
    concept_words: Optional[List[str]] = None,
    matching_mode: str = "hybrid",
    return_diagnostics: bool = False,
) -> Dict[str, Any]:
    """
    Measure whether a concept survives translation under optional intervention.

    Generalizes the original measure_translation_quality() to work with any
    concept domain by accepting concept_token_ids instead of sacred_token_ids.

    Args:
        sentences: Source sentences (strings)
        model: NLLB model
        tokenizer: NLLB tokenizer
        source_lang: Source language code
        target_lang: Target language code
        concept_token_ids: Token IDs to check for in outputs (used for
            probability estimation; may be empty if concept_words is provided)
        intervention: Optional active InterventionHook
        device: Compute device
        concept_words: Optional list of word strings. When provided, concept
            presence is determined by substring matching against the decoded
            translation rather than token ID lookup. This is more reliable for
            non-English languages where SentencePiece encodes words differently
            in isolation vs. in context.
        matching_mode: How to determine concept presence.
            - "substring": legacy behavior (lowercase substring match)
            - "word_boundary": regex boundary-aware match for whitespace scripts
            - "token_only": token-id only
            - "hybrid": token match OR lexical match with script-aware fallback

    Returns:
        {
            "concept_present_rate": float,
            "mean_concept_probability": float,
            "translations": List[str],
            "per_sentence": List[QualityMetrics],
        }
    """
    target_lang_id = tokenizer.convert_tokens_to_ids(target_lang)
    per_sentence: List[QualityMetrics] = []
    translations: List[str] = []
    match_info_per_sentence: List[Dict[str, bool]] = []

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", src_lang=source_lang).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                forced_bos_token_id=target_lang_id,
                max_length=50,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences[0]
        translation = tokenizer.decode(generated_ids, skip_special_tokens=True)
        translations.append(translation)

        token_present = any(t in generated_ids.tolist() for t in concept_token_ids) if concept_token_ids else False
        match_info = _match_concept(
            translation=translation,
            concept_words=concept_words or [],
            token_present=token_present,
            matching_mode=matching_mode,
        )
        match_info_per_sentence.append(match_info)
        concept_present = match_info["concept_present"]

        if hasattr(outputs, "scores") and outputs.scores:
            # Peak concept probability: max over all steps of P(any concept token | context).
            # This is the probability the model *most wanted* to generate a concept token
            # at its best opportunity — a soft, continuous analogue of binary presence.
            # Averaging over all steps (the old approach) was wrong: generic steps assign
            # ~1/vocab probability to concept tokens, swamping the signal.
            peak_concept_prob = 0.0
            avg_probs = []
            for i, step_scores in enumerate(outputs.scores):
                probs = F.softmax(step_scores[0], dim=0)
                if concept_token_ids:
                    step_max = probs[concept_token_ids].max().item()
                    if step_max > peak_concept_prob:
                        peak_concept_prob = step_max
                if i < len(generated_ids) - 1:
                    avg_probs.append(probs[generated_ids[i + 1]].item())

            concept_prob = peak_concept_prob
            avg_token_prob = float(np.mean(avg_probs)) if avg_probs else 0.0
        else:
            concept_prob = 0.0
            avg_token_prob = 0.0

        perplexity = float(np.exp(-np.log(avg_token_prob + 1e-10)))

        per_sentence.append(QualityMetrics(
            concept_token_present=concept_present,
            concept_token_probability=concept_prob,
            bleu_score=None,
            avg_token_prob=avg_token_prob,
            perplexity=perplexity,
            translation=translation,
        ))

    concept_present_rate = float(np.mean([m.concept_token_present for m in per_sentence]))
    mean_concept_prob = float(np.mean([m.concept_token_probability for m in per_sentence]))

    result = {
        "concept_present_rate": concept_present_rate,
        "mean_concept_probability": mean_concept_prob,
        "translations": translations,
        "per_sentence": per_sentence,
    }
    if return_diagnostics:
        mode = matching_mode.lower()
        token_hits = int(sum(1 for m in match_info_per_sentence if m["token_present"]))
        lexical_hits = int(sum(1 for m in match_info_per_sentence if m["lexical_present"]))
        hybrid_hits = int(sum(1 for m in match_info_per_sentence if m["concept_present"]))
        result["matching_diagnostics"] = {
            "matching_mode": mode,
            "n_sentences": len(match_info_per_sentence),
            "token_hit_rate": (token_hits / len(match_info_per_sentence)) if match_info_per_sentence else 0.0,
            "lexical_hit_rate": (lexical_hits / len(match_info_per_sentence)) if match_info_per_sentence else 0.0,
            "hybrid_hit_rate": (hybrid_hits / len(match_info_per_sentence)) if match_info_per_sentence else 0.0,
            "token_hits": token_hits,
            "lexical_hits": lexical_hits,
            "hybrid_hits": hybrid_hits,
        }
    return result


def _normalize_for_matching(text: str) -> str:
    """Normalize text for robust matching across Unicode forms."""
    return unicodedata.normalize("NFKC", text).casefold()


def _contains_cjk_or_arabic(text: str) -> bool:
    for ch in text:
        if ("\u4E00" <= ch <= "\u9FFF") or ("\u3400" <= ch <= "\u4DBF") or ("\u0600" <= ch <= "\u06FF"):
            return True
    return False


def _match_concept(
    translation: str,
    concept_words: List[str],
    token_present: bool,
    matching_mode: str,
) -> Dict[str, bool]:
    """Return token/lexical/concept match signals for a translation."""
    mode = matching_mode.lower()
    if mode == "token_only":
        return {
            "token_present": token_present,
            "lexical_present": False,
            "concept_present": token_present,
        }

    normalized_translation = _normalize_for_matching(translation)
    normalized_words = [_normalize_for_matching(w) for w in concept_words if w]
    lexical_present = False

    if normalized_words:
        if mode == "substring":
            lexical_present = any(w in normalized_translation for w in normalized_words)
        elif mode == "word_boundary":
            # For scripts without reliable whitespace boundaries, fall back to substring.
            if _contains_cjk_or_arabic(normalized_translation):
                lexical_present = any(w in normalized_translation for w in normalized_words)
            else:
                lexical_present = any(
                    re.search(rf"\b{re.escape(w)}\b", normalized_translation) is not None
                    for w in normalized_words
                )
        elif mode == "hybrid":
            if _contains_cjk_or_arabic(normalized_translation):
                lexical_present = any(w in normalized_translation for w in normalized_words)
            else:
                lexical_present = any(
                    re.search(rf"\b{re.escape(w)}\b", normalized_translation) is not None
                    for w in normalized_words
                )
        else:
            raise ValueError(f"Unknown matching_mode '{matching_mode}'")

    concept_present = (token_present or lexical_present) if mode == "hybrid" else lexical_present
    return {
        "token_present": token_present,
        "lexical_present": lexical_present,
        "concept_present": concept_present,
    }


def _contains_concept(
    translation: str,
    concept_words: List[str],
    token_present: bool,
    matching_mode: str,
) -> bool:
    """Backward-compatible boolean concept matcher wrapper."""
    return _match_concept(
        translation=translation,
        concept_words=concept_words,
        token_present=token_present,
        matching_mode=matching_mode,
    )["concept_present"]


# Backward-compatible alias
def measure_translation_quality(
    sentences: List[Dict],
    model,
    tokenizer,
    lang_code: str,
    target_lang: str,
    sacred_token_ids: List[int],
    intervention: Optional[InterventionHook] = None,
    device: str = "cuda",
) -> List[QualityMetrics]:
    """Legacy wrapper around measure_concept_deletion."""
    sentence_texts = [s["text"] if isinstance(s, dict) else s for s in sentences]
    result = measure_concept_deletion(
        sentence_texts, model, tokenizer, lang_code, target_lang,
        sacred_token_ids, intervention, device,
    )
    return result["per_sentence"]


# ---------------------------------------------------------------------------
# Necessity and sufficiency tests
# ---------------------------------------------------------------------------

def test_circuit_necessity(
    circuit,
    stimuli: Dict[str, List[Dict]],
    model,
    tokenizer,
    lang_code: str,
    target_lang: str,
    sacred_token_ids: List[int],
    device: str = "cuda",
) -> NecessityResults:
    """
    Test if circuit is NECESSARY for sacred concept translation.

    Conditions:
      1. Baseline (no intervention) on sacred sentences
      2. Circuit ablation on sacred sentences
      3. Circuit ablation on secular sentences (specificity control)
      4. Random neuron ablation on sacred sentences (negative control)
    """
    from intervention.hooks import validate_intervention_execution

    print("\n=== Testing Circuit Necessity ===")

    sacred_sentences = [s["text"] for s in stimuli["sacred"]]
    secular_sentences = [s["text"] for s in stimuli["secular"]]

    print("1. Measuring baseline...")
    baseline_sacred = measure_concept_deletion(
        sacred_sentences, model, tokenizer, lang_code, target_lang, sacred_token_ids, None, device,
    )["per_sentence"]
    baseline_secular = measure_concept_deletion(
        secular_sentences, model, tokenizer, lang_code, target_lang, sacred_token_ids, None, device,
    )["per_sentence"]

    print("2. Ablating circuit on sacred sentences...")
    intervention = InterventionHook()
    intervention.register_ablation_hook(model, circuit, "mlp")

    ablated_sacred = measure_concept_deletion(
        sacred_sentences, model, tokenizer, lang_code, target_lang, sacred_token_ids, intervention, device,
    )["per_sentence"]

    print("3. Validating intervention...")
    validation = validate_intervention_execution(
        model, circuit, sacred_sentences[0], tokenizer, lang_code, intervention, device,
    )

    print("4. Ablating circuit on secular sentences (specificity)...")
    ablated_secular = measure_concept_deletion(
        secular_sentences, model, tokenizer, lang_code, target_lang, sacred_token_ids, intervention, device,
    )["per_sentence"]

    intervention.cleanup()

    print("5. Random neuron ablation (negative control)...")
    intervention.register_random_ablation_hook(
        model,
        n_neurons_to_ablate=len(circuit.neurons),
        layers=circuit.get_critical_layers(),
    )
    random_ablated = measure_concept_deletion(
        sacred_sentences, model, tokenizer, lang_code, target_lang, sacred_token_ids, intervention, device,
    )["per_sentence"]
    intervention.cleanup()

    baseline_probs = np.array([m.concept_token_probability for m in baseline_sacred])
    ablated_probs = np.array([m.concept_token_probability for m in ablated_sacred])
    effect_size = compute_cohens_d(baseline_probs, ablated_probs)
    _, p_value = ttest_rel(baseline_probs, ablated_probs)

    results = NecessityResults(
        baseline_quality={"sacred": baseline_sacred},
        ablated_quality={"sacred": ablated_sacred},
        secular_baseline={"secular": baseline_secular},
        secular_ablated={"secular": ablated_secular},
        random_ablated={"sacred": random_ablated},
        effect_size=effect_size,
        p_value=p_value,
        significant=(p_value < 0.01 and abs(effect_size) > 0.5),
        validation_report=validation,
    )

    print(f"\nNecessity Test: d={effect_size:.3f}, p={p_value:.4f}, "
          f"significant={results.significant}, validated={validation['passed']}")
    return results


def test_vector_necessity(
    concept_vectors: dict,
    stimuli: dict,
    model,
    tokenizer,
    lang_code: str,
    target_lang: str,
    sacred_token_ids: list,
    layers: list,
    alpha: float = 0.25,
    device: str = "cuda",
) -> dict:
    """
    Supplementary necessity test using residual-stream vector subtraction.

    Compares dense concept-vector subtraction (1024-dim residual stream) to
    the sparse neuron ablation in test_circuit_necessity(). This tests whether
    the causal pathway is better captured by the dense concept direction or by
    the identified sparse neuron set.

    Args:
        concept_vectors: {layer: tensor[1024]} — sacred concept vectors per layer
        stimuli: {"sacred": [{"text": str}], "secular": [...]} — test sentences
        model: NLLB model
        tokenizer: NLLB tokenizer
        lang_code: Source language code
        target_lang: Target language code
        sacred_token_ids: Token IDs for sacred concept
        layers: Encoder layers to apply vector subtraction
        alpha: Scaling factor (use calibrated value; default 0.25)
        device: Compute device

    Returns:
        {
            "effect_size": float,
            "p_value": float,
            "significant": bool,
            "baseline_mean_prob": float,
            "ablated_mean_prob": float,
            "prob_delta": float,
        }
    """
    from intervention.hooks import InterventionHook
    from scipy.stats import ttest_rel

    print(f"\n=== Supplementary Vector Necessity Test (residual stream, alpha={alpha}) ===")

    sacred_sentences = [s["text"] for s in stimuli["sacred"]]

    # Average concept vector across specified layers
    available = [concept_vectors[l] for l in layers if l in concept_vectors]
    if not available:
        raise KeyError(f"No concept vectors for layers {layers}")
    mean_vec = torch.stack(available).mean(0).to(device)

    print("  1. Baseline...")
    baseline = measure_concept_deletion(
        sacred_sentences, model, tokenizer, lang_code, target_lang,
        sacred_token_ids, None, device,
    )

    print(f"  2. Vector subtraction (alpha={alpha})...")
    hook = InterventionHook()
    hook.register_scaled_vector_subtraction_hook(model, mean_vec, layers, alpha=alpha)
    ablated = measure_concept_deletion(
        sacred_sentences, model, tokenizer, lang_code, target_lang,
        sacred_token_ids, hook, device,
    )
    hook.cleanup()

    baseline_probs = np.array([m.concept_token_probability for m in baseline["per_sentence"]])
    ablated_probs = np.array([m.concept_token_probability for m in ablated["per_sentence"]])
    effect_size = compute_cohens_d(baseline_probs, ablated_probs)
    _, p_value = ttest_rel(baseline_probs, ablated_probs)

    result = {
        "effect_size": float(effect_size),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.01 and abs(effect_size) > 0.5),
        "baseline_mean_prob": float(baseline_probs.mean()),
        "ablated_mean_prob": float(ablated_probs.mean()),
        "prob_delta": float(baseline_probs.mean() - ablated_probs.mean()),
        "alpha": alpha,
        "layers": layers,
    }
    print(f"\n  Vector Necessity: d={effect_size:.3f}, p={p_value:.4f}, "
          f"significant={result['significant']}")
    print(f"  Prob delta: {result['prob_delta']:.4f} "
          f"({result['baseline_mean_prob']:.4f} → {result['ablated_mean_prob']:.4f})")
    return result


def test_circuit_sufficiency(
    circuit,
    stimuli: Dict[str, List[Dict]],
    model,
    tokenizer,
    lang_code: str,
    target_lang: str,
    sacred_token_ids: List[int],
    device: str = "cuda",
) -> SufficiencyResults:
    """
    Test if circuit ALONE is sufficient to restore behavior (placeholder).

    Full implementation requires activation patching infrastructure.
    """
    print("\n=== Testing Circuit Sufficiency (placeholder) ===")
    print("Note: Full sufficiency testing requires activation patching. Not yet implemented.")
    return SufficiencyResults(
        full_ablation_quality={},
        restored_quality={},
        baseline_quality={},
        restoration_percentage=0.0,
        sufficient=False,
        status="placeholder",
        implemented=False,
        note="Placeholder implementation: full activation patching sufficiency test is not yet implemented.",
    )
