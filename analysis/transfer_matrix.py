"""
Transfer Matrix Computation (Experiment 4).

Computes the full NxN directional transfer matrix where entry [i,j] is the
concept deletion rate when applying language_i's concept vector to
language_j's sentences during translation.

A transfer rate ≈ 1.0 means the concept vector from language i completely
erases the concept in language j's translations — strong cross-lingual transfer.

compute_transfer_matrix() returns both a binary deletion matrix AND a
probability-reduction matrix (continuous metric) to avoid ceiling effects.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from intervention.hooks import InterventionHook
from intervention.necessity import measure_concept_deletion


def compute_transfer_matrix(
    concept_vectors: Dict[str, Dict[int, torch.Tensor]],
    test_sentences: Dict[str, List[str]],
    concept_token_ids: Dict[str, List[int]],
    model,
    tokenizer,
    layers: List[int],
    alpha: float = 0.25,
    device: str = "cuda",
    concept_words_by_lang: Optional[Dict[str, List[str]]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compute the full NxN concept transfer matrix.

    Entry [i, j] = deletion rate when language_i's concept vector is
    subtracted from language_j's encoder activations during translation.

    The diagonal [i, i] is the same-language deletion rate (sanity check).

    Args:
        concept_vectors: {lang: {layer: tensor}} — concept vectors per language/layer
        test_sentences: {lang: [sentences containing the concept]}
        concept_token_ids: {lang: [token_ids]} for checking outputs
        model: NLLB model
        tokenizer: NLLB tokenizer
        layers: Which encoder layers to apply the intervention
        alpha: Vector subtraction scaling factor (use calibrated value; default 0.25)
        device: Compute device
        concept_words_by_lang: Optional {lang: [word_strings]} for string-based
            presence checking (more reliable than token-ID lookup for non-English).

    Returns:
        (deletion_matrix, prob_matrix, languages)
          deletion_matrix: NxN binary deletion rates (rows=source vector, cols=target lang)
          prob_matrix: NxN mean concept probability reductions (continuous metric)
          languages: Language list for axis labels
    """
    languages = sorted(
        set(concept_vectors.keys()) & set(test_sentences.keys()) & set(concept_token_ids.keys())
    )
    N = len(languages)
    deletion_matrix = np.zeros((N, N))
    prob_matrix = np.zeros((N, N))    # baseline_prob - intervened_prob per cell

    print(f"\n=== Transfer Matrix ({N}x{N}, alpha={alpha}) ===")
    print(f"Languages: {languages}")

    # Compute per-language baselines (no intervention)
    baselines = {}
    for lang_j in languages:
        sentences_j = test_sentences.get(lang_j, [])
        if not sentences_j:
            continue
        token_ids_j = concept_token_ids.get(lang_j, concept_token_ids.get("eng_Latn", []))
        words_j = concept_words_by_lang.get(lang_j) if concept_words_by_lang else None
        baseline = measure_concept_deletion(
            sentences_j, model, tokenizer,
            source_lang=lang_j, target_lang=lang_j,
            concept_token_ids=token_ids_j,
            intervention=None, device=device, concept_words=words_j,
        )
        baselines[lang_j] = baseline["mean_concept_probability"]

    for i, lang_i in enumerate(languages):
        try:
            vec_i = _get_layer_mean_vector(concept_vectors, lang_i, layers)
        except KeyError:
            print(f"  Skipping row {lang_i}: no concept vector")
            continue

        for j, lang_j in enumerate(languages):
            sentences_j = test_sentences.get(lang_j, [])
            if not sentences_j:
                print(f"  Skipping [{lang_i} → {lang_j}]: no test sentences")
                continue

            token_ids_j = concept_token_ids.get(lang_j, concept_token_ids.get("eng_Latn", []))
            words_j = concept_words_by_lang.get(lang_j) if concept_words_by_lang else None

            hook = InterventionHook()
            hook.register_vector_subtraction_hook(model, vec_i, layers, alpha=alpha)

            result = measure_concept_deletion(
                sentences_j, model, tokenizer,
                source_lang=lang_j, target_lang=lang_j,
                concept_token_ids=token_ids_j,
                intervention=hook,
                device=device,
                concept_words=words_j,
            )
            hook.cleanup()

            deletion_rate = 1.0 - result["concept_present_rate"]
            baseline_prob = baselines.get(lang_j, 0.0)
            prob_reduction = baseline_prob - result["mean_concept_probability"]

            deletion_matrix[i, j] = deletion_rate
            prob_matrix[i, j] = max(0.0, prob_reduction)
            print(f"  [{lang_i} → {lang_j}]: deletion={deletion_rate:.3f}, "
                  f"prob_reduction={prob_reduction:.4f}")

    return deletion_matrix, prob_matrix, languages


def _get_layer_mean_vector(
    concept_vectors: Dict[str, Dict[int, torch.Tensor]],
    lang: str,
    layers: List[int],
) -> torch.Tensor:
    """Average concept vectors across specified layers for a language."""
    lang_vecs = concept_vectors.get(lang, {})
    available = [lang_vecs[l] for l in layers if l in lang_vecs]
    if not available:
        raise KeyError(f"No concept vector for '{lang}' in layers {layers}")
    return torch.stack(available).mean(dim=0)


def compute_cross_lingual_transfer_scores(
    deletion_matrix: np.ndarray,
    languages: List[str],
) -> Dict[str, float]:
    """
    Compute cross-lingual transfer score for each off-diagonal cell.

    transfer_score[i,j] = deletion_matrix[i,j] / deletion_matrix[i,i]

    This normalises by the same-language deletion rate so that a score of 1.0
    means cross-lingual transfer is as effective as same-language transfer.
    Your proposal's success criterion is >70% (score > 0.7).

    Args:
        deletion_matrix: NxN deletion rate array
        languages: Language list for axis labels

    Returns:
        {"{src_lang}→{tgt_lang}": transfer_score} for all off-diagonal pairs
    """
    N = len(languages)
    scores = {}
    for i in range(N):
        diag = deletion_matrix[i, i]
        for j in range(N):
            if i == j:
                continue
            score = deletion_matrix[i, j] / diag if diag > 1e-6 else 0.0
            label = f"{languages[i].split('_')[0]}→{languages[j].split('_')[0]}"
            scores[label] = float(score)
    return scores


def interpret_transfer_matrix(matrix: np.ndarray, languages: List[str]) -> Dict:
    """
    Summarize key findings from the transfer matrix.

    Returns dict with:
      - mean_off_diagonal_deletion: mean deletion rate for cross-lingual cells
      - best_transfer_pair: (lang_i, lang_j) with highest off-diagonal rate
      - english_hub_score: mean transfer involving English vs others
      - mean_asymmetry: mean |matrix[i,j] - matrix[j,i]|
    """
    N = len(languages)
    off_diag = [(matrix[i, j], languages[i], languages[j])
                for i in range(N) for j in range(N) if i != j]

    best_rate, best_src, best_tgt = max(off_diag, key=lambda x: x[0]) if off_diag else (0, "", "")

    asymmetries = [abs(matrix[i, j] - matrix[j, i]) for i in range(N) for j in range(i + 1, N)]
    mean_asymmetry = float(np.mean(asymmetries)) if asymmetries else 0.0

    eng_idx = languages.index("eng_Latn") if "eng_Latn" in languages else None
    if eng_idx is not None:
        eng_row = [matrix[eng_idx, j] for j in range(N) if j != eng_idx]
        eng_col = [matrix[i, eng_idx] for i in range(N) if i != eng_idx]
        non_eng = [matrix[i, j] for i in range(N) for j in range(N)
                   if i != j and i != eng_idx and j != eng_idx]
        english_hub_score = float(np.mean(eng_row + eng_col)) / (float(np.mean(non_eng)) + 1e-10)
    else:
        english_hub_score = float("nan")

    return {
        "mean_off_diagonal_deletion": float(np.mean([r for r, _, _ in off_diag])),
        "best_transfer_pair": (best_src, best_tgt),
        "best_transfer_rate": float(best_rate),
        "mean_asymmetry": mean_asymmetry,
        "english_hub_score": english_hub_score,
    }
