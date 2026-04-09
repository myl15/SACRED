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


def _merge_matching_diagnostics(items: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate per-call matching diagnostics across matrix cells."""
    if not items:
        return {}
    n_sent = sum(int(d.get("n_sentences", 0)) for d in items)
    tok = sum(int(d.get("token_hits", 0)) for d in items)
    lex = sum(int(d.get("lexical_hits", 0)) for d in items)
    hyb = sum(int(d.get("hybrid_hits", 0)) for d in items)
    return {
        "n_sentences": n_sent,
        "token_hits": tok,
        "lexical_hits": lex,
        "hybrid_hits": hyb,
        "token_hit_rate": (tok / n_sent) if n_sent else 0.0,
        "lexical_hit_rate": (lex / n_sent) if n_sent else 0.0,
        "hybrid_hit_rate": (hyb / n_sent) if n_sent else 0.0,
    }


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
    output_lang: str = "eng_Latn",
    matching_mode: str = "hybrid",
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Dict[str, float]]]:
    """
    Compute the full NxN concept transfer matrix.

    Entry [i, j] = deletion rate when language_i's concept vector is
    subtracted from language_j's encoder activations during translation
    to `output_lang`.

    This matches the exp2 pivot-diagnosis setup: for each cell [i, j] we take
    language_j's test sentences, apply language_i's concept vector to the
    encoder, translate to `output_lang`, and measure whether the concept
    disappears from the output.  A fixed output language is required so the
    same concept_words/token_ids can be used for all cells.

    The diagonal [i, i] is the same-language deletion rate (sanity check):
    language_i's vector applied to language_i sentences → output_lang.

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
            presence checking. The output-language entry is used for ALL cells
            since every translation targets `output_lang`.
        output_lang: Fixed translation target for all cells (default: "eng_Latn").
            Using English ensures consistent concept_words matching across
            all source languages.

    Returns:
        (deletion_matrix, prob_matrix, languages, diagnostics)
          deletion_matrix: NxN binary deletion rates (rows=source vector lang,
                           cols=test-sentence lang; translation always → output_lang)
          prob_matrix: NxN mean concept probability reductions (continuous metric)
          languages: Language list for axis labels
    """
    languages = sorted(
        set(concept_vectors.keys()) & set(test_sentences.keys()) & set(concept_token_ids.keys())
    )
    N = len(languages)
    deletion_matrix = np.zeros((N, N))
    prob_matrix = np.zeros((N, N))    # baseline_prob - intervened_prob per cell
    diag_by_lang: Dict[str, List[Dict[str, float]]] = {}

    print(f"\n=== Transfer Matrix ({N}x{N}, alpha={alpha}) ===")
    print(f"Languages: {languages}")

    # Use output_lang concept words/tokens for checking concept presence in ALL cells.
    # Every translation targets output_lang, so the same word list applies throughout.
    out_token_ids = concept_token_ids.get(output_lang, concept_token_ids.get("eng_Latn", []))
    out_words = concept_words_by_lang.get(output_lang) if concept_words_by_lang else None

    # Compute per-language baselines (no intervention): lang_j → output_lang
    baselines = {}
    for lang_j in languages:
        sentences_j = test_sentences.get(lang_j, [])
        if not sentences_j:
            continue
        baseline = measure_concept_deletion(
            sentences_j, model, tokenizer,
            source_lang=lang_j, target_lang=output_lang,
            concept_token_ids=out_token_ids,
            intervention=None, device=device, concept_words=out_words,
            matching_mode=matching_mode,
            return_diagnostics=True,
        )
        baselines[lang_j] = baseline["mean_concept_probability"]
        if "matching_diagnostics" in baseline:
            diag_by_lang.setdefault(lang_j, []).append(baseline["matching_diagnostics"])

    for i, lang_i in enumerate(languages):
        # Pass the DICTIONARY of layer vectors, filtered to your target layers
        lang_vecs = concept_vectors.get(lang_i, {})
        vecs_i = {l: lang_vecs[l] for l in layers if l in lang_vecs}
        
        if not vecs_i:
            print(f"  Skipping row {lang_i}: no concept vectors for specified layers")
            continue

        for j, lang_j in enumerate(languages):
            sentences_j = test_sentences.get(lang_j, [])
            if not sentences_j:
                continue

            hook = InterventionHook()
            # Ensure your hook's register method is updated to accept a Dict[int, torch.Tensor]
            hook.register_vector_subtraction_hook(model, vecs_i, layers, alpha=alpha)

            result = measure_concept_deletion(
                sentences_j, model, tokenizer,
                source_lang=lang_j, target_lang=output_lang,
                concept_token_ids=out_token_ids,
                intervention=hook,
                device=device,
                concept_words=out_words,
                matching_mode=matching_mode,
                return_diagnostics=True,
            )
            hook.cleanup()

            deletion_rate = 1.0 - result["concept_present_rate"]
            baseline_prob = baselines.get(lang_j, 0.0)
            prob_reduction = baseline_prob - result["mean_concept_probability"]

            deletion_matrix[i, j] = deletion_rate
            # FIX: Keep negative transfer values
            prob_matrix[i, j] = prob_reduction
            if "matching_diagnostics" in result:
                diag_by_lang.setdefault(lang_j, []).append(result["matching_diagnostics"])
            print(f"  [{lang_i} → {lang_j}] Deletion: {deletion_rate:.3f}, "
                  f"Prob Reduction: {prob_reduction:.3f}")

    diagnostics = {
        "matching_by_target_language": {
            lang: _merge_matching_diagnostics(diag_items)
            for lang, diag_items in diag_by_lang.items()
        }
    }
    return deletion_matrix, prob_matrix, languages, diagnostics


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
        for j in range(N):
            if i == j:
                continue
            # FIX: Compare against target language's native deletion rate (j, j)
            target_native_diag = deletion_matrix[j, j]
            
            score = deletion_matrix[i, j] / target_native_diag if target_native_diag > 1e-6 else 0.0
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
        eng_absolute_mean = float(np.mean(eng_row + eng_col))
        non_eng_absolute_mean = float(np.mean(non_eng)) if non_eng else 0.0
        english_hub_score = eng_absolute_mean / (non_eng_absolute_mean + 1e-10)
        eng_saturation_rate = float(np.mean([x >= 0.55 for x in eng_row + eng_col])) if (eng_row or eng_col) else 0.0
    else:
        english_hub_score = float("nan")
        eng_absolute_mean = float("nan")
        non_eng_absolute_mean = float("nan")
        eng_saturation_rate = float("nan")

    row_marginals = {languages[i]: float(np.mean(matrix[i, :])) for i in range(N)}
    col_marginals = {languages[j]: float(np.mean(matrix[:, j])) for j in range(N)}
    off_diag_vals = [r for r, _, _ in off_diag]
    ceiling_rate_offdiag = float(np.mean([v >= 0.55 for v in off_diag_vals])) if off_diag_vals else 0.0

    return {
        "mean_off_diagonal_deletion": float(np.mean([r for r, _, _ in off_diag])),
        "best_transfer_pair": (best_src, best_tgt),
        "best_transfer_rate": float(best_rate),
        "mean_asymmetry": mean_asymmetry,
        "english_hub_score": english_hub_score,
        "english_hub_absolute_mean": eng_absolute_mean,
        "non_english_absolute_mean": non_eng_absolute_mean,
        "english_saturation_rate": eng_saturation_rate,
        "off_diagonal_ceiling_rate": ceiling_rate_offdiag,
        "row_marginals_mean_deletion": row_marginals,
        "column_marginals_mean_deletion": col_marginals,
    }
