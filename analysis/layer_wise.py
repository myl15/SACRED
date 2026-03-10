"""
Layer-Wise Convergence Analysis (Experiment 3).

Computes:
  - extract_parallel_representations: encoder hidden states per layer
  - compute_cka_similarity: Centered Kernel Alignment
  - compute_english_centricity: distance ratio to English centroid
  - compute_silhouette_by_language: language cluster separation score
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

from extraction.activation_capture import ActivationCapture


# ---------------------------------------------------------------------------
# Representation extraction
# ---------------------------------------------------------------------------

def extract_parallel_representations(
    parallel_sentences: Dict[str, List[str]],
    model,
    tokenizer,
    layers: List[int],
    device: str = "cuda",
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Extract encoder hidden states (residual stream) for parallel sentences.

    Args:
        parallel_sentences: {lang_code: [sentences]} — same sentences in each lang
        model: NLLB model
        tokenizer: NLLB tokenizer
        layers: Encoder layer indices
        device: Compute device

    Returns:
        {lang: {layer: tensor[n_sentences, hidden_dim]}}
    """
    reps: Dict[str, Dict[int, torch.Tensor]] = {}

    for lang, sentences in parallel_sentences.items():
        print(f"  Extracting representations for {lang}...")
        capture = ActivationCapture()
        capture.register_hooks(model, layers, component_type="residual")

        acts_by_layer: Dict[int, List[torch.Tensor]] = defaultdict(list)
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", src_lang=lang).to(device)
            with torch.no_grad():
                _ = model.model.encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
            for layer in layers:
                acts = capture.get_activations(layer, "residual")
                if acts is not None:
                    pooled = acts.mean(dim=1)          # [1, hidden_dim]
                    acts_by_layer[layer].append(pooled)
            capture.activations.clear()

        capture.cleanup()
        reps[lang] = {
            layer: torch.cat(tensors, dim=0)
            for layer, tensors in acts_by_layer.items()
            if tensors
        }

    return reps


# ---------------------------------------------------------------------------
# CKA similarity
# ---------------------------------------------------------------------------

def compute_cka_similarity(
    reps_a: torch.Tensor,
    reps_b: torch.Tensor,
) -> float:
    """
    Compute linear Centered Kernel Alignment (CKA) between two representation matrices.

    Args:
        reps_a: [n_sentences, hidden_dim]
        reps_b: [n_sentences, hidden_dim]

    Returns:
        CKA similarity in [0, 1]
    """
    X = reps_a.float().numpy()
    Y = reps_b.float().numpy()

    # Center
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Linear kernels
    K = X @ X.T
    L = Y @ Y.T

    # HSIC
    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n

    hsic_kl = np.trace(K @ H @ L @ H)
    hsic_kk = np.trace(K @ H @ K @ H)
    hsic_ll = np.trace(L @ H @ L @ H)

    denom = np.sqrt(hsic_kk * hsic_ll)
    return float(hsic_kl / denom) if denom > 1e-10 else 0.0


def compute_cka_matrix(
    reps_by_lang: Dict[str, Dict[int, torch.Tensor]],
    layer: int,
) -> Dict[tuple, float]:
    """Compute pairwise CKA for all language pairs at a given layer."""
    langs = list(reps_by_lang.keys())
    results = {}
    for i, la in enumerate(langs):
        for j, lb in enumerate(langs):
            if i >= j:
                continue
            ra = reps_by_lang[la].get(layer)
            rb = reps_by_lang[lb].get(layer)
            if ra is not None and rb is not None:
                results[(la, lb)] = compute_cka_similarity(ra, rb)
    return results


# ---------------------------------------------------------------------------
# English-centricity index
# ---------------------------------------------------------------------------

def compute_english_centricity(
    reps_by_lang: Dict[str, torch.Tensor],
    english_key: str = "eng_Latn",
) -> float:
    """
    Compute English-centricity index at a single layer.

    Index = mean distance to English centroid / mean distance to global centroid.
    < 1.0 → closer to English than to global mean (English-centric)
    > 1.0 → more language-neutral

    Args:
        reps_by_lang: {lang: [n_sentences, hidden_dim]} for a single layer
        english_key: Language code for English

    Returns:
        English-centricity index (float)
    """
    all_reps = torch.cat(list(reps_by_lang.values()), dim=0).float()
    global_centroid = all_reps.mean(dim=0)

    if english_key not in reps_by_lang:
        return float("nan")

    english_centroid = reps_by_lang[english_key].float().mean(dim=0)

    # Mean distance from each language's centroid to English vs global
    dist_to_english = []
    dist_to_global = []

    for lang, reps in reps_by_lang.items():
        if lang == english_key:
            continue
        centroid = reps.float().mean(dim=0)
        dist_to_english.append(torch.norm(centroid - english_centroid).item())
        dist_to_global.append(torch.norm(centroid - global_centroid).item())

    if not dist_to_english:
        return float("nan")

    mean_eng = float(np.mean(dist_to_english))
    mean_global = float(np.mean(dist_to_global))
    return mean_eng / mean_global if mean_global > 1e-10 else float("nan")


def compute_english_centricity_by_layer(
    reps: Dict[str, Dict[int, torch.Tensor]],
    layers: List[int],
    english_key: str = "eng_Latn",
) -> Dict[int, float]:
    """Compute English-centricity for each layer."""
    return {
        layer: compute_english_centricity(
            {lang: reps[lang][layer] for lang in reps if layer in reps[lang]},
            english_key,
        )
        for layer in layers
    }


# ---------------------------------------------------------------------------
# Silhouette score by language
# ---------------------------------------------------------------------------

def compute_silhouette_by_language(
    reps_by_lang: Dict[str, torch.Tensor],
) -> float:
    """
    Compute silhouette score measuring how well-separated language clusters are.

    High score (→ 1.0) = representations strongly cluster by language.
    Low/negative score = language boundaries are blurred.

    Args:
        reps_by_lang: {lang: [n_sentences, hidden_dim]} for a single layer

    Returns:
        Mean silhouette score
    """
    from sklearn.metrics import silhouette_score

    X_list = []
    labels = []
    lang_to_idx = {lang: i for i, lang in enumerate(reps_by_lang.keys())}

    for lang, reps in reps_by_lang.items():
        X_list.append(reps.float().numpy())
        labels.extend([lang_to_idx[lang]] * reps.shape[0])

    X = np.concatenate(X_list, axis=0)
    labels = np.array(labels)

    if len(np.unique(labels)) < 2:
        return float("nan")

    return float(silhouette_score(X, labels, metric="cosine"))
