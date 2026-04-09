"""
Layer-Wise Convergence Analysis (Experiment 3).

Computes:
  - extract_parallel_representations: encoder hidden states per layer
  - compute_cka_similarity: Centered Kernel Alignment
  - compute_english_centricity: distance ratio to English centroid
  - compute_silhouette_by_language: language cluster separation score
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from extraction.activation_capture import ActivationCapture
from extraction.concept_vectors import _pca_reading_vector

from sklearn.model_selection import KFold


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


# ---------------------------------------------------------------------------
# Concept direction geometry (PCA-based, for exp3)
# ---------------------------------------------------------------------------

def load_diff_matrices(
    domain: str,
    languages: List[str],
    vectors_dir: str = "outputs/vectors",
) -> Dict[str, Dict[str, Dict[int, torch.Tensor]]]:
    """
    Load per-pair activation difference matrices saved by exp1.

    Args:
        domain: "sacred" or "kinship"
        languages: NLLB language codes to load
        vectors_dir: Directory containing *_diffs.pt files

    Returns:
        {lang: {concept: {layer: tensor[n_pairs, hidden_dim]}}}
    """
    result = {}
    for lang in languages:
        path = Path(vectors_dir) / f"{domain}_{lang}_diffs.pt"
        if not path.exists():
            print(f"  Skipping {lang}: {path} not found (run exp1 first)")
            continue
        raw = torch.load(path, map_location="cpu")
        result[lang] = {
            concept: {int(l): d for l, d in layer_dict.items()}
            for concept, layer_dict in raw.items()
        }
    return result


def compute_concept_direction_alignment(
    diff_by_lang: Dict[str, Dict[str, Dict[int, torch.Tensor]]],
    layers: List[int],
) -> Dict[Tuple[str, str], Dict[int, float]]:
    """
    Compute cross-language alignment of PCA concept directions layer by layer.

    For each layer, extract the PCA reading vector per language (averaged across
    concepts), then compute pairwise cosine similarity between languages. A value
    near 1.0 means two languages encode the concept in the same residual-stream
    direction — the geometric prerequisite for cross-lingual causal transfer.

    We take the absolute cosine because PCA sign correction is a heuristic and
    can be inconsistent across languages.

    Args:
        diff_by_lang: {lang: {concept: {layer: diff[n_pairs, hidden_dim]}}}
        layers: Encoder layer indices to evaluate

    Returns:
        {(lang_a, lang_b): {layer: |cosine_similarity|}}
    """
    # Collect per-concept PCA directions per lang per layer
    pca_vecs_by_lang: Dict[str, Dict[int, List[torch.Tensor]]] = {}
    for lang, concepts in diff_by_lang.items():
        for concept, layer_diffs in concepts.items():
            for layer, diff in layer_diffs.items():
                if layer not in layers:
                    continue
                if diff.shape[0] < 2:
                    continue
                pca_vec = _pca_reading_vector(diff)   # unit vector [hidden_dim]
                pca_vecs_by_lang.setdefault(lang, {}).setdefault(layer, []).append(pca_vec)

    # Average across concepts to get one direction per lang × layer
    lang_avg_pca: Dict[str, Dict[int, torch.Tensor]] = {
        lang: {
            layer: torch.stack(vecs).mean(0)
            for layer, vecs in layer_dict.items()
        }
        for lang, layer_dict in pca_vecs_by_lang.items()
    }

    # Pairwise cosine similarity
    langs = sorted(lang_avg_pca.keys())
    alignment: Dict[Tuple[str, str], Dict[int, float]] = {}
    for i, la in enumerate(langs):
        for j, lb in enumerate(langs):
            if i >= j:
                continue
            pair = (la, lb)
            alignment[pair] = {}
            for layer in layers:
                va = lang_avg_pca.get(la, {}).get(layer)
                vb = lang_avg_pca.get(lb, {}).get(layer)
                if va is not None and vb is not None:
                    cos = F.cosine_similarity(
                        va.float().unsqueeze(0), vb.float().unsqueeze(0)
                    ).item()
                    alignment[pair][layer] = abs(cos)

    return alignment


def compute_projection_consistency(
    diff_by_lang: Dict[str, Dict[str, Dict[int, torch.Tensor]]],
    layers: List[int],
) -> Dict[str, Dict[int, float]]:
    """
    Measure how consistently contrastive pairs align with the PCA concept direction.

    For each lang × concept × layer, project every per-pair difference vector onto
    the PCA direction and compute the fraction with positive projection. A score
    near 1.0 means all pairs agree — the PCA direction is a reliable concept probe.
    A score near 0.5 means the direction is no better than random for that slice.

    Args:
        diff_by_lang: {lang: {concept: {layer: diff[n_pairs, hidden_dim]}}}
        layers: Encoder layer indices to evaluate

    Returns:
        {"{lang_prefix}/{concept}": {layer: fraction_positive}}
    """
    consistency: Dict[str, Dict[int, float]] = {}
    for lang, concepts in diff_by_lang.items():
        for concept, layer_diffs in concepts.items():
            label = f"{lang.split('_')[0]}/{concept}"
            consistency[label] = {}
            for layer in layers:
                if layer not in layer_diffs:
                    continue
                diff = layer_diffs[layer].float()   # [n_pairs, hidden_dim]
                if diff.shape[0] < 2:
                    continue
                pca_vec = _pca_reading_vector(diff)
                projections = diff.mv(pca_vec)       # [n_pairs]
                consistency[label][layer] = (projections > 0).float().mean().item()
    return consistency


def compute_cross_lingual_projection_transfer(
    diff_by_lang: Dict[str, Dict[str, Dict[int, torch.Tensor]]],
    layer: int,
) -> Tuple[np.ndarray, List[str]]:
    """
    Cross-lingual PCA direction generalization matrix at a single layer.

    Entry [i, j] = fraction of language_i's contrastive-pair differences that
    project positively onto language_j's PCA concept direction. The diagonal
    [i, i] is the self-consistency score. Off-diagonal entries tell us whether
    language j's concept direction separates language i's concept signal —
    the representation-space prerequisite for causal cross-lingual transfer.

    Args:
        diff_by_lang: {lang: {concept: {layer: diff[n_pairs, hidden_dim]}}}
        layer: Encoder layer to evaluate

    Returns:
        (matrix[N, N], language_list)
    """
    # Pool all concept diff rows and PCA directions per language
    pooled_diffs: Dict[str, torch.Tensor] = {}
    pooled_pca:   Dict[str, torch.Tensor] = {}

    for lang, concepts in diff_by_lang.items():
        diffs_for_lang, pcas_for_lang = [], []
        for concept, layer_diffs in concepts.items():
            diff = layer_diffs.get(layer)
            if diff is None or diff.shape[0] < 2:
                continue
            diffs_for_lang.append(diff.float())
            pcas_for_lang.append(_pca_reading_vector(diff))
        if diffs_for_lang:
            pooled_diffs[lang] = torch.cat(diffs_for_lang, dim=0)
            pooled_pca[lang]   = torch.stack(pcas_for_lang).mean(0)

    langs = sorted(set(pooled_diffs.keys()) & set(pooled_pca.keys()))
    N = len(langs)
    matrix = np.zeros((N, N))

    for i, la in enumerate(langs):
        diff_a = pooled_diffs[la]              # [n_pairs_a, hidden_dim]
        for j, lb in enumerate(langs):
            pca_b = pooled_pca[lb]             # [hidden_dim]
            projections = diff_a.mv(pca_b)     # [n_pairs_a]
            matrix[i, j] = (projections > 0).float().mean().item()

    return matrix, langs


def compute_linear_probe_cv_accuracy(
    diff_by_lang: Dict[str, Dict[str, Dict[int, torch.Tensor]]],
    layers: List[int],
    k: int = 5,
) -> Dict[str, Dict[int, float]]:
    """
    Circularity-free directional consistency accuracy via k-fold cross-validation.
    Measures if held-out contrastive differences align with the mean concept direction.

    The projection consistency metric is circular: the PCA direction is estimated
    from the same pairs it is then evaluated on. This function fixes that by using
    k-fold CV: each fold estimates the probe direction from the TRAINING pairs only
    (as the mean of the training diff rows), then evaluates fraction-positive on the
    held-out fold. The mean CV accuracy across folds is the reported score.

    Interpretation: 0.5 = chance (direction is no better than random on held-out
    data); 1.0 = perfect (all held-out pairs agree with the direction estimated from
    training data). Unlike projection consistency, a high score here is a genuine
    finding — not a statistical artifact.

    Args:
        diff_by_lang: {lang: {concept: {layer: diff[n_pairs, hidden_dim]}}}
        layers: Encoder layer indices to evaluate
        k: Number of cross-validation folds (default: 5)

    Returns:
        {"{lang_prefix}/{concept}": {layer: mean_cv_accuracy}}
    """
    cv_consistency: Dict[str, Dict[int, float]] = {}

    for lang, concepts in diff_by_lang.items():
        for concept, layer_diffs in concepts.items():
            label = f"{lang.split('_')[0]}/{concept}"
            cv_consistency[label] = {}

            for layer in layers:
                if layer not in layer_diffs:
                    continue
                
                diff = layer_diffs[layer].float()
                n = diff.shape[0]
                if n < k:
                    continue

                kf = KFold(n_splits=k, shuffle=True, random_state=42)
                fold_scores = []

                for train_idx, test_idx in kf.split(diff):
                    # Estimate direction from train
                    train_dir = diff[train_idx].mean(dim=0)
                    train_dir = train_dir / (train_dir.norm() + 1e-8)
                    
                    # Check alignment of test differences
                    test_diffs = diff[test_idx]
                    projections = test_diffs.mv(train_dir)
                    fold_scores.append((projections > 0).float().mean().item())

                cv_consistency[label][layer] = float(np.mean(fold_scores))

    return cv_consistency


def compute_cross_lingual_probe_transfer(
    diff_by_lang: Dict[str, Dict[str, Dict[int, torch.Tensor]]],
    layer: int,
) -> Tuple[np.ndarray, List[str]]:
    """
    Cross-lingual linear probe transfer matrix at a single layer.

    Entry [i, j] = fraction of language_j's diff rows that project positively onto
    the probe direction trained on ALL of language_i's diff rows at this layer.

    The diagonal [i, i] is the k-fold CV self-accuracy (circularity-free, k=5).
    Off-diagonal [i, j] is genuine cross-lingual generalization: language_i's probe
    direction is never exposed to language_j's data.

    This is a representation-space analogue of the Exp4 causal transfer matrix and
    can be displayed alongside it for direct comparison. High off-diagonal values
    (≥ 0.7) indicate that language_i's concept direction generalises to language_j —
    the linear-representation prerequisite for causal cross-lingual transfer.

    Args:
        diff_by_lang: {lang: {concept: {layer: diff[n_pairs, hidden_dim]}}}
        layer: Encoder layer to evaluate

    Returns:
        (matrix[N, N], language_list)
        matrix[i, j] = probe_i accuracy on language_j's data
    """
    K_FOLDS = 5

    # Pool diff rows across concepts per language
    pooled: Dict[str, torch.Tensor] = {}
    for lang, concepts in diff_by_lang.items():
        rows = [d.float() for c, ld in concepts.items()
                if (d := ld.get(layer)) is not None]
        if rows:
            pooled[lang] = torch.cat(rows, dim=0)

    langs = sorted(pooled.keys())
    N = len(langs)
    matrix = np.zeros((N, N))

    for i, lang_i in enumerate(langs):
        diff_i = pooled[lang_i]                         # [n_i, hidden_dim]
        # Probe direction: mean of ALL of language i's rows
        probe_dir = diff_i.mean(dim=0)
        probe_dir = probe_dir / (probe_dir.norm() + 1e-8)

        for j, lang_j in enumerate(langs):
            diff_j = pooled[lang_j]                     # [n_j, hidden_dim]

            if i == j:
                # Diagonal: k-fold CV self-accuracy to avoid circularity
                n = diff_i.shape[0]
                fold_size = max(1, n // K_FOLDS)
                fold_accs = []
                idx = np.arange(n)
                for fold in range(K_FOLDS):
                    test_idx  = idx[fold * fold_size : (fold + 1) * fold_size]
                    train_idx = np.concatenate(
                        [idx[:fold * fold_size], idx[(fold + 1) * fold_size:]]
                    )
                    if len(train_idx) == 0 or len(test_idx) == 0:
                        continue
                    train_dir = diff_i[train_idx].mean(dim=0)
                    train_dir = train_dir / (train_dir.norm() + 1e-8)
                    proj = diff_i[test_idx].mv(train_dir)
                    fold_accs.append((proj > 0).float().mean().item())
                matrix[i, j] = float(np.mean(fold_accs)) if fold_accs else 0.0
            else:
                # Off-diagonal: probe trained on lang_i, tested on lang_j
                proj = diff_j.mv(probe_dir)
                matrix[i, j] = (proj > 0).float().mean().item()

    return matrix, langs
