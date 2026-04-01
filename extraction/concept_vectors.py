"""
Concept Vector Extraction via Contrastive Activation Differencing.

Extracts dense concept vectors (directions in activation space) by:
  - method="mean": averaging the activation difference between pos/neg pairs
  - method="pca":  LAT-style first principal component of the difference matrix
  - method="both": returns {"mean": ..., "pca": ...} dicts

Key design decisions (per phase_one_plan.md):
  - Hooks the RESIDUAL STREAM (encoder.layers[i] output, 1024-dim) rather than
    fc1 (8192-dim MLP intermediate), for cleaner causal interventions.
  - Uses sequence mean-pooling by default (Approach B). Switch to
    pooling="token_aligned" if kinship results are noisy.
"""

import torch
import numpy as np
from typing import Dict, List, Literal, Optional, Union
from collections import defaultdict

from sklearn.decomposition import PCA

from extraction.activation_capture import ActivationCapture


def _pca_reading_vector(
    diff_matrix: torch.Tensor,  # [n_pairs, hidden_dim]
    sign_labels: Optional[torch.Tensor] = None,  # [n_pairs] — +1 for pos, -1 for neg
) -> torch.Tensor:
    """
    Compute a LAT-style reading vector via PCA on the difference matrix.

    The first principal component captures the direction of maximal variance
    across pairs, which is more robust than naive mean-differencing when
    individual pairs have noisy or partial concept signal.

    Sign correction: PCA is sign-ambiguous. We align the vector so that
    projecting positive-pair differences onto it yields positive scores.
    If sign_labels is None, we use the mean of the diff matrix as a proxy.

    Args:
        diff_matrix: [n_pairs, hidden_dim] tensor of (pos - neg) differences
        sign_labels: Optional [n_pairs] tensor for sign correction

    Returns:
        Reading vector of shape [hidden_dim]
    """
    X = diff_matrix.float().numpy()  # sklearn expects numpy

    if X.shape[0] < 2:
        # Fallback to mean if we have fewer than 2 pairs
        return diff_matrix.mean(dim=0)

    pca = PCA(n_components=1)
    scores = pca.fit_transform(X)          # [n_pairs, 1]
    reading_vec = torch.tensor(
        pca.components_[0], dtype=torch.float32
    )                                       # [hidden_dim]

    # Sign correction: ensure the PC points in the positive-concept direction
    if sign_labels is not None:
        alignment = (torch.tensor(scores[:, 0]) * sign_labels).mean()
    else:
        # Proxy: project mean difference onto the PC — should be positive
        mean_diff = diff_matrix.float().mean(dim=0)
        alignment = torch.dot(mean_diff, reading_vec)

    if alignment < 0:
        reading_vec = -reading_vec

    return reading_vec


def extract_concept_vectors(
    contrastive_pairs: List[Dict],
    model,
    tokenizer,
    lang_code: str,
    layers: List[int],
    component: str = "encoder_hidden",
    pooling: str = "mean",
    concept_token_positions: Optional[List[int]] = None,
    device: str = "cuda",
    method: Literal["mean", "pca", "both"] = "mean",
    return_diffs: bool = False,
) -> Union[Dict[int, torch.Tensor], Dict[str, Dict[int, torch.Tensor]]]:
    """
    Extract a concept vector per layer via contrastive activation differencing.

    Args:
        contrastive_pairs: List of {"positive": str, "negative": str, ...} dicts
        model: NLLB model
        tokenizer: NLLB tokenizer
        lang_code: Language code for tokenization
        layers: Encoder layer indices to extract from
        component: "encoder_hidden" (residual stream) or "mlp" (fc1 intermediate)
        pooling: "mean" (sequence mean-pool) or "token_aligned" (concept position)
        concept_token_positions: Token positions for token_aligned pooling
        device: Compute device
        method: Vector extraction method:
            "mean" — naive mean of (pos - neg) differences (default, backward-compat)
            "pca"  — LAT-style first principal component of the difference matrix
            "both" — returns {"mean": {layer: tensor}, "pca": {layer: tensor}}
        return_diffs: If True, the return value gains a "diffs" key containing
            {layer: tensor[n_pairs, hidden_dim]} — the raw per-pair difference
            matrices needed for explained-variance and scatter visualizations.
            Ignored when method is "mean" or "pca" (use method="both" to access).

    Returns:
        If method is "mean" or "pca":
            {layer_idx: concept_vector [hidden_dim]}
        If method is "both":
            {"mean": {layer: tensor}, "pca": {layer: tensor}}
        If method is "both" and return_diffs=True:
            {"mean": ..., "pca": ..., "diffs": {layer: tensor[n_pairs, hidden_dim]}}
    """
    comp_type = "residual" if component == "encoder_hidden" else "mlp"

    pos_sentences = [p["positive"] for p in contrastive_pairs]
    neg_sentences = [p["negative"] for p in contrastive_pairs]

    pos_acts = _collect_activations(
        pos_sentences, model, tokenizer, lang_code, layers, comp_type,
        pooling, concept_token_positions, device,
    )
    neg_acts = _collect_activations(
        neg_sentences, model, tokenizer, lang_code, layers, comp_type,
        pooling, None, device,
    )

    mean_vectors: Dict[int, torch.Tensor] = {}
    pca_vectors: Dict[int, torch.Tensor] = {}
    diff_matrices: Dict[int, torch.Tensor] = {}

    for layer in layers:
        if layer not in pos_acts or layer not in neg_acts:
            continue
        diff = pos_acts[layer] - neg_acts[layer]   # [n_pairs, hidden_dim]
        diff_matrices[layer] = diff.cpu()
        mean_vectors[layer] = diff.mean(dim=0)     # [hidden_dim]
        pca_vectors[layer] = _pca_reading_vector(diff) if diff.shape[0] >= 2 else mean_vectors[layer]

    if method == "mean":
        return mean_vectors
    elif method == "pca":
        return pca_vectors
    else:  # "both"
        result: Dict = {"mean": mean_vectors, "pca": pca_vectors}
        if return_diffs:
            result["diffs"] = diff_matrices
        return result


def _collect_activations(
    sentences: List[str],
    model,
    tokenizer,
    lang_code: str,
    layers: List[int],
    comp_type: str,
    pooling: str,
    concept_token_positions: Optional[List[int]],
    device: str,
) -> Dict[int, torch.Tensor]:
    """Collect (optionally pooled) activations for a list of sentences."""
    capture = ActivationCapture()
    capture.register_hooks(model, layers, component_type=comp_type)

    acts_by_layer: Dict[int, List[torch.Tensor]] = defaultdict(list)

    for i, sentence in enumerate(sentences):
        inputs = tokenizer(sentence, return_tensors="pt", src_lang=lang_code).to(device)
        with torch.no_grad():
            _ = model.model.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )

        for layer in layers:
            raw = capture.get_activations(layer, comp_type)  # [1, seq_len, dim]
            if raw is None:
                continue

            if pooling == "token_aligned" and concept_token_positions is not None:
                pos = concept_token_positions[i] if i < len(concept_token_positions) else 0
                pos = min(pos, raw.shape[1] - 1)
                pooled = raw[:, pos, :]              # [1, dim]
            else:
                pooled = raw.mean(dim=1)             # [1, dim]

            acts_by_layer[layer].append(pooled)

        capture.activations.clear()

    capture.cleanup()

    return {
        layer: torch.cat(tensors, dim=0)
        for layer, tensors in acts_by_layer.items()
        if tensors
    }


def save_concept_vectors(
    vectors: Dict[str, Dict[str, Dict[int, torch.Tensor]]],
    output_path: str,
) -> None:
    """
    Save concept vectors to disk.

    Args:
        vectors: {domain: {lang: {layer: tensor}}}
        output_path: Path for .pt file
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    serializable = {
        domain: {
            lang: {str(layer): vec.cpu() for layer, vec in layer_vecs.items()}
            for lang, layer_vecs in lang_vecs.items()
        }
        for domain, lang_vecs in vectors.items()
    }
    torch.save(serializable, output_path)
    print(f"Concept vectors saved to {output_path}")


def load_concept_vectors(path: str) -> Dict[str, Dict[str, Dict[int, torch.Tensor]]]:
    """Load concept vectors saved by save_concept_vectors."""
    raw = torch.load(path, map_location="cpu")
    return {
        domain: {
            lang: {int(layer): vec for layer, vec in layer_vecs.items()}
            for lang, layer_vecs in lang_vecs.items()
        }
        for domain, lang_vecs in raw.items()
    }
