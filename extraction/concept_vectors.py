"""
Concept Vector Extraction via Contrastive Activation Differencing.

Extracts dense concept vectors (directions in activation space) by averaging
the activation difference between positive/negative sentence pairs.

Key design decisions (per phase_one_plan.md):
  - Hooks the RESIDUAL STREAM (encoder.layers[i] output, 1024-dim) rather than
    fc1 (8192-dim MLP intermediate), for cleaner causal interventions.
  - Uses sequence mean-pooling by default (Approach B). Switch to
    pooling="token_aligned" if kinship results are noisy.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

from extraction.activation_capture import ActivationCapture


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
) -> Dict[int, torch.Tensor]:
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

    Returns:
        {layer_idx: concept_vector} where concept_vector has shape [hidden_dim]
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

    concept_vectors: Dict[int, torch.Tensor] = {}
    for layer in layers:
        if layer in pos_acts and layer in neg_acts:
            diff = pos_acts[layer] - neg_acts[layer]   # [n_pairs, hidden_dim]
            concept_vectors[layer] = diff.mean(dim=0)  # [hidden_dim]

    return concept_vectors


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
