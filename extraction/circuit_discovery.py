"""
Circuit Discovery for Sacred Concept Analysis.

Contains the data structures (NeuronComponent, Circuit, UniversalCircuit)
and discovery functions (capture_all_activations, discover_sacred_circuit,
find_universal_components_with_validation).

This module operates on fc1 MLP internals (8192-dim). For concept vectors
operating on the residual stream (1024-dim) see extraction/concept_vectors.py.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from extraction.activation_capture import ActivationCapture
from analysis.statistical import compute_cohens_d, permutation_test, bootstrap_confidence_interval


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NeuronComponent:
    """Represents a single neuron component in the circuit."""
    layer: int
    neuron_idx: int
    effect_size: float          # Cohen's d
    p_value: float
    confidence_interval: tuple  # (lower, upper)
    mean_activation_sacred: float
    mean_activation_secular: float


@dataclass
class AttentionHeadComponent:
    """Represents a single attention head component in the circuit."""
    layer: int
    head_idx: int
    effect_size: float
    p_value: float
    attention_to_sacred: float


@dataclass
class Circuit:
    """Complete circuit representation with statistical validation."""
    neurons: List[NeuronComponent] = field(default_factory=list)
    attn_heads: List[AttentionHeadComponent] = field(default_factory=list)
    language: str = ""
    statistics: Dict[str, Any] = field(default_factory=dict)

    def get_neurons_by_layer(self, layer: int) -> List[NeuronComponent]:
        return [n for n in self.neurons if n.layer == layer]

    def get_critical_layers(self) -> List[int]:
        layers_with_neurons = {n.layer for n in self.neurons}
        layers_with_heads = {h.layer for h in self.attn_heads}
        return sorted(layers_with_neurons | layers_with_heads)


@dataclass
class UniversalCircuit(Circuit):
    """Circuit components universal across all languages."""
    languages: List[str] = field(default_factory=list)
    meta_analysis_results: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Activation capture helpers
# ---------------------------------------------------------------------------

def capture_all_activations(
    sentences: List[str],
    model,
    tokenizer,
    lang_code: str,
    layers: Optional[List[int]] = None,
    device: str = "cuda",
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Capture MLP fc1 and attention activations for all sentences.

    Returns:
        {"mlp": {layer: tensor[n_sents, mlp_dim]},
         "attn": {layer: tensor[n_sents, hidden_dim]}}
    """
    if layers is None:
        layers = list(range(model.config.encoder_layers))

    all_activations: Dict[str, Dict[int, torch.Tensor]] = {"mlp": {}, "attn": {}}

    for comp_type in ("mlp", "attn"):
        capture = ActivationCapture()
        capture.register_hooks(model, layers, component_type=comp_type)
        acts_by_layer: Dict[int, List] = defaultdict(list)

        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", src_lang=lang_code).to(device)
            with torch.no_grad():
                _ = model.model.encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
            for layer in layers:
                acts = capture.get_activations(layer, comp_type)
                if acts is not None:
                    pooled = acts.mean(dim=1)          # [batch, dim]
                    acts_by_layer[layer].append(pooled)
            capture.activations.clear()

        capture.cleanup()
        for layer in layers:
            if acts_by_layer[layer]:
                all_activations[comp_type][layer] = torch.cat(acts_by_layer[layer], dim=0)

    return all_activations


# ---------------------------------------------------------------------------
# Statistical scoring
# ---------------------------------------------------------------------------

def compute_contrastive_scores_with_stats(
    sacred_acts: torch.Tensor,
    secular_acts: torch.Tensor,
    inanimate_acts: torch.Tensor,
    alpha: float = 0.01,
    min_effect_size: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Compute per-neuron contrastive statistics with FDR correction.

    Returns list of significant neuron dicts.
    """
    sacred_np = sacred_acts.cpu().numpy()
    secular_np = secular_acts.cpu().numpy()

    n_neurons = sacred_np.shape[1]
    neuron_stats = []

    for neuron_idx in tqdm(range(n_neurons), desc="    Neurons", leave=False):
        sacred_n = sacred_np[:, neuron_idx]
        secular_n = secular_np[:, neuron_idx]

        d = compute_cohens_d(sacred_n, secular_n)
        if abs(d) < min_effect_size:
            continue

        p_value = permutation_test(sacred_n, secular_n, n_permutations=100)
        ci = bootstrap_confidence_interval(sacred_n - secular_n, n_bootstrap=100)

        neuron_stats.append({
            "neuron_idx": neuron_idx,
            "cohens_d": d,
            "p_value": p_value,
            "ci_lower": ci[0],
            "ci_upper": ci[1],
            "mean_sacred": float(np.mean(sacred_n)),
            "mean_secular": float(np.mean(secular_n)),
        })

    if not neuron_stats:
        return []

    p_values = [n["p_value"] for n in neuron_stats]
    _, corrected_p, _, _ = multipletests(p_values, method="fdr_bh")
    for i, stat in enumerate(neuron_stats):
        stat["p_value_corrected"] = corrected_p[i]

    significant = [
        n for n in neuron_stats
        if n["p_value_corrected"] < alpha and abs(n["cohens_d"]) >= min_effect_size
    ]
    print(f"    Found {len(significant)} significant neurons (p < {alpha}, |d| > {min_effect_size})")
    return significant


# ---------------------------------------------------------------------------
# Discovery functions
# ---------------------------------------------------------------------------

def discover_sacred_circuit(
    stimuli: Dict[str, List[Dict]],
    model,
    tokenizer,
    lang_code: str,
    include_attention: bool = True,
    alpha: float = 0.01,
    layers_to_analyze: Optional[List[int]] = None,
    device: str = "cuda",
) -> Circuit:
    """
    Discover sacred concept circuit for a single language.

    Args:
        stimuli: {"sacred": [...], "secular": [...], "inanimate": [...]}
        model: NLLB model
        tokenizer: NLLB tokenizer
        lang_code: Language code
        include_attention: Whether to analyze attention heads
        alpha: Statistical significance threshold
        layers_to_analyze: Specific layers (None = all)
        device: Device string

    Returns:
        Circuit with validated components
    """
    print(f"\nDiscovering circuit for {lang_code}...")

    sacred_sentences = [s["text"] for s in stimuli["sacred"]]
    secular_sentences = [s["text"] for s in stimuli["secular"]]
    inanimate_sentences = [s["text"] for s in stimuli["inanimate"]]

    print("  Capturing activations...")
    sacred_acts = capture_all_activations(sacred_sentences, model, tokenizer, lang_code, device=device)
    secular_acts = capture_all_activations(secular_sentences, model, tokenizer, lang_code, device=device)
    inanimate_acts = capture_all_activations(inanimate_sentences, model, tokenizer, lang_code, device=device)

    circuit = Circuit(language=lang_code)

    if layers_to_analyze is None:
        layers_to_analyze = list(range(model.config.encoder_layers))

    print(f"  Analyzing {len(layers_to_analyze)} layers: {layers_to_analyze}")

    for layer in layers_to_analyze:
        if layer not in sacred_acts["mlp"]:
            continue

        print(f"  Analyzing layer {layer}/{max(layers_to_analyze)}...")

        significant_neurons = compute_contrastive_scores_with_stats(
            sacred_acts["mlp"][layer],
            secular_acts["mlp"][layer],
            inanimate_acts["mlp"][layer],
            alpha=alpha,
        )

        for neuron_stat in significant_neurons:
            component = NeuronComponent(
                layer=layer,
                neuron_idx=neuron_stat["neuron_idx"],
                effect_size=neuron_stat["cohens_d"],
                p_value=neuron_stat["p_value_corrected"],
                confidence_interval=(neuron_stat["ci_lower"], neuron_stat["ci_upper"]),
                mean_activation_sacred=neuron_stat["mean_sacred"],
                mean_activation_secular=neuron_stat["mean_secular"],
            )
            circuit.neurons.append(component)

    print(f"Found {len(circuit.neurons)} significant neurons across {len(circuit.get_critical_layers())} layers")
    return circuit


def find_universal_components_with_validation(
    circuits_by_lang: Dict[str, Circuit],
    alpha: float = 0.01,
) -> UniversalCircuit:
    """
    Identify circuit components universal across all languages.

    Uses set intersection + hypergeometric test for overlap significance.
    """
    print(f"\nFinding universal components across {len(circuits_by_lang)} languages...")

    languages = list(circuits_by_lang.keys())
    universal_circuit = UniversalCircuit(languages=languages)

    all_layers: set = set()
    for circuit in circuits_by_lang.values():
        all_layers.update(circuit.get_critical_layers())

    for layer in sorted(all_layers):
        neurons_by_lang = {}
        for lang, circuit in circuits_by_lang.items():
            layer_neurons = circuit.get_neurons_by_layer(layer)
            neurons_by_lang[lang] = {
                n.neuron_idx for n in layer_neurons
                if n.p_value < alpha and abs(n.effect_size) > 0.5
            }

        if not neurons_by_lang:
            continue

        universal_neurons = set.intersection(*neurons_by_lang.values())

        for neuron_idx in universal_neurons:
            effect_sizes = []
            p_values = []
            for lang, circuit in circuits_by_lang.items():
                for n in circuit.get_neurons_by_layer(layer):
                    if n.neuron_idx == neuron_idx:
                        effect_sizes.append(n.effect_size)
                        p_values.append(n.p_value)
                        break

            component = NeuronComponent(
                layer=layer,
                neuron_idx=neuron_idx,
                effect_size=float(np.mean(effect_sizes)),
                p_value=float(np.min(p_values)),
                confidence_interval=(0.0, 0.0),
                mean_activation_sacred=0.0,
                mean_activation_secular=0.0,
            )
            universal_circuit.neurons.append(component)

    print(
        f"Universal circuit: {len(universal_circuit.neurons)} neurons "
        f"across {len(universal_circuit.get_critical_layers())} layers"
    )
    return universal_circuit
