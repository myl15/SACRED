"""
InterventionHook: Hook management for causal intervention experiments.

Supports:
  - register_ablation_hook        — zero out specific MLP neurons
  - register_random_ablation_hook — zero out random neurons (negative control)
  - register_patching_hook        — patch in clean activations (sufficiency)
  - register_vector_subtraction_hook — subtract concept vector from residual stream
    (NEW: primary method for cross-lingual transfer experiments)

Also provides validate_intervention_execution() to verify that hooks fired.
"""

import torch
import numpy as np
from typing import Callable, Dict, List, Optional
from collections import defaultdict

from extraction.activation_capture import ActivationCapture


class InterventionHook:
    """
    Manages intervention hooks with proper lifecycle management.
    Always calls cleanup() before registering new hooks to avoid stacking.
    """

    def __init__(self):
        self.handles: List = []
        self.intervention_type: Optional[str] = None
        self.intervention_params: Dict = {}

    # ------------------------------------------------------------------
    # Ablation hooks (circuit neurons, fc1 space)
    # ------------------------------------------------------------------

    def register_ablation_hook(self, model, circuit, component_type: str = "mlp"):
        """Zero out specific circuit neurons in fc1 activations."""
        self.cleanup()
        self.intervention_type = "ablation"

        neurons_by_layer: Dict[int, List[int]] = defaultdict(list)
        for neuron in circuit.neurons:
            neurons_by_layer[neuron.layer].append(neuron.neuron_idx)

        for layer_idx, neuron_indices in neurons_by_layer.items():
            if component_type == "mlp":
                target_module = model.model.encoder.layers[layer_idx].fc1
            else:
                raise NotImplementedError(f"Component type {component_type} not yet supported")

            handle = target_module.register_forward_hook(
                self._make_ablation_hook(neuron_indices)
            )
            self.handles.append(handle)

        self.intervention_params["neurons_by_layer"] = dict(neurons_by_layer)

    def _make_ablation_hook(self, neuron_indices: List[int]) -> Callable:
        def hook_fn(module, input, output):
            output_copy = output.clone()
            output_copy[:, :, neuron_indices] = 0.0
            return output_copy
        return hook_fn

    def register_random_ablation_hook(
        self,
        model,
        n_neurons_to_ablate: int,
        layers: List[int],
        intermediate_dim: int = None,
    ):
        """Zero out random neurons as a negative control."""
        self.cleanup()
        self.intervention_type = "random_ablation"

        for layer_idx in layers:
            fc1 = model.model.encoder.layers[layer_idx].fc1
            # Derive actual fc1 output size from the module rather than a constant
            dim = intermediate_dim if intermediate_dim is not None else fc1.out_features
            n_ablate = min(n_neurons_to_ablate, dim)
            random_neurons = np.random.choice(dim, size=n_ablate, replace=False).tolist()
            handle = fc1.register_forward_hook(
                self._make_ablation_hook(random_neurons)
            )
            self.handles.append(handle)

    # ------------------------------------------------------------------
    # Patching hook (sufficiency testing)
    # ------------------------------------------------------------------

    def register_patching_hook(self, model, circuit, clean_activations: Dict[int, torch.Tensor]):
        """Patch clean activations into specified circuit neurons."""
        self.cleanup()
        self.intervention_type = "patching"

        neurons_by_layer: Dict[int, List[int]] = defaultdict(list)
        for neuron in circuit.neurons:
            neurons_by_layer[neuron.layer].append(neuron.neuron_idx)

        for layer_idx, neuron_indices in neurons_by_layer.items():
            if layer_idx not in clean_activations:
                continue
            target_module = model.model.encoder.layers[layer_idx].fc1
            handle = target_module.register_forward_hook(
                self._make_patching_hook(neuron_indices, clean_activations[layer_idx])
            )
            self.handles.append(handle)

    def _make_patching_hook(self, neuron_indices: List[int], clean_acts: torch.Tensor) -> Callable:
        def hook_fn(module, input, output):
            output_copy = output.clone()
            output_copy[:, :, neuron_indices] = clean_acts[:, :, neuron_indices]
            return output_copy
        return hook_fn

    # ------------------------------------------------------------------
    # Vector subtraction hooks (concept vector transfer experiments)
    # ------------------------------------------------------------------

    def register_scaled_vector_subtraction_hook(
        self,
        model,
        concept_vector: torch.Tensor,
        layers: List[int],
        alpha: float = 1.0,
        target: str = "residual",
        vector_method: str = "mean",
    ):
        """
        Subtract alpha * concept_vector from encoder activations.

        Extends register_vector_subtraction_hook with:
          - fc1 target support (8192-dim MLP intermediate)
          - Dimension validation (raises ValueError on mismatch)

        Works with both mean-differencing and PCA reading vectors.

        Args:
            model: NLLB model
            concept_vector: [hidden_dim] tensor; dim must match target.
                            Can be a mean-difference or PCA reading vector.
            layers: Encoder layer indices to apply the subtraction
            alpha: Scaling factor — lower values reduce intervention strength
            target: "residual" for layer output (1024-dim) or
                    "fc1" for MLP intermediate (fc1.out_features)
            vector_method: Informational — "mean" or "pca". Stored in
                           intervention_params for bookkeeping only.
        """
        self.cleanup()
        self.intervention_type = "scaled_vector_subtraction"
        self.intervention_params = {"alpha": alpha, "layers": layers, "target": target, "vector_method": vector_method}

        for layer_idx in layers:
            if target == "residual":
                expected_dim = model.config.d_model
                if concept_vector.shape[0] != expected_dim:
                    raise ValueError(
                        f"concept_vector dim {concept_vector.shape[0]} does not match "
                        f"residual stream dim {expected_dim} (d_model) at layer {layer_idx}. "
                        f"Extract concept vectors with component='encoder_hidden'."
                    )
                target_module = model.model.encoder.layers[layer_idx]
                handle = target_module.register_forward_hook(
                    self._make_subtraction_hook(concept_vector, alpha)
                )
            elif target == "fc1":
                fc1 = model.model.encoder.layers[layer_idx].fc1
                expected_dim = fc1.out_features
                if concept_vector.shape[0] != expected_dim:
                    raise ValueError(
                        f"concept_vector dim {concept_vector.shape[0]} does not match "
                        f"fc1 intermediate dim {expected_dim} at layer {layer_idx}. "
                        f"Extract concept vectors with component='mlp'."
                    )
                handle = fc1.register_forward_hook(
                    self._make_subtraction_hook(concept_vector, alpha)
                )
            else:
                raise ValueError(f"Unknown target '{target}'. Choose 'residual' or 'fc1'.")

            self.handles.append(handle)

    def register_vector_subtraction_hook(
        self,
        model,
        concept_vector: torch.Tensor,
        layers: List[int],
        alpha: float = 1.0,
        vector_method: str = "mean",
    ):
        """
        Subtract a concept vector from encoder residual stream at specified layers.

        This is the core intervention for cross-lingual transfer testing:
          1. Extract concept vector from language A (e.g., Arabic sacred).
          2. Subtract it from language B's encoder activations during translation.
          3. Measure whether the concept disappears from the output.

        Works with both mean-differencing and PCA reading vectors — both have
        shape [hidden_dim] and are causal interventions in the same space.
        Pass the appropriate vector from extract_concept_vectors(method=...).

        Args:
            model: NLLB model
            concept_vector: [hidden_dim] tensor (residual stream dimension).
                            Can be a mean-difference or PCA reading vector.
            layers: Encoder layer indices to apply the subtraction
            alpha: Scaling factor (1.0 = full subtraction)
            vector_method: Informational — "mean" or "pca". Stored in
                           intervention_params for bookkeeping only.
        """
        self.cleanup()
        self.intervention_type = "vector_subtraction"
        self.intervention_params = {"alpha": alpha, "layers": layers, "vector_method": vector_method}

        for layer_idx in layers:
            target_module = model.model.encoder.layers[layer_idx]
            handle = target_module.register_forward_hook(
                self._make_subtraction_hook(concept_vector, alpha)
            )
            self.handles.append(handle)

    def _make_subtraction_hook(self, vector: torch.Tensor, alpha: float) -> Callable:
        def hook_fn(module, input, output):
            # encoder layer output is a tuple (hidden_states, ...)
            if isinstance(output, tuple):
                hidden = output[0]
                modified = hidden - alpha * vector.to(hidden.device)
                return (modified,) + output[1:]
            else:
                return output - alpha * vector.to(output.device)
        return hook_fn

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []


# ---------------------------------------------------------------------------
# Intervention validation
# ---------------------------------------------------------------------------

def validate_intervention_execution(
    model,
    circuit,
    test_sentence: str,
    tokenizer,
    lang_code: str,
    intervention_hook: InterventionHook,
    device: str = "cuda",
) -> Dict:
    """
    Verify that the intervention hook actually modified activations.

    Runs two encoder passes (clean, intervened) and checks that the targeted
    neurons were zeroed (for ablation) or modified (for vector subtraction).

    Returns:
        {"passed": bool, "errors": [...], "neuron_checks": [...]}
    """
    report = {"passed": True, "errors": [], "neuron_checks": []}

    layers = circuit.get_critical_layers()
    inputs = tokenizer(test_sentence, return_tensors="pt", src_lang=lang_code).to(device)

    # Baseline pass: capture hooks only, no ablation
    capture_baseline = ActivationCapture()
    capture_baseline.register_hooks(model, layers, component_type="mlp")
    intervention_hook.cleanup()
    with torch.no_grad():
        _ = model.model.encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
    baseline_acts = {}
    for layer in layers:
        acts = capture_baseline.get_activations(layer, "mlp")
        if acts is not None:
            baseline_acts[layer] = acts.clone()
    capture_baseline.cleanup()

    # Intervened pass: register ablation hooks FIRST, then capture hooks
    # so the capture sees post-ablation values (hooks fire in registration order).
    if intervention_hook.intervention_type == "ablation":
        intervention_hook.register_ablation_hook(model, circuit, "mlp")
    elif intervention_hook.intervention_type == "random_ablation":
        intervention_hook.register_random_ablation_hook(
            model,
            n_neurons_to_ablate=max(1, len(circuit.neurons) // max(len(layers), 1)),
            layers=layers,
        )

    capture_intervened = ActivationCapture()
    capture_intervened.register_hooks(model, layers, component_type="mlp")

    with torch.no_grad():
        _ = model.model.encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )

    intervened_acts = {}
    for layer in layers:
        acts = capture_intervened.get_activations(layer, "mlp")
        if acts is not None:
            intervened_acts[layer] = acts.clone()

    capture_intervened.cleanup()

    # Check ablation zeroed the correct neurons
    if intervention_hook.intervention_type in ("ablation", "random_ablation"):
        neurons_by_layer: Dict[int, List[int]] = defaultdict(list)
        for neuron in circuit.neurons:
            neurons_by_layer[neuron.layer].append(neuron.neuron_idx)

        for layer in layers:
            if layer not in baseline_acts or layer not in intervened_acts:
                continue
            for neuron_idx in neurons_by_layer.get(layer, []):
                baseline_val = baseline_acts[layer][0, 0, neuron_idx].item()
                intervened_val = intervened_acts[layer][0, 0, neuron_idx].item()
                if abs(intervened_val) > 1e-5:
                    report["passed"] = False
                    report["errors"].append(
                        f"Layer {layer}, Neuron {neuron_idx}: expected ~0, got {intervened_val:.6f}"
                    )
                report["neuron_checks"].append({
                    "layer": layer, "neuron": neuron_idx,
                    "baseline": baseline_val, "intervened": intervened_val,
                    "zeroed": abs(intervened_val) < 1e-5,
                })

    return report
