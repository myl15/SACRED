"""
ActivationCapture: Forward-hook management for NLLB encoder layers.

Fixes the hook-stacking bug present in the original notebook by calling
cleanup() before re-registering hooks.

Supports two component types:
  "mlp"  — hooks model.model.encoder.layers[i].fc1  (8192-dim MLP intermediate)
  "attn" — hooks model.model.encoder.layers[i].self_attn
  "residual" — hooks model.model.encoder.layers[i] output (1024-dim residual stream)
"""

import torch
from typing import Callable, Dict, List, Optional
from collections import defaultdict


class ActivationCapture:
    """
    Proper activation capture with handle management.
    Fixes the hook stacking bug in the original notebook.
    """

    def __init__(self):
        self.handles: List = []
        self.activations: Dict = defaultdict(list)
        self.current_layer: Optional[int] = None

    def register_hooks(
        self,
        model,
        layers: List[int],
        component_type: str = "mlp",
    ):
        """
        Register forward hooks on specified layers.

        Args:
            model: NLLB model
            layers: Layer indices to hook
            component_type: "mlp", "attn", or "residual"
        """
        self.cleanup()

        for layer_idx in layers:
            if component_type == "mlp":
                target_module = model.model.encoder.layers[layer_idx].fc1
            elif component_type == "attn":
                target_module = model.model.encoder.layers[layer_idx].self_attn
            elif component_type == "residual":
                target_module = model.model.encoder.layers[layer_idx]
            else:
                raise ValueError(f"Unknown component_type: {component_type}")

            handle = target_module.register_forward_hook(
                self._make_hook_fn(layer_idx, component_type)
            )
            self.handles.append(handle)

    def _make_hook_fn(self, layer_idx: int, component_type: str) -> Callable:
        """Create hook function for specific layer and component type."""

        def hook_fn(module, input, output):
            if component_type == "mlp":
                # fc1 output: [batch, seq_len, intermediate_dim]
                self.activations[f"mlp_{layer_idx}"].append(output.detach().cpu())
            elif component_type == "attn":
                # self_attn output is a tuple; first element is the hidden states
                acts = output[0] if isinstance(output, tuple) else output
                self.activations[f"attn_{layer_idx}"].append(acts.detach().cpu())
            elif component_type == "residual":
                # Full encoder layer output — residual stream
                acts = output[0] if isinstance(output, tuple) else output
                self.activations[f"residual_{layer_idx}"].append(acts.detach().cpu())

        return hook_fn

    def cleanup(self):
        """Remove all registered hooks and clear stored activations."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.activations.clear()

    def get_activations(
        self, layer: int, component_type: str = "mlp"
    ) -> Optional[torch.Tensor]:
        """
        Get captured activations for a specific layer.

        Returns:
            Concatenated tensor [n_samples, seq_len, dim] or None if empty.
        """
        key = f"{component_type}_{layer}"
        if key in self.activations and self.activations[key]:
            return torch.cat(self.activations[key], dim=0)
        return None
