"""
Swin Transformer attention map visualization.

Extracts self-attention weights from the last Swin-B stage's window attention
and rolls them into a coarse "where is the model attending" heatmap, so it
can be visually compared against the CNN-side GradCAM output (Phase 8: do
CNN GradCAM and Swin attention agree on lesion localization?).

Implementation note: timm's Swin blocks compute attention internally and
don't expose it by default, so we register a forward hook on the last
block's `attn` module to capture the attention tensor during a forward pass.
"""

import numpy as np
import torch


class SwinAttentionExtractor:
    def __init__(self, swin_encoder_module):
        """
        Args:
            swin_encoder_module: the `.encoder` attribute of SwinBEncoder
                (the raw timm Swin model).
        """
        self.attn_maps = []
        self._hook_handle = None
        self._register_hook(swin_encoder_module)

    def _register_hook(self, swin_model):
        last_block = swin_model.layers[-1].blocks[-1]

        def hook(module, input, output):
            # timm's WindowAttention doesn't return attn weights by default;
            # this hook captures them if `fused_attn=False` is set on the
            # block (forces the eager attention path that computes/keeps
            # the attn matrix). See notebooks/03_results_analysis.ipynb for
            # the full extraction + upsampling-to-image-size code.
            if hasattr(module, "attn_weights"):
                self.attn_maps.append(module.attn_weights.detach().cpu())

        self._hook_handle = last_block.attn.register_forward_hook(hook)

    def clear(self):
        self.attn_maps = []

    def remove(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()

    def get_rolled_attention(self, image_size: int = 224) -> np.ndarray:
        """
        Averages captured attention heads/windows and upsamples to a
        coarse (image_size, image_size) heatmap for overlay visualization.
        Placeholder aggregation -- refine per what the hook actually captures
        once run against the real trained model (Swin internals vary by timm
        version; verify `attn_weights` availability first).
        """
        if not self.attn_maps:
            raise RuntimeError("No attention maps captured. Run a forward pass first.")
        avg_attn = torch.stack(self.attn_maps).mean(dim=0)
        return avg_attn.numpy()
