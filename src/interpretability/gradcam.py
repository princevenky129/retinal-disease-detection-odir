"""
Manual GradCAM implementation (no external pytorch_grad_cam dependency).

We implement GradCAM by hand because the pytorch_grad_cam library's
automatic target/backward handling does not reliably collapse to a scalar
for this model's multi-stage architecture (backbone -> CBAM -> FPN ->
bridge -> Swin -> head), causing "grad can be implicitly created only for
scalar outputs". Selecting the class logit explicitly and calling
.backward() on it ourselves sidesteps that entirely.

Core GradCAM algorithm (Selvaraju et al., 2017):
1. Forward hook on the target conv layer captures its activations.
2. Backward hook on the same layer captures the gradient of the selected
   class logit with respect to those activations.
3. Global-average-pool the gradients per channel -> per-channel weight.
4. Weighted sum of activation channels, then ReLU (keep only features that
   positively support the class).
5. Upsample to image size, overlay as a heatmap.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class RetinalGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int, rgb_image: np.ndarray):
        self.model.zero_grad()
        logits = self.model(input_tensor)        # (1, num_classes)
        target_score = logits[0, class_idx]        # scalar - this is the key fix

        target_score.backward()

        activations = self.activations[0]           # (C, h, w)
        gradients = self.gradients[0]                 # (C, h, w)

        weights = gradients.mean(dim=(1, 2))            # (C,)
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (rgb_image.shape[1], rgb_image.shape[0]))

        # Suppress border artifacts: fundus images are circular crops inside
        # a square frame, and the padding/edges outside the circle can
        # produce spurious high-gradient activations unrelated to any real
        # clinical feature. Zeroing a thin margin keeps the heatmap focused
        # on the actual fundus content.
        h, w = cam.shape
        margin_h, margin_w = int(h * 0.05), int(w * 0.05)
        border_mask = np.zeros_like(cam)
        border_mask[margin_h:h - margin_h, margin_w:w - margin_w] = 1.0
        cam = cam * border_mask

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        overlay = 0.5 * rgb_image + 0.5 * heatmap
        overlay = np.uint8(255 * overlay / overlay.max())
        return overlay