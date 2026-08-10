"""
GradCAM visualization on the EfficientNet backbone.

Shows WHICH regions of the fundus image most influenced a given class
prediction, by weighting the backbone's final feature map channels by the
gradient of the target class's logit with respect to those channels.

We hook the LAST EfficientNet stage (deepest, most semantic features) that
feeds into CBAM/FPN, since that's where "what pattern triggered this
prediction" is most interpretable. Uses the `grad-cam` pip package
(pip install grad-cam) which supports arbitrary nn.Module targets.
"""

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class RetinalGradCAM:
    def __init__(self, model, target_layer):
        """
        Args:
            model: the full HybridRetinalModel (in eval mode).
            target_layer: the nn.Module to hook -- e.g.
                model.backbone.backbone.blocks[-1] (last EfficientNet block).
        """
        self.model = model
        self.cam = GradCAM(model=model, target_layers=[target_layer])

    def generate(self, input_tensor: torch.Tensor, class_idx: int, rgb_image: np.ndarray):
        """
        Args:
            input_tensor: (1, 3, H, W) preprocessed image tensor.
            class_idx: which of the 8 classes to explain (0=N ... 7=O).
            rgb_image: (H, W, 3) float array in [0, 1], the same image
                (unnormalized) for overlay visualization.
        Returns:
            (H, W, 3) uint8 image with the GradCAM heatmap overlaid.
        """
        targets = [ClassifierOutputTarget(class_idx)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)[0]
        visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
        return visualization
