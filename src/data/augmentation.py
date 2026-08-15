"""
Augmentation pipelines built with Albumentations.

Train-time augmentation: HFlip, VFlip, Rotation(+-15deg), ColorJitter.
Fundus images are rotation/flip-invariant in terms of diagnosis (a hemorrhage
is still a hemorrhage upside down), which is why these particular augmentations
are safe to use here -- unlike, say, natural photos where flipping might change
meaning.

Val/test-time: only resizing + normalization (no randomness), so evaluation
numbers are stable and reproducible.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int, mean, std, rotation_degrees: int = 15,
                          brightness: float = 0.2, contrast: float = 0.2,
                          saturation: float = 0.1, hue: float = 0.02,
                          scale_range: tuple = (0.9, 1.1)):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=rotation_degrees, border_mode=0, p=0.7),
        A.RandomScale(scale_limit=(scale_range[0] - 1, scale_range[1] - 1), p=0.5),
        A.Resize(image_size, image_size),  # re-resize after RandomScale changes dimensions
        A.ColorJitter(brightness=brightness, contrast=contrast,
                       saturation=saturation, hue=hue, p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_eval_transforms(image_size: int, mean, std):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
