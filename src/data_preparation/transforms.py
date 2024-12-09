"""
Transformation functions for the liver tumor segmentation dataset.
"""

import torch
import numpy as np
from transformers import SegformerImageProcessor

# processor = SegformerImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")  # ViT
processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")  # Segformer


def process_column(col) -> list[torch.Tensor]:
    """
    Converts a column of PIL images to a list of torch tensors.
    """
    return [torch.tensor(np.array(x.convert("RGB")), dtype=torch.float32) for x in col]


def base_transforms(example_batch, processor=processor):
    """
    Base transformation function for the liver tumor segmentation dataset.
    """
    images = process_column(example_batch["image"])
    labels = [
        torch.tensor(np.array(x), dtype=torch.float32) for x in example_batch["label"]
    ]
    inputs = processor(images, labels)
    return inputs


def train_transforms(example_batch):
    """
    Transforms the training data to the format expected by the model.
    """
    return base_transforms(example_batch)


def val_transforms(example_batch):
    """
    Transforms the validation data to the format expected by the model.
    """
    return base_transforms(example_batch)
