"""
CNN Encoder — Pretrained feature extractor for video frames.

Uses ResNet50 (or ResNet18) as backbone.
Removes the final classification layer and outputs a feature vector
of dimension `feature_dim` for each input frame.

NOT trainable by default (freeze_backbone=True).
"""

import torch
import torch.nn as nn
from torchvision import models

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CNNEncoder(nn.Module):
    """
    Pretrained CNN backbone for spatial feature extraction.

    Input:  (B, C, H, W) or (B, T, C, H, W)
    Output: (B, feature_dim) or (B, T, feature_dim)
    """

    BACKBONES = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
    }

    def __init__(
        self,
        backbone: str = config.CNN_BACKBONE,
        feature_dim: int = config.CNN_FEATURE_DIM,
        pretrained: bool = config.CNN_PRETRAINED,
        freeze_backbone: bool = config.FREEZE_CNN,
    ):
        super().__init__()

        if backbone not in self.BACKBONES:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose from {list(self.BACKBONES.keys())}")

        model_fn, weights, native_dim = self.BACKBONES[backbone]

        # Load pretrained model
        base = model_fn(weights=weights if pretrained else None)

        # Remove final FC layer → feature extractor
        self.features = nn.Sequential(*list(base.children())[:-1])  # output: (B, native_dim, 1, 1)
        self.native_dim = native_dim

        # Optional projection to desired feature_dim
        if feature_dim != native_dim:
            self.projection = nn.Sequential(
                nn.Linear(native_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            )
        else:
            self.projection = nn.Identity()

        self.feature_dim = feature_dim

        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze all backbone parameters (no gradient update)."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, num_layers: int = 0):
        """
        Optionally unfreeze the last `num_layers` of the backbone
        for fine-tuning.
        """
        if num_layers == 0:
            for param in self.features.parameters():
                param.requires_grad = True
        else:
            # Unfreeze last N children
            children = list(self.features.children())
            for child in children[-num_layers:]:
                for param in child.parameters():
                    param.requires_grad = True

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a single batch of images.

        Args:
            x: (B, C, H, W)
        Returns:
            features: (B, feature_dim)
        """
        feat = self.features(x)          # (B, native_dim, 1, 1)
        feat = feat.flatten(start_dim=1)  # (B, native_dim)
        feat = self.projection(feat)      # (B, feature_dim)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass supporting both single images and video clips.

        Args:
            x: (B, C, H, W) or (B, T, C, H, W)
        Returns:
            (B, feature_dim) or (B, T, feature_dim)
        """
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)       # merge batch & time
            feat = self.forward_single(x_flat)      # (B*T, feature_dim)
            feat = feat.view(B, T, self.feature_dim)  # (B, T, feature_dim)
            return feat
        else:
            return self.forward_single(x)


class AppearanceEncoder(nn.Module):
    """
    Lightweight encoder for object crops (bounding box regions).
    Used to generate appearance embeddings for graph nodes.

    Input:  (N, C, H, W) — cropped object images
    Output: (N, embed_dim)
    """

    def __init__(self, embed_dim: int = config.GRAPH_NODE_FEATURE_DIM):
        super().__init__()

        # Use a smaller backbone for object crops
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.projection = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.embed_dim = embed_dim

        # Freeze backbone
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) — cropped & resized object images
        Returns:
            (N, embed_dim)
        """
        feat = self.features(x).flatten(start_dim=1)
        return self.projection(feat)


if __name__ == "__main__":
    # Quick test
    encoder = CNNEncoder(backbone="resnet50", feature_dim=2048)
    print(f"CNN Encoder loaded. Feature dim: {encoder.feature_dim}")

    dummy_image = torch.randn(2, 3, 224, 224)
    out = encoder(dummy_image)
    print(f"Single image output: {out.shape}")  # (2, 2048)

    dummy_clip = torch.randn(2, 16, 3, 224, 224)
    out = encoder(dummy_clip)
    print(f"Video clip output:   {out.shape}")  # (2, 16, 2048)
