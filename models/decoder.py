# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:46:00 2026

@author: Exaud
"""

"""Edge-server decoder: fuses the K received features and classifies.
Missing UAVs (not yet transmitted) are zero-filled by the caller."""
import torch.nn as nn


class FusionDecoder(nn.Module):
    def __init__(self, K, bottleneck_dim, num_classes, teacher_dim=None):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(K * bottleneck_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.classifier = nn.Linear(128, num_classes)
        # projection into the teacher space; only exists when a teacher is used
        self.proj = nn.Linear(128, teacher_dim) if teacher_dim else None

    def forward(self, y_cat):
        h = self.fusion(y_cat)
        logits = self.classifier(h)
        proj = self.proj(h) if self.proj is not None else None
        return logits, proj
