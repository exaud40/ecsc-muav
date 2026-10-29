# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:46:41 2026

@author: Exaud
"""

import torch.nn as nn


class UAVEncoder(nn.Module):
    def __init__(self, bottleneck_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 128, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, bottleneck_dim)

    def forward(self, v):
        return self.fc(self.conv(v).flatten(1))
