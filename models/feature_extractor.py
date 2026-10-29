# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:50:47 2026

@author: Exaud
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPTeacher(nn.Module):
    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, name="openai/clip-vit-base-patch32", out_dim=512):
        super().__init__()
        self.stub = (name == "stub")
        if self.stub:   # tiny random network: smoke tests only, NEVER for reported results
            g = torch.Generator().manual_seed(0)
            self.net = nn.Sequential(nn.Conv2d(3, 8, 7, stride=8), nn.ReLU(),
                                     nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, out_dim))
            with torch.no_grad():
                for p in self.net.parameters():
                    p.copy_(torch.randn(p.shape, generator=g) * 0.1)
        else:
            from transformers import CLIPModel
            self.model = CLIPModel.from_pretrained(name, use_safetensors=True)
            assert self.model.config.projection_dim == out_dim, "teacher_dim must match CLIP projection_dim"
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1))

    def train(self, mode=True):      # always stay in eval mode
        return super().train(False)

    @torch.no_grad()
    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std                      # CLIP's own normalisation, x in [0,1]
        if self.stub:
            f = self.net(x)
        else:
            out = self.model.vision_model(pixel_values=x)
            f = self.model.visual_projection(out.pooler_output)
        return F.normalize(f, dim=-1)
