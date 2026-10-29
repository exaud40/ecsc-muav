# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:52:26 2026

@author: Exaud
"""

"""Emulation of K UAV observations from one single-view image.

  patch   : the image is cut into K non-overlapping vertical strips, one per UAV.
            Views carry DIFFERENT content; no redundancy between them. Strips get
            narrow as K grows, so a K-sweep with this mode is confounded by
            information per view.
  overlap : each UAV sees a random square crop (side 50-80 % of the image),
            resized to out_size x out_size, so views share content as partially
            overlapping footprints do.
"""
import torch
from torchvision.ops import roi_align


def make_views(x, K, mode, generator=None, out_size=128, scale=(0.5, 0.8)):
    B, _, H, W = x.shape
    if mode == "patch":
        edges = [round(k * W / K) for k in range(K + 1)]
        return [x[..., edges[k]:edges[k + 1]] for k in range(K)]
    if mode == "overlap":
        views = []
        idx = torch.arange(B, device=x.device, dtype=x.dtype)
        for _ in range(K):
            r = lambda: torch.rand(B, generator=generator, device=x.device, dtype=x.dtype)
            side = (scale[0] + (scale[1] - scale[0]) * r()) * min(H, W)
            x1 = r() * (W - side)
            y1 = r() * (H - side)
            rois = torch.stack([idx, x1, y1, x1 + side, y1 + side], dim=1)
            views.append(roi_align(x, rois, (out_size, out_size), spatial_scale=1.0,
                                   sampling_ratio=2, aligned=True))
        return views
    raise ValueError(f"unknown view_mode '{mode}'")
