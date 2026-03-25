# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 05:05:39 2026

@author: Exaud
"""

import torch.nn as nn

from models.layers import UAVEncoder, EdgeServer



class MultiUAVSystem(nn.Module):

    def __init__(self, bottleneck_dim=64, num_classes=21):

        super().__init__()

        self.encoder1 = UAVEncoder(bottleneck_dim)
        self.encoder2 = UAVEncoder(bottleneck_dim)

        self.edge = EdgeServer(bottleneck_dim, num_classes)


    def split_views(self,x):

        w = x.size(3)

        v1 = x[:,:,:,:w//2]
        v2 = x[:,:,:,w//2:]

        return v1,v2

    def forward(self,x):

        v1,v2 = self.split_views(x)

        f1 = self.encoder1(v1)
        f2 = self.encoder2(v2)

        logits,fused_clip = self.edge(f1,f2)

        return logits,fused_clip,[f1,f2]




