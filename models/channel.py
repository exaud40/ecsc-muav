# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:43:42 2026

@author: Exaud
"""

"""Wireless channel.

Conventions:
  * Transmitted features have unit average power PER REAL ELEMENT
    (power_normalize), so E||w||^2 = d'.
  * psi = 10^(SNR/10); noise variance per real element is 1/psi, so the SNR is
    defined per real element.
  * Rayleigh / Rician: block fading, one complex coefficient per UAV per
    transmission, real features paired into d'/2 complex symbols, perfect CSI
    at the server with an MMSE equaliser.
  * Transmission ratio rho: a random subset of symbols is sent (positions known
    to the receiver through a shared pseudo-random seed). Untransmitted symbols
    receive no noise and are zero at the receiver.
"""
import math
import torch
import torch.nn as nn


class Channel(nn.Module):
    def __init__(self, rician_k_lin: float = 10.0):
        super().__init__()
        self.rician_k = rician_k_lin

    @staticmethod
    def power_normalize(x):
        return x / (x.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-8)

    def forward(self, x, snr_db, ratio=1.0, chan="awgn"):
        B, D = x.shape
        assert D % 2 == 0, "feature dimension must be even"
        psi = 10.0 ** (float(snr_db) / 10.0)

        if chan == "awgn":
            y = x + torch.randn_like(x) / math.sqrt(psi)
        elif chan in ("rayleigh", "rician"):
            s = torch.complex(x[:, :D // 2], x[:, D // 2:])
            n = torch.complex(torch.randn_like(s.real), torch.randn_like(s.real)) / math.sqrt(psi)
            h = torch.complex(torch.randn(B, 1, device=x.device),
                              torch.randn(B, 1, device=x.device)) / math.sqrt(2.0)   # CN(0,1)
            if chan == "rician":
                k = self.rician_k
                h = math.sqrt(k / (k + 1.0)) + h / math.sqrt(k + 1.0)               # E|h|^2 = 1
            y = h * s + n
            y = y * h.conj() / (h.abs() ** 2 + 1.0 / psi)                           # MMSE, perfect CSI
            y = torch.cat([y.real, y.imag], dim=1)
        else:
            raise ValueError(f"unknown channel '{chan}'")

        if ratio < 1.0:
            m = (torch.rand(B, D // 2, device=x.device) < ratio).float()
            y = y * torch.cat([m, m], dim=1)
        return y
