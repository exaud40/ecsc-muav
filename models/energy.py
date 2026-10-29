# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:47:32 2026

@author: Exaud
"""

"""EnergyNet: pairwise compatibility energy between a teacher embedding z_i and a
projected server feature h_j.Shared encoder, then an MLP on [z~, h~, z~ * h~]; a tanh bounds the score.

Convention: E = -tanh(g), g = interact([z~, h~, z~*h~]), so LOWER energy = MORE compatible, and the loss is
    L_cont = 1/(B(B-1)) sum_i sum_{j!=i} max(0, m + E_ii - E_ij). Negatives are the other
samples of the minibatch (B = batch size). """
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyNet(nn.Module):
    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.interact = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1))

    def forward(self, z, h):
        """z: (B, D) teacher embeddings; h: (B, D) projected features. Returns E: (B, B), E[i, j] = E(z_i, h_j)."""
        ez, eh = self.embed(z), self.embed(h)
        B = ez.size(0)
        ez_ = ez.unsqueeze(1).expand(-1, B, -1)
        eh_ = eh.unsqueeze(0).expand(B, -1, -1)
        feat = torch.cat([ez_, eh_, ez_ * eh_], dim=-1)
        return -torch.tanh(self.interact(feat).squeeze(-1))


def contrastive_energy_loss(E, margin=0.5):
    B = E.size(0)
    pos = torch.diag(E).unsqueeze(1)                       # E_ii as a column
    viol = F.relu(margin + pos - E)                        # max(0, m + E_ii - E_ij)
    off_diag = 1.0 - torch.eye(B, device=E.device)
    return (viol * off_diag).sum() / (B * (B - 1))


@torch.no_grad()
def pairwise_score(net, t, p, chunk=32):
    """Compatibility logit g for every (teacher t_j, student p_i) pair: returns (Nt, Np), E = -tanh(g)."""
    net.eval()
    et, ep = net.embed(t), net.embed(p)                    # (Nt, H), (Np, H)
    out = []
    for s in range(0, ep.size(0), chunk):
        e = ep[s:s + chunk]
        a = et.unsqueeze(1).expand(-1, e.size(0), -1)      # (Nt, n, H) teacher side
        b = e.unsqueeze(0).expand(et.size(0), -1, -1)      # (Nt, n, H) student side
        out.append(net.interact(torch.cat([a, b, a * b], dim=-1)).squeeze(-1))
    return torch.cat(out, dim=1)
