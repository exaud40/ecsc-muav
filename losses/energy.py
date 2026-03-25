import torch
import torch.nn as nn
import torch.nn.functional as F



def contrastive_energy_loss(energy_matrix, margin=0.5):
    """
    energy_matrix: (B, B)
    """
    B = energy_matrix.size(0)

    pos = torch.diag(energy_matrix)  # E_ii

    loss = 0
    for i in range(B):
        neg = energy_matrix[i]  # E_ij
        mask = torch.ones(B, device=energy_matrix.device)
        mask[i] = 0

        loss_i = F.relu(margin + neg - pos[i]) * mask
        loss += loss_i.sum()

    return loss / (B * (B - 1))


def bitrate_loss(features):
    loss = 0
    K = len(features)

    for k in range(K):
        f = features[k]  # (B, D)

        # treat each dimension as packet t
        loss += torch.mean(torch.abs(f))   # L1 proxy

    return loss / K


# Compression Loss
class CompressedFeatureLoss(nn.Module):

    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta

    def compute_entropy(self, x):

        x = (x - x.min()) / (x.max() - x.min() + 1e-8)

        hist = torch.histc(x, bins=256, min=0, max=1)

        p = hist / hist.sum()

        entropy = -torch.sum(p * torch.log2(p + 1e-8))

        return entropy

    def forward(self, features):

        sparsity = 0
        entropy = 0

        for f in features:

            sparsity += torch.mean(torch.abs(f))
            entropy += self.compute_entropy(f)

        sparsity /= len(features)
        entropy /= len(features)

        return sparsity + self.beta * entropy
    


