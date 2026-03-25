# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 05:55:28 2026

@author: Exaud
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.layers import EnergyNet
from losses.energy import CompressedFeatureLoss, bitrate_loss, contrastive_energy_loss


def train(model, clip_extractor, loader, device, epochs=10, save_path="best_model.pth"):

    optimizer = optim.Adam(model.parameters(),
                           lr=1e-4,
                           weight_decay=1e-4)

    energy_net = EnergyNet(512).to(device)
    optimizer_energy = optim.Adam(energy_net.parameters(), lr=1e-4)

    ce_loss = nn.CrossEntropyLoss()
    compress_loss = CompressedFeatureLoss()

    best_acc = 0

    model.train()

    for epoch in range(epochs):

        total_loss = 0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(loader)

        for imgs, labels in pbar:

            imgs = imgs.to(device)
            labels = labels.to(device)

            # Progressive SNR
            min_snr = max(0, 10 - epoch//5)
            max_snr = min(30, 10 + epoch//3)

            snr = min_snr + (max_snr-min_snr)*torch.rand(1).item()
            model.edge.channel.set_snr(snr)

            logits, fused_clip, compressed = model(imgs)
            
            # Split features
            f1, f2 = compressed

            # ---- τ = 1 (partial info) ----
            logits_tau1, _ = model.edge(f1, torch.zeros_like(f2))
            
            # ---- τ = 2 (full info) ----
            logits_tau2 = logits
            
            loss_tau1 = ce_loss(logits_tau1, labels)
            loss_tau2 = ce_loss(logits_tau2, labels)
            
            loss_task = loss_tau1 + loss_tau2
            
            loss_rate = bitrate_loss(compressed)
            
            #loss_cls = ce_loss(logits, labels)
            loss_compress = compress_loss(compressed)

            with torch.no_grad():
                clip_feat = clip_extractor(imgs)

            #energy = energy_net(clip_feat, fused_clip)
            #loss_energy = energy.mean()
            
            energy_matrix = energy_net(clip_feat, fused_clip)
            loss_energy = contrastive_energy_loss(energy_matrix)

            #loss = loss_cls + 0.1 * loss_energy + 0.01 * loss_compress
            
            lambda1 = 0.1
            lambda2 = 0.01
            eta = 0.01
            
            loss = (
                loss_task                     # Στ
                + eta * loss_rate             # Σk Σt
                + lambda1 * loss_energy       # contrastive
                + lambda2 * loss_compress     # structural sparsity
            )

            optimizer.zero_grad()
            optimizer_energy.zero_grad()

            loss.backward()

            optimizer.step()
            optimizer_energy.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            acc = total_correct / total_samples

            pbar.set_description(
                f"Epoch {epoch+1} Loss {total_loss/(pbar.n+1):.4f} Acc {acc:.4f}"
            )

        epoch_loss = total_loss / len(loader)
        epoch_acc = total_correct / total_samples

        print(f"Epoch {epoch+1} Loss {epoch_loss:.4f} Accuracy {epoch_acc:.4f}")

        # ---------------------------------
        # SAVE BEST MODEL
        # ---------------------------------
        if epoch_acc > best_acc:

            best_acc = epoch_acc

            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': best_acc,
                'epoch': epoch
            }, save_path)

            print(f" Best model saved (Acc={best_acc:.4f})")
            
            
"""        
def train(model, clip_extractor, loader, device, epochs=100, save_path="checkpoints/best_model.pth"):

    optimizer = optim.Adam(model.parameters(),
                           lr=1e-4,
                           weight_decay=1e-4)

    energy_net = EnergyNet(512).to(device)
    optimizer_energy = optim.Adam(energy_net.parameters(), lr=1e-4)

    ce_loss = nn.CrossEntropyLoss()
    compress_loss = CompressedFeatureLoss()

    best_acc = 0
    model.train()

    for epoch in range(epochs):

        total_loss = 0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(loader)

        for imgs, labels in pbar:

            imgs = imgs.to(device)
            labels = labels.to(device)

            # ---------------------------------
            #  Progressive SNR (curriculum learning)
            # ---------------------------------
            min_snr = max(0, 10 - epoch // 5)
            max_snr = min(30, 10 + epoch // 3)

            snr = min_snr + (max_snr - min_snr) * torch.rand(1).item()
            model.edge.channel.set_snr(snr)

            # ---------------------------------
            #  Forward pass
            # ---------------------------------
            logits, fused_clip, compressed = model(imgs)

            f1, f2 = compressed

            # ---------------------------------
            # τ-based inference (key contribution)
            # ---------------------------------
            # τ = 1 (partial transmission)
            logits_tau1, _ = model.edge(f1, torch.zeros_like(f2))

            # τ = 2 (full transmission)
            logits_tau2 = logits

            loss_tau1 = ce_loss(logits_tau1, labels)
            loss_tau2 = ce_loss(logits_tau2, labels)

            loss_task = loss_tau1 + loss_tau2

            # ---------------------------------
            # Rate constraint (bit minimization)
            # ---------------------------------
            loss_rate = bitrate_loss(compressed)

            # ---------------------------------
            # 🧩 Structural compression
            # ---------------------------------
            loss_compress = compress_loss(compressed)

            # ---------------------------------
            #  Semantic alignment (contrastive)
            # ---------------------------------
            with torch.no_grad():
                clip_feat = clip_extractor(imgs)

            energy_matrix = energy_net(clip_feat, fused_clip)
            loss_energy = contrastive_energy_loss(energy_matrix)

            # ---------------------------------
            # Final Loss (FULL OBJECTIVE)
            # ---------------------------------
            lambda1 = 0.1   # semantic alignment
            lambda2 = 0.01  # compression
            eta = 0.01      # rate penalty

            loss = (
                loss_task
                + eta * loss_rate
                + lambda1 * loss_energy
                + lambda2 * loss_compress
            )

            # ---------------------------------
            #  Backprop
            # ---------------------------------
            optimizer.zero_grad()
            optimizer_energy.zero_grad()

            loss.backward()

            optimizer.step()
            optimizer_energy.step()

            # ---------------------------------
            #  Metrics
            # ---------------------------------
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            acc = total_correct / total_samples

            pbar.set_description(
                f"Epoch {epoch+1} Loss {total_loss/(pbar.n+1):.4f} Acc {acc:.4f}"
            )

        epoch_loss = total_loss / len(loader)
        epoch_acc = total_correct / total_samples

        print(f"Epoch {epoch+1} Loss {epoch_loss:.4f} Accuracy {epoch_acc:.4f}")

        # ---------------------------------
        #  Save Best Model
        # ---------------------------------
        if epoch_acc > best_acc:

            best_acc = epoch_acc

            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': best_acc,
                'epoch': epoch
            }, save_path)

            print(f"Best model saved (Acc={best_acc:.4f})")
"""