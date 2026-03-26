import os
import torch

from models.feature_extractor import CLIPFeatureExtractor_RS
from data.dataloader import get_loader_rs
from models.model import MultiUAVSystem
from utils.train import train
from utils.plots import evaluate
from utils.plots import plot_snr_for_tx_ratios,plot_power_vs_bits, plot_latency_vs_snr
from checkpoints.load_model import load_best_model



def main():
    
    epochs = 100
    bottleneck_dim = 32

    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_loader_rs("D:/Research/Datasets/RSSCN7")

    model = MultiUAVSystem().to(device)
    clip = CLIPFeatureExtractor_RS(device)
    
    
    train(
        model, 
        clip, 
        train_loader, 
        device, 
        epochs,
        save_path="checkpoints/best_uav_model_RSSCN7.pth"
        )
    

    model=load_best_model(
        model,
        "checkpoints/best_uav_model_RSSCN7.pth",
        device
    )
    

    evaluate(model, test_loader, device)

    print("\nGenerating Plots...\n")
    
    
    plot_snr_for_tx_ratios(
        model, 
        test_loader,
        device,
        bottleneck_dim,
        filename="results/RSSCN7_acc_vs_SNR.pdf"
        )
    
    
    plot_power_vs_bits(
        model,
        test_loader,
        device,
        bottleneck_dim,
        filename="results/RSSCN7_power_vs_bits.pdf"
    )
    
    plot_latency_vs_snr(
    model,
    test_loader,
    device,
    bottleneck_dim,
    filename="results/RSSCN7_latency_vs_snr.pdf"
    )
    

if __name__ == "__main__":
    main()