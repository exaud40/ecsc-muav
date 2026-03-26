import os
import torch
from data.dataloader import get_loader
from models.feature_extractor import CLIPFeatureExtractor_LU
from models.model import MultiUAVSystem
from utils.train import train
from utils.plots import evaluate_snr
from utils.plots import plot_latency_vs_snr, plot_power_vs_bits,plot_snr_for_tx_ratios
from checkpoints.load_model import load_best_model


def main():
    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = "D:/Research/Datasets/UCMerced_LandUse/Images"

    epochs = 100
    bottleneck_dim = 32
    snr_values = [-5, 0, 5, 10, 15, 20, 25]

    loader = get_loader(dataset_path)

    model = MultiUAVSystem(bottleneck_dim).to(device)

    clip_extractor = CLIPFeatureExtractor_LU(device)


    # TRAIN
    train(
        model,
        clip_extractor,
        loader,
        device,
        epochs,
        save_path="checkpoints/best_uav_model_LandUse.pth"
    )
    
    
    # LOAD BEST MODEL
    model = load_best_model(
        model,
        "checkpoints/best_uav_model_LandUse.pth",
        device
    )
    
    
    
    _ = evaluate_snr(model, loader, device, snr_values)
    
    
    print("\nGenerating Plots...\n")
    
    plot_snr_for_tx_ratios(
        model,
        loader,
        device,
        bottleneck_dim,
        filename="results/UCMerced_LandUse_acc_vs_SNR.pdf"
    )
    
    
    plot_power_vs_bits(
        model,
        loader,
        device,
        bottleneck_dim,
        filename="results/UCMerced_LandUse_power_vs_bits.pdf"
    )
    
    
    plot_latency_vs_snr(
    model,
    loader,
    device,
    bottleneck_dim,
    filename="results/UCMerced_LandUse_latency_vs_snr.pdf"
)
    
if __name__ == "__main__":
    main()