import os
import torch

from models.feature_extractor import CLIPFeatureExtractor_RS
from data.dataloader import get_loader_rs
from models.model import MultiUAVSystem
from utils.train import train
from utils.plots import evaluate
from utils.plots import plot_snr_for_tx_ratios,plot_power_vs_bits_rs, plot_latency_error_rs




def main():
    
    epochs = 200

    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_loader_rs("D:/Research/Datasets/RSSCN7")

    model = MultiUAVSystem().to(device)
    clip = CLIPFeatureExtractor_RS(device)

    train(model, clip, train_loader, device, epochs)
    evaluate(model, test_loader, device)

    print("\nGenerating Plots...\n")

    plot_snr_for_tx_ratios(model, test_loader, device)
    plot_power_vs_bits_rs()
    plot_latency_error_rs(model, test_loader, device)


if __name__ == "__main__":
    main()