import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_accuracy_vs_snr(model, loader, device, bottleneck_dim, snr_range):

    accuracies = []

    model.eval()

    for snr in snr_range:

        model.edge.channel.set_snr(snr)

        correct = 0
        total = 0

        with torch.no_grad():

            for imgs, labels in loader:

                imgs = imgs.to(device)
                labels = labels.to(device)

                logits, _, _ = model(imgs)

                preds = torch.argmax(logits, dim=1)

                correct += (preds == labels).sum().item()

                total += labels.size(0)

        acc = 100 * correct / total
        accuracies.append(acc)

        print(f"SNR {snr} dB → Accuracy {acc:.2f}%")

    plt.figure(figsize=(8,6))

    plt.plot(snr_range, accuracies, 'bo-', linewidth=2)

    plt.xlabel("SNR (dB)", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.title("UAV System Accuracy vs SNR")

    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_accuracy_vs_bits(model, loader, device, bottleneck_dim):

    tx_ratios = [0.25,0.5,0.75,1.0]

    bits = []

    # feature size estimate
    feature_dim = bottleneck_dim * 2  # 2 UAVs

    for r in tx_ratios:
        bits.append(int(feature_dim * r))

    accuracies = []

    model.eval()

    for ratio in tx_ratios:

        model.edge.channel.set_tx_ratio(ratio)

        correct = 0
        total = 0

        with torch.no_grad():

            for imgs, labels in loader:

                imgs = imgs.to(device)
                labels = labels.to(device)

                logits, _, _ = model(imgs)

                preds = torch.argmax(logits, dim=1)

                correct += (preds == labels).sum().item()

                total += labels.size(0)

        acc = 100 * correct / total
        accuracies.append(acc)

        print(f"Bits {ratio*feature_dim:.0f} → Accuracy {acc:.2f}%")

    plt.figure(figsize=(8,6))

    plt.plot(bits, accuracies, 'ro-', linewidth=2)

    plt.xlabel("Transmitted Bits", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.title("Accuracy vs Transmission Bits")

    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_snr_for_tx_ratios(model, loader, device, bottleneck_dim, filename):

    plt.rcParams['font.family'] = 'Times New Roman'

    snr_range = np.arange(0, 21, 5)
    tx_ratios = [0.25, 0.5, 0.75, 1.0]

    # Different markers and colors
    markers = ['o', 's', '^', 'D']
    colors = ['b', 'g', 'r', 'm']

    plt.figure(figsize=(8,6))

    for i, ratio in enumerate(tx_ratios):

        model.edge.channel.set_tx_ratio(ratio)

        accuracies = []

        for snr in snr_range:

            model.edge.channel.set_snr(snr)

            correct = 0
            total = 0

            with torch.no_grad():

                for imgs, labels in loader:

                    imgs = imgs.to(device)
                    labels = labels.to(device)

                    logits,_,_ = model(imgs)

                    preds = torch.argmax(logits, dim=1)

                    correct += (preds==labels).sum().item()
                    total += labels.size(0)

            acc = 100 * correct / total
            accuracies.append(acc)

            print(f"Ratio {ratio:.2f} | SNR {snr} → {acc:.2f}%")

        bits = int(ratio * bottleneck_dim * 2)

        plt.plot(
            snr_range,
            accuracies,
            marker=markers[i],
            color=colors[i],
            linestyle='--',
            linewidth=2.5,
            markersize=8,
            label=f'bits={bits}'
        )

    plt.xlabel("SNR (dB)", fontsize=26)
    plt.ylabel("Accuracy (%)", fontsize=26)

    plt.xticks(snr_range, fontsize=20)
    plt.yticks(fontsize=20)

    #plt.title("Accuracy vs SNR for Different Bit Budgets")

    plt.legend(
        loc='lower right',
        fontsize=18,
        frameon=True
    )

    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()

   
def plot_power_vs_bits(model, loader, device, bottleneck_dim, filename):

    plt.rcParams['font.family'] = 'Times New Roman'

    tx_ratios = [0.25, 0.5, 0.75, 1.0]

    bits_list = []
    tx_power_list = []
    comp_power_list = []

    # -----------------------------
    # Communication parameters
    # -----------------------------
    bandwidth = 2e5        # 200 kHz
    noise_power = 1e-8     # W
    channel_gain = 1e-4
    tx_time = 2e-3         # 2 ms

    # -----------------------------
    # Computation parameters
    # -----------------------------
    mac_energy = 5e-12     # 5 pJ per MAC
    compute_time = 1e-2    # 10 ms inference

    for ratio in tx_ratios:

        model.edge.channel.set_tx_ratio(ratio)

        # transmitted bits
        bits = int(ratio * bottleneck_dim * 64)
        bits_list.append(bits/32)

        # -----------------------------
        # Communication power
        # -----------------------------
        rate = bits / tx_time

        snr = 2 ** (rate / bandwidth) - 1

        tx_power = (snr * noise_power) / channel_gain
        tx_power_mw = tx_power * 100000

        tx_power_list.append(tx_power_mw)

        # -----------------------------
        # Computation power
        # -----------------------------
        semantic_dim = ratio * bottleneck_dim

        macs = semantic_dim ** 2

        energy_comp = macs * mac_energy

        comp_power_mw = (energy_comp / compute_time) * 100000000

        comp_power_list.append(comp_power_mw)

        print(f"Bits {bits} | TX {tx_power_mw:.2f} mW | Comp {comp_power_mw:.2f} mW")

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax1 = plt.subplots(figsize=(8,6))
    ax2 = ax1.twinx()

    ax1.plot(bits_list, tx_power_list,
             marker='o', linewidth=2.5,
             label='Transmit Power')

    ax2.plot(bits_list, comp_power_list,
             marker='s', linestyle='--', linewidth=2.5,
             label='Computation Power')

    ax1.set_xlabel("Transmitted Bits", fontsize=24)
    ax1.set_ylabel("Average Transmit Power (mW)", fontsize=22)
    ax2.set_ylabel("Average Computation Power (mW)", fontsize=22)

    ax1.tick_params(axis='both', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='lower right', fontsize=18)

    ax1.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_latency_vs_psnr(model, loader, device, bottleneck_dim, filename):

    plt.rcParams['font.family'] = 'Times New Roman'

    # PSNR range (dynamic channel)
    snr_range = np.arange(0, 21, 5)

    # Communication parameters
    bandwidth = 2e5        # 200 kHz UAV link
    bits = bottleneck_dim * 2 * 32   # two UAV features and 32 FP Quantization

    latencies = []
    error_rates = []

    model.eval()

    for snr in snr_range:

        # update channel condition
        model.edge.channel.set_snr(snr)

        correct = 0
        total = 0

        with torch.no_grad():

            for imgs, labels in loader:

                imgs = imgs.to(device)
                labels = labels.to(device)

                logits, _, _ = model(imgs)

                preds = torch.argmax(logits, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        error_rate = (1 - accuracy) * 100
        error_rates.append(error_rate)

        # ----------------------------
        # Communication latency
        # ----------------------------
        snr_linear = 10 ** (snr / 10)

        capacity = bandwidth * np.log2(1 + snr_linear)

        latency = bits / capacity
        latency_ms = latency * 1000

        latencies.append(latency_ms)

        print(f"PSNR {snr} dB | Latency {latency_ms:.3f} ms | Error {error_rate:.2f}%")

    # ----------------------------
    # Plot (Dual Axis)
    # ----------------------------
    fig, ax1 = plt.subplots(figsize=(8,6))
    ax2 = ax1.twinx()

    ax1.plot(snr_range, latencies,
             marker='o', linewidth=2.5,
             color='b', label='Latency')

    ax2.plot(snr_range, error_rates,
             marker='s', linestyle='--', linewidth=2.5,
             color='r', label='Error Rate')

    ax1.set_xlabel("SNR (dB)", fontsize=24)

    ax1.set_ylabel("Latency (ms)", fontsize=22)
    ax2.set_ylabel("Error Rate (%)", fontsize=22)

    ax1.tick_params(axis='both', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper right', fontsize=16)

    ax1.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_snr_for_tx_ratios_rs(model, loader, device):

    snr_range = [0,5,10,15,20]
    ratios = [0.25,0.5,0.75,1.0]

    plt.figure(figsize=(8,6))

    for r in ratios:

        model.edge.channel.set_tx_ratio(r)
        accs=[]

        for snr in snr_range:

            model.edge.channel.set_snr(snr)

            cor=0; tot=0
            with torch.no_grad():
                for x,y in loader:
                    x,y = x.to(device), y.to(device)
                    logits, _, _ = model(x)
                    cor += (logits.argmax(1)==y).sum().item()
                    tot += y.size(0)

            accs.append(100*cor/tot)

        bits = int(r*32*2)

        plt.plot(snr_range,accs,'o--',label=f'{bits} bits')

    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid()
    plt.savefig("results/acc_vs_snr.pdf", dpi=300, bbox_inches='tight')
    plt.show()


def plot_power_vs_bits_rs():

    bits = [10,12,14,16,18]

    tx=[]; comp=[]

    for b in bits:

        rate = b / 0.002
        snr = 2**(rate/2e5)-1

        tx.append((snr*1e-8/1e-4)*1e5)

        mac = b**2
        comp.append((mac*5e-12/1e-2)*1e8)

    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(bits,tx,'o-',label='TX Power')
    ax2.plot(bits,comp,'s--',label='Comp Power')

    ax1.set_ylabel("Transmit Power (mW)")
    ax2.set_ylabel("Computation Power (mW)")

    plt.savefig("results/RSSCN7_power.pdf", dpi=300, bbox_inches='tight')
    plt.show()


def plot_latency_error_rs(model, loader, device):

    snr=[0,5,10,15,20]
    lat=[]; err=[]

    for s in snr:

        model.edge.channel.set_snr(s)

        cor=0; tot=0
        with torch.no_grad():
            for x,y in loader:
                x,y = x.to(device), y.to(device)
                logits,_,_ = model(x)
                cor += (logits.argmax(1)==y).sum().item()
                tot += y.size(0)

        acc = cor/tot
        err.append((1-acc)*100)

        cap = 2e5*np.log2(1+10**(s/10))
        lat.append((32*2*32)/cap*1000)

    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(snr,lat,'o-',label='Latency')
    ax2.plot(snr,err,'s--',label='Error')

    ax1.set_ylabel("Latency (ms)")
    ax2.set_ylabel("Error (%)")

    plt.savefig("results/RSSCN7_latency.pdf", dpi=300, bbox_inches='tight')
    plt.show()






def evaluate(model, loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:

            imgs = imgs.to(device)
            labels = labels.to(device)

            logits, _, _ = model(imgs)

            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")


def evaluate_bits(model, loader, device):

    tx_ratios = [0.25,0.5,0.75,1.0]

    model.eval()

    for ratio in tx_ratios:

        model.edge.channel.set_tx_ratio(ratio)

        correct = 0
        total = 0

        with torch.no_grad():

            for imgs, labels in loader:

                imgs = imgs.to(device)
                labels = labels.to(device)

                logits,_,_ = model(imgs)

                preds = torch.argmax(logits,dim=1)

                correct += (preds==labels).sum().item()

                total += labels.size(0)

        acc = correct/total

        print(f"TX ratio {ratio:.2f} → Accuracy {acc:.4f}")


def evaluate_snr(model, loader, device, snr_list):

    model.eval()

    results = {}

    for snr in snr_list:

        # update SNR inside edge channel
        model.edge.channel.set_snr(snr)

        correct = 0
        total = 0

        with torch.no_grad():

            for imgs, labels in loader:

                imgs = imgs.to(device)
                labels = labels.to(device)

                logits, _, _ = model(imgs)

                preds = torch.argmax(logits, dim=1)

                correct += (preds == labels).sum().item()

                total += labels.size(0)

        acc = correct / total

        results[snr] = acc

        print(f"SNR {snr} dB → Accuracy {acc:.4f}")

    return results