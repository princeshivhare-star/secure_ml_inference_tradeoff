import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.model import SimpleCNN
from src.leakage_analyzer import collect_classwise_timings
from src.device import get_device


def main():
    device = get_device()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    model = SimpleCNN()
    model.load_state_dict(torch.load("models/mnist_cnn.pth", map_location=device))
    model.to(device)
    model.eval()

    print(f"Collecting class-wise timing data on device: {device}")

    df = collect_classwise_timings(
        model,
        test_loader,
        device=device,
        max_samples=300
    )

    os.makedirs("results/csv", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    df.to_csv("results/csv/classwise_timings.csv", index=False)

    grouped = df.groupby("true_class")["latency_sec"].mean()
    print("\nAverage latency by class:")
    print(grouped)

    plt.figure(figsize=(8, 5))
    grouped.plot(kind="bar")
    plt.title(f"Average Inference Latency by Class ({device.upper()})")
    plt.xlabel("MNIST Class")
    plt.ylabel("Latency (sec)")
    plt.tight_layout()
    plt.savefig("results/plots/classwise_latency.png")
    plt.close()

    print("\nSaved:")
    print("results/csv/classwise_timings.csv")
    print("results/plots/classwise_latency.png")


if __name__ == "__main__":
    main()