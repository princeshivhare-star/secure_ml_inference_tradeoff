import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.model import SimpleCNN
from src.inference import run_multiple_inference, evaluate_accuracy
from src.profiler import summarize_latencies, save_results_to_csv, get_cpu_memory_usage_mb
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

    print(f"Running baseline inference on device: {device}")

    latencies = run_multiple_inference(
        model,
        test_loader,
        num_batches=50,
        device=device
    )

    summary = summarize_latencies(latencies)

    accuracy = evaluate_accuracy(
        model,
        test_loader,
        device=device,
        max_batches=50
    )

    memory_mb = get_cpu_memory_usage_mb()

    results = [{
        "mode": "baseline",
        "device": device,
        "accuracy": accuracy,
        "memory_mb": memory_mb,
        **summary
    }]

    print("\n=== Baseline Summary ===")
    for k, v in results[0].items():
        print(f"{k}: {v}")

    os.makedirs("results/csv", exist_ok=True)
    save_results_to_csv(results, "results/csv/baseline_results.csv")


if __name__ == "__main__":
    main()