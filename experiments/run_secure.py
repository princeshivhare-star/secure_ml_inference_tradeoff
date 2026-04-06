import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.model import SimpleCNN
from src.profiler import summarize_latencies, save_results_to_csv, get_cpu_memory_usage_mb
from src.defenses import secure_inference_wrapper
from src.inference import evaluate_accuracy
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
        batch_size=16,
        shuffle=False
    )

    model = SimpleCNN()
    model.load_state_dict(torch.load("models/mnist_cnn.pth", map_location=device))
    model.to(device)
    model.eval()

    print(f"Running secure inference on device: {device}")

    latencies = []

    for i, (inputs, labels) in enumerate(test_loader):
        if i >= 50:
            break

        _, latency = secure_inference_wrapper(
            model,
            inputs,
            device=device,
            use_dummy_compute=True
        )
        latencies.append(latency)

    summary = summarize_latencies(latencies)

    accuracy = evaluate_accuracy(
        model,
        test_loader,
        device=device,
        max_batches=50
    )

    memory_mb = get_cpu_memory_usage_mb()

    results = [{
        "mode": "secure",
        "device": device,
        "accuracy": accuracy,
        "memory_mb": memory_mb,
        **summary
    }]

    print("\n=== Secure Summary ===")
    for k, v in results[0].items():
        print(f"{k}: {v}")

    os.makedirs("results/csv", exist_ok=True)
    save_results_to_csv(results, "results/csv/secure_results.csv")


if __name__ == "__main__":
    main()