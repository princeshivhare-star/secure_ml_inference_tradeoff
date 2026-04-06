import time
import torch
import pandas as pd


def synchronize_device(device):
    if device == "cuda":
        torch.cuda.synchronize()


def collect_classwise_timings(model, dataloader, device="cpu", max_samples=200, warmup_samples=20):
    model.eval()
    model.to(device)

    results = []
    count = 0
    warm_count = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            for i in range(inputs.size(0)):
                x = inputs[i:i+1].to(device)
                label = labels[i].item()

                if warm_count < warmup_samples:
                    _ = model(x)
                    synchronize_device(device)
                    warm_count += 1
                    continue

                if count >= max_samples:
                    return pd.DataFrame(results)

                start = time.perf_counter()
                _ = model(x)
                synchronize_device(device)
                end = time.perf_counter()

                results.append({
                    "true_class": label,
                    "latency_sec": end - start
                })
                count += 1

    return pd.DataFrame(results)