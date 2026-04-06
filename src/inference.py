import time
import torch


def synchronize_device(device):
    if device == "cuda":
        torch.cuda.synchronize()
    # MPS does not have a direct synchronize call like CUDA in standard PyTorch user API


def warmup_model(model, dataloader, device="cpu", warmup_batches=5):
    model.eval()
    model.to(device)

    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= warmup_batches:
                break
            inputs = inputs.to(device)
            _ = model(inputs)
            synchronize_device(device)


def run_inference(model, inputs, device="cpu"):
    model.eval()
    model.to(device)
    inputs = inputs.to(device)

    with torch.no_grad():
        if device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            outputs = model(inputs)
            end_event.record()

            torch.cuda.synchronize()
            latency = start_event.elapsed_time(end_event) / 1000.0
        else:
            start = time.perf_counter()
            outputs = model(inputs)
            synchronize_device(device)
            end = time.perf_counter()
            latency = end - start

    return outputs, latency


def run_multiple_inference(model, dataloader, num_batches=20, device="cpu", warmup_batches=5):
    latencies = []
    model.eval()
    model.to(device)

    warmup_model(model, dataloader, device=device, warmup_batches=warmup_batches)

    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_batches:
                break

            inputs = inputs.to(device)

            if device == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                _ = model(inputs)
                end_event.record()

                torch.cuda.synchronize()
                latency = start_event.elapsed_time(end_event) / 1000.0
            else:
                start = time.perf_counter()
                _ = model(inputs)
                synchronize_device(device)
                end = time.perf_counter()
                latency = end - start

            latencies.append(latency)

    return latencies


def evaluate_accuracy(model, dataloader, device="cpu", max_batches=None):
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        return 0.0

    return correct / total