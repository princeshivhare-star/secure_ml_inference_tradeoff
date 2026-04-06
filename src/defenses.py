import torch
import time


def fixed_batch_padding(inputs, target_batch_size=32):
    current_batch_size = inputs.size(0)

    if current_batch_size >= target_batch_size:
        return inputs[:target_batch_size]

    pad_size = target_batch_size - current_batch_size
    padding = torch.zeros(
        (pad_size, *inputs.shape[1:]),
        dtype=inputs.dtype,
        device=inputs.device
    )
    padded_inputs = torch.cat([inputs, padding], dim=0)
    return padded_inputs


def dummy_compute(device="cpu", size=256):
    x = torch.randn(size, size, device=device)
    y = torch.randn(size, size, device=device)
    _ = torch.mm(x, y)

    if device == "cuda":
        torch.cuda.synchronize()


def secure_inference_wrapper(model, inputs, device="cpu", use_dummy_compute=True):
    model.eval()
    model.to(device)
    inputs = inputs.to(device)
    inputs = fixed_batch_padding(inputs, target_batch_size=32)

    with torch.no_grad():
        if device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            outputs = model(inputs)

            if use_dummy_compute:
                dummy_compute(device=device, size=128)

            end_event.record()
            torch.cuda.synchronize()
            latency = start_event.elapsed_time(end_event) / 1000.0
        else:
            start = time.perf_counter()
            outputs = model(inputs)

            if use_dummy_compute:
                dummy_compute(device=device, size=128)

            end = time.perf_counter()
            latency = end - start

    return outputs, latency