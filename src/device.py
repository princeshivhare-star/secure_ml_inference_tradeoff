import torch

def get_device():
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using Apple Silicon MPS")
        return "mps"
    else:
        print("Using CPU")
        return "cpu"