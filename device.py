import torch

def get_device():
    if torch.backends.mps.is_available():
        return "mps"   # Apple Silicon GPU
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"