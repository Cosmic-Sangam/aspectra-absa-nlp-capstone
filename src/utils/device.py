"""Device selection utility — CPU / CUDA."""
import torch


def get_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("Using CPU")
    return dev
