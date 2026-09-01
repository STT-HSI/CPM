import torch


def resolve_torch_device(gpu_id) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(gpu_id)}")
    return torch.device("cpu")
