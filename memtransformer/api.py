from pathlib import Path
import torch

from .config import MemTransformerConfig
from .model import MemTransformer


def auto_device(device="auto"):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def create_memtransformer(config: MemTransformerConfig, device="auto") -> MemTransformer:
    """Create an untrained MemTransformer architecture instance."""
    device = auto_device(device)
    return MemTransformer(config).to(device)


def save_architecture_config(config: MemTransformerConfig, path):
    config.save_json(path)


def load_architecture_config(path) -> MemTransformerConfig:
    return MemTransformerConfig.from_json(path)


def load_weights(model: MemTransformer, checkpoint_path, device="auto"):
    """Optional helper for users who have trained their own weights."""
    device = auto_device(device)
    obj = torch.load(Path(checkpoint_path), map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        state = obj["model_state_dict"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    else:
        state = obj
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
