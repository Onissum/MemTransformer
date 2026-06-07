import torch

class MemoryGenePolicy:
    VALID_MODES = {"off", "research", "safe"}

    def __init__(self, config):
        self.config = config

    def decide(self, mode: str, found: bool, alpha: torch.Tensor, context01: torch.Tensor):
        mode = str(mode).lower().strip()
        if mode not in self.VALID_MODES:
            raise ValueError("Invalid memory_gene_mode. Use off, research, or safe.")
        if mode == "off":
            return torch.zeros_like(alpha, dtype=torch.bool), "mode_off"
        if not found:
            return torch.zeros_like(alpha, dtype=torch.bool), "no_memory"
        if mode == "research":
            return torch.ones_like(alpha, dtype=torch.bool), "research_apply"
        apply = (alpha >= float(self.config.safe_min_alpha)) & (context01 >= float(self.config.safe_min_score))
        return apply, "safe_policy"
