from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class NativeMemoryCore(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.n_embd = int(n_embd)
        self.register_buffer("memory_keys", torch.empty(0, self.n_embd))
        self.register_buffer("memory_values", torch.empty(0, self.n_embd))
        self.register_buffer("memory_targets", torch.empty(0, dtype=torch.long))

    @property
    def size(self) -> int:
        return int(self.memory_keys.shape[0])

    def clear(self):
        device = self.memory_keys.device
        self.memory_keys = torch.empty(0, self.n_embd, device=device)
        self.memory_values = torch.empty(0, self.n_embd, device=device)
        self.memory_targets = torch.empty(0, dtype=torch.long, device=device)

    @torch.no_grad()
    def set_memory(self, keys: torch.Tensor, values: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None):
        if values is None:
            values = keys
        if keys.ndim != 2 or values.ndim != 2:
            raise ValueError("keys and values must have shape [M, D].")
        if keys.shape != values.shape:
            raise ValueError("keys and values must have the same shape.")
        if keys.shape[1] != self.n_embd:
            raise ValueError(f"Expected hidden dim {self.n_embd}, got {keys.shape[1]}.")
        device = self.memory_keys.device
        self.memory_keys = keys.detach().to(device)
        self.memory_values = values.detach().to(device)
        if targets is None:
            targets = torch.full((keys.shape[0],), -1, dtype=torch.long)
        self.memory_targets = targets.detach().long().to(device)

    def retrieve(self, query: torch.Tensor) -> Dict[str, Any]:
        if query.ndim != 2:
            raise ValueError("query must have shape [B, D].")
        B, D = query.shape
        if self.size == 0:
            zeros = torch.zeros(B, D, device=query.device, dtype=query.dtype)
            ids = torch.full((B,), -1, device=query.device, dtype=torch.long)
            score = torch.zeros(B, device=query.device, dtype=query.dtype)
            return {"found": False, "memory_values": zeros, "memory_ids": ids, "memory_targets": ids, "score": score, "context01": score}
        q = F.normalize(query, dim=-1, eps=1e-8)
        k = F.normalize(self.memory_keys.to(query.device, query.dtype), dim=-1, eps=1e-8)
        sims = q @ k.t()
        score, ids = sims.max(dim=-1)
        values = self.memory_values.to(query.device, query.dtype)[ids]
        targets = self.memory_targets.to(query.device)[ids]
        context01 = ((score + 1.0) * 0.5).clamp(0.0, 1.0)
        return {"found": True, "memory_values": values, "memory_ids": ids, "memory_targets": targets, "score": score, "context01": context01}
