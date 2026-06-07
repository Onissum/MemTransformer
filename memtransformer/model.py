from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MemTransformerConfig
from .memory_core import NativeMemoryCore
from .policy import MemoryGenePolicy


class CausalSelfAttention(nn.Module):
    def __init__(self, config: MemTransformerConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        self.n_head = config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):
    def __init__(self, config: MemTransformerConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: MemTransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MemTransformer(nn.Module):
    """
    Architecture-only reference implementation of MemTransformer.

    Transformer backbone + NativeMemoryCore + MemoryToHidden projection
    + Residual MemoryGene + off/research/safe policy.
    """

    def __init__(self, config: MemTransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.native_memory = NativeMemoryCore(config.n_embd)
        self.memory_to_hidden = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.alpha_gate = nn.Sequential(
            nn.Linear(3 * config.n_embd, config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.n_embd, 1, bias=True),
        )
        self.policy = MemoryGenePolicy(config)
        self.apply(self._init_weights)

    @property
    def device(self):
        return next(self.parameters()).device

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_memory(self, keys: torch.Tensor, values: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None):
        self.native_memory.set_memory(keys, values=values, targets=targets)

    def clear_memory(self):
        self.native_memory.clear()

    def _injection_layer_index(self):
        if self.config.injection_layer == -1:
            return len(self.blocks) - 1
        idx = int(self.config.injection_layer)
        if idx < 0 or idx >= len(self.blocks):
            raise ValueError("Invalid injection_layer index.")
        return idx

    def _last_positions(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        if attention_mask is None:
            return torch.full((input_ids.shape[0],), input_ids.shape[1] - 1, dtype=torch.long, device=input_ids.device)
        return attention_mask.long().sum(dim=1).clamp(min=1) - 1

    def _apply_memory_gene(self, h, input_ids, attention_mask, memory_gene_mode):
        B, T, D = h.shape
        last_pos = self._last_positions(input_ids, attention_mask)
        batch_idx = torch.arange(B, device=h.device)
        query = h[batch_idx, last_pos, :]
        retrieval = self.native_memory.retrieve(query)
        memory_raw = retrieval["memory_values"]
        memory_hidden = self.memory_to_hidden(memory_raw)
        gate_input = torch.cat([query, memory_hidden, query * memory_hidden], dim=-1)
        alpha = torch.sigmoid(self.alpha_gate(gate_input)).squeeze(-1)
        apply_mask, reason = self.policy.decide(
            mode=memory_gene_mode,
            found=retrieval["found"],
            alpha=alpha,
            context01=retrieval["context01"],
        )
        strength = float(self.config.research_strength if memory_gene_mode == "research" else self.config.memory_strength)
        h2 = h.clone()
        if apply_mask.any():
            direction = F.normalize(memory_hidden, dim=-1, eps=1e-8)
            if self.config.scale_by_hidden_norm:
                h_norm = torch.norm(query, dim=-1, keepdim=True).clamp(min=1e-6)
                injection = strength * alpha.unsqueeze(-1) * h_norm * direction
            else:
                injection = strength * alpha.unsqueeze(-1) * direction
            injection = injection * apply_mask.float().unsqueeze(-1)
            h2[batch_idx, last_pos, :] = h2[batch_idx, last_pos, :] + injection
        debug = {
            "memory_gene_mode": memory_gene_mode,
            "memory_gene_reason": reason,
            "memory_gene_applied": apply_mask.detach(),
            "memory_alpha": alpha.detach(),
            "memory_score": retrieval["score"].detach(),
            "memory_context01": retrieval["context01"].detach(),
            "memory_ids": retrieval["memory_ids"].detach(),
            "memory_targets": retrieval["memory_targets"].detach(),
        }
        return h2, debug

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        memory_gene_mode: str = "safe",
        return_debug: bool = False,
    ):
        B, T = input_ids.shape
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.config.block_size}.")
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        h = self.dropout(self.token_embedding(input_ids) + self.position_embedding(pos)[None, :, :])
        inject_idx = self._injection_layer_index()
        gene_debug = None
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i == inject_idx:
                h, gene_debug = self._apply_memory_gene(h, input_ids, attention_mask, memory_gene_mode)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.config.pad_id,
            )
        if return_debug:
            return logits, loss, {"architecture": "MemTransformer", "memory_gene": gene_debug}
        return logits, loss

    @torch.no_grad()
    def predict_next_id(self, input_ids: torch.Tensor, memory_gene_mode: str = "safe"):
        logits, _, debug = self.forward(input_ids, memory_gene_mode=memory_gene_mode, return_debug=True)
        next_id = torch.argmax(logits[:, -1, :], dim=-1)
        return next_id, debug

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 20, memory_gene_mode: str = "safe"):
        self.eval()
        out = input_ids
        for _ in range(max_new_tokens):
            idx_cond = out[:, -self.config.block_size:]
            logits, _ = self.forward(idx_cond, memory_gene_mode=memory_gene_mode, return_debug=False)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            out = torch.cat([out, next_id], dim=1)
        return out
