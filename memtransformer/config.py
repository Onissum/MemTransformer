from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class MemTransformerConfig:
    vocab_size: int
    block_size: int = 128
    n_embd: int = 256
    n_head: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    bias: bool = True
    pad_id: int = 0

    memory_strength: float = 1.0
    research_strength: float = 1.0
    injection_layer: int = -1

    safe_min_alpha: float = 0.15
    safe_min_score: float = 0.50
    safe_max_logit_gap: float = 8.0
    scale_by_hidden_norm: bool = True

    def to_dict(self):
        return asdict(self)

    def save_json(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
