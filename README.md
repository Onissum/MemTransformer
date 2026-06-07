# MemTransformer

**MemTransformer is not a chatbot.**

MemTransformer is a reference implementation of the **MemTransformer architecture**:
a Transformer variant designed to integrate retrieved memories into the model's internal residual stream.

It is not RAG.  
It is not prompt memory.  
It is not final-logit patching.

It is native memory injection inside the Transformer computation.

## Architecture

```text
MemTransformer =
Transformer backbone
+ NativeMemoryCore
+ Contextual Memory Retrieval
+ MemoryToHidden Projection
+ Residual MemoryGene
+ Safe Memory Policy
```

## Installation

Local editable install:

```bash
pip install -e .
```

## Quickstart

```python
import torch
from memtransformer import MemTransformerConfig, create_memtransformer

config = MemTransformerConfig(
    vocab_size=6000,
    block_size=32,
    n_embd=96,
    n_head=4,
    n_layers=3,
)

model = create_memtransformer(config, device="auto")

x = torch.randint(0, config.vocab_size, (1, 16), device=model.device)

logits, loss, debug = model(
    x,
    memory_gene_mode="safe",
    return_debug=True,
)
```

## Memory bank

A memory bank can be attached as hidden vectors:

```python
memory_keys = torch.randn(100, config.n_embd, device=model.device)
memory_values = torch.randn(100, config.n_embd, device=model.device)
model.set_memory(memory_keys, memory_values)
```

## Modes

```text
off       -> no memory injection
research  -> aggressive memory injection
safe      -> default safe memory policy
```

## Important

This repository defines the architecture.

It does not include a production model.  
It does not include large trained weights.  
A checkpoint is a trained instance of the architecture, not the architecture itself.

## Project status

`v0.1.0-alpha`

Reference implementation of the MemTransformer architecture.
