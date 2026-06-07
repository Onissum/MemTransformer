# MemTransformer Architecture

MiniMem defines a MemTransformer architecture.

It is not a chatbot.  
It is not a production LLM.  
It is a reference implementation of a Transformer variant with native memory injection.

## Definition

MemTransformer =

- Transformer backbone
- NativeMemoryCore
- Contextual memory retrieval
- MemoryToHidden projection
- Residual MemoryGene
- Safe / Research / Off policy

## Core principle

The memory is not appended to the prompt.  
The memory is not patched into final logits.  
The memory is projected into hidden space and injected into the residual stream.

```text
retrieved memory
-> memory_to_hidden
-> residual stream
-> normal Transformer output path
```

## Modes

### off

No memory injection.

### research

Memory is injected aggressively when available.

### safe

Memory is injected only if retrieval confidence and MemoryGene alpha pass the default thresholds.

## Status

v0.1.0-alpha is architecture-only.

No trained checkpoint is included.
