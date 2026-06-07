# Final Release v0.1.0-alpha

This release freezes the architecture API of MemTransformer as a MemTransformer reference implementation.

It defines:

- MemTransformerConfig
- MemTransformer
- NativeMemoryCore
- Residual MemoryGene
- Safe / Research / Off modes

## What this release is not

It is not a chatbot.  
It is not a production LLM.  
It is not a trained model release.

## Main idea

A MemTransformer is a Transformer variant where memory enters the internal residual stream.

```text
memory_to_hidden -> residual stream -> lm_head
```

The goal is not to compete with large LLMs.  
The goal is to define an architecture that can be used to build models with native memory.
