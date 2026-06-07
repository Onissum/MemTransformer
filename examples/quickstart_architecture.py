import torch
from memtransformer import MemTransformerConfig, create_memtransformer


def main():
    config = MemTransformerConfig(
        vocab_size=6000,
        block_size=32,
        n_embd=96,
        n_head=4,
        n_layers=3,
    )

    model = create_memtransformer(config, device="auto")

    print("Created:", type(model).__name__)
    print("Device:", model.device)
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    x = torch.randint(0, config.vocab_size, (1, 16), device=model.device)

    # Fake memory bank, only to verify the architecture runs.
    memory_keys = torch.randn(8, config.n_embd, device=model.device)
    memory_values = torch.randn(8, config.n_embd, device=model.device)
    model.set_memory(memory_keys, memory_values)

    logits, loss, debug = model(x, memory_gene_mode="safe", return_debug=True)

    print("Logits shape:", tuple(logits.shape))
    print("Memory debug:", debug["memory_gene"])

    next_id, next_debug = model.predict_next_id(x, memory_gene_mode="safe")
    print("Next token id:", int(next_id[0]))


if __name__ == "__main__":
    main()
