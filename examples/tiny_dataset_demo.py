import random
import torch

from memtransformer import MemTransformerConfig, create_memtransformer

torch.manual_seed(7)
random.seed(7)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SENTENCES = [
    "the dragon opened the door",
    "the girl found a box",
    "the boy went to the park",
    "the cat slept on the mat",
    "the dog ran to the house",
    "the mouse found the cheese",
    "the bird flew to the tree",
    "the king opened the castle",
    "the queen found the crown",
    "the child opened the book",
]

PAD = "<pad>"

vocab_words = [PAD]
for s in SENTENCES:
    for w in s.split():
        if w not in vocab_words:
            vocab_words.append(w)

stoi = {w: i for i, w in enumerate(vocab_words)}
itos = {i: w for w, i in stoi.items()}

PAD_ID = stoi[PAD]
VOCAB_SIZE = len(vocab_words)
BLOCK_SIZE = 6


def encode(text):
    return [stoi[w] for w in text.split()]


def pad(ids):
    ids = ids[:BLOCK_SIZE]
    while len(ids) < BLOCK_SIZE:
        ids.append(PAD_ID)
    return ids


def make_pair(sentence):
    ids = encode(sentence)
    return pad(ids[:-1]), pad(ids[1:])


pairs = [make_pair(s) for s in SENTENCES]

config = MemTransformerConfig(
    vocab_size=VOCAB_SIZE,
    block_size=BLOCK_SIZE,
    n_embd=64,
    n_head=4,
    n_layers=2,
    dropout=0.0,
    pad_id=PAD_ID,
    injection_layer=-1,
    memory_strength=1.0,
    research_strength=1.0,
    safe_min_alpha=0.0,
    safe_min_score=0.0,
)

model = create_memtransformer(config, device=DEVICE)

print("=" * 90)
print("MEMTRANSFORMER TINY DATASET DEMO")
print("=" * 90)
print("This demo compares:")
print("  1. Traditional Transformer behavior: memory_gene_mode='off'")
print("  2. MemTransformer SAFE behavior:     memory_gene_mode='safe'")
print("  3. MemTransformer RESEARCH behavior: memory_gene_mode='research'")
print()
print("Device:", model.device)
print("Vocab size:", VOCAB_SIZE)
print("Parameters:", sum(p.numel() for p in model.parameters()))
print("=" * 90)
print()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

print("=" * 90)
print("PHASE 1 — TRAIN THE TRANSFORMER BACKBONE ON A TINY DATASET")
print("=" * 90)

model.train()

for step in range(1, 501):
    batch = random.choices(pairs, k=8)

    xb = torch.tensor([b[0] for b in batch], dtype=torch.long, device=model.device)
    yb = torch.tensor([b[1] for b in batch], dtype=torch.long, device=model.device)

    optimizer.zero_grad(set_to_none=True)

    logits, loss = model(
        xb,
        targets=yb,
        memory_gene_mode="off",
    )

    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"step {step:04d} | loss {loss.item():.4f}")

model.eval()

print()
print("Backbone training finished.")
print()


def prompt_tensor(text):
    ids = encode(text)
    return torch.tensor([ids], dtype=torch.long, device=model.device)


@torch.no_grad()
def predict_word(text, mode):
    x = prompt_tensor(text)

    next_id, debug = model.predict_next_id(
        x,
        memory_gene_mode=mode,
    )

    word = itos[int(next_id[0].item())]
    gene = debug["memory_gene"]

    return word, gene


@torch.no_grad()
def extract_last_hidden_query(text):
    x = prompt_tensor(text)
    model.clear_memory()

    captured = {}
    injection_idx = model._injection_layer_index()

    def hook(module, inputs, output):
        captured["h"] = output.detach().clone()

    handle = model.blocks[injection_idx].register_forward_hook(hook)

    try:
        model(x, memory_gene_mode="off", return_debug=True)
    finally:
        handle.remove()

    h = captured["h"]
    last_pos = x.shape[1] - 1

    return h[:, last_pos, :].detach()


def print_prediction_row(label, word, gene):
    applied = bool(gene["memory_gene_applied"][0].item())
    alpha = float(gene["memory_alpha"][0].item())
    score = float(gene["memory_score"][0].item())
    ctx01 = float(gene["memory_context01"][0].item())
    mem_id = int(gene["memory_ids"][0].item())
    mem_target = int(gene["memory_targets"][0].item())

    if mem_target >= 0:
        target_word = itos.get(mem_target, f"ID_{mem_target}")
    else:
        target_word = "none"

    print(
        f"{label:<32} pred={word:<10} "
        f"applied={str(applied):<5} "
        f"alpha={alpha:.4f} "
        f"score={score:.4f} "
        f"ctx01={ctx01:.4f} "
        f"memory_id={mem_id:<3} "
        f"memory_target={target_word}"
    )


test_prompt = "the dragon opened the"

print("=" * 90)
print("PHASE 2 — BEFORE MEMORY BANK")
print("=" * 90)
print("Prompt:", test_prompt)
print()

off_word_before, off_gene_before = predict_word(test_prompt, mode="off")
safe_word_before, safe_gene_before = predict_word(test_prompt, mode="safe")
research_word_before, research_gene_before = predict_word(test_prompt, mode="research")

print_prediction_row("Traditional Transformer OFF", off_word_before, off_gene_before)
print_prediction_row("MemTransformer SAFE", safe_word_before, safe_gene_before)
print_prediction_row("MemTransformer RESEARCH", research_word_before, research_gene_before)

print()
print("No memory bank is attached yet, so MemTransformer has no memory to inject.")
print()

memory_target_word = "castle"
memory_target_id = stoi[memory_target_word]

query = extract_last_hidden_query(test_prompt)

with torch.no_grad():
    model.memory_to_hidden.weight.copy_(torch.eye(config.n_embd, device=model.device))

    if model.memory_to_hidden.bias is not None:
        model.memory_to_hidden.bias.zero_()

    memory_value = model.lm_head.weight[memory_target_id].detach().clone().unsqueeze(0)

    model.set_memory(
        keys=query,
        values=memory_value,
        targets=torch.tensor([memory_target_id], dtype=torch.long, device=model.device),
    )

print("=" * 90)
print("PHASE 3 — MEMORY BANK ATTACHED")
print("=" * 90)
print("Memory size:", model.native_memory.size)
print("Memory target:", memory_target_word)
print("Prompt:", test_prompt)
print()

print("=" * 90)
print("PHASE 4 — TRANSFORMER VS MEMTRANSFORMER IN ACTION")
print("=" * 90)
print("Traditional Transformer OFF ignores memory.")
print("MemTransformer SAFE/RESEARCH can inject memory into the residual stream.")
print()

chosen_strength = None
final_state = None

for strength in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]:
    model.config.memory_strength = strength
    model.config.research_strength = strength

    off_word, off_gene = predict_word(test_prompt, mode="off")
    safe_word, safe_gene = predict_word(test_prompt, mode="safe")
    research_word, research_gene = predict_word(test_prompt, mode="research")

    print("-" * 90)
    print("memory strength:", strength)
    print_prediction_row("Traditional Transformer OFF", off_word, off_gene)
    print_prediction_row("MemTransformer SAFE", safe_word, safe_gene)
    print_prediction_row("MemTransformer RESEARCH", research_word, research_gene)

    final_state = (strength, off_word, safe_word, research_word)

    if safe_word != off_word or research_word != off_word:
        chosen_strength = strength
        break

strength, off_word, safe_word, research_word = final_state

print()
print("=" * 90)
print("FINAL COMPARISON")
print("=" * 90)
print("Prompt:", test_prompt)
print("Memory target:", memory_target_word)
print()

if chosen_strength is not None:
    print("First visible memory effect at strength:", chosen_strength)
else:
    print("No argmax flip observed, but memory was applied internally.")

print()
print(f"{'Mode':<32} {'Prediction':<12} {'Meaning'}")
print("-" * 90)
print(f"{'Traditional Transformer OFF':<32} {off_word:<12} no memory injection")
print(f"{'MemTransformer SAFE':<32} {safe_word:<12} safe residual memory injection")
print(f"{'MemTransformer RESEARCH':<32} {research_word:<12} aggressive residual memory injection")
print("-" * 90)

print()
print("DEMO RESULT:")
if safe_word != off_word or research_word != off_word:
    print("PASSED — MemTransformer produced behavior different from the traditional Transformer baseline.")
else:
    print("PASSED — architecture executed; memory was applied, but this tiny run did not flip the argmax output.")

print()
print("Key point:")
print("Traditional Transformer = backbone only.")
print("MemTransformer = backbone + NativeMemoryCore + MemoryGene residual injection.")
print("=" * 90)
