import random
import sys
import torch

from memtransformer import MemTransformerConfig, create_memtransformer


# ============================================================
# Explicit Memory Win Test
#
# Goal:
# show one clean case where:
#
# - Traditional Transformer gives the learned answer
# - MemTransformer uses an injected memory and wins the memory task
#
# This is not a general language benchmark.
# It is a targeted architectural test:
#
# Can native memory change the output through the residual stream?
# ============================================================

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
for sentence in SENTENCES:
    for word in sentence.split():
        if word not in vocab_words:
            vocab_words.append(word)

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


pairs = [make_pair(sentence) for sentence in SENTENCES]

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

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

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

model.eval()


def prompt_tensor(text):
    ids = encode(text)
    return torch.tensor([ids], dtype=torch.long, device=model.device)


@torch.no_grad()
def predict_word(text, mode):
    x = prompt_tensor(text)
    next_id, debug = model.predict_next_id(x, memory_gene_mode=mode)
    word = itos[int(next_id[0].item())]
    return word, debug["memory_gene"]


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


def short_gene(gene):
    return (
        f"applied={bool(gene['memory_gene_applied'][0].item())} "
        f"alpha={float(gene['memory_alpha'][0].item()):.4f} "
        f"score={float(gene['memory_score'][0].item()):.4f} "
        f"ctx01={float(gene['memory_context01'][0].item()):.4f}"
    )


prompt = "the dragon opened the"

learned_expected = "door"
memory_expected = "castle"

# 1. Traditional Transformer before memory.
off_before, off_before_gene = predict_word(prompt, "off")

# 2. Attach one memory:
#    same prompt should retrieve memory target "castle".
query = extract_last_hidden_query(prompt)
target_id = stoi[memory_expected]

with torch.no_grad():
    model.memory_to_hidden.weight.copy_(torch.eye(config.n_embd, device=model.device))
    if model.memory_to_hidden.bias is not None:
        model.memory_to_hidden.bias.zero_()

    memory_value = model.lm_head.weight[target_id].detach().clone().unsqueeze(0)

    model.set_memory(
        keys=query,
        values=memory_value,
        targets=torch.tensor([target_id], dtype=torch.long, device=model.device),
    )

winner_strength = None
final = None

for strength in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]:
    model.config.memory_strength = strength
    model.config.research_strength = strength

    off_word, off_gene = predict_word(prompt, "off")
    safe_word, safe_gene = predict_word(prompt, "safe")
    research_word, research_gene = predict_word(prompt, "research")

    final = {
        "strength": strength,
        "off_word": off_word,
        "safe_word": safe_word,
        "research_word": research_word,
        "off_gene": off_gene,
        "safe_gene": safe_gene,
        "research_gene": research_gene,
    }

    if safe_word == memory_expected or research_word == memory_expected:
        winner_strength = strength
        break

print("=" * 90)
print("MEMTRANSFORMER EXPLICIT MEMORY WIN TEST")
print("=" * 90)
print("Prompt:", prompt)
print()
print("Learned continuation in tiny dataset:")
print("  the dragon opened the door")
print()
print("Injected memory target:")
print(" ", memory_expected)
print()
print("Task:")
print("  Use the injected memory.")
print()
print("=" * 90)
print("RESULTS")
print("=" * 90)
print(f"{'System':<34} {'Prediction':<12} {'Memory task'}")
print("-" * 90)

off_ok = final["off_word"] == memory_expected
safe_ok = final["safe_word"] == memory_expected
research_ok = final["research_word"] == memory_expected

print(f"{'Traditional Transformer OFF':<34} {final['off_word']:<12} {'WIN' if off_ok else 'FAIL'}")
print(f"{'MemTransformer SAFE':<34} {final['safe_word']:<12} {'WIN' if safe_ok else 'FAIL'}")
print(f"{'MemTransformer RESEARCH':<34} {final['research_word']:<12} {'WIN' if research_ok else 'FAIL'}")
print("-" * 90)
print()
print("Debug:")
print("  OFF     ", short_gene(final["off_gene"]))
print("  SAFE    ", short_gene(final["safe_gene"]))
print("  RESEARCH", short_gene(final["research_gene"]))
print()
print("Strength used:", final["strength"])
print()

if (not off_ok) and (safe_ok or research_ok):
    print("FINAL VERDICT:")
    print("  Traditional Transformer did NOT win the memory task.")
    print("  MemTransformer DID win the memory task.")
    print()
    print("PASSED")
    sys.exit(0)

print("FINAL VERDICT:")
print("  Test executed, but did not produce a clean MemTransformer-only win.")
print("FAILED")
sys.exit(1)
