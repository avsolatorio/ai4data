"""Checks for grouped-query attention and the attention gate in tiny/model.py.

1. The deployed checkpoint loads unchanged and gives the same greedy output as before (default config).
2. For each variant, KV-cached batch decoding equals full recomputation (`generate`).
3. Parameter counts of the shapes to compare."""
import torch

from relevance_explainer.tiny.export import load_int8
from relevance_explainer.tiny.model import Config, TinyLM

model, tk = load_int8("runs/v2g-from-dapt-8ep-dpo/explainer-int8.pt")
ids = tk.encode("Search: youth unemployment in Kenya\nRecord: Kenya Labour Force Survey 2019\n").ids
out = model.generate(torch.tensor([ids]), eos=0, max_new=20)
print("deployed model loads; greedy tokens:", out[:10])

torch.manual_seed(0)
for kv, gate in [(1, ""), (7, "head"), (7, "elem"), (1, "head")]:
    m = TinyLM(Config(vocab=500, d_model=448, n_layer=2, n_head=7, n_kv_head=kv, attn_gate=gate, dropout=0.0)).eval()
    prompts = [list(range(5, 25)), list(range(30, 42))]
    batch = m.generate_batch(prompts, eos=1, max_new=12, no_repeat=0)
    single = [m.generate(torch.tensor([p]), eos=1, max_new=12, no_repeat=0) for p in prompts]
    print(f"kv={kv} gate={gate or '-'}: cache matches full recompute: {batch == single}")

for name, L, kv, gate in [("32k 7L", 7, 0, ""), ("32k 8L GQA kv1", 8, 1, ""), ("32k 7L gate head", 7, 0, "head"),
                          ("32k 7L gate elem", 7, 0, "elem"), ("32k 8L GQA kv1 gate head", 8, 1, "head")]:
    m = TinyLM(Config(vocab=32000, d_model=448, n_layer=L, n_head=7, n_kv_head=kv, attn_gate=gate, d_ff=1024))
    print(f"{name}: {m.n_params() / 1e6:.2f}M params")
