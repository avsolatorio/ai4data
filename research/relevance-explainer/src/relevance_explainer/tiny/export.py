"""Export a trained model as one INT8 file and load it back for inference.

The file holds per-row symmetric INT8 weights with fp16 row scales for every matrix, fp16 norm weights, the model
configuration and the tokenizer. Loading dequantizes to float tensors, which is numerically what the evaluation
scripts measure (see `quantize` in train.py).

  python -m relevance_explainer.tiny.export --run runs/f1-pre2 --out runs/f1-pre2/explainer-int8.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .model import Config, TinyLM
from .scriptguard import banned_for, token_masks

EOS = 1


def export_int8(state: dict, config: dict, tokenizer_json: str, path: Path) -> int:
    tensors = {}
    for name, w in state.items():
        w = w.detach().float().cpu()
        if w.dim() == 2:
            scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
            tensors[name] = {"q": (w / scale).round().clamp(-127, 127).to(torch.int8), "scale": scale.half()}
        else:
            tensors[name] = {"fp16": w.half()}
    torch.save({"config": config, "tensors": tensors, "tokenizer": tokenizer_json}, path)
    return path.stat().st_size


def load_int8(path: str | Path, device: str = "cpu") -> tuple[TinyLM, Tokenizer]:
    blob = torch.load(path, map_location="cpu")
    state = {}
    for name, t in blob["tensors"].items():
        state[name] = t["q"].float() * t["scale"].float() if "q" in t else t["fp16"].float()
    model = TinyLM(Config(**blob["config"]))
    model.load_state_dict(state)
    return model.to(device).eval(), Tokenizer.from_str(blob["tokenizer"])


def explain(model: TinyLM, tk: Tokenizer, inputs: list[str], max_new: int = 260, batch: int = 32,
            script_guard: bool = True) -> list[tuple[str, str]]:
    """Return (verdict, narrative) for each rendered input text. `script_guard` blocks stray characters of scripts
    that neither the requested language nor the input uses (see scriptguard)."""
    sep = "\n\n### Judgment\n"
    prompts = [tk.encode(t + sep).ids for t in inputs]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    masks = token_masks(tk) if script_guard else None
    res: dict[int, tuple[str, str]] = {}
    for s in range(0, len(order), batch):
        ch = order[s : s + batch]
        ban = torch.stack([banned_for(masks, inputs[i]) for i in ch]) if masks else None
        outs = model.generate_batch([prompts[i] for i in ch], EOS, max_new, banned=ban)
        for i, o in zip(ch, outs):
            head, _, rest = tk.decode(o).strip().partition("\n")
            res[i] = (head.strip(), rest.strip())
    return [res[i] for i in range(len(inputs))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    ck = torch.load(run / "model_fp32.pt", map_location="cpu")
    size = export_int8(ck["state"], ck["config"], (run / "tokenizer.json").read_text(), Path(a.out))
    print(f"wrote {a.out}: {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
