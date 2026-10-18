"""Open-weights models that can label, write or judge, with how each is served and called.

The teacher (Qwen3.8-27B) and Gemma 4 are the models behind the released data; their servers are started by
scripts/keepalive.sh and scripts/serve_vllm.sh, and the data scripts name them directly. The other entries were added
as further judges, one per model family absent from the data; on the judge panel (scripts/judge_panel.py, notebook
2026-10-18) gpt-oss-120b agreed with the references nearly as well as Gemma 4 and the other two did not. Each is
started on its own:

  python -m relevance_explainer.synth.models serve gpt-oss-120b [--gpus 1]
  python -m relevance_explainer.synth.models list

`request` holds the extra fields of every chat request to that model (llm.chat(..., extra=...)): each turns the
model's reasoning off, as the project's label scripts do for the teacher, or to "low" for gpt-oss, which has no off. Weights are read from RELEXP_HF_CACHE when
set (scripts/local.env), else from the default Hugging Face cache.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
# The FlashInfer FP8 MoE kernels are compiled at start-up and fail with the nvcc installed here; Triton's do not need it.
TRITON_MOE = ["--moe-backend", "triton"]

MODELS: dict[str, dict] = {
    "qwen3.8-27b": {
        "repo": "Qwen/Qwen3.8-27B-FP8", "family": "Qwen (Alibaba)", "licence": "Apache-2.0", "port": 8000,
        "urls": ["http://localhost:8000/v1", "http://localhost:8001/v1"], "request": NO_THINKING,
        "serve": ["--max-model-len", "4096", "--language-model-only"], "mem": 0.55, "role": "teacher",
    },
    "gemma-4-31b": {
        "repo": "RedHatAI/gemma-4-31B-it-FP8-dynamic", "family": "Gemma (Google)", "licence": "Apache-2.0", "port": 8002,
        "urls": ["http://localhost:8002/v1"], "request": NO_THINKING,
        "serve": ["--max-model-len", "3072", "--language-model-only"], "mem": 0.5, "role": "second judge",
    },
    "gpt-oss-120b": {
        "repo": "openai/gpt-oss-120b", "family": "gpt-oss (OpenAI)", "licence": "Apache-2.0", "port": 8003,
        "urls": ["http://localhost:8003/v1"], "request": {"reasoning_effort": "low"},
        "serve": ["--max-model-len", "8192"], "mem": 0.85, "role": "judge",
        # gpt-oss has no reasoning-off mode; "low" is its shortest, so max_tokens must leave room for the reasoning.
        "max_tokens": 2000,
    },
    "nemotron-3.5-lightning": {
        "repo": "RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8", "family": "Nemotron (NVIDIA)",
        "licence": "OpenMDW-1.1", "port": 8004, "urls": ["http://localhost:8004/v1"], "request": NO_THINKING,
        "serve": ["--max-model-len", "8192", "--trust-remote-code", *TRITON_MOE], "mem": 0.6,
        "role": "not used",  # 48.5% of designed challenge verdicts; with reasoning on 79.7%, at 36 times the tokens
    },
    "mistral-small-4": {
        "repo": "mistralai/Mistral-Small-4-119B-2603", "family": "Mistral", "licence": "Apache-2.0", "port": 8005,
        "urls": ["http://localhost:8005/v1"], "request": {"reasoning_effort": "none"},
        # FP8 weights, split over both GPUs, loaded from the Mistral-format files (consolidated-*.safetensors,
        # params.json, tekken.json; hf download ... --include "consolidated*" params.json tekken.json): vllm 0.30.0
        # maps them to MistralLarge3ForCausalLM, and finds no text architecture in the Hugging Face config.
        "serve": ["--max-model-len", "8192", "--tensor-parallel-size", "2", "--attention-backend", "FLASH_ATTN_MLA",
                  "--tokenizer-mode", "mistral", "--config-format", "mistral", "--load-format", "mistral",
                  "--language-model-only", *TRITON_MOE],
        # vllm 0.30.0 cannot load it with the locked transformers 5.18.0 (scripts/vllm_overlay.sh).
        "mem": 0.9, "gpus": "0,1", "overlay": True,
        "role": "not used",  # 47.7% of designed challenge verdicts; 65.3% with reasoning high
    },
}


def model_dir(repo: str) -> Path:
    roots = [os.environ.get("RELEXP_HF_CACHE"), os.environ.get("HF_HUB_CACHE"), str(Path.home() / ".cache/huggingface/hub")]
    for root in filter(None, roots):
        snaps = sorted((Path(root) / f"models--{repo.replace('/', '--')}" / "snapshots").glob("*"))
        if snaps:
            return snaps[0]
    raise FileNotFoundError(f"{repo} is not downloaded (hf download {repo})")


def serve(name: str, gpus: str | None) -> None:
    m = MODELS[name]
    base = os.environ.get("RELEXP_VLLM", "vllm")
    vllm = base
    if m.get("overlay"):
        vllm = os.environ.get("RELEXP_VLLM_OVERLAY") or sys.exit(f"{name} needs RELEXP_VLLM_OVERLAY (scripts/vllm_overlay.sh)")
    # Kernels compiled at start-up (torch inductor, for Nemotron's Mamba layers) call ninja from the vllm environment.
    path = os.pathsep.join([str(Path(base).parent)] * (os.sep in base) + [os.environ.get("PATH", "")])
    env = {**os.environ, "PATH": path, "CUDA_VISIBLE_DEVICES": gpus or m.get("gpus", "0"),
           # As in scripts/serve_vllm.sh: DeepGEMM and the FlashInfer sampler JIT-compile with an nvcc not installed here.
           "VLLM_USE_DEEP_GEMM": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    argv = [vllm, "serve", str(model_dir(m["repo"])), "--served-model-name", name,
            "--port", str(m["port"]), "--gpu-memory-utilization", str(m["mem"]), "--max-num-seqs", "128",
            "--enable-prefix-caching", *m["serve"]]
    print(" ".join(argv), flush=True)
    os.execvpe(argv[0], argv, env)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("name", choices=sorted(MODELS))
    s.add_argument("--gpus", help="CUDA_VISIBLE_DEVICES (default: the model's own)")
    sub.add_parser("list")
    a = ap.parse_args()
    if a.cmd == "serve":
        serve(a.name, a.gpus)
    for name, m in MODELS.items():
        try:
            where = str(model_dir(m["repo"]))
        except FileNotFoundError:
            where = "not downloaded"
        print(f"{name:24s} {m['family']:18s} {m['licence']:12s} port {m['port']}  {m['role']:12s} {where}", file=sys.stdout)


if __name__ == "__main__":
    main()
