"""Run the app's answer pipeline (webapp.server._events: greedy decoding, number check, answer confidence, retry) on a
test set on CPU and save each final answer with its score and attempts.

  python scripts/app_answers.py data/eval_real/itemfirst_nosim_v4 runs/v2g-from-dapt-8ep-dpo/app_answers_realtest.jsonl 12
"""
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

MODEL = "Realistic, no retrieval score"


def work(args):
    i, text = args
    from relevance_explainer.webapp import server as S
    done = None
    for line in S._events(text, MODEL, 0.0, True):
        ev = json.loads(line)
        if ev["type"] in ("done", "error"):
            done = ev
    return i, done


def init():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(2)


if __name__ == "__main__":
    data, out, procs = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    rows = [json.loads(l) for l in (data / "test.jsonl").open()]
    done = {json.loads(l)["i"] for l in out.open()} if out.exists() else set()
    jobs = [(i, r["input"]) for i, r in enumerate(rows) if i not in done]
    with mp.get_context("spawn").Pool(procs, initializer=init) as pool, out.open("a") as f:
        for n, (i, ev) in enumerate(pool.imap_unordered(work, jobs, chunksize=4)):
            r = rows[i]
            f.write(json.dumps({"i": i, "label": r["verdict"], "unanimous": r.get("unanimous"), "language": r["language"],
                                "source": r.get("source"), "event": ev}, ensure_ascii=False) + "\n")
            if n % 100 == 0:
                print(f"{n}/{len(jobs)}", flush=True)
    print("APP-ANSWERS-DONE", flush=True)
