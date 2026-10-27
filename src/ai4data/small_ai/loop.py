"""The feedback loop: rounds of generation, training, and measurement on real text.

A run folder holds one arm of an experiment:

    run.json            settings: task, arm, requests per round, rounds, seed, prices
    reference/          the real reference set, split into feedback, threshold, and report files
    rounds/rNN/         per round: plan.jsonl, gen/ (raw replies, examples), check.json,
                        model/, feedback_scores.jsonl, eval.json, gap.json, decision.json
    log.jsonl           one line per event, with the time

Round 0 is a balanced plan. In the targeted arm, each later round takes a
plan from the orchestrator: items with labels, a number of requests, a
mode, and a note. The plan must use exactly the requests of the round,
so that the targeted arm and the control arm add the same amount of
data. In the control arm, each later round is a balanced plan of the
same size that counts the existing examples. Every round trains a new
model from the base encoder on all accepted examples so far and scores
it on the feedback file.

A round is accepted when its headline score on the feedback file is no
more than `tolerance` below the best accepted round. A rejected round's
examples are left out of later rounds, and the orchestrator is told.

The report file is never read during the rounds. final() trains the
last accepted data with each seed, sets the threshold on the threshold
file (multi-label tasks), and scores the report file once.
"""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from . import check as checks
from . import evaluate as ev
from . import generate as gen
from . import model as mdl
from . import plan as plans
from .task import Task, load_task


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: Path) -> pd.DataFrame:
    return (
        pd.read_json(path, lines=True, dtype={"id": str})
        if path.exists() and path.stat().st_size
        else pd.DataFrame()
    )


def write_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_json(path, orient="records", lines=True, force_ascii=False)


class Run:
    def __init__(self, folder: str | Path):
        self.folder = Path(folder).resolve()
        self.settings = json.loads(
            (self.folder / "run.json").read_text(encoding="utf-8")
        )
        self.task: Task = load_task(self.settings["task"])

    # -- bookkeeping
    def log(self, event: str, **data) -> None:
        with (self.folder / "log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"time": now(), "event": event, **data}, ensure_ascii=False)
                + "\n"
            )

    def round_dir(self, k: int) -> Path:
        return self.folder / "rounds" / f"r{k:02d}"

    def rounds(self) -> list[int]:
        return sorted(
            int(p.name[1:])
            for p in (self.folder / "rounds").glob("r*")
            if (p / "decision.json").exists()
        )

    def decision(self, k: int) -> dict:
        return json.loads(
            (self.round_dir(k) / "decision.json").read_text(encoding="utf-8")
        )

    def accepted(self) -> list[int]:
        return [k for k in self.rounds() if self.decision(k)["accepted"]]

    def stopped(self) -> dict | None:
        path = self.folder / "stop.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def examples(self, rounds: list[int]) -> pd.DataFrame:
        frames = [
            read_jsonl(self.round_dir(k) / "gen" / "examples.jsonl") for k in rounds
        ]
        frames = [f for f in frames if len(f)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def cost(self) -> dict:
        price_in, price_out = self.settings["price_per_million"]
        total = {"prompt_tokens": 0, "completion_tokens": 0}
        for p in (self.folder / "rounds").glob("r*/gen/raw.jsonl"):
            if self.settings.get("shared_baseline") and p.parent.parent.name == "r00":
                continue
            for role in gen.usage(p).values():
                for k in total:
                    total[k] += role[k]
        total["dollars"] = round(
            total["prompt_tokens"] * price_in / 1e6
            + total["completion_tokens"] * price_out / 1e6,
            2,
        )
        return total

    def status(self) -> dict:
        rows = []
        for k in self.rounds():
            d = self.decision(k)
            rows.append(
                {
                    "round": k,
                    "accepted": d["accepted"],
                    "headline": d["headline"],
                    "examples_added": d["examples_added"],
                    "train_examples": d["train_examples"],
                    "note": d.get("note", ""),
                }
            )
        best = max((r["headline"] for r in rows if r["accepted"]), default=None)
        return {
            "task": self.task.id,
            "arm": self.settings["arm"],
            "rounds_done": len(rows),
            "max_rounds": self.settings["rounds"],
            "requests_per_round": self.settings["round_requests"],
            "best_feedback_score": best,
            "rounds": rows,
            "spent": self.cost(),
            "budget_dollars": self.settings.get("budget_dollars"),
            "stopped": self.stopped(),
            "score_name": "micro F1" if self.task.multi_label else "accuracy",
        }

    # -- reference data
    def reference(self, part: str) -> pd.DataFrame:
        return read_jsonl(self.folder / "reference" / f"{part}.jsonl")

    # -- one round
    def next_round(self) -> int:
        done = self.rounds()
        return (done[-1] + 1) if done else 0

    def plan_for(self, k: int, items: list[dict] | None) -> list[dict]:
        s, task = self.settings, self.task
        siblings = (
            {x: task.siblings(x) for x in task.leaves}
            if task.generation.get("group") == "siblings"
            else None
        )
        per_request = int(task.generation.get("labels_per_request", 8))
        if k == 0:
            return plans.balanced(
                task.leaves,
                s["baseline_requests"],
                per_request,
                seed=s["seed"],
                siblings=siblings,
                prefix="r00",
            )
        if s["arm"] == "control":
            existing = Counter(
                x for ls in self.examples(self.accepted())["labels"] for x in ls
            )
            per_offer = (
                float(task.generation.get("per_call", 20))
                / per_request
                * (1.5 if task.multi_label else 1.0)
            )
            return plans.balanced(
                task.leaves,
                s["round_requests"],
                per_request,
                existing,
                per_offer,
                seed=s["seed"] + k,
                siblings=siblings,
                prefix=f"r{k:02d}",
            )
        if not items:
            raise ValueError("the targeted arm needs a plan for this round")
        planned = plans.targeted(items, set(task.leaves), prefix=f"r{k:02d}")
        if len(planned) != s["round_requests"]:
            raise ValueError(
                f"the plan has {len(planned)} requests; this round takes exactly {s['round_requests']}"
            )
        if not task.multi_label and any(r["mode"] == "background" for r in planned):
            raise ValueError("background requests are for multi-label tasks only")
        return planned

    def run_round(
        self, items: list[dict] | None = None, note: str = "", workers: int = 32
    ) -> dict:
        if self.stopped():
            raise RuntimeError("this run has been stopped")
        k = self.next_round()
        if k > self.settings["rounds"]:
            raise RuntimeError(f"all {self.settings['rounds']} rounds are done")
        spent = self.cost()["dollars"]
        if (
            self.settings.get("budget_dollars")
            and spent >= self.settings["budget_dollars"]
        ):
            raise RuntimeError(
                f"the budget of {self.settings['budget_dollars']} dollars is spent ({spent})"
            )
        rd = self.round_dir(k)
        rd.mkdir(parents=True, exist_ok=True)
        planned = self.plan_for(k, items)
        write_jsonl(planned, rd / "plan.jsonl")
        if items is not None:
            (rd / "orchestrator_plan.json").write_text(
                json.dumps(
                    {"items": items, "note": note}, indent=2, ensure_ascii=False
                ),
                encoding="utf-8",
            )
        self.log("round_start", round=k, requests=len(planned), note=note)
        per_call = int(self.task.generation.get("per_call", 20))
        result = gen.generate(
            self.task,
            planned,
            rd / "gen",
            per_call,
            workers=workers,
            verify=bool(self.task.generation.get("verify", True)),
        )
        # one retry for requests that failed
        if result["failed"]:
            result = gen.generate(
                self.task,
                planned,
                rd / "gen",
                per_call,
                workers=workers,
                verify=bool(self.task.generation.get("verify", True)),
            )
        return self.finish_round(k, note)

    def finish_round(self, k: int, note: str = "") -> dict:
        rd = self.round_dir(k)
        rounds = [*self.accepted(), k]
        raw = self.examples(rounds)
        report, clean, _ = checks.check(
            raw,
            self.task,
            require_verified=bool(self.task.generation.get("verify", True)),
        )
        new = read_jsonl(rd / "gen" / "examples.jsonl")
        new_report, _, _ = checks.check(new, self.task)
        (rd / "check.json").write_text(
            json.dumps({"cumulative": report, "this_round": new_report}, indent=2),
            encoding="utf-8",
        )
        write_jsonl(checks.agreement(new), rd / "agreement.jsonl")
        splits = mdl.split(clean.reset_index(drop=True), seed=self.settings["seed"])
        config = mdl.train(
            self.task,
            splits,
            rd / "model",
            seed=self.settings["seed"],
            bf16=self.settings.get("bf16", False),
        )
        feedback = self.reference("feedback")
        scores, _ = mdl.score(self.task, rd / "model", feedback)
        write_jsonl(scores, rd / "feedback_scores.jsonl")
        threshold = None
        if self.task.multi_label:
            threshold = max(ev.sweep(feedback, scores), key=lambda tf: tf[1])[0]
        result = ev.evaluate(self.task, feedback, scores, threshold)
        (rd / "eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        counts = Counter(x for ls in splits["train"]["labels"] for x in ls)
        agree = {
            r["label"]: r["agreement"]
            for r in checks.agreement(self.examples(rounds)).to_dict("records")
        }
        gap = ev.gap_report(self.task, result, counts, agree)
        (rd / "gap.json").write_text(
            json.dumps(gap, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        best = max(
            (self.decision(j)["headline"] for j in self.accepted()), default=None
        )
        tolerance = float(self.settings.get("tolerance", 0.01))
        accepted = best is None or gap["headline"] >= best - tolerance
        decision = {
            "round": k,
            "accepted": accepted,
            "headline": gap["headline"],
            "best_before": best,
            "threshold": threshold,
            "examples_added": int(new_report["clean examples"]),
            "train_examples": len(splits["train"]),
            "train_seconds": config["train_seconds"],
            "synthetic_test": config["scores"]["test"],
            "note": note,
            "time": now(),
        }
        (rd / "decision.json").write_text(
            json.dumps(decision, indent=2), encoding="utf-8"
        )
        self.log("round_end", **decision)
        shutil.rmtree(rd / "model" / "checkpoints", ignore_errors=True)
        if not self.settings.get("keep_models"):
            # a round's weights are not needed again: final() trains the chosen data afresh
            for weights in (rd / "model").glob("*.safetensors"):
                weights.unlink()
        return {"decision": decision, "gap": gap}

    # -- the end of the run
    def stop(self, reason: str) -> None:
        (self.folder / "stop.json").write_text(
            json.dumps({"reason": reason, "time": now()}, indent=2), encoding="utf-8"
        )
        self.log("stop", reason=reason)

    def final(
        self, seeds: list[int], rounds: list[int] | None = None, tag: str = "final"
    ) -> dict:
        rounds = rounds if rounds is not None else self.accepted()
        clean = checks.check(
            self.examples(rounds),
            self.task,
            require_verified=bool(self.task.generation.get("verify", True)),
        )[1].reset_index(drop=True)
        out = {"rounds": rounds, "examples": len(clean), "seeds": {}}
        for seed in seeds:
            folder = self.folder / tag / f"seed{seed}"
            splits = mdl.split(clean, seed=self.settings["seed"])
            if not (folder / "model" / "classifier.json").exists():
                mdl.train(
                    self.task,
                    splits,
                    folder / "model",
                    seed=seed,
                    bf16=self.settings.get("bf16", False),
                )
                shutil.rmtree(folder / "model" / "checkpoints", ignore_errors=True)
            threshold = None
            if self.task.multi_label:
                calib = self.reference("threshold")
                cs, _ = mdl.score(self.task, folder / "model", calib)
                threshold = max(ev.sweep(calib, cs), key=lambda tf: tf[1])[0]
            report = self.reference("report")
            rs, timing = mdl.score(self.task, folder / "model", report)
            write_jsonl(rs, folder / "report_scores.jsonl")
            result = ev.evaluate(self.task, report, rs, threshold)
            out["seeds"][str(seed)] = {
                "headline": ev.headline(self.task, result),
                "threshold": threshold,
                "summary": ev.summary(self.task, result),
                "timing": timing,
                **{
                    k: v
                    for k, v in result.items()
                    if k not in ("per_label", "confusions")
                },
            }
        heads = [v["headline"] for v in out["seeds"].values()]
        out["mean_headline"] = round(sum(heads) / len(heads), 4)
        out["spent"] = self.cost()
        (self.folder / f"{tag}.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8"
        )
        self.log(tag, mean_headline=out["mean_headline"], rounds=rounds)
        return out


def init_run(
    task_folder: str | Path,
    folder: str | Path,
    arm: str,
    baseline_requests: int,
    round_requests: int,
    rounds: int,
    seed: int = 42,
    price=(0.15, 0.60),
    budget: float | None = None,
    baseline_from: str | Path | None = None,
    tolerance: float = 0.01,
    bf16: bool = False,
    feedback_share: float = 0.67,
) -> Run:
    """Create a run folder, with its reference split, optionally reusing round 0 of another run."""
    task = load_task(task_folder)
    folder = Path(folder)
    if (folder / "run.json").exists():
        raise FileExistsError(f"{folder} already holds a run")
    folder.mkdir(parents=True, exist_ok=True)
    settings = {
        "task": str(task.folder),
        "arm": arm,
        "baseline_requests": baseline_requests,
        "round_requests": round_requests,
        "rounds": rounds,
        "seed": seed,
        "price_per_million": list(price),
        "budget_dollars": budget,
        "tolerance": tolerance,
        "bf16": bf16,
        "created": now(),
        "shared_baseline": bool(baseline_from),
    }
    (folder / "run.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    split_reference(task, folder / "reference", feedback_share)
    if baseline_from:
        src = Path(baseline_from) / "rounds" / "r00"
        dst = folder / "rounds" / "r00"
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("checkpoints"))
    run = Run(folder)
    run.log("init", **settings)
    return run


def split_reference(task: Task, out: Path, feedback_share: float = 0.67) -> dict:
    """Split the task's calibration file into feedback and threshold files, and copy the report file.

    The split is by a hash of the id, so it does not depend on the order of the file.
    """
    import hashlib

    out.mkdir(parents=True, exist_ok=True)
    calib = read_jsonl(task.path("calibration"))
    key = calib["id"].map(
        lambda i: int(hashlib.sha256(str(i).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    )
    write_jsonl(calib[key < feedback_share], out / "feedback.jsonl")
    write_jsonl(calib[key >= feedback_share], out / "threshold.jsonl")
    shutil.copy(task.path("report"), out / "report.jsonl")
    return {
        "feedback": int((key < feedback_share).sum()),
        "threshold": int((key >= feedback_share).sum()),
    }
