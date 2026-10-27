"""Command line of the harness: python -m ai4data.small_ai <command> ...

Task and data steps:

    task TASK                         check a task folder and print its labels and reference files
    generate TASK PLAN OUT            run the writer and labeller over a request plan
    check TASK EXAMPLES [--out CLEAN] print the check report and write the cleaned set
    train TASK CLEAN MODEL            split by request and train
    score TASK MODEL DOCS OUT         score documents or short texts
    evaluate TASK REFERENCE SCORES    compare with reference labels
    baseline TASK TEXTS OUT           a large model labels the texts directly, for comparison

Feedback loop (one run folder per arm):

    init RUN --task TASK --arm targeted|control ...
    round RUN [--plan PLAN.json]      run the next round (the targeted arm needs a plan after round 0)
    status RUN                        rounds, scores, and spending, as JSON
    gap RUN [--round K]               the gap report of a round, as JSON
    labels RUN --query Q              labels whose code or name matches, with definitions
    examples RUN --label X            generated examples of a label from the accepted rounds
    errors RUN --label X              feedback texts the model got wrong for a label (if the task allows it)
    stop RUN --reason TEXT            end the rounds
    final RUN --seeds 42,1            train on the accepted data and score the report file once
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


def dump(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def main(argv: list[str] | None = None) -> int:
    from . import check as checks
    from . import evaluate as ev
    from . import generate as gen
    from . import loop
    from . import model as mdl
    from .task import load_task

    p = argparse.ArgumentParser(
        prog="python -m ai4data.small_ai", description=__doc__.splitlines()[0]
    )
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("task")
    a.add_argument("task")
    a = sub.add_parser("generate")
    a.add_argument("task")
    a.add_argument("plan")
    a.add_argument("out")
    a.add_argument("--workers", type=int, default=16)
    a.add_argument("--limit", type=int)
    a.add_argument("--no-verify", action="store_true")
    a = sub.add_parser("check")
    a.add_argument("task")
    a.add_argument("examples")
    a.add_argument("--out")
    a = sub.add_parser("train")
    a.add_argument("task")
    a.add_argument("clean")
    a.add_argument("model")
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--bf16", action="store_true")
    a = sub.add_parser("score")
    a.add_argument("task")
    a.add_argument("model")
    a.add_argument("documents")
    a.add_argument("out")
    a.add_argument("--device")
    a = sub.add_parser("evaluate")
    a.add_argument("task")
    a.add_argument("reference")
    a.add_argument("scores")
    a.add_argument("--threshold", type=float)
    a.add_argument("--top", type=int, default=5)
    a.add_argument("--sweep", action="store_true")
    a = sub.add_parser("baseline")
    a.add_argument("task")
    a.add_argument("texts")
    a.add_argument("out")
    a.add_argument("--batch", type=int, default=20)
    a.add_argument("--workers", type=int, default=8)
    a = sub.add_parser("init")
    a.add_argument("run")
    a.add_argument("--task", required=True)
    a.add_argument("--arm", choices=["targeted", "control"], required=True)
    a.add_argument("--baseline-requests", type=int, default=1000)
    a.add_argument("--round-requests", type=int, default=150)
    a.add_argument("--rounds", type=int, default=5)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument(
        "--price",
        default="0.15,0.60",
        help="dollars per million input and output tokens",
    )
    a.add_argument("--budget", type=float)
    a.add_argument("--baseline-from")
    a.add_argument("--tolerance", type=float, default=0.01)
    a.add_argument("--bf16", action="store_true")
    a = sub.add_parser("round")
    a.add_argument("run")
    a.add_argument("--plan")
    a.add_argument("--workers", type=int, default=32)
    a = sub.add_parser("finish-round")
    a.add_argument("run")
    a.add_argument("round", type=int)
    for name in ("status",):
        sub.add_parser(name).add_argument("run")
    a = sub.add_parser("gap")
    a.add_argument("run")
    a.add_argument("--round", type=int)
    a = sub.add_parser("labels")
    a.add_argument("run")
    a.add_argument("--query", required=True)
    a = sub.add_parser("examples")
    a.add_argument("run")
    a.add_argument("--label", required=True)
    a.add_argument("--n", type=int, default=10)
    a = sub.add_parser("errors")
    a.add_argument("run")
    a.add_argument("--label", required=True)
    a.add_argument("--n", type=int, default=10)
    a = sub.add_parser("stop")
    a.add_argument("run")
    a.add_argument("--reason", required=True)
    a = sub.add_parser("final")
    a.add_argument("run")
    a.add_argument("--seeds", default="42")
    a.add_argument(
        "--rounds", help="comma-separated rounds to use, default: the accepted rounds"
    )
    a.add_argument("--tag", default="final")
    args = p.parse_args(argv)

    if args.command == "task":
        task = load_task(args.task)
        dump(
            {
                "id": task.id,
                "title": task.title,
                "multi_label": task.multi_label,
                "labels": len(task.labels),
                "leaves": len(task.leaves),
                "levels": task.levels,
                "first_leaves": [task.labels[x].prompt_line() for x in task.leaves[:3]],
                "reference": {
                    k: str(task.path(k))
                    for k in ("calibration", "report")
                    if k in task.reference
                },
                "privacy": task.privacy,
            }
        )
        return 0
    if args.command == "generate":
        task = load_task(args.task)
        plan = pd.read_json(args.plan, lines=True).to_dict("records")
        for r in plan:
            r.setdefault("mode", "positive")
            r.setdefault("note", "")
        result = gen.generate(
            task,
            plan,
            Path(args.out),
            int(task.generation.get("per_call", 20)),
            args.workers,
            verify=not args.no_verify,
            limit=args.limit,
        )
        dump(result | {"usage": gen.usage(Path(args.out) / "raw.jsonl")})
        return 1 if result["failed"] else 0
    if args.command == "check":
        task = load_task(args.task)
        report, clean, unknown = checks.check(
            pd.read_json(args.examples, lines=True), task
        )
        print(checks.format_report(report))
        if unknown:
            print(
                "most frequent unknown labels: "
                + ", ".join(f"{n} ({c})" for n, c in unknown.most_common(6))
            )
        if args.out:
            loop.write_jsonl(clean, Path(args.out))
            print(f"wrote {args.out}")
        return 0
    if args.command == "train":
        task = load_task(args.task)
        clean = pd.read_json(args.clean, lines=True)
        config = mdl.train(
            task,
            mdl.split(clean, seed=42),
            Path(args.model),
            seed=args.seed,
            bf16=args.bf16,
        )
        dump(
            {
                k: config[k]
                for k in ("threshold", "scores", "train_seconds", "train_examples")
            }
        )
        return 0
    if args.command == "score":
        task = load_task(args.task)
        scores, timing = mdl.score(
            task,
            Path(args.model),
            pd.read_json(args.documents, lines=True, dtype={"id": str}),
            args.device,
        )
        loop.write_jsonl(scores, Path(args.out))
        dump(timing)
        return 0
    if args.command == "evaluate":
        task = load_task(args.task)
        gold = pd.read_json(args.reference, lines=True, dtype={"id": str})
        pred = pd.read_json(args.scores, lines=True, dtype={"id": str}).to_dict(
            "records"
        )
        if args.sweep:
            for t, f in ev.sweep(gold, pred, args.top):
                print(f"threshold {t:<6g} micro F1 {f:.3f}")
            return 0
        print(ev.summary(task, ev.evaluate(task, gold, pred, args.threshold, args.top)))
        return 0
    if args.command == "baseline":
        from . import baseline

        task = load_task(args.task)
        if not task.allows_external("baseline"):
            print(
                "this task does not allow real texts to be sent to an outside model (privacy.real_text_to_external.baseline)"
            )
            return 1
        texts = pd.read_json(args.texts, lines=True, dtype={"id": str})
        records, info = baseline.run(task, texts, args.batch, args.workers)
        loop.write_jsonl(records, Path(args.out))
        dump(info)
        return 0
    if args.command == "init":
        price = tuple(float(x) for x in args.price.split(","))
        run = loop.init_run(
            args.task,
            args.run,
            args.arm,
            args.baseline_requests,
            args.round_requests,
            args.rounds,
            args.seed,
            price,
            args.budget,
            args.baseline_from,
            args.tolerance,
            args.bf16,
        )
        dump(run.status())
        return 0

    run = loop.Run(args.run)
    if args.command == "round":
        items, note = None, ""
        if args.plan:
            spec = json.loads(Path(args.plan).read_text(encoding="utf-8"))
            items, note = spec["items"], spec.get("note", "")
        out = run.run_round(items, note, workers=args.workers)
        dump({"decision": out["decision"], "summary": out["gap"]["summary"]})
        return 0
    if args.command == "finish-round":
        out = run.finish_round(args.round)
        dump({"decision": out["decision"], "summary": out["gap"]["summary"]})
        return 0
    if args.command == "status":
        dump(run.status())
        return 0
    if args.command == "gap":
        k = args.round if args.round is not None else run.rounds()[-1]
        dump(json.loads((run.round_dir(k) / "gap.json").read_text(encoding="utf-8")))
        return 0
    if args.command == "labels":
        q = args.query.lower()
        hits = [
            x
            for x in run.task.leaves
            if x.lower().startswith(q) or q in run.task.labels[x].name.lower()
        ]
        dump(
            {
                "query": args.query,
                "matches": len(hits),
                "labels": [run.task.labels[x].prompt_line() for x in hits[:25]],
            }
        )
        return 0
    if args.command == "examples":
        data = run.examples(run.accepted())
        rows = data[data["labels"].map(lambda ls: args.label in ls)]
        pick = (
            rows.sample(min(args.n, len(rows)), random_state=0) if len(rows) else rows
        )
        dump(
            {
                "label": args.label,
                "name": run.task.labels[args.label].name,
                "examples_in_accepted_rounds": len(rows),
                "sample": [
                    {"text": t, "labels": ls, "mode": m}
                    for t, ls, m in zip(
                        pick["text"],
                        pick["labels"],
                        pick.get("mode", pd.Series(["positive"] * len(pick))),
                    )
                ],
            }
        )
        return 0
    if args.command == "errors":
        if not run.task.allows_external("orchestrator"):
            dump(
                {
                    "error": "this task does not allow real reference texts to be shown to the orchestrator"
                }
            )
            return 1
        k = run.accepted()[-1]
        feedback = run.reference("feedback")
        scores = loop.read_jsonl(run.round_dir(k) / "feedback_scores.jsonl")
        data = feedback.merge(scores, on="id")
        decision = run.decision(k)
        out = []
        for text, gold, sc in zip(data["text"], data["labels"], data["scores"]):
            pred = (
                ev.choose(sc, decision["threshold"], 5)
                if run.task.multi_label
                else [max(sc, key=sc.get)]
            )
            missed = args.label in gold and args.label not in pred
            false = args.label in pred and args.label not in gold
            if missed or false:
                out.append(
                    {
                        "kind": "missed" if missed else "wrongly assigned",
                        "text": str(text)[:600],
                        "reference": gold,
                        "predicted": pred,
                    }
                )
        random.Random(0).shuffle(out)
        dump(
            {
                "label": args.label,
                "name": run.task.labels[args.label].name,
                "round": k,
                "counts": dict(Counter(o["kind"] for o in out)),
                "sample": out[: args.n],
            }
        )
        return 0
    if args.command == "stop":
        run.stop(args.reason)
        dump(run.status())
        return 0
    if args.command == "final":
        rounds = [int(x) for x in args.rounds.split(",")] if args.rounds else None
        dump(run.final([int(s) for s in args.seeds.split(",")], rounds, args.tag))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
