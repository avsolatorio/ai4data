# ai4data.small_ai

`ai4data.small_ai` trains small text classifiers on examples written by a
large language model, for labelling tasks described in a task folder. The
same steps serve topic tagging of documents, occupation coding to ISCO-08,
and expenditure coding to COICOP. The method and its measured results are
described in the cookbook *Practical Guide to Small AI from Synthetic
Data*; this page describes the package.

## Steps

| Step | Module | What it does |
|---|---|---|
| Task | `task.py` | Reads `task.json` and the label file: the classification, the unit of text, one label or several, the code levels, the generation settings, the reference files, and which steps may send real text to an outside service. |
| Plan | `plan.py` | Balanced plans that offer every label about equally often, optionally grouped with sibling codes, and targeted plans from the feedback loop. |
| Generation | `generate.py` | A writer model writes labelled examples for each request, restricted by a JSON schema to the request's labels. A labeller model labels the same texts from a wider list of candidates, and an example is kept when the two agree. |
| Checks | `check.py` | Unknown labels, repeated texts, several labels in a single-label task, labeller disagreement, and the cleaned set. |
| Model | `model.py` | Split by request, fine-tuning of a small encoder (DeBERTa-v3-small by default), and scoring of documents or short texts. |
| Evaluation | `evaluate.py` | Scores on real reference labels: micro and macro F1 for multi-label tasks; accuracy by code level, top-3 accuracy, macro F1, and accuracy on the most confident predictions for single-label tasks. The gap report for the feedback loop. |
| Feedback loop | `loop.py` | Rounds of targeted generation, training, and measurement on a feedback set, with a control arm, acceptance of rounds, and a final score on a report set that the rounds never read. |
| Orchestrator | `orchestrator/` | A pi extension that gives an agent the tools of the loop and no others. |

## Task folder

A task folder holds `task.json`, a label file, and a `data/` folder with
the real reference files, which are built by the task's own script and
are not committed. The label file has the columns `id`, `name`, `parent`,
`description`, `includes`, `excludes`, and `aliases`. The main fields of
`task.json` are:

| Field | Meaning |
|---|---|
| `statement` | The task as a curator or coder would state it; the labeller sees it. |
| `multi_label` | `true` when a text can take several labels. |
| `levels` | Code lengths of the hierarchy, for example `[1, 2, 3, 4]` for ISCO-08. |
| `generation` | The kind of text, rules for the writer, labels per request, examples per request, whether to group sibling codes, and whether the labeller checks the examples. |
| `documents` | How real texts are read: `whole`, or `sentence` with an aggregation (`mean` or `max`) for long documents. |
| `reference` | The calibration and report files and their source. |
| `privacy.real_text_to_external` | Per step, whether real reference text may be sent to an outside service. With `orchestrator: false`, the orchestrator cannot see real texts. |

The tasks in `tasks/` are `wb_topics` (World Bank topics for working
papers), `isco` (occupations to ISCO-08), and `coicop` (household
expenditure items to COICOP 2018).

## Command line

```bash
python -m ai4data.small_ai task TASK
python -m ai4data.small_ai generate TASK plan.jsonl gen/
python -m ai4data.small_ai check TASK gen/examples.jsonl --out clean.jsonl
python -m ai4data.small_ai train TASK clean.jsonl model/
python -m ai4data.small_ai score TASK model/ texts.jsonl scores.jsonl
python -m ai4data.small_ai evaluate TASK reference.jsonl scores.jsonl
```

The generation steps read `SYNTH_BASE_URL`, `SYNTH_API_KEY`, `SYNTH_MODEL`,
and optionally `SYNTH_LABEL_MODEL` and `SYNTH_EXTRA_BODY`. Any server with
the OpenAI chat completions interface works, including a local vLLM
server for tasks whose data may not leave the organization.

## Feedback loop

```bash
# round 0 is shared by the two arms
python -m ai4data.small_ai init runs/isco-targeted --task tasks/isco --arm targeted
python -m ai4data.small_ai round runs/isco-targeted
python -m ai4data.small_ai init runs/isco-control --task tasks/isco --arm control \
    --baseline-from runs/isco-targeted

# the orchestrator plans and runs the targeted rounds
src/ai4data/small_ai/orchestrator/run.sh runs/isco-targeted

# the control arm runs the same number of rounds without a plan
for k in 1 2 3 4 5; do python -m ai4data.small_ai round runs/isco-control; done

# each arm is trained with two seeds on its accepted data and scored once on the report set
python -m ai4data.small_ai final runs/isco-targeted --seeds 42,1
python -m ai4data.small_ai final runs/isco-control --seeds 42,1
```

The orchestrator runs in pi with its built-in tools switched off: no
shell, no file reading or writing, no MCP servers, skills, or context
files. Its seven tools run commands of this package on the run folder
that the operator names. `run.sh` records the event stream in
`orchestrator.jsonl`, and every round keeps the orchestrator's plan, its
notes, the generated data, the model's scores, and the decision. pi
itself runs with the permissions of the account that starts it, and its
documentation recommends a container for untrusted work.

## Installation

```bash
uv pip install "ai4data[small-ai]"
cd src/ai4data/small_ai/orchestrator && npm ci    # only for the orchestrator
```

The training libraries are pinned: with transformers 5.20 and torch 2.14,
DeBERTa-v3 did not train (the loss became NaN within ten steps).
