"""Record the answers of the small model during training.  python learn_record.py SEED

Trains the `all` arm exactly as arms.py does (same texts, split, seed and settings) and scores the
1,000 survey-style test answers at fixed points of the training run: before the first step, at
fractions of a pass over the training texts, and with the saved model at the end. The record holds
the top 1 and top 5 scores and the three highest-scoring codes of every test answer at each point."""
import json, shutil, sys

import numpy as np
import pandas as pd
import torch
import transformers

from common import OUT, RUN, TASK, pool, read_jsonl
from ai4data.small_ai import check as checks, loop, model as mdl

seed = int(sys.argv[1])
out = OUT / 'learn' / f'seed{seed}'
if (out / 'record.json').exists():
    sys.exit(0)
out.mkdir(parents=True, exist_ok=True)
PASSES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6]

run = loop.Run(str(RUN))
defs = checks.check(run.examples(run.accepted()), run.task, require_verified=True)[1][['text', 'labels', 'request_id']].assign(source='defs')
p = pool()
titles = pd.DataFrame({'text': p['text'], 'labels': p['labels'], 'request_id': p['id'], 'source': 'titles'})
rows = [r for r in read_jsonl(OUT / 'rewrites/examples.jsonl') if r['verified'] is True]
survey = pd.DataFrame(rows)[['text', 'labels', 'request_id']].assign(source='survey').drop_duplicates('text')
data = pd.concat([defs, titles, survey], ignore_index=True).drop_duplicates('text').reset_index(drop=True)
splits = mdl.split(data, seed=42)

test = pd.DataFrame(read_jsonl(OUT / 'test_survey/examples.jsonl'))
gold = [ls[0] for ls in test['labels']]
labels = list(TASK.leaves)
max_length = int(TASK.documents.get('max_length', TASK.model.get('max_length', 128)))
steps_per_pass = int(np.ceil(len(splits['train']) / int(TASK.model.get('batch', 32))))
at = {int(round(f * steps_per_pass)): f for f in PASSES}
points = []


def rank(model, tokenizer):
    was_training = model.training
    model.eval()
    order = []
    with torch.no_grad():
        for i in range(0, len(test), 128):
            enc = tokenizer(test['text'].iloc[i:i + 128].tolist(), truncation=True, max_length=max_length, padding=True, return_tensors='pt').to(model.device)
            order.append(model(**enc).logits.float().topk(5, dim=1).indices.cpu().numpy())
    model.train(was_training)
    return [[labels[j] for j in row] for row in np.concatenate(order)]


def point(step, passes, top):
    hit1 = sum(g == t[0] for g, t in zip(gold, top))
    hit5 = sum(g in t for g, t in zip(gold, top))
    points.append({'step': step, 'passes': passes, 'texts_read': None if step is None else step * int(TASK.model.get('batch', 32)),
                   'top1': hit1 / len(test), 'top5': hit5 / len(test), 'top3': [t[:3] for t in top]})
    print(json.dumps({k: v for k, v in points[-1].items() if k != 'top3'}), flush=True)


class Record(transformers.TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_train_begin(self, args, state, control, model=None, **kw):
        point(0, 0, rank(model, self.tokenizer))

    def on_step_end(self, args, state, control, model=None, **kw):
        if state.global_step in at:
            point(state.global_step, at[state.global_step], rank(model, self.tokenizer))


# The lazy transformers module ignores a replaced `Trainer` attribute on `from transformers import Trainer`,
# so the callback is added in the constructor of the class itself.
_init = transformers.Trainer.__init__


def init(self, *a, **kw):
    kw['callbacks'] = list(kw.get('callbacks') or []) + [Record(kw['processing_class'])]
    _init(self, *a, **kw)


transformers.Trainer.__init__ = init
config = mdl.train(TASK, splits, out / 'model', seed=seed)
rs, _ = mdl.score(TASK, out / 'model', test)
assert [r['id'] for r in rs] == test['id'].astype(str).tolist()
top = [sorted(r['scores'], key=lambda k: -r['scores'][k])[:5] for r in rs]
assert len(points) > 1, 'the callback did not record any point'
point(None, config['epochs_run'], top)
record = {'seed': seed, 'train_examples': config['train_examples'], 'steps_per_pass': steps_per_pass, 'epochs_run': config['epochs_run'],
          'best_checkpoint': config['best_checkpoint'], 'train_seconds': config['train_seconds'], 'ids': test['id'].astype(str).tolist(), 'points': points}
(out / 'record.json').write_text(json.dumps(record))
shutil.rmtree(out / 'model', ignore_errors=True)
