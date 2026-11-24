"""Train one arm of the survey-style experiment and score it on three test sets.  python arms.py ARM SEED

Arms: defs (synthetic titles written from the ISCO-08 definitions, the 15,865 cases of isco-l-targeted),
titles (labelled job-board titles outside the test files), survey (verified survey-style rewrites of
those titles), titles_survey, all (defs + titles + survey).
Tests: the 1,000 report job titles; the same titles rewritten as survey answers by Gemini 3.8 Flash;
the hand-written survey-style responses of the deck that have an expected code."""
import json, shutil, sys

import pandas as pd

from common import OUT, RUN, TASK, pool, read_jsonl
from ai4data.small_ai import check as checks, loop, model as mdl

arm, seed = sys.argv[1], int(sys.argv[2])
out = OUT / 'arms' / arm / f'seed{seed}'
if (out / 'scores.json').exists():
    sys.exit(0)
out.mkdir(parents=True, exist_ok=True)


def defs():
    run = loop.Run(str(RUN))
    clean = checks.check(run.examples(run.accepted()), run.task, require_verified=True)[1]
    return clean[['text', 'labels', 'request_id']].assign(source='defs')


def titles():
    p = pool()
    return pd.DataFrame({'text': p['text'], 'labels': p['labels'], 'request_id': p['id'], 'source': 'titles'})


def survey():
    rows = [r for r in read_jsonl(OUT / 'rewrites/examples.jsonl') if r['verified'] is True]
    f = pd.DataFrame(rows)[['text', 'labels', 'request_id']].assign(source='survey')
    return f.drop_duplicates('text')


PARTS = {'defs': [defs], 'titles': [titles], 'survey': [survey], 'titles_survey': [titles, survey], 'all': [defs, titles, survey]}
data = pd.concat([f() for f in PARTS[arm]], ignore_index=True).drop_duplicates('text').reset_index(drop=True)
splits = mdl.split(data, seed=42)
config = mdl.train(TASK, splits, out / 'model', seed=seed)

report = pd.read_json(RUN / 'reference/report.jsonl', lines=True, dtype=False)
test_survey = pd.DataFrame(read_jsonl(OUT / 'test_survey/examples.jsonl'))
hand = pd.DataFrame(json.loads((OUT / 'handwritten.json').read_text()))
tests = {'report_titles': report, 'survey_style': test_survey, 'handwritten': hand}
scores = {'arm': arm, 'seed': seed, 'examples': len(data), 'by_source': data['source'].value_counts().to_dict(),
          'train_examples': config['train_examples'], 'train_seconds': config['train_seconds'],
          'epochs_run': config['epochs_run'], 'synthetic_test': config['scores']['test'], 'tests': {}}
for name, docs in tests.items():
    rs, _ = mdl.score(TASK, out / 'model', docs)
    gold = dict(zip(docs['id'].astype(str), docs['labels']))
    hit = {1: 0, 3: 0, 5: 0, 10: 0}
    ranks = {}
    for r in rs:
        order = sorted(r['scores'], key=lambda k: -r['scores'][k])
        ranks[r['id']] = order.index(gold[r['id']][0]) + 1
        for k in hit:
            hit[k] += gold[r['id']][0] in order[:k]
    scores['tests'][name] = {'n': len(rs), **{f'top{k}': round(v / len(rs), 4) for k, v in hit.items()}}
    (out / f'ranks_{name}.json').write_text(json.dumps(ranks))
(out / 'scores.json').write_text(json.dumps(scores, indent=1))
if seed != 42:
    shutil.rmtree(out / 'model', ignore_errors=True)
print(json.dumps(scores))
