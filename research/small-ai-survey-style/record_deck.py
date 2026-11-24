"""Record the suggestions of the small occupation model (arm all, seed 42) for the slides of session 02.

Writes deck/record.json: the five best groups for the eight responses of slide s02 and for every survey-style
test answer, with the probability of each.  python record_deck.py"""
import json

import numpy as np
import pandas as pd

from common import OUT, SLIDES, TASK, read_jsonl
from ai4data.small_ai import model as mdl

config, tokenizer, model, device = mdl.load(OUT / 'arms/all/seed42/model', 'cuda')
labels = config['labels']
name = lambda c: TASK.labels[c].name


def top5(texts):
    p = mdl.probabilities(texts, config, tokenizer, model, device, 64)
    order = np.argsort(-p, axis=1)
    return [[{'code': labels[j], 'name': name(labels[j]), 'score': round(float(p[i, j]), 4)} for j in row[:5]]
            for i, row in enumerate(order)], order


responses = json.load(open(SLIDES / 'occupation.json'))['responses']
sug, _ = top5([r['text'] for r in responses])
docs = pd.DataFrame(read_jsonl(OUT / 'test_survey/examples.jsonl'))
tsug, order = top5(docs['text'].tolist())
gold = docs['labels'].map(lambda x: x[0]).tolist()
rank = [[labels[j] for j in row].index(g) + 1 for g, row in zip(gold, order)]
print('label object', vars(TASK.labels[gold[0]]) if hasattr(TASK.labels[gold[0]], '__dict__') else TASK.labels[gold[0]])
print('top1', np.mean([r == 1 for r in rank]), 'top3', np.mean([r <= 3 for r in rank]), 'top5', np.mean([r <= 5 for r in rank]))
out = {'responses': [{**{k: r[k] for k in ('text', 'expected', 'expectedName')}, 'suggestions': s} for r, s in zip(responses, sug)],
       'test': [{'id': i, 'title': t if isinstance(t, str) else None, 'text': x, 'code': g, 'name': name(g), 'rank': r, 'suggestions': s}
                for i, t, x, g, r, s in zip(docs['id'], docs.get('title', [None] * len(docs)), docs['text'], gold, rank, tsug)]}
(OUT / 'deck/record.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
for r in out['responses']:
    print(r['text'], '|', r['expected'], '|', [(s['code'], round(s['score'], 2)) for s in r['suggestions']])
