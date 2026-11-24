"""Write the training record of slide s29 of session 02 from the three recorded runs.  python learn_deck.py

Reads learn/seed<N>/record.json (written by learn_record.py) and writes slides/src/learn.json: the mean scores of
the three runs at each recorded point of training, the scores of the saved models, and the first code of the
seed-42 model at each point for the six test answers of slide s31."""
import json

from common import OUT, SLIDES, TASK

SEEDS = [42, 1, 2]
ANSWERS = ['hahu-03772', 'hahu-03915', 'hahu-03045', 'hahu-03548', 'hahu-03974', 'hahu-03880']  # the test answers of s31
BATCH = int(TASK.model.get('batch', 32))

recs = {s: json.loads((OUT / f'learn/seed{s}/record.json').read_text()) for s in SEEDS}
ref = recs[42]
steps = [p['step'] for p in ref['points']]
assert all([p['step'] for p in r['points']] == steps and r['ids'] == ref['ids'] for r in recs.values())
assert steps[-1] is None and steps[0] == 0

# The saved model of each run must give the scores recorded when the arm was first trained.
for s, r in recs.items():
    old = json.loads((OUT / f'arms/all/seed{s}/scores.json').read_text())['tests']['survey_style']
    new = r['points'][-1]
    assert abs(old['top1'] - new['top1']) < 1e-9 and abs(old['top5'] - new['top5']) < 1e-9, (s, old, new)

sc = lambda t: t[0] + t[1:].lower()
name = lambda c: sc(TASK.labels[c].name)
mean = lambda k, i: round(sum(r['points'][i][k] for r in recs.values()) / len(recs), 3)
per_pass, n = ref['steps_per_pass'], ref['train_examples']
texts = lambda step: step * BATCH - (step // per_pass) * (per_pass * BATCH - n)
test = {r['id']: r for r in map(json.loads, open(OUT / 'test_survey/examples.jsonl'))}
row = {i: k for k, i in enumerate(ref['ids'])}

out = {
    'trainTexts': n, 'testAnswers': len(ref['ids']), 'runs': len(recs),
    'points': [{'passes': ref['points'][i]['passes'], 'texts': texts(steps[i]), 'top1': mean('top1', i), 'top5': mean('top5', i)}
               for i in range(len(steps) - 1)],
    'saved': {'top1': mean('top1', -1), 'top5': mean('top5', -1),
              'passes': sorted(int(r['best_checkpoint'].rsplit('-', 1)[1]) // per_pass for r in recs.values())},
    'answers': [{'text': test[i]['text'], 'code': test[i]['labels'][0], 'name': name(test[i]['labels'][0]),
                 'first': [{'code': p['top3'][row[i]][0], 'name': name(p['top3'][row[i]][0])} for p in ref['points']]}
                for i in ANSWERS],
}
open(SLIDES / 'learn.json', 'w').write(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
for p in out['points']: print(p)
print(out['saved'])
for a in out['answers']: print(a['text'][:50], a['code'], [f['code'] for f in a['first']])
