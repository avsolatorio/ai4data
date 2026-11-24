"""Large models asked directly on the survey-style test set (1,000 answers written by Gemini 3.8 Flash from
the report job titles).  python direct.py NAME

Three measurements per model, each resumable:
  one    one code per answer, 20 answers per request (the setting of the job-title baseline of 2026-11-03)
  five   five codes per answer, most likely first, 20 answers per request
  single one answer per request, one request at a time, first 50 answers: seconds per answer
The connection comes from the environment: SYNTH_BASE_URL, SYNTH_API_KEY, SYNTH_MODEL and SYNTH_EXTRA_BODY."""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pandas as pd
from openai import OpenAI

from common import OUT, TASK, read_jsonl
from ai4data.small_ai import baseline
from ai4data.small_ai.generate import call

name = sys.argv[1]
G = OUT / 'direct'
G.mkdir(exist_ok=True)
task = replace(TASK, statement='Assign the ISCO-08 unit group (four-digit code) to an answer to the survey question '
                               '"What kind of work do you do in your main job?".')
docs = pd.DataFrame(read_jsonl(OUT / 'test_survey/examples.jsonl'))
gold = dict(zip(docs['id'], docs['labels'].map(lambda x: x[0])))
client = OpenAI(base_url=os.environ['SYNTH_BASE_URL'], api_key=os.environ['SYNTH_API_KEY'], timeout=600)
model = os.environ['SYNTH_MODEL']
workers = int(os.environ.get('WORKERS', 8))
setting = os.environ.get('BASELINE_TEMPERATURE', '0')
temperature = None if setting.lower() == 'none' else float(setting)


def ask(batch, top):
    """Codes for each answer of the batch, most likely first; and the raw record of the request."""
    labels = '\n'.join(f'- {task.labels[x].prompt_line(False)}' for x in task.leaves)
    want = 'the one code that applies' if top == 1 else f'the {top} most likely codes, most likely first'
    prompt = baseline.PROMPT.format(statement=task.statement, labels=labels, cardinality=want,
                                    texts='\n'.join(f'[{i}] {t}' for i, t in enumerate(batch['text'])))
    item = {'type': 'object', 'additionalProperties': False, 'required': ['index', 'labels'],
            'properties': {'index': {'type': 'integer'},
                           'labels': {'type': 'array', 'items': {'type': 'string', 'enum': list(task.leaves)},
                                      'minItems': 1, 'maxItems': top}}}
    schema = {'name': 'labels', 'strict': True, 'schema': {'type': 'object', 'additionalProperties': False,
              'required': ['items'], 'properties': {'items': {'type': 'array', 'items': item}}}}
    raw, choice = call(client, model, prompt, schema, temperature, max_tokens=16384)
    chosen = {}
    if choice.finish_reason == 'stop':
        try:
            for it in json.loads(raw['content']).get('items', []):
                chosen[it.get('index')] = [x for x in it.get('labels', []) if x in task.labels][:top]
        except json.JSONDecodeError:
            pass
    raw['finish_reason'] = choice.finish_reason
    return [{'id': i, 'codes': chosen.get(k, [])} for k, i in enumerate(batch['id'])], raw


def run(kind, top):
    path = G / f'{name}_{kind}.json'
    if path.exists():
        return json.loads(path.read_text())
    batches = [docs.iloc[i:i + 20] for i in range(0, len(docs), 20)]
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda b: ask(b, top), batches))
    rows = [r for out, _ in results for r in out]
    hit = {k: sum(gold[r['id']] in r['codes'][:k] for r in rows) / len(rows) for k in (1, 3, 5) if k <= top}
    out = {'model': model, 'kind': kind, 'answers': len(rows), 'unlabelled': sum(not r['codes'] for r in rows),
           'fewer_than_asked': sum(len(r['codes']) < top for r in rows),
           'finish': pd.Series([raw['finish_reason'] for _, raw in results]).value_counts().to_dict(),
           'seconds': round(time.time() - started, 1), 'workers': workers,
           'prompt_tokens': sum(raw['usage'].get('prompt_tokens') or 0 for _, raw in results),
           'completion_tokens': sum(raw['usage'].get('completion_tokens') or 0 for _, raw in results),
           **{f'top{k}': round(v, 4) for k, v in hit.items()}, 'rows': rows}
    path.write_text(json.dumps(out))
    return out


def single():
    path = G / f'{name}_single.json'
    if path.exists():
        return json.loads(path.read_text())
    times, right, tokens, bad = [], 0, [], 0
    for k in range(52):   # two requests to warm the connection and the prefix cache, then 50 measured
        rows, raw = ask(docs.iloc[k:k + 1], 1)
        if k < 2:
            continue
        times.append(raw['seconds'])
        tokens.append((raw['usage'].get('prompt_tokens') or 0, raw['usage'].get('completion_tokens') or 0))
        bad += not rows[0]['codes']
        right += gold[rows[0]['id']] in rows[0]['codes']
    s = pd.Series(times)
    out = {'model': model, 'requests': len(times), 'mean_seconds': round(s.mean(), 3), 'median_seconds': round(s.median(), 3),
           'min_seconds': round(s.min(), 3), 'max_seconds': round(s.max(), 3), 'unlabelled': bad, 'top1': right / len(times),
           'prompt_tokens_mean': round(sum(t[0] for t in tokens) / len(tokens)), 'completion_tokens_mean': round(sum(t[1] for t in tokens) / len(tokens), 1)}
    path.write_text(json.dumps(out))
    return out


for kind, top in (('one', 1), ('five', 5)):
    r = run(kind, top)
    print(name, kind, {k: v for k, v in r.items() if k != 'rows'}, flush=True)
print(name, 'single', single(), flush=True)
