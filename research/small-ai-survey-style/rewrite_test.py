"""Survey-style test set: each of the 1,000 report job titles rewritten as one survey answer by a
second large model (Gemini 3.8 Flash on OpenRouter), with a prompt that differs from the training
prompt. No answer is filtered. Resumable.  python rewrite_test.py [LIMIT]"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pandas as pd
from openai import OpenAI

from common import OUT, RUN, TASK, read_jsonl
from ai4data.small_ai import generate as gen

PROMPT = """In a labour force survey the interviewer asks "What kind of work do you do in your main job?" and writes down what the respondent says.

The respondent's job, as an employer would advertise it: {title}
Occupation group of this job: {name}

Write the answer this respondent would give, as spoken: between three and twenty words that describe the work. The answer must not repeat the advertised job title word for word."""
SCHEMA = {'name': 'answer', 'strict': True, 'schema': {'type': 'object', 'additionalProperties': False,
          'required': ['answer'], 'properties': {'answer': {'type': 'string'}}}}
G = OUT / 'test_survey'
G.mkdir(parents=True, exist_ok=True)
report = pd.read_json(RUN / 'reference/report.jsonl', lines=True, dtype=False)
done = {r['request_id'] for r in read_jsonl(G / 'raw.jsonl')}
todo = report[~report['id'].isin(done)]
if len(sys.argv) > 1:
    todo = todo.head(int(sys.argv[1]))
client = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=os.environ['OPENROUTER_API_KEY'], timeout=300)
lock, failed = Lock(), []


def work(row):
    code = row['labels'][0]
    try:
        raw, choice = gen.call(client, 'google/gemini-3.8-flash', PROMPT.format(title=row['text'], name=TASK.labels[code].name),
                               SCHEMA, 1.0, {'provider': {'require_parameters': True}}, 4096)
        raw.update(role='writer', request_id=row['id'])
        if choice.finish_reason != 'stop':
            failed.append((row['id'], choice.finish_reason)); return
        answer = ' '.join(str(json.loads(raw['content'])['answer']).split())
    except Exception as error:  # noqa: BLE001
        failed.append((row['id'], str(error)[:200])); return
    with lock:
        with (G / 'raw.jsonl').open('a', encoding='utf-8') as h:
            h.write(json.dumps(raw, ensure_ascii=False) + '\n')
        with (G / 'examples.jsonl').open('a', encoding='utf-8') as h:
            h.write(json.dumps({'id': row['id'], 'text': answer, 'labels': [code], 'title': row['text']}, ensure_ascii=False) + '\n')


with ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(work, todo.to_dict('records')))
rows = read_jsonl(G / 'examples.jsonl')
print('sent', len(todo), 'failed', len(failed), failed[:3], 'answers', len(rows), 'usage', gen.usage(G / 'raw.jsonl'))
