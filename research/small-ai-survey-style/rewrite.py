"""Rewrite labelled job-board titles as answers to a survey question on the main job.

The writer receives one title with its ISCO-08 unit group and writes three answers in a respondent's
words. The labeller reads each answer without the code and chooses a unit group from the true one, its
siblings, and random others; an answer is verified when the labeller chooses the code of the title.
Resumable: titles already in raw.jsonl are skipped.  python rewrite.py [LIMIT]"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from common import OUT, SURVEY_TASK, TASK, pool, read_jsonl
from ai4data.small_ai import generate as gen

PROMPT = """A household survey asks two questions about work: "What is your main job?" and "What are your main tasks or duties in this job?". The interviewer writes the answer down in the respondent's own words.

A respondent holds this job:
Job title in a job advertisement: {title}
Occupation group: {label}

Write three different answers that this respondent could give.
- Answer 1: a short answer of one to four words, the way many respondents answer.
- Answer 2: the job together with the place of work or what the employer does, in four to twelve words.
- Answer 3: the main tasks in everyday words, in six to twenty words, without the job title.

Rules:
- Use the plain words of a respondent, and avoid the wording of a job advertisement. Do not copy the job title as it is written.
- Write the way an interviewer records an answer: mostly lower case, short, sometimes an incomplete sentence, in the first person or with no subject.
- Leave out the seniority grade, the company name, and anything a respondent would not say.
- Each answer must describe this job, and must fit this occupation group better than any related group."""

SCHEMA = {'name': 'answers', 'strict': True, 'schema': {
    'type': 'object', 'additionalProperties': False, 'required': ['answers'],
    'properties': {'answers': {'type': 'array', 'minItems': 3, 'maxItems': 3, 'items': {'type': 'string'}}}}}

G = OUT / 'rewrites'
G.mkdir(parents=True, exist_ok=True)
titles = pool()
done = {r['request_id'] for r in read_jsonl(G / 'raw.jsonl') if r.get('role') == 'writer'}
todo = titles[~titles['id'].isin(done)]
if len(sys.argv) > 1:
    todo = todo.head(int(sys.argv[1]))
(client, writer, wx), (lclient, labeller, lx) = gen.connection()
lock, failed = Lock(), []


def work(row):
    code = row['labels'][0]
    request = {'request_id': row['id'], 'labels': [code]}
    try:
        raw, choice = gen.call(client, writer, PROMPT.format(title=row['text'], label=TASK.labels[code].prompt_line(True)), SCHEMA, 1.0, wx, 400)
        raw.update(role='writer', request_id=row['id'])
        if choice.finish_reason != 'stop':
            failed.append((row['id'], choice.finish_reason)); return
        answers = [' '.join(str(a).split()) for a in json.loads(raw['content'])['answers']]
        check, verdicts = gen.label_texts(SURVEY_TASK, lclient, labeller, request, answers, lx)
        check.update(role='labeller', request_id=row['id'])
        rows = [{'text': a, 'labels': [code], 'request_id': row['id'], 'title': row['text'], 'style': i + 1,
                 'labeller_labels': v, 'verified': None if v is None else v == [code]}
                for i, (a, v) in enumerate(zip(answers, verdicts or [None] * 3)) if a]
    except Exception as error:  # noqa: BLE001
        failed.append((row['id'], str(error)[:200])); return
    with lock:
        with (G / 'raw.jsonl').open('a', encoding='utf-8') as h:
            h.write(json.dumps(raw, ensure_ascii=False) + '\n' + json.dumps(check, ensure_ascii=False) + '\n')
        with (G / 'examples.jsonl').open('a', encoding='utf-8') as h:
            for r in rows:
                h.write(json.dumps(r, ensure_ascii=False) + '\n')


with ThreadPoolExecutor(max_workers=128) as ex:
    list(ex.map(work, todo.to_dict('records')))
ex_rows = read_jsonl(G / 'examples.jsonl')
print('titles in pool', len(titles), 'sent', len(todo), 'failed', len(failed), failed[:3])
print('answers', len(ex_rows), 'verified', sum(r['verified'] is True for r in ex_rows), 'usage', gen.usage(G / 'raw.jsonl'))
