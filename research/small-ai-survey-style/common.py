"""Shared pieces of the survey-style experiment on ISCO-08 (session 02).

Paths come from the environment. SMALL_AI_RUNS is the folder of the run folders (default: runs/ in the
repository); it holds isco-l-targeted (the harness run on the ISCO task) and isco-survey (the outputs of these
scripts). HAHU_TITLES is the file occupations_training_set.csv of the dataset tabiya/occupation_titles_esco.
SESSION02_SLIDES is the folder slides/src of training session 02."""
import json, os, sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from ai4data.small_ai.task import load_task  # noqa: E402

RUNS = Path(os.environ.get('SMALL_AI_RUNS', REPO / 'runs'))
OUT = RUNS / 'isco-survey'
HAHU = Path(os.environ.get('HAHU_TITLES', RUNS / 'refdata/occupations_training_set.csv'))
RUN = RUNS / 'isco-l-targeted'
SLIDES = Path(os.environ.get('SESSION02_SLIDES', REPO / 'training/session-02-small-ai-synthetic-data/slides/src'))
TASK = load_task(REPO / 'src/ai4data/small_ai/tasks/isco')
SURVEY_TASK = replace(TASK, generation={**TASK.generation, 'items': 'answers to a survey question on the main job', 'item': 'answer'})


def frame() -> pd.DataFrame:
    """All job-board titles with one unit group, in the order of tasks/isco/build.py (seed 0)."""
    table = pd.read_csv(HAHU, dtype=str)
    table['isco'] = table['esco_code'].str.extract(r'^(\d{4})')[0]
    codes = table.dropna(subset=['isco']).groupby('title')['isco'].agg(lambda s: sorted(set(s)))
    single = codes[codes.map(len) == 1].map(lambda s: s[0])
    single = single[single.isin(set(TASK.leaves))]
    f = pd.DataFrame({'text': single.index.str.strip(), 'labels': single.map(lambda c: [c]).values})
    f = f[f['text'].str.len() > 1].drop_duplicates('text').sample(frac=1.0, random_state=0)
    f.insert(0, 'id', [f'hahu-{i:05d}' for i in range(len(f))])
    return f.reset_index(drop=True)


def pool() -> pd.DataFrame:
    """Titles outside the calibration (first 3,000) and report (next 1,000) files."""
    f = frame()
    report = pd.read_json(RUN / 'reference/report.jsonl', lines=True, dtype=False)
    assert f.iloc[3000:4000]['text'].tolist() == report['text'].tolist(), 'report file does not match the rebuilt order'
    return f.iloc[4000:].reset_index(drop=True)


def read_jsonl(path) -> list:
    path = Path(path)
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
