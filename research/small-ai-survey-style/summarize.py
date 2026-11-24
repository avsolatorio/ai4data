"""Print the mean and range of the scores of each arm over its seeds.  python summarize.py"""
import json, statistics as st

from common import OUT

root = OUT / 'arms'
for arm in ['defs', 'titles', 'survey', 'titles_survey', 'all']:
    runs = [json.loads(p.read_text()) for p in sorted((root / arm).glob('seed*/scores.json'))]
    if not runs: print(arm, 'no runs'); continue
    print(f"{arm}: {len(runs)} runs, seeds {[r['seed'] for r in runs]}, examples {runs[0]['examples']}, train {runs[0].get('train_examples')}, seconds {[r['train_seconds'] for r in runs]}, epochs {[r['epochs_run'] for r in runs]}")
    for test in ['report_titles', 'survey_style', 'handwritten']:
        row = []
        for k in ['top1', 'top3', 'top5', 'top10']:
            v = [r['tests'][test][k] for r in runs]
            row.append(f"{k} {st.mean(v):.3f} [{min(v):.3f}-{max(v):.3f}]")
        print('   ', test.ljust(14), ' | '.join(row))
