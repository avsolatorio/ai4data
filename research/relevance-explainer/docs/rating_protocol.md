# Human rating protocol for the relevance explainer

This protocol describes a small study in which catalog or statistical-office staff rate the explainer's answers. It
answers three questions that the automatic evaluation cannot:

1. How often are the verdict and the explanation right, according to people who know the data?
2. Does the answer confidence shown in the app mean what it says when people are the judges? For example, are answers
   shown at 90% right about 90% of the time?
3. How far can the LLM teacher and judge be trusted as stand-ins for people? Every automatic number in the lab
   notebook rests on them.

## Sample

The sample has 300 answers from the deployed app plus 60 answers written by the teacher, 360 items in all.
`scripts/rating_sample.py` draws it from the app's answers (`scripts/app_answers.py`) on three test sets and writes one
row per item. The first draw is in `docs/rating_sample/study1_sheet.csv` (for raters) and `study1_key.csv` (kept by the
study lead):

```
python scripts/rating_sample.py --languages English French Spanish Arabic --teacher 60 --out docs/rating_sample/study1 \
  --source realistic runs/v2g-from-dapt-8ep-dpo/app_answers_realtest.jsonl data/eval_real/itemfirst_nosim_v4 45 \
  --source challenge runs/v2g-from-dapt-8ep-dpo/app_answers_challenge.jsonl data/eval_challenge/itemfirst_nosim_v4 20 \
  --source tables runs/v2g-from-dapt-8ep-dpo/app_answers_tables.jsonl data/eval_tables/itemfirst_nosim 10
```

- **Languages.** English, French, Spanish and Arabic in the first draw; change `--languages` to match the raters.
  Each language has 75 model answers.
- **Sets.** Per language, 45 answers come from the realistic test, 20 from the challenge suite (hard phrasings such as
  negation, look-alike topics and proxies) and 10 from the held-out Eurostat tables. The automatic evaluation found the
  answer confidence overconfident on the last two sets (lab notebook, 2026-10-11 00:21 UTC), so people need to check it
  there.
- **Confidence bands.** Within each set and language, answers are split evenly across three bands of the answer
  confidence: below 0.5, 0.5 to 0.8, and 0.8 or above. This over-samples doubtful answers. Realistic searches in
  English, French and Spanish have few low-confidence answers (3 to 8 per language), so those bands give what they have
  and the rest goes to the other bands. Each row of the key carries a weight, so results can be reweighted to each
  set's real mix.
- **Teacher answers.** 60 items show the teacher's own answer for a realistic-test input instead of the model's.
  Raters are not told which is which. These items measure how often the teacher is right according to people.

With 300 model answers, the share rated correct has a 95% confidence interval of about plus or minus 5.5 points. The
three confidence bands have 75, 114 and 111 answers (low, middle, high), about plus or minus 9 to 11 points.

## Raters

- At least two raters per language, each fluent in the language and familiar with statistical metadata.
- A fifth of the items in each language (about 20 per language) are rated by both raters, to measure agreement.
- Raters see the record's metadata as the model saw it, the search, and the answer. They do not see the answer
  confidence, the teacher's label, or which system wrote the answer.

## What raters judge

| Question | Answer options |
|---|---|
| 1. Is the verdict right for this search and this record? | right; acceptable (a defensible borderline call); wrong |
| 2. If the verdict is wrong, what should it be? | relevant; partly relevant; irrelevant |
| 3. Is every statement in the explanation supported by the record's metadata? | all supported; one minor unsupported detail; a wrong or invented statement |
| 4. Is the explanation clear and in the requested language? | yes; understandable but awkward; no |
| 5. Would this explanation help a user decide whether to open the record? | yes; somewhat; no |
| 6. Comment (optional) | free text, for example the wrong statement |

An answer counts as **fully correct** when the verdict is right or acceptable, every statement is supported, and the
explanation is clear and in the requested language. This mirrors the automatic definition used for the answer
confidence: verdict matches the teacher, faithfulness at least 4, coherence at least 4, consistent, right language, no
invented number.

## Analysis

`scripts/rating_analysis.py` (to be written when ratings come back) reports:

1. **Accuracy by people:** the share of fully correct answers, verdict accuracy and the share with a wrong or invented
   statement, overall and per language, reweighted to the confidence bands' real shares.
2. **Calibration against people:** for each confidence band, mean predicted confidence against the share rated fully
   correct, with intervals. Expected calibration error.
3. **Teacher accuracy by people:** the same measures on the 60 teacher items.
4. **Agreement:** agreement between raters (Cohen's kappa) on the doubly rated items; agreement between raters and the
   LLM judge on verdict and faithfulness.
5. **Error types:** the comments grouped into categories (wrong period, wrong place, look-alike topic, negation
   ignored, invented detail, poor language), with counts.

## Decisions the results feed

- If the answer confidence is calibrated against people, the app keeps it as shown. If it is over- or under-confident,
  the head is refitted on the human ratings, with the 300 answers as calibration data.
- If people rate the teacher much below the automatic numbers suggest, the teacher prompt and the verdict rules are
  revised before more data is generated.
- The error categories set the targets of the next improvement round.

## Practical notes

- One rating takes about one to two minutes, so 360 items take about 6 to 12 hours of rater time, plus about 80
  doubly rated items.
- Ratings are collected in a spreadsheet with one row per item and the six questions as columns. A rating page in the
  web app can replace it later.
- Raters' names are not stored with the ratings; a rater code is enough for the agreement analysis.
