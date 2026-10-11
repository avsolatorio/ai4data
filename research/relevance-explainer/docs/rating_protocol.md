# Human rating protocol for the relevance explainer

This protocol describes a small study in which catalog or statistical-office staff rate the explainer's answers. It
answers three questions that the automatic evaluation cannot:

1. How often are the verdict and the explanation right, according to people who know the data?
2. Does the answer confidence shown in the app mean what it says when people are the judges? For example, are answers
   shown at 90% right about 90% of the time?
3. How far can the LLM teacher and judge be trusted as stand-ins for people? Every automatic number in the lab
   notebook rests on them.

## Sample

The sample has 300 answers from the deployed model plus 60 answers written by the teacher, 360 items in all.
`scripts/rating_sample.py` draws it from the realistic test set and writes one row per item.

- **Languages.** Choose three or four languages the raters read well. English plus two or three of French, Spanish,
  Arabic and Russian is a practical set. The sample is split evenly across the chosen languages.
- **Confidence bands.** Within each language, a third of the answers come from each band of the answer confidence:
  below 0.5, 0.5 to 0.8, and 0.8 or above. This over-samples doubtful answers, so the low bands have enough items to
  measure. Results for the whole population are reweighted to the bands' real shares.
- **Item types.** Each language's sample is spread across indicators, studies, datasets and documents in proportion to
  the test set.
- **Teacher answers.** 60 items show the teacher's own answer for the same input instead of the model's. Raters are not
  told which is which. These items measure how often the teacher is right according to people.

With 300 model answers, the share rated correct has a 95% confidence interval of about plus or minus 5.5 points. Each
confidence band has about 100 answers, about plus or minus 10 points.

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
