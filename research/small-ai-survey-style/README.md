# Survey-style occupation answers: run scripts

This folder contains the scripts that produced the occupation-coding data and figures of training session 02
(`training/session-02-small-ai-synthetic-data`). The scripts call the small AI harness in `src/ai4data/small_ai`.
The run outputs are not stored in the repository.

## Scripts

| Script | Purpose |
|---|---|
| `rewrite.py` | Rewrites each labelled job title as three survey-style answers. A second model chooses a code for each answer, and the answer is marked as verified when the two codes agree. |
| `rewrite_test.py` | Rewrites the 1,000 test titles as one survey-style answer each, with a second large model and a different prompt. |
| `rewrite_prompt.py` | The two prompts of the rewriting step. |
| `arms.py` | Trains one combination of the training sets with one seed and scores it on three test sets. |
| `summarize.py` | Prints the mean and range of the scores of each combination over its seeds. |
| `direct.py` | Asks large models for codes on the survey-style test set and records accuracy, tokens, and time. |
| `speed.py` | Times the small model per answer on a processor and on a graphics card. |
| `learn_record.py` | Trains the model on all three sets and scores the test answers at fixed points of training. |
| `learn_deck.py` | Writes the training record of the slides, `slides/src/learn.json`, from the three recorded runs. |
| `record_deck.py` | Records the five best groups of the small model for the responses listed in `slides/src/occupation.json` and for the test answers. |
| `run_arms.sh`, `run_arms2.sh`, `run_learn.sh`, `run_direct.sh` | Launch the runs on a chosen graphics card and write a log. |

The outputs of `rewrite.py`, `rewrite_test.py`, and `arms.py` are the source of the session files
`assets/occupation/answers.jsonl.gz`, `test-answers.jsonl.gz`, and `results.json`. The steps that copied the outputs
into those files and into `slides/src/occupation.json` were one-off edits and are not part of this folder.

## Settings

The scripts read their paths and connections from the environment.

| Variable | Meaning |
|---|---|
| `SMALL_AI_RUNS` | Folder of the run folders. The default is `runs/` in the repository. It contains `isco-l-targeted`, the harness run on the `isco` task, and `isco-survey`, the outputs of these scripts. |
| `HAHU_TITLES` | Path of `occupations_training_set.csv` from the dataset [tabiya/occupation_titles_esco](https://huggingface.co/datasets/tabiya/occupation_titles_esco). |
| `SESSION02_SLIDES` | Folder `slides/src` of session 02. The default is the folder in the repository. |
| `PY` | Python of the harness environment, used by the launch scripts. |
| `SYNTH_BASE_URL`, `SYNTH_API_KEY`, `SYNTH_MODEL`, `SYNTH_EXTRA_BODY` | Connection to the writer model, as described in the harness README. |
| `OPENROUTER_API_KEY` | Key for the second large model of `rewrite_test.py` and `run_direct.sh`. |

## Order of a full run

1. A harness run on the `isco` task with the targeted arm, in `$SMALL_AI_RUNS/isco-l-targeted`. The scripts in `research/small-ai-loop-runs` started this run. It provides the titles written from the ISCO-08 definitions and the file `reference/report.jsonl` with the 1,000 test titles.
2. `rewrite.py` and `rewrite_test.py`.
3. `run_arms.sh` for the five combinations and three seeds, then `summarize.py`.
4. `run_direct.sh`.
5. `run_learn.sh`, then `learn_deck.py` and `record_deck.py`.

## Differences from the scripts as run

The runs of November 2026 used copies of these scripts with fixed paths on the machine of the runs. The copies
in this folder take the same paths from the environment. Two of them were run again on the recorded outputs after
the change: `summarize.py` printed the same scores, and `learn_deck.py` wrote a file identical to
`slides/src/learn.json`. The scripts that call a model or train one were not run again.

No trained coder reviewed the codes of the job titles or of the synthetic answers. The session's
`assets/README.md` describes each data file and its source.
