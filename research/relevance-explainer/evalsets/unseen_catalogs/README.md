# Records from catalogs absent from training

This directory holds a test set of 49 labelled searches over records from four catalogs that are not among the
training sources (Data360, Microdata Library, World Bank Documents and Reports, HDX, and the record pools of
`synth/pools.py`):

| Source | Records | Rendered as | Terms of the copied metadata |
|---|---|---|---|
| Our World in Data (charts) | 8 | indicator series | CC BY 4.0 |
| open.canada.ca (CKAN) | 7 | opendata (`schemas.ckan_record`) | Open Government Licence – Canada |
| data.gov.au (CKAN) | 7 | opendata (`schemas.ckan_record`) | the licence stated on each dataset page |
| Zenodo (datasets) | 7 | opendata | metadata CC0 |

- `records.jsonl`: the records, fetched on 2026-10-16 by `scripts/unseen_catalogs.py fetch`.
- `cases.tsv`: one search per line with the record id, the language, the label and an accepted second label for
  borderline cases. 45 searches are in the model's eleven languages; 4 (German, Tagalog, Japanese, Vietnamese) are
  in languages it does not support. The labels were written before the model was run on them. One label was corrected
  after the first run: the Our World in Data literacy series has no row for Turkey.

```
python scripts/unseen_catalogs.py run --bundle <bundle dir> --out answers.jsonl     # render version 5
python scripts/unseen_catalogs.py run --bundle <bundle dir> --out answers_v4.jsonl --render-version 4 --kind dataset
python scripts/unseen_catalogs.py report answers.jsonl answers_v4.jsonl
```

`--render-version 4 --kind dataset` reproduces the inputs of the first run (the training render version, with the
portal datasets shown as HDX datasets). The results are in the lab notebook entries of 2026-10-16.
