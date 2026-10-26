# GISTEmbed example batch

Data behind the GISTEmbed explainer video (`stages/gist.html`). `batch.py`
encodes a batch of four (query, positive, negative) triplets, written for the
video, with three models and writes the pairwise cosine similarities to
`similarities.json`:

- `WhereIsAI/UAE-Large-V1` (335 million parameters), the guide model of the
  GISTEmbed paper;
- `sentence-transformers/all-MiniLM-L6-v2` (22.7 million parameters), the
  smallest base model in the paper;
- `avsolatorio/GIST-all-MiniLM-L6-v2`, that model after GISTEmbed fine-tuning.

The matrices follow the paper's notation: `qp` (query to positive), `qn`
(query to negative), `qq` (query to query), and `pp` (positive to positive).
The video uses the guide model's scores for the first query, "Capital
cities": a text whose score is above the score of the query and its own
positive, "Ottawa", is removed from the negatives of that query.

```
python -m venv .venv-gist && .venv-gist/bin/pip install sentence-transformers
.venv-gist/bin/python batch.py similarities.json
```

Paper: Solatorio, A. V. (2024). GISTEmbed: Guided In-sample Selection of
Training Negatives for Text Embedding Fine-tuning. arXiv:2402.16829.
