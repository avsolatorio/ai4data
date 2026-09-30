# Relevance Explainer

Research project on a very small (about 10-30M parameter, roughly 15-30 MB quantized) model that verbalizes why a catalog search result matches a natural-language query.

## Research question

Can retrieval reasoning be separated from linguistic explanation so that an extremely small language model provides useful conversational search explanations for Data360 and NADA catalog results?

## Design

The retriever performs the semantic matching. A match-decomposition step converts retriever output into structured evidence, and the small model turns that evidence into text.

```
query -> semantic retriever -> top-k records -> match decomposition -> small model -> explanation
```

Match decomposition produces:

- query concepts
- matched metadata fields (indicator, sex, age, residence, and so on)
- concept correspondences (for example, employment and labor force participation)
- missing constraints
- similarity signals

Stronger models supply the domain knowledge at indexing and training time (synonyms, concept hierarchies, dimensional relationships). The small model only consumes these signals at inference time.

## Planned behaviors

- explain strong and partial matches
- name a missing concept
- distinguish title, description, and metadata-field matches
- explain synonyms
- explain why another result ranked lower
- answer simple structured follow-ups ("What is missing?", "Why is this second?")

## Training approach

1. Generate synthetic (query, record, diagnostics) triples covering the case types below.
2. Label them with explanations from a larger teacher model.
3. Distill into a small encoder-decoder, then quantize to INT8 or INT4.

Case types: exact match, semantic synonym, broader concept, narrower concept, missing geography, missing demographic, wrong unit, wrong time period, wrong frequency, partial dimensional match, metadata-only match, weak semantic match.

## Out of scope

Multi-indicator methodological comparison and inferring conceptual relations that are absent from the supplied evidence.

## Target

On-device, offline inference with a model of about 20 MB and generation startup under roughly 100 ms.

## Layout

- `notes/lab_notebook.md`: dated, append-only experiment log.
