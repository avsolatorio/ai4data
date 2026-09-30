# Lab notebook: Relevance Explainer

Append-only. Entry headers carry a UTC date and time taken from a network source. Each entry names the git commit its results were produced against.

## 2026-09-30 22:36 UTC — Project scoping

Commit: 40ee179 (branch `relevance-explainer`, no code yet)

**Question/Context.** Can a model of about 20 MB explain why a catalog result matches a query, if the retriever already performs the semantic matching?

**What was done.** Design discussion only. No experiments run. Proposed a hybrid pipeline: retriever, then match decomposition into structured evidence (matched concepts, matched fields, missing constraints, similarity), then a small encoder-decoder (10-30M parameters, INT8/INT4) that verbalizes the evidence. Training data would come from distilling a larger teacher model over synthetic (query, record, diagnostics) examples.

**Result.** None yet. The size and latency targets (about 20 MB, sub-100 ms startup) are untested assumptions.

**Implication.** The design decision to keep reasoning out of the small model (precompute relations at indexing time) needs empirical support. First steps: define the diagnostics schema, build a small synthetic set across the twelve case types listed in the README, and establish a baseline (template-based explanations) for comparison.
