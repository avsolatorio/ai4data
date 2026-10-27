"""ai4data.small_ai - small classifiers trained on generated examples, for any labelling task.

A task folder (task.json and a label file) describes the classification:
topics for documents, occupation codes, expenditure codes. The package
generates labelled examples with a writer and a labeller model, checks
them, trains a small encoder, measures it on real reference data, and
runs a feedback loop in which an orchestrator directs the next round of
generation at the labels the model gets wrong. The command line is
`python -m ai4data.small_ai`; see cli.py.

Install with: uv pip install "ai4data[small-ai]"
"""

from .task import Label, Task, load_task

__all__ = ["Label", "Task", "load_task"]
