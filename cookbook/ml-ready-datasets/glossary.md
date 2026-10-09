---
id: glossary
title: Glossary
sidebar_label: Glossary
sidebar_position: 11
description: Terms used in this guide, in plain language.
---

# Glossary

**Benchmark (population).** The share of each group in the population, from the census or population estimates, against which a dataset's composition is compared.

**Checksum.** A fingerprint of a file's content (here SHA-256) that changes if one byte of the file changes, used to prove that a file is as released.

**Croissant.** The MLCommons format for describing a machine-learning dataset so that tools can load it: its files, their checksums, the fields of each record, and the splits, as a JSON-LD document on schema.org.

**Dataset card.** The one document that tells a user what a dataset is, why it exists, how it was collected and labelled, whom it represents, what it is for and not for, its licence, its version, and how to cite it.

**DOI.** Digital object identifier, a persistent identifier registered with DataCite that resolves to one version of a dataset.

**Feature.** A column a model may use as input.

**Group key.** The column that names the unit whose records must stay together in a split: a household, a firm, a document.

**Grouped split.** A draw of whole groups (households, documents) into the parts of a dataset, so that no group is divided between training and test.

**JSON-LD.** JSON with a vocabulary attached, so that each key has a defined meaning that different tools read the same way.

**Label.** The column a model is trained to predict: the occupation code, the transcribed value, the series a question refers to.

**Leakage.** Any way the answer reaches a model other than through its features: shared groups across splits, a feature that encodes the label, duplicated inputs.

**Manifest.** A file that lists every file of a release with its size and checksum.

**ML-ready dataset.** A fixed, versioned set of examples with a documented structure, a stated purpose, a split for training and testing, a statement of whom the examples represent, and a licence that covers model training.

**Reference model.** A simple model trained on the dataset and scored on its test part, published as a baseline so that users can compare their scores and see the error by group.

**Representativeness.** The degree to which a dataset's composition matches the population it will be used on, measured group by group and stated in the card.

**Split.** The division of a dataset into a training part (the model learns from it), a validation part (the model builder chooses settings on it), and a test part (touched once, to report a score).

**Thin group.** A group with too few examples for a model to learn its patterns; listed in the card with the minimum count used.

**Time-based test part.** A test part made of the latest round of the source, so that a score describes what a model will do on the next round.

**Weight (survey).** The number of population members a record stands for in a survey, kept as a column for population estimates; not a training weight by default.
