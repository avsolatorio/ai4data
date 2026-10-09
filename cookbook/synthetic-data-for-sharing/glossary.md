---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in the Practical Guide to Synthetic Data for Sharing.
---

# Glossary

**API.** Application programming interface: the address at which a program asks a system for data and receives it in a structured form.

**Attribute inference.** The risk that a sensitive attribute of real people can be predicted from their other characteristics through the synthetic file; measured against a baseline.

**Closest-record distance.** The distance from a synthetic record to its nearest real record on the quasi-identifiers; compared with the distance between two halves of the real file.

**Confidence.** A number between 0 and 1 that a model returns with its output to say how sure it is; the basis of a threshold rule.

**Cramér's V.** A measure of association between two categorical variables, from 0 (unrelated) to 1 (fully determined).

**Disclosure control.** The measurement and reduction of the risk that a released file identifies a person or reveals something about one; applied before any release.

**DOI.** Digital object identifier, a persistent identifier registered with DataCite that resolves to a dataset or a version of it and makes citations countable.

**Exact copy.** A synthetic record identical to a real one; a failed check that sends the file back to the method.

**Fully synthetic.** A file in which every value is generated; the subject of this guide.

**Gate.** The rule that decides whether a change ships, applied to a comparison of the evaluation suite before and after the change.

**GPU.** A graphics processing unit, the processor on which language models run fast; its memory decides which model fits.

**GSBPM.** The Generic Statistical Business Process Model, the shared description of the steps of statistical production from specifying needs to disseminating and evaluating.

**ISCO-08.** The International Standard Classification of Occupations, whose four-digit codes classify jobs; the label of the coding examples in this guide.

**Partially synthetic.** A file in which only sensitive variables are replaced; a disclosure control method documented with the public use file.

**Precision and recall.** Precision is the share of what a system found that is right; recall is the share of what is there that the system found. F1 is the average that combines the two.

**Provenance.** The record of where a value came from: the series, the release, the document and page, the method, and the version, carried with the value.

**Quasi-identifier.** A variable an outsider could know about a person from elsewhere (region, sex, age, education), whose combination can identify someone in a file.

**Relational synthesis.** Generation of a child table (persons) conditional on a generated parent table (households), preserving the structure.

**Sequential synthesis.** Fitting and generating one variable at a time conditional on the previous ones, with CART or parametric models.

**Synthesis record.** The document that accompanies a synthetic file: source, method and seed, utility and risk results, intended and prohibited uses, label, licence, contact.

**Target analysis.** A statistic or model that users of the file will compute, compared between the real and the synthetic file in the utility report.

**Threshold.** The value of a score (a confidence, a similarity, a garble rate) at which the rule changes what happens to an item; set from measurements.

**Total variation distance.** A measure of the difference between two categorical distributions, from 0 (identical) to 1 (no overlap).

**Utility.** How closely the synthetic file reproduces the real one at the marginal, association, and analysis levels.

**Visit sequence.** The order in which variables are synthesized in a sequential method.
