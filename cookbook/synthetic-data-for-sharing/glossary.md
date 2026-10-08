---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in the Practical Guide to Synthetic Data for Sharing.
---

# Glossary

**Attribute inference.** The risk that a sensitive attribute of real people can be predicted from their other characteristics through the synthetic file; measured against a baseline.

**Closest-record distance.** The distance from a synthetic record to its nearest real record on the quasi-identifiers; compared with the distance between two halves of the real file.

**Exact copy.** A synthetic record identical to a real one; a failed check that sends the file back to the method.

**Fully synthetic.** A file in which every value is generated; the subject of this guide.

**Partially synthetic.** A file in which only sensitive variables are replaced; a disclosure control method documented with the public use file.

**Relational synthesis.** Generation of a child table (persons) conditional on a generated parent table (households), preserving the structure.

**Sequential synthesis.** Fitting and generating one variable at a time conditional on the previous ones, with CART or parametric models.

**Synthesis record.** The document that accompanies a synthetic file: source, method and seed, utility and risk results, intended and prohibited uses, label, licence, contact.

**Target analysis.** A statistic or model that users of the file will compute, compared between the real and the synthetic file in the utility report.

**Utility.** How closely the synthetic file reproduces the real one at the marginal, association, and analysis levels.

**Visit sequence.** The order in which variables are synthesized in a sequential method.
