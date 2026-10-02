---
id: govern
title: "8. Operate responsibly"
sidebar_label: "8. Govern"
sidebar_position: 8
description: Governance, human oversight, security, privacy, model choice, and vendor dependence.
---

# 8. Operate responsibly

**Question:** Can we operate AI-supported dissemination responsibly?

## Why this matters

Statistical offices work under the
[Fundamental Principles of Official Statistics](https://unstats.un.org/fpos/),
which include professional independence, impartiality, confidentiality of
individual data, and transparency of methods. AI components change how data
reach users, and each component needs to be consistent with these principles.

## What good looks like

- A named owner is accountable for each AI component.
- AI-generated content is labeled, and a person reviews it before it becomes official metadata.
- Confidential microdata never reach external AI services unless the legal and technical basis is documented.
- The office knows which models and providers it depends on and can replace them.

## Maturity levels

| Level | Steps |
|---|---|
| **Foundational** | Write a short policy for AI use: permitted uses, prohibited uses, and who approves new ones. Keep confidential and unpublished data out of external AI tools. Classify data by sensitivity. |
| **AI-ready** | Keep a register of AI components with purpose, owner, model, provider, data used, and review date. Require human review for generated metadata and published text. Apply disclosure control to any dataset exposed through an API or agent. Use read-only credentials for agent access. |
| **AI-native** | Log all model inputs and outputs for audit. Test for prompt injection in any component that reads external text. Keep an exit plan for each provider. Review the register on a fixed schedule. |

## Implementation options

- **Human oversight:** AI proposes, a person approves. Record the approver and the date.
- **Privacy and confidentiality:** apply statistical disclosure control before data are made available to any tool, and check query results for small cells.
- **Security:** limit agent tools to what the task needs, use read-only access, rate-limit requests, and treat text retrieved from outside sources as untrusted input.
- **Model choice:** compare hosted models, open-weight models run locally, and small task-specific models. Sensitivity of the data, cost, connectivity, and language coverage all affect the choice.
- **Vendor dependence:** use standard interfaces (for example, an API format supported by several providers) so that a model can be replaced with limited changes.
- **Transparency:** publish what AI is used for, and describe the methods in the metadata.

## World Bank examples

- [Efficient and Inclusive AI Applications](/docs/inclusive-ai/) describes provider-agnostic configuration and the use of locally hosted open-weight models.
- The [Metadata Reviewer](/docs/metadata-reviewer/overview) keeps the reviewer in the loop through a review board for AI suggestions.

## How to test it

- **Register check:** pick any AI component and confirm that the register names its owner, model, and data.
- **Data-flow check:** trace a confidential field through the system and confirm that it never leaves the permitted boundary.
- **Replacement drill:** switch one component to a different model and measure the change in the evaluation suite ([chapter 6](./evaluate.md)).
- **Injection test:** place an instruction inside a document the system reads and confirm that the system ignores it.

## Checklist

- [ ] AI use policy approved and published
- [ ] Register of AI components with owners
- [ ] Human review required for generated metadata and text
- [ ] Confidential data excluded from external services
- [ ] Disclosure control applied before data exposure
- [ ] Agent access limited and read-only
- [ ] Prompt-injection test performed
- [ ] Exit plan for each model provider
