---
id: index
title: Authoring guide for cookbooks
sidebar_label: Authoring guide
description: The standard structure, writing rules, and tooling for a cookbook on this site, with the scaffold and checker that enforce them.
hide_table_of_contents: false
---

# Authoring guide for cookbooks

A cookbook is a practical guide on one problem domain for one audience. It
is organized around the questions that audience asks, and each chapter
answers one question with recipes: tasks with a defined result, the skills
and time they need, and the code or template to carry them out. This guide
sets the standard that every cookbook on the site follows, so that readers
can move between cookbooks without relearning the structure and so that new
cookbooks can be produced quickly.

Two tools support the standard. `scripts/docs/new_cookbook.py` creates a
cookbook from the template in `cookbook/_template/` and registers it on the
site. `scripts/docs/check_cookbooks.py` verifies that every cookbook has the
required structure and follows the writing rules; it runs in continuous
integration and can be run locally.

## Definition and scope

| Element | Standard |
|---|---|
| Subject | One problem domain, for example AI-ready data dissemination, microdata documentation, or extraction of data from documents. |
| Audience | One primary audience, named on the overview page and on the landing card, for example national statistical organizations or data curators. |
| Organizing principle | The questions the audience asks about its own situation. Each chapter answers one question. |
| Length | Six to ten chapters. A domain that needs more is two cookbooks. |
| Relation to documentation | A cookbook states what to do and why, and links to the documentation for how each method or tool works. It does not duplicate the documentation. |
| Register | A guidance document: formal, specific, and descriptive. See [Writing rules](#writing-rules). |

## Required parts

### Overview page (`index.mdx`)

1. Title and subtitle. The title names the guide ("Practical Guide to ..."); the subtitle names the audience ("A cookbook for ...").
2. Version and date, and a note stating that the guide is a working draft where that is the case.
3. An introduction of two or three paragraphs: the problem, the organizing questions, the standards the recipes build on.
4. A "Chapters and questions" table: chapter, question, practical topics.
5. The maturity levels, using the shared vocabulary below.
6. The running example: one small, fictional dataset or case that every chapter uses, with its files.
7. "Scope of the examples": a statement that the examples are small on purpose and that each recipe states its limits.
8. A files table listing every downloadable file with the chapter that uses it.
9. Related resources, a suggested citation, and a version history.

### Chapters

Each chapter is one `.mdx` file with this structure, in this order. The
checker verifies the headings.

| Section | Content |
|---|---|
| Front matter | `id`, `title` ("N. Noun phrase"), `sidebar_label` ("N. Short"), `sidebar_position`, `description`. |
| H1 and question | The title, then a line `**Question:** ...` with the chapter's organizing question. |
| Rationale | Why the question matters, in two or three paragraphs. |
| Target state | The condition the chapter aims at, as a bulleted list. |
| Maturity levels | A table with one row per level and the steps that reach it. |
| Running example | An `:::info[Running example]` admonition that says what the chapter does to the example. |
| Recipes | Two to four recipes (see below). |
| Implementation options | Open standards and patterns, with alternatives. |
| World Bank examples | Program workstreams, tools, and reference implementations that apply, with links. Optional where none exist. |
| Common mistakes | Three to five short items. |
| Verification | How to test that the chapter's steps worked. |
| Checklist | A `<Checklist id="..." items={[...]} />` component. |

### Recipes

A recipe uses the `Recipe` component:

```mdx
<Recipe title="N.M Imperative title naming the result" level="Foundational | AI-ready | AI-native" skills="Who can do this" time="How long">

**Result:** one sentence on what exists when the recipe is done.

1. Step.
2. Step, with code or a template where useful.
3. How to check that it worked.

**What this does not do.** Where the example stops and what the full version adds.

</Recipe>
```

- The title is an imperative that names the result ("Publish code lists and crosswalks"). Procedures are the one place where the imperative is the standard form.
- `level` uses the shared vocabulary. `skills` names a role. `time` is a realistic estimate.
- Code is embedded from a file in `website/static/cookbook-files/<cookbook-id>/` with a code-import block, so the page and the download are one file:

  ````mdx
  ```python file=../../website/static/cookbook-files/<cookbook-id>/script.py
  ```
  ````

- Every recipe ends with a **What this does not do** paragraph. Simplified examples stay useful only when they say where they stop.

### Standards page (`standards.md`)

A table per area: the standard, its use, where the World Bank uses it, and
where the cookbook applies it. Prefer the standards the World Bank has
adopted; link each to its source. The checker warns when the page is
missing.

### Glossary and contributing pages

A glossary of the terms the cookbook introduces, and a contributing page
with the recipe template and the criteria for a recipe. Both can start from
the template files.

### Optional parts

- A self-assessment page using the `SelfAssessment` component with a
  question per chapter (`questions` and `base` props).
- An "Application by data type" page when the domain spans data types.
- Profiles, templates, and example records as downloadable files.

## Shared vocabulary

Maturity levels are the same in every cookbook, so that a reader who knows
one cookbook knows them all:

| Level | Meaning |
|---|---|
| **Foundational** | The practice exists in a form a person can use and a crawler can read. |
| **AI-ready** | The practice is structured, identified, and accessible through documented interfaces, so that software, including AI systems, can use it without manual steps. |
| **AI-native** | The practice offers AI-oriented interfaces or methods, with provenance and verification built in. |

Use "organization" for the audience institution, as the AI-readiness
assessment framework does, and name the specific kind on first use
("national statistical organization").

## Code and files

- Files for download live in `website/static/cookbook-files/<cookbook-id>/`. The page embeds them by code import and links to them with `pathname:///cookbook-files/<cookbook-id>/<file>`.
- Scripts use the Python standard library unless a dependency is stated in the docstring and on the files table. Each script has a module docstring with purpose, usage, exit status, and a "what this does not check" statement.
- Scripts pass `ruff check` under the repository configuration and `ruff format`.
- Each cookbook with scripts has a test file `tests/test_cookbook_<id>.py` that runs them on the example files. Fixtures that would otherwise need the network are pinned under `tests/fixtures/`.
- The example data are fictional and say so. Real data are used only where the source is cited and the licence allows it.

## Writing rules

The repository's style rules apply in full. The points that matter most in
a cookbook:

- Titles and headings are noun phrases that name the section ("Target state", "Progression between maturity levels"). They are never a statement plus a comma and a tag, a teaser, or a slogan.
- Recipe titles are imperatives that name the result.
- The register is that of a guidance document: declarative sentences, no second person outside prompt templates and policy templates, no contractions, no intensifiers.
- Claims about improvement or scale are made only where the cookbook links to the measurement or the source.
- Standards are cited with a link to the publishing body. Where the World Bank has adopted a standard, the cookbook says so.
- Before finishing, run the greps in the repository's `CLAUDE.md` or run `scripts/docs/check_cookbooks.py`, which includes them.

## Creating a cookbook

```bash
python scripts/docs/new_cookbook.py \
  --id microdata-documentation \
  --title "Practical Guide to AI-Ready Microdata Documentation" \
  --audience "Data curators in national statistical organizations" \
  --chapters "variables:Variables and value labels:Can AI interpret the variables?" \
             "questionnaire:Questionnaire and concepts:Can AI relate questions to concepts?"
```

The script copies the template into `cookbook/<id>/`, writes one chapter
file per `slug:Title:Question` argument, creates
`website/static/cookbook-files/<id>/`, adds a sidebar in
`website/sidebars-cookbook.js`, and adds a card in
`website/src/content/cookbooks.js`. It then prints the files to edit.

## Checking a cookbook

```bash
python scripts/docs/check_cookbooks.py            # all cookbooks
python scripts/docs/check_cookbooks.py --id microdata-documentation
```

The checker reports errors (missing required parts, unregistered cookbook,
banned constructions in headings) and warnings (missing optional parts,
recipes without a scope note, second person in prose). Errors fail the
continuous-integration job for changes under `cookbook/`.

## Review before publication

- Every standard named on the standards page has been checked against its source.
- Every external claim links to its evidence.
- Every script runs on the example files and has a test.
- Every recipe has a scope note.
- Translations and non-English examples have been reviewed by a speaker of the language.
- The overview names the version, the date, and the draft status.
- The landing card's description, audience, and chapter count match the overview.
