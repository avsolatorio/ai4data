# Relevance explainer for realistic searches: design

Status: proposal and pilot (2026-10-04). Supersedes the input format of the first data rounds for new work.

## 1. The task, restated

At explanation time a search system has: the text the user typed (any language, from two keywords to a full
question), any facets the user set in the interface (often none), one retrieved catalog record with whatever
metadata that catalog provides, and the retriever's score. The explainer returns a verdict (relevant, partly
relevant, irrelevant) and two or three sentences that say what the record contains and how it meets or misses each
thing the user asked for, whether the user typed it or set it as a facet.

The first data rounds assumed that constraints arrive as structured filters. In those rounds 98% of searches that
name a country also carry it as a filter, which real searches rarely do. Constraints usually live in the text:
"pobreza rural Bolivia 2015", "malaria under-5 Nampula", "latest unemployment MENA".

## 2. Pipeline at inference

1. Query understanding, deterministic and multilingual, no model weights: finds places (countries in 11 languages,
   World Bank regions and groupings, other common groupings, first-level subnational units, larger cities), time
   (years, ranges, decades, relative expressions such as "latest" or "since 2015"), frequency, population groups
   (sex, age, urban or rural, wealth, displacement) and requested data types. Facets are merged in.
2. Matching, deterministic: each understood constraint is compared with the record (country lists, region
   membership, years covered, frequency, text evidence for groups and subnational detail) and gets a status:
   covered, partly covered, not covered, or cannot be confirmed from the metadata.
3. Explanation by the 20 MB model, from the user's own words, the understood constraints and the checks.

Membership and coverage lookups are exact and cheap outside the model; the model's capacity goes to judging the
topic and writing in the user's language. Constraints the extractor misses are still visible to the model in the
verbatim search text.

## 3. Input representation

```
Respond in: Spanish
Search: pobreza rural en Cochabamba despues de 2015
Facets: none
Understood: place=Cochabamba (department of Bolivia) [query]; time=2016 or later [query]; group=rural [query]
Checks: place: Bolivia covered, department-level detail not stated | time: covered (2000-2021) | group rural: stated in description
Retrieval similarity: 0.58
Item: survey microdata study
Title: Encuesta de Hogares 2018
Producer: Instituto Nacional de Estadística
Coverage: national; urban and rural areas
Years: 2018
Abstract: ...
```

- `Search` is verbatim, including typos and mixed languages.
- `Facets` lists interface filters when set; `Understood` merges facets and text and tags each constraint with its
  source, so a narrative can say "your search mentions 2015" or "the country filter".
- `Checks` has one entry per understood constraint, with the record value that decided it.
- The item block lists only the fields the catalog actually has, in a fixed order; missing fields are omitted.
  Descriptions keep the catalog's own length up to a cap.

## 4. Training data

Diversity is set by code, not left to the teacher's habits. For each query a spec is sampled and the teacher writes
a search that realizes it, plus a structured parse used as ground truth for the extractor.

| Dimension | Values (target share) |
|---|---|
| Style | 1-2 keywords (20%), keyword string of 3-7 words (35%), natural phrase (25%), full question or request (15%), long specific request (5%) |
| Surface | correct (70%), typos or missing accents (15%), all lowercase (part of the above), English terms inside another language (15% of non-English) |
| Geography | none (25%), one country (35%), several countries (8%), World Bank region or income group (8%), other grouping such as Sahel, MENA, ASEAN, small island states (6%), subnational unit or city (12%), global or "developing countries" (6%) |
| Country draw | near-uniform over about 217 economies, so small and less-studied countries appear |
| Time | none (40%), one year (20%), range (12%), decade (5%), relative such as latest, recent, since 2015, before COVID (18%), survey round such as "DHS 2018" (5%) |
| Population group | none (60%), sex, age, urban or rural, wealth, displacement, disability (40% together) |
| Data type | unspecified (55%), indicator or time series, microdata or survey, paper or report, dataset or file |
| Facets set in the interface | none (75%), duplicating the text (10%), adding a constraint not in the text (15%) |
| Language | 11 languages, flatter than the first rounds |

Records come from the four catalogs as before. Verdict balance is controlled with record-anchored generation: given
a record, the teacher writes a search of a requested relation to it (answers it fully; right topic but wrong place,
time or group; related concept; same place but different topic; unrelated). Theme searches with retrieval at
several ranks add naturally distributed pairs.

Record rendering varies as catalogs do: optional fields are dropped at catalog-typical rates, descriptions range
from one line to long text, documents sometimes have no retrieved passage.

Labels keep the original verdict definitions: relabelling 50,000 examples under a tightened partly-relevant rubric
lowered verdict accuracy against every label set (lab notebook, 2026-10-04 05:37 UTC). The teacher sees the raw
search, the understood constraints and the checks, and flags constraints the extractor missed, which feeds
extractor fixes.

## 5. Evaluation

- A realistic test set from held-out themes, with an unseen-item subset.
- Verdict against labels where the teacher, its thinking mode and a second model agree, and against the majority.
- Narrative faithfulness with two judges and pairwise usefulness, as before.
- Constraint coverage: does the narrative state the status of each understood constraint?
- Extractor precision and recall against the teacher's parse, by constraint type and language.

The current model is measured on the realistic test set first, with the realistic searches rendered in the old
format (no filters unless facets were set), to size the gap before regenerating training data.
