# Public RAG example: World Bank country economic updates

This folder holds the data preparation, retrieval runs, and answer recording
for an explainer video on retrieval-augmented generation (RAG) built on
published World Bank reports. The existing RAG explainer
(`videos/retrieval-augmented-generation.json`) uses the eight workshop briefs
of the training session. This example uses the same retrieval stack and
prompt over 13 editions of two report series, and its failure cases come from
the rankings of that collection. No document was edited and no distractor was
added.

| File | Contents |
|---|---|
| `sources.json` | The 13 reports: title, edition, repository handle and DOI, licence, PDF address, size and SHA-256 |
| `fetch.py` | Downloads the PDFs and checks their SHA-256 |
| `build_store.py` | Extracts the text, splits it into chunks, embeds them, and writes the store |
| `common.py` | Paths, citation form, prompt, and the `make_rag` chain of notebook 03 |
| `runs.py` | Runs the retrieval cases and checks the figures; writes `retrieval_runs.json` |
| `retrieval_runs.json` | Every run: question, k, filter, ranked passages with scores, report, edition, page, and text |
| `record_answers.py` | Sends the chosen cases to a live model; writes `answers.json` |
| `render_pages.py` | Renders covers, inside pages and cited pages for the videos, with passage and phrase positions; writes `local/rag-public/pages/` |
| `requirements.txt` | Python packages for this example |

The PDFs, the extracted text (`chunks.jsonl`) and the store (`store.json`)
are written to `local/rag-public/` at the repository root and stay out of git.

## Collection

Two series of World Bank economic updates, each published about twice a
year: seven editions of the Kenya Economic Update (December 2022 to July
2026) and six editions of the Uganda Economic Update (June 2023 to June
2026). Each edition reports the latest growth, inflation, fiscal and poverty
figures of its time and revises those of earlier editions, so the collection
holds several answers to a question about "the latest" figure. All files come
from the World Bank Open Knowledge Repository. Two editions of the period
were not in the repository when the files were downloaded (Kenya, December
2023; Uganda, 23rd edition, June 2024) and are left out.

| Id | Title | Licence | Handle |
|---|---|---|---|
| keu-2022-12 | Kenya Economic Update, December 2022: Continued Rebound, but Storms Cloud the Horizon | CC BY 3.0 IGO | [10986/38386](https://hdl.handle.net/10986/38386) |
| keu-2023-06 | Kenya Economic Update, June 2023: Securing Growth | CC BY-NC 3.0 IGO | [10986/39930](https://hdl.handle.net/10986/39930) |
| keu-2024-06 | Kenya Economic Update, June 2024: Fostering Trade for Robust Growth and Dynamic Job Creation | CC BY-NC 3.0 IGO | [10986/41706](https://hdl.handle.net/10986/41706) |
| keu-2024-12 | Kenya Economic Update, December 2024: Special Focus on Women's Economic Empowerment | CC BY-NC 3.0 IGO | [10986/42541](https://hdl.handle.net/10986/42541) |
| keu-2025-05 | Kenya Economic Update, May 2025: Special Focus on Poverty and Distributional Impacts of Fiscal Policy in Kenya | CC BY-NC 3.0 IGO | [10986/43417](https://hdl.handle.net/10986/43417) |
| keu-2025-11 | Kenya Economic Update, November 2025: Special Focus on Competition Policy | CC BY-NC 3.0 IGO | [10986/44020](https://hdl.handle.net/10986/44020) |
| keu-2026-07 | Kenya Economic Update, July 2026: Sustaining Macroeconomic Gains, Advancing Fiscal Reforms | CC BY-NC 3.0 IGO | [10986/45196](https://hdl.handle.net/10986/45196) |
| ueu-2023-06 | Uganda Economic Update, 21st Edition, June 2023: Leveraging Sustainable Tourism to Support Growth and Diversification | CC BY-NC 3.0 IGO | [10986/40750](https://hdl.handle.net/10986/40750) |
| ueu-2023-12 | Uganda Economic Update, 22nd Edition, December 2023 | CC BY-NC 3.0 IGO | [10986/41043](https://hdl.handle.net/10986/41043) |
| ueu-2025-02 | Uganda Economic Update, 24th Edition, February 2025: Investing in Early Childhood Development for Transformation of Human Capital in Uganda | CC BY-NC 3.0 IGO | [10986/42909](https://hdl.handle.net/10986/42909) |
| ueu-2025-09 | Uganda Economic Update, 25th Edition, September 2025: Increasing Uganda's Fiscal Space through Improved Revenue Mobilization and Enhanced Efficiency of Spending and Service Delivery | CC BY-NC 3.0 IGO | [10986/43815](https://hdl.handle.net/10986/43815) |
| ueu-2025-12 | Uganda Economic Update, December 2025: Cultivating Prosperity through Agro-Industrialization | CC BY-NC 3.0 IGO | [10986/44123](https://hdl.handle.net/10986/44123) |
| ueu-2026-06 | Uganda Economic Update, 27th Edition, June 2026: Building on Urban Transformation, Construction as a Jobs Engine | CC BY-NC 3.0 IGO | [10986/45168](https://hdl.handle.net/10986/45168) |

The id names the series and the edition month; it is also the citation
label in the prompt, for example `[ueu-2026-06 · p. 22]`. Page numbers are
PDF pages counted from 1, which differ from the printed page numbers.

Reference values come from World Development Indicators
(`api.worldbank.org/v2`), licensed CC BY 4.0.

### Licences and attribution

The licence in the table is the one stated in each report's repository
record. CC BY-NC 3.0 IGO permits use for noncommercial purposes with
attribution. The rights page inside every PDF, including the December 2022
Kenya edition, permits reproduction in whole or in part for noncommercial
purposes with full attribution. A video that shows passages or figures from
these reports therefore carries an attribution for each report it quotes, in
this form:

> World Bank. 2026. Uganda Economic Update, 27th Edition, June 2026: Building
> on Urban Transformation, Construction as a Jobs Engine. Washington, DC: World
> Bank. https://hdl.handle.net/10986/45168. License: CC BY-NC 3.0 IGO.

For indicator values: "Source: World Bank, World Development Indicators.
License: CC BY 4.0." The cover photographs of some reports are credited to
third parties and are not covered by the report licence. They are left out
of any video built from this example.

## Rebuilding

From this folder, with any Python 3.10 or later environment that has
`requirements.txt` installed (the runs recorded here used Python 3.11,
langchain-core 1.7, sentence-transformers 6.2 and PyMuPDF 2.0 on CPU):

```
python -m venv .venv-rag && .venv-rag/bin/pip install -r requirements.txt
.venv-rag/bin/python fetch.py          # download the 13 PDFs and check their SHA-256
.venv-rag/bin/python build_store.py    # text, chunks and store under local/rag-public/
.venv-rag/bin/python render_pages.py   # page images for the videos, under local/rag-public/pages/
.venv-rag/bin/python runs.py           # retrieval_runs.json
```

`fetch.py --record` rewrites the sizes and SHA-256 values in `sources.json`
after a deliberate change of files. The store takes about half a minute to
build on CPU.

### Store

| Measure | Value |
|---|---|
| Reports | 13 |
| PDF pages | 769 |
| Pages with chunks | 595 |
| Chunks | 3,440 |
| Characters | 1,906,358 |
| Tokens (word pieces of the embedding model) | 433,267 |
| Chunks longer than the model's 256-token window | 197 |

`build_store.py` extracts text with PyMuPDF and splits each page into chunks
of at most 800 characters at paragraph and sentence boundaries, with 80
characters of overlap. A chunk never crosses a page, so each chunk carries one
page number. The build removes running headers and footers, page numbers and
photo credits, and leaves out front matter (contents, lists of figures,
abbreviations, acknowledgements) and reference lists. The metadata of each
chunk holds the report id, series, country, title, edition, date (YYYY-MM)
and page. The embedding model is `sentence-transformers/all-MiniLM-L6-v2`,
the model of the session notebooks, and the store is LangChain's
`InMemoryVectorStore`, which scores by cosine similarity. The 197 long chunks
are mostly statistical tables, whose numbers split into many word pieces; the
model embeds the first 256 word pieces of each.

## The cases in retrieval_runs.json

| Run | Question | k, filter | What the ranking shows |
|---|---|---|---|
| `main_one_search`, `main_ranking` | How did the conflict in the Middle East affect Uganda, and how much did poverty fall? | 4 | The four passages are on poverty. The first passage on the conflict is fifth, at 0.633 against 0.653 for the fourth. |
| `main_part_1`, `main_part_2`, `main_merged` | The two parts as separate searches | 4 each | Eight merged passages, which hold the conflict figures (24 percent rise in energy prices, 15 percent of imports) and the poverty figures (51.5 and 52.9 percent). |
| `latest_no_filter` | What is the latest inflation rate in Uganda? | 2 | Passages from the December 2023 and February 2025 editions. The April 2026 figure (3.0 percent) is 19th. |
| `latest_reworded` | Uganda headline inflation April 2026, percent year-on-year | 2 | The same two older passages, with higher scores; the April 2026 passage moves to 11th. |
| `latest_filtered` | The latest question | 2, date ≥ 2026-01 | Both passages come from the June 2026 edition, and the second holds 3.0 percent for April 2026. |
| `inside`, `outside` | How much did Uganda's economy grow in FY25? / How much did Rwanda's economy grow in 2025? | 3 | Best scores 0.792 and 0.631. The first Uganda passage states 6.3 percent. The collection holds no Rwanda report, and the Rwanda search returns a chart legend and passages on growth in Africa and Kenya. |
| `cutoff` | 17 labelled questions | 1 | A cut-off of 0.663 classifies 14 of 17 correctly. One answerable question scores below it, and two questions outside the collection score above it. |
| `tests` | 8 test questions | 4, with and without the date filter | The checks of notebook 03 on the passages alone. |

`runs.py` also checks every figure the cases rely on: each quote must occur
on the stated page of the stated report, and where an indicator exists the
World Development Indicators values are fetched and stored next to it
(`checks` in the JSON). Kenya's growth of 4.6 percent in 2025 (keu-2026-07,
p. 13) matches the WDI value of 4.63, and Uganda's growth of 6.3 percent in
FY25 (ueu-2025-12, p. 12) matches 6.33.

In this collection a score cut-off separates answerable from unanswerable
questions less cleanly than in the workshop collection. Questions near the
topics of the reports, such as growth in Tanzania in 2025, score above some
answerable questions, and only the instruction to say when the passages do
not answer covers them.

The number check of notebook 03 (`numbers_present`) accepts any number
within its tolerance. In report text a value such as 24 also matches
"2023/24", and a growth rate for another period matches the value needed
(Kenya's historical average of 4.6 percent). The tests therefore also report
`figures_as_percent`, which requires the value written as a percentage. The
expected-report check is the one that detects an outdated passage.

## Recording answers

`record_answers.py` runs six cases through the chain of notebook 03 with a
live model: `main_one_search`, `main_merged`, `latest_no_filter`,
`latest_filtered`, `inside` and `outside`. It reads the same environment
variables as the session's `record_responses.py`, never prints the key, and
stops with a message when `WORKSHOP_API_KEY` is not set.

```
WORKSHOP_API_KEY=<key> \
WORKSHOP_BASE_URL=https://openrouter.ai/api/v1 \
WORKSHOP_MODEL=anthropic/claude-haiku-5.5 \
.venv-rag/bin/python record_answers.py
```

`--dry-run` prints the prompts without a key and writes nothing;
`--only main_merged,outside` records some cases and keeps the others in
`answers.json`. Each answer is stored with its question, the passages given,
the messages sent, the raw answer, the finish reason, the token counts, the
model name and the date.
