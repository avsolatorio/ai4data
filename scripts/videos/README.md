# Guide to producing explainer videos

This guide records how the Proof-Carrying Numbers explainer (4 min 40 s,
accepted on 2026-10-24) was produced,
and the rules that a new video must follow to look and sound the same. It is
written for a person or an AI agent who is given a new topic. The PCN
explainer is the reference: open `stages/pcn.html` and
`videos/pcn-explainer.json`, record the video with the command in section 2,
and match them.

The repository holds the tool, the stages and the step scripts. The Python
environment, the voice models and the recordings stay out of git
(`.gitignore` in this folder); section 2 describes how to set them up.

## 1. Two kinds of video

| Kind | Use for | Built from | Example |
|---|---|---|---|
| Concept explainer | A method or idea: what it is, how it works, why it matters | A stage page of animated scenes (`stages/<name>.html`) | `pcn-explainer` |
| Site walkthrough | How to use a feature of the AI for Data site | The live site, driven by Playwright with an overlay pointer and captions | `document-collection` |

The program team's direction for methods and notebooks is a concept explainer with
its own visuals. A walkthrough of a notebook page was tried for PCN and
rejected: it showed the page but did not explain the method. Most of this
guide is about concept explainers; section 9 covers walkthroughs.

Concept explainers are generic. They do not name a training session, a
notebook, or a workshop, in the narration or on screen. The site page that
embeds the video carries those links.

## 2. Files and tools

```
scripts/videos/
  README.md               this guide
  requirements.txt        Python packages
  record.py               plays a step script, records the browser, adds voice and music
  captions.py             divides the narration into subtitle cues for the .vtt file
  series.py               checks the shorts of a series against its plan
  thumbnail.py            renders thumbnails/<name>.html to a 1280x720 PNG for YouTube
  thumbnails/             thumbnail pages, one per published video
  review.py               contact sheets of the stage (before recording) and of the video (after)
  music.py                synthesises the background track (no licence needed)
  overlay.js              pointer, ring, caption and title cards for site walkthroughs
  stages/kit/stage.css    theme and shared components for every stage
  stages/kit/stage.js     go(), later(), wire(), flow(), carry(), fly(), typeText(), stream()
  stages/kit/collection.* map of the steps of RAG, report pages, passage cards, moveCard(), vectors
  stages/_template.html   starting point for a new stage
  stages/_short-template.html  starting point for a short in a series
  stages/pcn.html         the reference stage
  stages/preparing-knowledge.html   a second explainer, made from this guide alone
  stages/model-to-application.html  a third explainer (instructions and context in a prompt)
  stages/retrieval-augmented-generation.html  a fourth explainer (retrieval, the prompt, failure cases)
  stages/rag-public-reports.html    the same method on 13 published World Bank reports
  videos/_template.json   starting point for a new step script
  videos/_short-template.json  starting point for the step script of a short
  videos/series/          plans of series of shorts (parts, shorts, ideas, order)
  videos/pcn-explainer.json
  videos/preparing-knowledge.json
  videos/model-to-application.json
  videos/retrieval-augmented-generation.json
  videos/rag-public-reports.json
  videos/document-collection.json   a site walkthrough
  examples/rag-public/    sources, retrieval runs and recorded answers behind rag-public-reports
  voices/                 Piper voices (*.onnx) and Kokoro model files (voices/kokoro/)
  out/<name>.mp4, .vtt    results; out/<name>/ holds narration clips and review sheets
```

Setup, from `scripts/videos/` (no step needs root; `ffmpeg` must be on the
PATH):

```
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m playwright install chromium
mkdir -p voices/kokoro
curl -L -o voices/kokoro/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o voices/kokoro/voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

Kokoro is released under the Apache 2.0 licence. Piper voices (`*.onnx`
files in `voices/`) are optional; each voice has its own licence on its
model card. The music is synthesised by `music.py`, so no licence applies.

Stage paths in a step script (`"url": "stages/pcn.html"`) are relative to
this folder. The stages load the program logos from `website/static/img/`.

Commands, run from `scripts/videos/`:

```
.venv/bin/python review.py stage videos/<name>.json          # sheets of every beat, no audio
.venv/bin/python record.py videos/<name>.json --voice af_heart --speed 1.1 --music
.venv/bin/python review.py video videos/<name>.json          # sheets of the finished video
```

A recording takes about as long as the video. Run it in the foreground or as
a background task and wait for it; it does not need tmux.

## 3. Process

1. **Study the source.** Read the notebook, documentation, library code and
   paper behind the method. Write down the facts the video will state, with
   where each one comes from. Prefer values recorded in the source (a
   notebook's saved outputs). When a value has to be computed again, use a
   separate environment built from the source's own requirements file (for
   a training notebook, its `requirements.txt`); the video environment
   holds only the recording tools. Use real values from the source data for the
   worked example (PCN used internet use in Kenya and Rwanda from World
   Development Indicators, with the values that notebook 06 returns). Check
   any claim about the library against its code; for example, PCN claim ids
   can be any string, which the PCN docs show with `clm 7ef6`.
2. **Write the outline** as a list of scenes in the sequence of section 4,
   one line per scene with its heading and what moves on screen.
3. **Write the narration** in `videos/<name>.json`, one step per beat, under
   the rules of section 6. Read it through as a whole before building.
4. **Build the stage**: copy `stages/_template.html` to `stages/<name>.html`,
   add one `<section>` per scene and one hook per animated beat (section 5).
5. **Review the stage** with `review.py stage` and read every sheet. Fix
   overlaps, wrapped identifiers, empty areas and text that crosses other
   text before recording.
6. **Record** with the command above.
7. **Review the video** with `review.py video`. Each frame is the state on
   screen at the end of a narration line; check that the animation of each
   line has finished by then. Check the length and the levels (section 7).
8. **Send** the MP4 and the VTT for review, with a short note of what
   changed. Add decisions and feedback to the history in section 10.

## 4. Scene sequence

The PCN explainer follows this sequence. A new explainer keeps the order and
the kinds of scene; the number of method steps changes with the method.

| # | Scene | Eyebrow | Purpose | PCN example |
|---|---|---|---|---|
| 1 | Title (dark) | none; kicker "AI for Data – Data for AI · AREA" | Name the method and state what it does in one sentence | "Proof-Carrying Numbers" |
| 2 | Context diagram | Context | Where the method applies, as a flow of three nodes | Statistics → AI chatbot → People |
| 3 | Context example | Context | A concrete instance, animated (streaming, typing) | A question typed, an answer streamed word by word |
| 4 | The problem | The problem | What goes wrong without the method, shown as motion | Three origins sending values into one figure |
| 5–8 | Method steps | `Step k of N · Name` with the progress bar | One scene per step: inputs, format, mechanism | Data with claim ids; instructions; claim tag; verifier |
| 9 | Result for the user | In the chatbot (or the equivalent) | What a person sees when the method runs | Marks and hover popovers |
| 10 | Scope and judgement | Scope of the check | What the method checks, what remains a human judgement, and how the method helps that judgement | "The reader makes the final judgement" |
| 11 | Closing (dark) | none | Code, paper, program addresses | github.com/worldbank/pcn, arXiv, worldbank.github.io/ai4data |

Scene ids follow the PCN stage: `s1` is the title, `s2` the context
diagram, `s2b` the context example, then `s3`, `s4`, … in the order of the
table, and the closing is the last id. The step script calls `go(n, beat)`
with the part of the id after `s`: `go(3, 1)` for `s3`, and `go('2b', 1)`
for `s2b`, with `HOOKS['2b']` for its hooks. Copy the numbering of
`stages/pcn.html` or `stages/preparing-knowledge.html`.

The kicker on the title and closing names one of the six areas of the
program hub (`website/src/content/tools.js`), in noun form: Assessment,
Documentation, Access, Verification, Implementation, or Adoption. Pick the
area where the method's tools are listed; PCN is Verification, and the
preparation of a document collection for retrieval is Implementation.

The closing lists addresses that the program maintains. A method with its
own program library and paper lists Code, Paper and Program, as PCN does.
A topic built on general third-party tools (LangChain, an embedding model),
or a general method with no program library (the parts of a prompt), shows
the Program address only. Third-party projects can be named in the
narration where the method uses them.

The narration does not read out or point to the addresses. A closing with a
program library has one line that names the library; a closing with the
Program address only has no narration and stays on screen for four seconds
(`"min": 4.0`) under the music.

Rules that came from review feedback on PCN:

- Show how the method is produced, starting from its inputs. For PCN this
  meant the data carrying claim ids and the instructions that require the
  tag format, before the tag and the verifier.
- Describe each mechanism precisely. The verifier scene shows that the
  verifier reads the model's response as it streams, parses each tag in the
  PCN format, and checks the extracted value mechanically against the data
  that the application added to the context.
- Note real variants that a viewer might need. The claim id was shown both
  as a readable key and as a hash.
- An example inside an instruction or template differs from the values that
  the video later checks. PCN used `{claim_id}` and `12.3` in the instruction
  example.
- Close the argument on the human role. A method that automates a check
  still leaves a judgement to the reader; say so, and say how the method
  makes that judgement easier.
- Failure cases go into the mechanism scene as further beats. They do not
  need scenes of their own. PCN shows verified, flagged and unmarked in one
  verifier scene. A method whose failure cases each have their own fix can
  give each failure a method step, with the failure and its fix as beats of
  that scene; `retrieval-augmented-generation` does this for two-part
  questions, dates, and questions outside the collection.
- A method that can be tested on its own (retrieval without the model) may
  add one scene between the result scene and the scope scene, with the
  eyebrow naming the test ("Testing retrieval").

### Short explainers in a series

A long explainer covers a method in 3 to 5 minutes. A reviewer found the
long version of retrieval-augmented generation on published reports too
dense: 664 words, about 40 spoken figures, and five ideas. A series divides
a topic into shorts of 60 to 90 seconds, each with one idea, in an order
where each short builds on the one before it. The plan of a series is a
file in `videos/series/`. `videos/series/ai-applications.json` lists 27
shorts in six parts, from a model with no context to the verification of
figures, all on one example: published World Bank material on Kenya and
Uganda.

A short follows the rules of this guide, with these changes:

| Item | Long explainer | Short in a series |
|---|---|---|
| Length | 3 to 5 minutes | 60 to 90 seconds, 150 to 230 words |
| Ideas | One method with its steps and failure cases | The one idea in the plan entry |
| Title | Dark title scene with a narrated first line | Dark title card with the series title, the area, the title of the short and its place in the part; silent, `"min": 3.0` |
| Scenes | The sequence in the table above | Three to five light scenes: the example, the mechanism or failure, the result; a scope or judgement scene where the idea needs one |
| Figures | Where the method needs them | At most one spoken figure per narration line. Scores, ranks and counts stay on screen when they are not the idea of the short |
| Terms | The method's own terms | Plain words: "closest in meaning" for similarity scores, "the passages kept for the prompt" for top k. A technical term is named only when the short is about it |
| Closing | Addresses, with one line naming a program library | Silent end screen, `"min": 4.0`: series title, area, "Next in the series" with the title of the next short, the Program address, and the source line |

A short stands on its own, because the site embeds it on the page about its
idea. The narration does not refer to an earlier or a later video. The
first scene may open with one sentence that states the idea of the previous
short as a fact ("The search returns the passages closest in meaning to the
question.").

Every short in a series uses the same example material, so that a viewer
of several shorts sees one application grow. The shorts on documents use
the document collection of `examples/rag-public/`; its runs and recorded answers are
the source of every figure on screen.

Files: copy `stages/_short-template.html` and `videos/_short-template.json`.
The step script is `videos/<short id>.json` with the id, the title, and
the next id of the plan (`"next": null` for the last short of the series).
`series.py` checks every short of a plan: the fields, the end screen, the
word count, and lines that speak more than one figure.

```
.venv/bin/python series.py videos/series/ai-applications.json
```

## 5. Visual rules

### Theme

The stage follows the site theme in `website/src/css/custom.css`. All tokens
are in `stages/kit/stage.css`; do not introduce new colours.

| Token | Value | Use |
|---|---|---|
| `--canvas` | #ffffff | Scene background, with a dot grid (26 px) |
| `--ink` | #16191d | Text |
| `--muted`, `--subtle` | #565f68, #8a929a | Secondary text, labels |
| `--hairline` | #e2e5e8 | Card borders |
| `--primary` | #1558d6 | Main flow, highlights, eyebrow, step bar |
| `--primary-subtle` | #eaf0fd | Highlighted cells and words |
| `--teal` | #12a08a | Data in context, people, second flow |
| `--ok`, `--bad` | #1f7a4d, #b42318 | Verified and flagged results only |
| purple #6d4bd8 | | The model's memory or training data |
| `--block` | #12151a | Title and closing scenes, code panels, the model node |

- Type: Inter for text (headings 700, 40 px, letter-spacing -0.025em);
  JetBrains Mono for code, identifiers and values from data. Prose on
  screen (sentences in cards and chat bubbles) is at least 16 px. Data
  lines, identifiers, table cells and chart labels in JetBrains Mono are
  12–15 px, as in PCN. Labels are at least 11.5 px uppercase with letter
  spacing.
- Surfaces are flat: white cards with 1 px hairline borders and 10 px radius,
  no drop shadows. Emphasis comes from a coloured top border, a coloured
  ring (`box-shadow: 0 0 0 3px var(--primary-subtle)`), or a tinted
  background.
- Frame: 1280×720. Scene padding 52 px top, 80 px sides. Every light scene
  has an eyebrow line and an `h2` heading at the top left. The program logo
  (`program-logo.png`) sits at the top right of context scenes;
  `program-logo-dark.png` is on the title and closing scenes.
- Icons are inline SVG line drawings (stroke 1.6–2), in the colour of their
  role. No emoji.
- Decorative elements (the drifting code on the title) stay faint and do not
  cross text.
- Values come from the source data. A visual that cannot be real, such as
  positions on a map of meaning drawn in two dimensions, carries a label on
  screen that says so ("positions drawn for illustration").
- Published material from outside the program (report text, figures,
  passages) carries a source line in 12 px at the bottom of every scene
  that shows it, with the publisher, series, years, repository and
  licence, and the closing repeats it under the addresses. Model answers
  recorded for the video are named on the closing with the model and the
  temperature. `stages/kit/collection.js` holds the line (`SRC`) and fills
  every `.srcline` element.
- Show the real source. A viewer understands a document collection from
  its covers and pages, and a citation from the page that it leads to.
  `examples/rag-public/render_pages.py` renders covers, inside pages and
  cited pages from the downloaded PDFs into `local/rag-public/pages/`
  (outside git, with the PDFs), with the position of each passage and of
  each marked phrase. `pageView()` shows a page and zooms to a region;
  `pageBox()` marks a passage or a phrase on it. A page is shown at a size
  where its text can be read, or as a recognisable page in a collage.
- One focal object per beat. A scene of two or three panels of small
  monospaced text cannot be read on a phone, and the eye has no place to
  start. Prose on screen is at least 16 px and, in the main object of a
  scene, 18 to 21 px.
- Shorts on retrieval-augmented generation show the map of its steps
  (`ragMap()` and `mapAt()` in `stages/kit/collection.js`) under the
  heading of every light scene, with the current step marked: Documents,
  Text, Passages, Vectors and Store, prepared once; Question, Search,
  Prompt and Answer, for each question.
- A store of thousands of chunks is drawn as one small square per chunk,
  grouped by document in store order, so that the retrieved and the needed
  chunks can be marked at their real positions (`rag-public-reports`,
  3,440 squares). One tile per chunk with a label works up to a few dozen.
- Empty space: early beats of a scene may leave part of the frame empty,
  because later beats fill it. In the final state of every scene, the
  content spans the frame; no half of the frame (left, right, top or
  bottom) is empty.

### Motion

Every scene has something that moves; a scene in which elements only fade in
is not finished. The kit provides the motions used in PCN:

| Helper | Motion | Use in PCN |
|---|---|---|
| `data-b="k"` | Element slides up and fades in at beat k | Every card |
| `wire(svg, a, b, opts)` | A curved connector draws itself between two elements | Flows, annotations |
| `flow(svg, path, colour)` | Dots travel along a connector in a loop | Data moving to the model |
| `carry(path, html, colour, ms, onStep, onEnd)` | A labelled value travels along a connector once and can change on the way | 34.9762 becoming 35.0; a digit flipping |
| `fly(from, to, html, ms)` | A chip moves from one element to another | A tag moving to the parser; a value moving from the data table |
| `typeText(el, text, cps)` | Typing with a caret | The user's question |
| `stream(el, tokens, ms, onToken)` | Tokens appended one at a time with a caret | Model output, in a chat bubble or a code panel |
| `.model.busy` | Spinning gradient ring | The model while it writes |
| `.pm.stamp` | A mark stamps onto a figure | Verification marks |

An element that moves between two places keeps its size: `moveCard()`
in `stages/kit/collection.js` moves a copy of a card from one element to
another. Scaling a small element to the box of a large one, as an earlier
helper did with a 6 px square sent to a list row, draws a stretched blue
lens across the frame.

A stage can add its own motion code where the kit has no helper; keep it in
the stage's script, use `later()` for its timers and `rectOf()` for
positions. `stages/preparing-knowledge.html` moves clones of whole elements
(tiles flying into a store) and draws straight dashed lines on a chart; a
helper used by a second stage moves into the kit.

The progress bar in the eyebrow (`.steps4`) takes one `<i>` per method
step, so it works for any number of steps; `model-to-application` has five.

`stream()` appends each token with `insertAdjacentHTML`, so every token must
be complete HTML. A table streamed as an opening `<table>` token followed by
row tokens is closed by the browser after the first token, and the rows land
outside it as plain text. Stream rows as complete elements (for example one
`<div>` grid row per token).

Beats are driven from the step script: `go(scene, beat)` reveals elements
with `data-b ≤ beat` and runs `HOOKS[scene][beat]`. Timers must use
`later(ms, fn)` so that they stop when the scene changes. Keep each
animation shorter than the narration of its beat, or give the step a `min`
so that the recorder waits for it.

A hook that uses an element written by `stream()` in an earlier beat
(a citation to point at, for example) finds nothing if the stream has not
finished. Give the streaming beat a `min` longer than the stream, including
the 1.6× delay after tokens longer than 30 characters, and let the later
hook return when the element is missing. `review.py stage` waits only
`min` (or `--wait`) on each beat, so it shows such a race as a missing
popover or pointer.

Model output streams. When the method involves an AI system, show its answer
arriving token by token, in the chat window for what a user sees and in a
dark code panel for the raw response. Use answers recorded from a real
model where the method has one (`examples/rag-public/answers.json`). A long
recorded answer may be shortened with "…" on screen; the figures and
citations that remain are copied unchanged.

## 6. Narration rules

The narration is read by a voice and shown as subtitles, so it follows the
writing rules of the project (the global and project `CLAUDE.md`):

- One idea per sentence, with a named subject and a verb. Two sentences per
  beat at most.
- No antithesis ("X, not Y", "rather than", "instead of", "not just"), no
  rhetorical questions, no signposting ("there are three steps"), no
  aphorisms or punchline closers, no lists kept for rhythm, no "you".
- Consecutive beats do not open on the same frame. "The model can also ..."
  twice in a row was rewritten.
- Each step of a method gets one line on why it is needed: the problem
  that the step solves, with the example on screen ("A PDF records where
  text sits on the page, so the application first restores the reading
  order of columns and tables"). A reviewer found the first RAG shorts
  clear on what each step does and silent on why.
- Leave out counts and measurements that carry no idea of the short (the
  number of pages, passages, or vector dimensions, similarity scores).
  Keep a count only where the short is about it.
- State facts that the source supports. "Data producers now commonly offer
  AI chatbots" was the framing requested in review for the context scene.
- Describe what is on screen as it appears; the narration and the animation
  of a beat refer to the same thing.
- No reference to a session or notebook. The closing names the library, if
  there is one, and says nothing about the addresses shown.

Text for the voice: put the caption in `say` and, where it differs, the
spoken form in `speak`. Write numbers out in words in `speak` ("thirty-four
point nine seven six two", "twenty twenty-four") and space out acronyms that
are read as letters ("P C N").

Length: Kokoro `af_heart` at speed 1.1 reads about 3.1 words per second.
Each step also adds about 1 second of settle and pause. PCN has 763 spoken
words in 34 steps and lasts 4 min 40 s. Aim for 3–5 minutes.

## 7. Voice, music, and timing

- Voice: Kokoro `af_heart` at `--speed 1.1`, the voice of the accepted PCN
  version. It replaced Piper after the reviewer asked for a livelier voice. Five
  other Kokoro voices were sampled (`af_bella`, `am_michael`, `am_puck`,
  `bf_emma`, `bm_george`) and have not been compared in review, so keep
  `af_heart` unless asked.
- Narration is normalised to -16 LUFS (`loudnorm`) in `record.py`.
- Music: `--music` adds the track from `music.py`: 112 BPM, C–G–Am–F, soft
  kick, off-beat hat, bass, plucked chords, at gain 0.45 (about 6 dB under
  the voice; it was 0.3 until a reviewer found it too quiet under dense
  narration) with light ducking
  under the voice. A reviewer found the first, slow and quiet pad inaudible
  and asked for "a little upbeat". The mixed video measures about -20 dB
  mean volume; check with
  `ffmpeg -i out/<name>.mp4 -af volumedetect -f null -`.
- Video: stages are recorded at 1920x1080. The page keeps its 1280x720
  layout and is zoomed by 1.5; `record.py` corrects element boxes for the
  zoom and captures Chromium's screencast as JPEG frames at about 30 per
  second. Site walkthroughs stay at 1280x720.
- Audio is AAC at 48 kHz stereo. A 22 kHz mono track played silent on a
  reviewer's device.
- Subtitles: `record.py` divides each narration line with `captions.py` into
  cues of at most two lines of 37 characters, and shares the time of the
  line and its pause between them. The result passes `check_captions.py` of
  the explainer videos cookbook (37 characters, two lines, 20 characters per
  second). The checker lives on the cookbook branch; on a branch without it,
  run `git show cookbook/explainer-videos:website/static/cookbook-files/explainer-videos/check_captions.py > <scratch>/check_captions.py`
  and run that copy on `out/<name>.vtt`. A `.vtt` file written before this change is rewrapped with
  `python captions.py out/<name>.vtt out/<name>.vtt`.
- Step timing fields: `settle` (seconds after the beat starts before the
  narration, default 0.4; use 0.8–1.0 on a new scene), `pause` (after the
  narration; 0.35 for most steps, 2.0 on the closing), `min` (the least time
  on the beat, for animations longer than the line).

## 8. Review checklist

Before sending a video, confirm each item:

- [ ] `review.py stage` sheets read: no overlapping text, no identifier
      wrapped mid-token, no element off the frame, no large empty area.
- [ ] Every scene has motion; connectors attach to the edges of their cards.
- [ ] `review.py video` sheets read: each animation has finished by the end
      of its line.
- [ ] No page errors printed by `review.py`.
- [ ] Title and closing name no session or notebook.
- [ ] Narration passes the greps of the global `CLAUDE.md` (antithesis,
      second person, hedges, cadence shapes); run them on the `say` fields.
- [ ] Every figure and claim matches the source.
- [ ] Length 3–5 minutes (a short: 60–90 seconds, and `series.py` reports no
      errors); voice and music levels as in section 7.
- [ ] The MP4 and the VTT are sent together.

## 9. Site walkthroughs

A walkthrough script uses the step kinds `goto`, `click`, `hover`, `scroll`,
`highlight`, `wait` and `card` against the live site, with `caption`,
`show_url`, `then_show` and `spotlight` options (see `record.py` and
`videos/document-collection.json`). `overlay.js` draws the pointer, the
highlight ring, the caption bar, the address pill and title cards in the
same palette. The narration rules of section 6 and the audio settings of
section 7 apply unchanged.

## 10. History of decisions

| Date | Feedback or finding | Change |
|---|---|---|
| 2026-10-24 | First video silent on a reviewer's device | 48 kHz stereo AAC |
| 2026-10-24 | Notebook walkthrough not wanted for methods | Concept explainers with own visuals |
| 2026-10-24 | Explainer skipped how PCN is produced | Scenes for data with claim ids and for the instructions |
| 2026-10-24 | Music inaudible; asked for upbeat | 112 BPM track at gain 0.3 |
| 2026-10-24 | Needed: chatbots as a data channel, a more dynamic origins scene, streaming, site theme, hash claim ids, a different instruction example, a precise verifier, judgement with PCN's help, livelier voice | Current stage and Kokoro voice |
| 2026-10-24 | Intro and outro too tied to the session | Generic title and closing |
| 2026-10-24 | Connectors drawn during a card's slide-in pointed a few pixels off | `rectOf()` in the kit measures the resting position |
| 2026-10-24 | A second explainer (preparing documents for retrieval) made by an agent from this guide alone matched the reference; the agent listed gaps | Rules for scene ids, the kicker area, closing addresses, type sizes, illustrative visuals, empty space, stage-local motion, and the environment for checking values |
| 2026-10-24 | Closing of that video listed third-party projects | Program address only for topics built on third-party tools |
| 2026-10-24 | One cue per narration line exceeded the subtitle limits of the explainer videos cookbook | `captions.py` divides the lines into cues of two lines of 37 characters |
| 2026-10-25 | A third explainer (instructions and context in a prompt) made from this guide; the agent found no location for the caption checker, no rule for a general method without a library, and a streamed table that broke apart | Checker location, closing rule for general methods, five-step progress bar, complete-HTML tokens for `stream()` |
| 2026-10-25 | Reviewer saw no reason for a line saying that the addresses are on screen | Closing narration names the library only; Program-only closings are silent |
| 2026-10-25 | A fourth explainer (retrieval-augmented generation) made from this guide; it needed a step per failure case, a scene for testing retrieval alone, and found a pointer hook that ran before the stream it pointed into had finished | Rules for failure steps and a test scene; timing note for hooks that use streamed elements |
| 2026-10-25 | A second RAG explainer built on 13 published World Bank reports (`rag-public-reports`); the guide had no rule for attributing licensed report material, and one tile per chunk does not scale to 3,440 chunks | Source line on scenes with report material and on the closing; one square per chunk for a large store |
| 2026-10-25 | Reviewer found the long RAG explainer on published reports too technical and dense | Series of shorts of 60 to 90 seconds with one idea each; plan in `videos/series/`, templates for shorts, `series.py` |
| 2026-10-26 | Reviewer could not tell what the document source looked like, saw a stretched blue shape when passages moved, and asked whether the RAG shorts were a serious explanation of its steps | Real covers and pages from `render_pages.py`, a map of the steps of RAG in every scene, one focal object per beat, `moveCard()` without scaling; "document collection" in place of "archive"; a first short on the steps |
| 2026-10-26 | Reviewer found the RAG shorts lacking the reason for each step and carrying unneeded counts, and asked where parsing is explained | One why line per step; counts removed; Text step in the map of RAG; short "Getting text out of documents" first in part 2 |
| 2026-10-26 | Reviewer asked for at least 1080p and found the music too quiet under the narration | Stages recorded at 1920x1080 from the screencast; music gain 0.45 |
