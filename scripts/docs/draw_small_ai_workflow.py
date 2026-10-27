"""Animated SVG of the workflow in the cookbook "Practical Guide to Small AI from Synthetic Data".

The picture shows the seven chapters as stations and steps through them one scene at a time. Each scene gives
the figures of the running example, as the chapters report them. The animation is SMIL, built into the SVG: it
loops, needs no packages, and shows its last frame where animation is not supported. The drawing helpers follow
training/session-01-building-ai-applications/notebooks/explainers.py.

    python scripts/docs/draw_small_ai_workflow.py    # writes website/static/img/cookbook/small-ai-from-synthetic-data/workflow.svg
"""
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "website" / "static" / "img" / "cookbook" / "small-ai-from-synthetic-data" / "workflow.svg"

INK, BLUE, PALE, GREY, LIGHT = "#16315a", "#2f5f9e", "#e8f0fa", "#6b7280", "#f3f4f6"
GREEN, AMBER, LINE = "#2f8f5b", "#d9822b", "#e1e6ee"
SANS = "Helvetica, Arial, sans-serif"
MONO = "Menlo, Consolas, 'DejaVu Sans Mono', monospace"
FADE = 0.4
W, H = 1000, 460
SCENE = 7.0

STATIONS = [
    ("1", "Task and", "label set"),
    ("2", "Generation", "of examples"),
    ("3", "Checks on", "examples"),
    ("4", "Training of", "the small model"),
    ("5", "Evaluation", "on real text"),
    ("6", "Application", "to a collection"),
    ("7", "Release and", "maintenance"),
]


def anim(s, e, total, base):
    """An element visible from s to e seconds of a loop of `total` seconds; `base` is its opacity without animation."""
    pts = [(0.0, 0.0)]
    if s > 0:
        pts += [(s, 0.0)]
    pts += [(s + FADE, 1.0), (max(e, s + FADE), 1.0)]
    if e < total:
        pts += [(min(e + FADE, total), 0.0)]
    if pts[-1][0] < total:
        pts += [(total, pts[-1][1])]
    keys, vals = [], []
    for t, v in pts:
        if keys and abs(t - keys[-1]) < 1e-9:
            vals[-1] = v
            continue
        keys.append(t)
        vals.append(v)
    kt = ";".join(f"{k / total:.4f}" for k in keys)
    vv = ";".join(f"{v:g}" for v in vals)
    return f'opacity="{base}"><animate attributeName="opacity" dur="{total}s" repeatCount="indefinite" keyTimes="{kt}" values="{vv}"/>'


def group(inner, s, e, total, base):
    return f"<g {anim(s, e, total, base)}{inner}</g>"


def lines(x, y, rows, size=13.5, lh=20, fill=INK, font=SANS, weight="400"):
    return "".join(f'<text x="{x}" y="{y + i * lh}" font-size="{size}" font-family="{font}" font-weight="{weight}" '
                   f'fill="{fill}" xml:space="preserve">{escape(r)}</text>' for i, r in enumerate(rows))


def bars(x, y, rows, vmax, width=300, unit=""):
    """Horizontal bars: rows of (label, value, colour, shown value)."""
    out = []
    for i, (label, value, col, shown) in enumerate(rows):
        by = y + i * 44
        out.append(f'<text x="{x}" y="{by}" font-size="12.5" fill="{GREY}">{escape(label)}</text>')
        out.append(f'<rect x="{x}" y="{by + 7}" width="{width}" height="16" rx="3" fill="{LIGHT}"/>')
        out.append(f'<rect x="{x}" y="{by + 7}" width="{width * value / vmax:.1f}" height="16" rx="3" fill="{col}"/>')
        out.append(f'<text x="{x + width + 10}" y="{by + 20}" font-size="13" font-weight="700" fill="{col}">{escape(shown)}{unit}</text>')
    return "".join(out)


def scene_details():
    """Per scene: the heading of the detail panel, its text lines, and the drawing on the right."""
    rx, ry = 560, 262
    d = []
    # 1 task and labels
    d.append(("The task and its labels", [
        "The task assigns topics to working papers, as suggestions",
        "for curators. The labels come from the World Bank's topic",
        "list: its 95 names become 86 labels once names for the",
        "same topic are merged. Each label records its parent topic.",
    ], lines(rx, ry, ["label,parents,is_parent", "Agribusiness and Value Chains,", "    Agriculture and Food,no",
                       "Agriculture and Food,,yes", "Biodiversity,Environment,no", "..."], size=12.5, lh=19, font=MONO)
       + f'<text x="{rx}" y="{ry - 26}" font-size="12" font-weight="700" fill="{GREY}">labels.csv</text>'))
    # 2 generation
    d.append(("Generation with a large model", [
        "A plan of 1,000 requests offers each label 92 to 95 times.",
        "gpt-oss-120b writes 20 sentences per request. A JSON schema",
        "lets it use only names from the label file. The run gave",
        "20,074 sentences for 1.96 US dollars at the listed price.",
    ], f'<text x="{rx}" y="{ry - 26}" font-size="12" font-weight="700" fill="{GREY}">ONE GENERATED EXAMPLE</text>'
       + f'<rect x="{rx}" y="{ry - 14}" width="410" height="128" rx="8" fill="{PALE}" stroke="{BLUE}"/>'
       + lines(rx + 12, ry + 8, ['"text": "The education ministry\'s', '  policy brief argues that reducing', '  poverty requires expanding free',
                                 '  primary schooling to rural', '  low-income communities.",', '"labels": ["Education", "Poverty"]'],
               size=12, lh=18, font=MONO)))
    # 3 checks
    d.append(("Checks on the generated examples", [
        "The checker counts unknown labels, repeated texts and texts",
        "that name a label, and removes the faulty rows. Each set then",
        "trains a model, scored on 1,492 real abstracts. The set with",
        "topic names allowed trained the strongest model.",
    ], f'<text x="{rx}" y="{ry - 26}" font-size="12" font-weight="700" fill="{GREY}">MICRO F1 ON REAL ABSTRACTS, BEST RUN</text>'
       + bars(rx, ry, [("Public dataset of 2024", 0.406, GREY, "0.406"),
                       ("gpt-oss-120b, topic names avoided", 0.387, GREY, "0.387"),
                       ("gpt-oss-120b, topic names allowed", 0.446, BLUE, "0.446")], 0.55)))
    # 4 training
    d.append(("Training of the small model", [
        "The examples are split by request, so that sentences from one",
        "reply never sit on both sides of the split. DeBERTa-v3-small",
        "is fine-tuned for four epochs: 2 minutes 10 seconds on one",
        "H100 GPU. The threshold is chosen on the dev split.",
    ], f'<text x="{rx}" y="{ry - 26}" font-size="12" font-weight="700" fill="{GREY}">SPLIT OF THE PUBLIC DATASET BY REQUEST</text>'
       + bars(rx, ry, [("Train: 1,611 requests", 32712, BLUE, "32,712"),
                       ("Dev: 201 requests", 4109, GREY, "4,109"),
                       ("Test: 202 requests", 4089, GREY, "4,089")], 36000, unit=" sentences", width=250)))
    # 5 evaluation
    d.append(("Evaluation on real documents", [
        "1,792 working-paper abstracts keep their catalogue subjects,",
        "mapped to 45 of the labels by a 71-row crosswalk. The",
        "threshold is set on 1,492 abstracts and the scores are",
        "reported on the other 300, with a large model as baseline.",
    ], f'<text x="{rx}" y="{ry - 26}" font-size="12" font-weight="700" fill="{GREY}">MICRO F1 ON THE 300 REPORTING ABSTRACTS</text>'
       + bars(rx, ry, [("Small model, public dataset", 0.426, GREY, "0.426"),
                       ("Small model, topic names allowed", 0.460, BLUE, "0.460"),
                       ("gpt-oss-120b, asked directly", 0.511, AMBER, "0.511")], 0.6)))
    # 6 application
    d.append(("Application to a document collection", [
        "Each abstract is scored sentence by sentence and the scores",
        "are averaged: 0.17 seconds per abstract on four CPU cores.",
        "A frequency check compares how often each label is suggested",
        "with the reference. Over-assigned labels go back to step 2.",
    ], f'<text x="{rx}" y="{ry - 26}" font-size="12" font-weight="700" fill="{GREY}">ABSTRACTS WITH "MEASURING POVERTY", OF 1,792</text>'
       + bars(rx, ry, [("Small model, public dataset", 1298, AMBER, "1,298"),
                       ("Small model, topic names allowed", 216, BLUE, "216"),
                       ("Reference subjects", 48, GREEN, "48")], 1400)))
    # 7 release
    card = (lambda x, title, rows: f'<rect x="{x}" y="{ry - 14}" width="195" height="128" rx="8" fill="#ffffff" stroke="{BLUE}" stroke-width="1.4"/>'
            f'<text x="{x + 12}" y="{ry + 8}" font-size="13" font-weight="700" fill="{INK}">{title}</text>'
            + lines(x + 12, ry + 32, rows, size=12, lh=18, fill=GREY))
    d.append(("Release and maintenance", [
        "A dataset card names the generating model, its terms and the",
        "prompt version. A model card reports the scores on real",
        "documents and the known errors. The model is retrained when",
        "the labels change or its score falls by more than 0.03.",
    ], card(rx, "Dataset card", ["generator and licence", "prompt version", "check counts", "known issues"])
       + card(rx + 215, "Model card", ["encoder and licence", "real-document scores", "known errors", "intended use"])))
    return d


CAPTIONS = [
    "1  ·  A narrow task with a fixed list of labels",
    "2  ·  A large model writes labelled training sentences",
    "3  ·  The checks remove faulty examples. Each set is then compared on real text",
    "4  ·  A small encoder learns the labels from the generated sentences",
    "5  ·  Real documents with independent labels measure the model",
    "6  ·  The small model runs alone over the collection",
    "7  ·  Two cards document the model and its data. A rule says when to retrain",
]


def workflow_svg():
    n = len(STATIONS)
    T = SCENE * n
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" style="max-width:100%;height:auto" font-family="{SANS}" role="img" aria-labelledby="t d">',
         '<title id="t">Workflow of the guide</title>',
         f'<desc id="d">{escape(ALT)}</desc>',
         (f'<defs><marker id="ar" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{BLUE}"/></marker>'
         f'<marker id="am" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{AMBER}"/></marker></defs>'),
         f'<rect width="{W}" height="{H}" rx="12" fill="#ffffff" stroke="{LINE}"/>',
         f'<text x="24" y="34" font-size="17" font-weight="700" fill="{INK}">From a label list to a small model</text>']
    bw, gap, x0, y0, bh = 120, 18, 26, 64, 66
    xs = [x0 + i * (bw + gap) for i in range(n)]
    # who does the work: the large model in steps 2 and 5, the small model from step 4
    for i, label, col in [(1, "large model", AMBER), (4, "large model as baseline", AMBER), (3, "small model", BLUE), (5, "small model", BLUE)]:
        p.append(f'<text x="{xs[i] + bw / 2}" y="{y0 + bh + 18}" text-anchor="middle" font-size="11" fill="{col}">{label}</text>')
    for i, (num, a, b) in enumerate(STATIONS):
        x = xs[i]
        p.append(f'<rect x="{x}" y="{y0}" width="{bw}" height="{bh}" rx="10" fill="{PALE}" stroke="{BLUE}" stroke-width="1.4"/>'
                 f'<text x="{x + 12}" y="{y0 + 22}" font-size="13" font-weight="700" fill="{BLUE}">{num}</text>'
                 f'<text x="{x + 12}" y="{y0 + 40}" font-size="12.5" font-weight="700" fill="{INK}">{escape(a)}</text>'
                 f'<text x="{x + 12}" y="{y0 + 56}" font-size="12.5" font-weight="700" fill="{INK}">{escape(b)}</text>')
        hi = f'<rect x="{x - 3}" y="{y0 - 3}" width="{bw + 6}" height="{bh + 6}" rx="12" fill="none" stroke="{INK}" stroke-width="3"/>'
        p.append(group(hi, i * SCENE, (i + 1) * SCENE, T, 0))
        if i < n - 1:
            p.append(f'<line x1="{x + bw + 1}" y1="{y0 + bh / 2}" x2="{x + bw + gap - 2}" y2="{y0 + bh / 2}" stroke="{BLUE}" stroke-width="2" marker-end="url(#ar)"/>')
    # feedback from the frequency check (6) to generation (2), drawn under the stations
    fx1, fx2, fy = xs[5] + bw / 2, xs[1] + bw / 2, y0 + bh + 48
    loop = (f'<path d="M{fx1},{y0 + bh + 26} V{fy} H{fx2} V{y0 + bh + 26}" fill="none" stroke="{AMBER}" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#am)"/>'
            f'<text x="{(fx1 + fx2) / 2}" y="{fy + 18}" text-anchor="middle" font-size="12" fill="{AMBER}">an over-assigned label is fixed in the training data (recipe 6.3)</text>')
    p.append(f'<g opacity="0.35">{loop}</g>')
    p.append(group(loop, 5 * SCENE + 3.0, T, T, 0))
    # detail panel
    py = 200
    p.append(f'<rect x="24" y="{py}" width="{W - 48}" height="200" rx="10" fill="#ffffff" stroke="{LINE}"/>')
    for i, (head, text, right) in enumerate(scene_details()):
        inner = (f'<text x="44" y="{py + 34}" font-size="15.5" font-weight="700" fill="{INK}">{escape(head)}</text>'
                 + lines(44, py + 72, text, size=13.5, lh=24) + right)
        p.append(group(inner, i * SCENE + 0.4, (i + 1) * SCENE - 0.2, T, 0))
    for i, label in enumerate(CAPTIONS):
        p.append(group(f'<text x="24" y="{H - 30}" font-size="14.5" font-weight="700" fill="{BLUE}">{escape(label)}</text>',
                       i * SCENE, (i + 1) * SCENE, T, 0))
    p.append("</svg>")
    return "".join(p)


ALT = ("Animation of the seven steps of the guide. 1: the World Bank topic list gives 86 labels. 2: gpt-oss-120b writes "
       "20,074 labelled sentences for 1.96 dollars. 3: the checks remove faulty rows; on 1,492 real abstracts the set with "
       "topic names allowed trains the best model, micro F1 0.446 against 0.406 for the public dataset and 0.387 with names "
       "avoided. 4: DeBERTa-v3-small is fine-tuned on sentences split by request, in about two minutes on one GPU. 5: on "
       "300 real abstracts the small model scores 0.460 and gpt-oss-120b 0.511. 6: the small model scores an abstract in "
       "0.17 seconds on four CPU cores; the first model suggested Measuring Poverty for 1,298 of 1,792 abstracts, the "
       "names-allowed model for 216, against 48 in the reference, and over-assigned labels are fixed in the training data. "
       "7: a dataset card and a model card document the release, and a rule says when to retrain.")


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(workflow_svg())
    print(f"wrote {OUT.relative_to(ROOT)}")
