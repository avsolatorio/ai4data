"""Animated SVG of the feedback loop in chapter 8 of the cookbook "Practical Guide to Small AI from Synthetic Data".

The picture shows the targeted arm as a cycle (gap report, agent's plan, generation with a labeller,
training and scoring) beside the control arm, and steps through them one scene at a time with the
figures of the tests reported in the chapter. The last scene gives the scores on the report sets.
The drawing helpers come from draw_small_ai_workflow.py.

    python scripts/docs/draw_small_ai_loop.py    # writes website/static/img/cookbook/small-ai-from-synthetic-data/loop.svg
"""

from html import escape

from draw_small_ai_workflow import (
    AMBER,
    BLUE,
    GREEN,
    GREY,
    INK,
    LINE,
    PALE,
    ROOT,
    SANS,
    bars,
    group,
    lines,
)

OUT = ROOT / "website" / "static" / "img" / "cookbook" / "small-ai-from-synthetic-data" / "loop.svg"
W, H = 1000, 560
SCENE = 7.0

NODES = [  # (key, x, y, two lines)
    ("gap", 40, 70, ("1  Gap report", "on the feedback set")),
    ("plan", 290, 70, ("2  Agent plans", "the round (pi)")),
    ("gen", 290, 190, ("3  Writer and", "labeller generate")),
    ("train", 40, 190, ("4  Train to convergence,", "score the feedback set")),
]
BW, BH = 200, 62


def scenes() -> list[tuple[str, str, list[str], str]]:
    """(highlighted node, heading, text lines, drawing on the right) per scene."""
    rx, ry = 560, 330
    out = []
    out.append(("gap", "The gap report", [
        "The latest model is scored on the feedback set, two thirds",
        "of the real calibration texts. The report lists the weakest",
        "labels, the most frequent confusions, over-assigned labels,",
        "labels with few examples, and the labeller's agreement.",
    ], lines(rx, ry, ["weakest:  Indigenous Peoples; Oceans, Fisheries, ...; Debt",
                      "confused (reference -> assigned, abstracts):",
                      "  Inequality and Shared Prosperity -> Measuring Poverty, 48",
                      "  Macroeconomics -> Migration, 43",
                      "over-assigned:  Migration, 152 assigned, 10 in reference"], size=12, lh=20)))
    out.append(("plan", "The agent's plan", [
        "The agent has seven tools of the harness and no others: no",
        "shell, no files. In the first topic round it put 65 of 150",
        "requests on weak labels and 85 on five confused pairs, with",
        "a note for the writer on each item.",
    ], bars(rx, ry, [("positive items, six weak labels", 65, BLUE, "65"),
                     ("contrast items, five confused pairs", 85, AMBER, "85")], 150, width=250, unit=" requests")))
    out.append(("gen", "Generation with a labeller", [
        "The writer produces labelled texts for each request. The",
        "labeller labels them again without the writer's labels, and",
        "texts on which the two disagree are dropped: 2.8 percent of",
        "ISCO titles and 4.5 percent of COICOP descriptions.",
    ], bars(rx, ry, [("ISCO-08, round 0: 18,061 titles", 97.2, GREEN, "97.2%"),
                     ("COICOP, round 0: 10,035 descriptions", 95.5, GREEN, "95.5%")], 100, width=250, unit=" agree")))
    out.append(("train", "Training and the decision", [
        "Each round trains a new model on all accepted examples, to",
        "convergence. At four epochs the COICOP model was under-",
        "trained (0.382 against 0.529 on the same data). A round is",
        "kept when it is no more than 0.03 below the best round.",
    ], bars(rx, ry, [("COICOP round 0, four epochs", 0.382, GREY, "0.382"),
                     ("COICOP round 0, to convergence", 0.529, BLUE, "0.529")], 0.7, width=250)))
    out.append(("control", "The control arm", [
        "A control arm adds the same number of requests in each",
        "round, spread evenly over the labels. Only the difference",
        "between the two arms shows what targeting adds: more data",
        "alone raised topics by 0.024 and COICOP by 0.042.",
    ], lines(rx, ry, ["same round size, balanced plan", "same training, same feedback set",
                      "final models: three seeds each", "report set scored once, at the end"], size=12.5, lh=20)))
    out.append(("report", "Scores on the report sets", [
        "Targeted rounds scored the same as the control arm in four",
        "comparisons: +0.019 and +0.004 on topics, +0.000 on ISCO,",
        "+0.001 on COICOP, with fewer examples. The large model",
        "asked directly led on COICOP by 0.16.",
    ], ""))
    return out


RESULTS = [  # task, control, targeted, large model
    ("Topics, replicate 1 (micro F1)", 0.484, 0.502, 0.511),
    ("ISCO-08 (accuracy)", 0.330, 0.331, 0.353),
    ("COICOP (accuracy)", 0.561, 0.563, 0.726),
]


def results_chart(x: int, y: int) -> str:
    out = []
    width = 250
    for i, (task, control, targeted, large) in enumerate(RESULTS):
        by = y + i * 56
        out.append(f'<text x="{x}" y="{by}" font-size="12" fill="{GREY}">{escape(task)}</text>')
        for j, (value, col) in enumerate([(control, GREY), (targeted, BLUE), (large, AMBER)]):
            out.append(f'<rect x="{x}" y="{by + 6 + j * 13}" width="{width * value / 0.8:.1f}" height="10" rx="2" fill="{col}"/>')
            out.append(f'<text x="{x + width * value / 0.8 + 6:.1f}" y="{by + 15 + j * 13}" font-size="11" fill="{col}">{value:.3f}</text>')
    legend = [("control", GREY), ("targeted", BLUE), ("gpt-oss-120b", AMBER)]
    for k, (name, col) in enumerate(legend):
        out.append(f'<rect x="{x + k * 110}" y="{y - 30}" width="10" height="10" fill="{col}"/>'
                   f'<text x="{x + 14 + k * 110}" y="{y - 21}" font-size="11.5" fill="{GREY}">{name}</text>')
    return "".join(out)


def loop_svg() -> str:
    data = scenes()
    T = SCENE * len(data)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" style="max-width:100%;height:auto" font-family="{SANS}" role="img" aria-labelledby="t d">',
         '<title id="t">Feedback loop with a control arm</title>', f'<desc id="d">{escape(ALT)}</desc>',
         (f'<defs><marker id="ar" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{BLUE}"/></marker>'
          f'<marker id="am" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{GREY}"/></marker></defs>'),
         f'<rect width="{W}" height="{H}" rx="12" fill="#ffffff" stroke="{LINE}"/>',
         f'<text x="24" y="34" font-size="17" font-weight="700" fill="{INK}">Targeted rounds beside a control arm</text>',
         f'<text x="40" y="58" font-size="12" font-weight="700" fill="{BLUE}">TARGETED ARM</text>',
         f'<text x="600" y="58" font-size="12" font-weight="700" fill="{GREY}">CONTROL ARM</text>']
    pos = {}
    for key, x, y, (a, b) in NODES:
        pos[key] = (x, y)
        p.append(f'<rect x="{x}" y="{y}" width="{BW}" height="{BH}" rx="10" fill="{PALE}" stroke="{BLUE}" stroke-width="1.4"/>'
                 f'<text x="{x + 12}" y="{y + 26}" font-size="13" font-weight="700" fill="{INK}">{escape(a)}</text>'
                 f'<text x="{x + 12}" y="{y + 45}" font-size="12.5" fill="{INK}">{escape(b)}</text>')
    arrow = f'stroke="{BLUE}" stroke-width="2" marker-end="url(#ar)"'
    p.append(f'<line x1="242" y1="101" x2="288" y2="101" {arrow}/>'
             f'<line x1="390" y1="134" x2="390" y2="188" {arrow}/>'
             f'<line x1="288" y1="221" x2="242" y2="221" {arrow}/>'
             f'<line x1="140" y1="188" x2="140" y2="134" {arrow}/>')
    # control arm: rounds in a row
    cx, cy = 600, 70
    p.append(f'<rect x="{cx}" y="{cy}" width="370" height="{BH}" rx="10" fill="#f3f4f6" stroke="{GREY}" stroke-width="1.4"/>'
             f'<text x="{cx + 12}" y="{cy + 26}" font-size="13" font-weight="700" fill="{INK}">Balanced rounds of the same size</text>'
             f'<text x="{cx + 12}" y="{cy + 45}" font-size="12.5" fill="{INK}">train to convergence, score the feedback set</text>')
    rx_, ry_ = 600, 190
    p.append(f'<rect x="{rx_}" y="{ry_}" width="370" height="{BH}" rx="10" fill="#ffffff" stroke="{INK}" stroke-width="1.4"/>'
             f'<text x="{rx_ + 12}" y="{ry_ + 26}" font-size="13" font-weight="700" fill="{INK}">Report set, scored once at the end</text>'
             f'<text x="{rx_ + 12}" y="{ry_ + 45}" font-size="12.5" fill="{INK}">three seeds per arm, interval over texts</text>')
    p.append(f'<line x1="785" y1="134" x2="785" y2="188" stroke="{GREY}" stroke-width="2" marker-end="url(#am)"/>'
             f'<path d="M242,240 C420,300 520,260 598,232" fill="none" {arrow}/>')
    boxes = {**{k: (x, y, BW, BH) for k, (x, y) in pos.items()}, "control": (cx, cy, 370, BH), "report": (rx_, ry_, 370, BH)}
    py = 290
    p.append(f'<rect x="24" y="{py}" width="{W - 48}" height="226" rx="10" fill="#ffffff" stroke="{LINE}"/>')
    for i, (key, head, text, right) in enumerate(data):
        x, y, w, h = boxes[key]
        p.append(group(f'<rect x="{x - 3}" y="{y - 3}" width="{w + 6}" height="{h + 6}" rx="12" fill="none" stroke="{INK}" stroke-width="3"/>',
                       i * SCENE, (i + 1) * SCENE, T, 0))
        drawing = right if key != "report" else results_chart(560, 350)
        inner = (f'<text x="44" y="{py + 32}" font-size="15.5" font-weight="700" fill="{INK}">{escape(head)}</text>'
                 + lines(44, py + 66, text, size=13.5, lh=24) + drawing)
        p.append(group(inner, i * SCENE + 0.4, (i + 1) * SCENE - 0.2, T, 0))
    p.append(f'<text x="24" y="{H - 18}" font-size="12" fill="{GREY}">Tests of chapter 8: topics (two replicates), ISCO-08 job titles, COICOP item descriptions; gpt-oss-120b as writer, labeller, and agent.</text>')
    p.append("</svg>")
    return "".join(p)


ALT = ("Animation of the feedback loop. The targeted arm is a cycle: a gap report on the feedback set, a plan by an "
       "agent with seven tools of the harness and no shell or files (in the first topic round, 65 of 150 requests on "
       "weak labels and 85 on five confused pairs), generation by a writer and a labeller that drops texts on which the "
       "two disagree (2.8 percent of ISCO titles and 4.5 percent of COICOP descriptions), and training to convergence "
       "(the COICOP model scored 0.382 at four epochs and 0.529 at convergence). A control arm adds rounds of the same "
       "size spread evenly. Both are scored once on a report set. Targeted minus control: +0.019 and +0.004 on topics, "
       "+0.000 on ISCO, +0.001 on COICOP. Report scores for control, targeted, and gpt-oss-120b: topics 0.484, 0.502, "
       "0.511; ISCO 0.330, 0.331, 0.353; COICOP 0.561, 0.563, 0.726.")


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(loop_svg())
    print(f"wrote {OUT.relative_to(ROOT)}")
