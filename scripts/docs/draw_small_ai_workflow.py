"""Animated SVG of the workflow in the cookbook "Practical Guide to Small AI from Synthetic Data".

The picture shows the seven chapters as numbered nodes on one line, with one figure of the running example under
each, and the feedback from the frequency check of chapter 6 to generation. The colour of a node says which model
does the work. A highlight moves from node to node with CSS animation, which browsers stop when the reader has
asked for reduced motion; the picture is complete without it. The helpers at the top are shared with
draw_small_ai_loop.py.

    python scripts/docs/draw_small_ai_workflow.py    # writes website/static/img/cookbook/small-ai-from-synthetic-data/workflow.svg
"""

from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "website" / "static" / "img" / "cookbook" / "small-ai-from-synthetic-data" / "workflow.svg"

INK, BLUE, AMBER, GREY, LINE, PALE = "#16315a", "#2f5f9e", "#c96d16", "#6b7280", "#d5dbe5", "#f5f7fa"
SANS = "Helvetica, Arial, sans-serif"
STEP = 3.0  # seconds per node

# (name, figure, unit, who): who is "large" (large model), "small" (small model) or "" (data and people)
STATIONS = [
    ("Labels", "86", "labels", ""),
    ("Generate", "20,074", "sentences", "large"),
    ("Check", "0.446", "micro F1, best set", ""),
    ("Train", "2 min", "on one GPU", "small"),
    ("Evaluate", "0.460", "micro F1 on 300 abstracts", ""),
    ("Apply", "0.17 s", "per abstract on CPU", "small"),
    ("Release", "2 cards", "dataset and model", ""),
]
COLOUR = {"large": AMBER, "small": BLUE, "": GREY}


def svg_open(w: int, h: int, title: str, desc: str, css: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'style="max-width:100%;height:auto" font-family="{SANS}" role="img" aria-labelledby="t d">'
            f'<title id="t">{escape(title)}</title><desc id="d">{escape(desc)}</desc>'
            f'<style>{css}@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; }} }}</style>'
            f'<rect width="{w}" height="{h}" rx="12" fill="#ffffff" stroke="{LINE}"/>')


def marker(name: str, colour: str) -> str:
    return (f'<marker id="{name}" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="8" refY="5" '
            f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{colour}"/></marker>')


def text(x, y, s, size=14, fill=INK, weight=400, anchor="middle") -> str:
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}">{escape(s)}</text>')


def legend(x: int, y: int, items: list[tuple[str, str]]) -> str:
    out, cx = [], x
    for label, colour in items:
        out.append(f'<circle cx="{cx + 6}" cy="{y - 4}" r="6" fill="{colour}"/>' + text(cx + 18, y, label, 13, GREY, anchor="start"))
        cx += 30 + 7.2 * len(label)
    return "".join(out)


def workflow_svg() -> str:
    n = len(STATIONS)
    total = STEP * n
    W, H, y, r = 1000, 345, 165, 27
    xs = [80 + i * 140 for i in range(n)]
    span = xs[-1] - xs[0]
    css = (f"@keyframes on {{ 0% {{ opacity: 0 }} 3% {{ opacity: 1 }} {100 / n - 1:.2f}% {{ opacity: 1 }} "
           f"{100 / n + 2:.2f}% {{ opacity: 0 }} 100% {{ opacity: 0 }} }}"
           f".ring {{ opacity: 0; animation: on {total}s linear infinite; }}"
           f"@keyframes run {{ from {{ stroke-dashoffset: {span} }} to {{ stroke-dashoffset: 0 }} }}"
           f".run {{ stroke-dasharray: {span}; stroke-dashoffset: {span}; animation: run {total}s linear infinite; }}")
    p = [svg_open(W, H, "Workflow of the guide", ALT, css),
         f'<defs>{marker("fb", AMBER)}</defs>',
         text(28, 38, "From a label list to a small model", 18, INK, 700, "start")]
    # feedback from Apply (6) to Generate (2)
    a, b = xs[5], xs[1]
    p.append(f'<path d="M{a},{y - r - 4} C{a},{y - 92} {b},{y - 92} {b},{y - r - 6}" fill="none" stroke="{AMBER}" '
             f'stroke-width="2" stroke-dasharray="6 5" marker-end="url(#fb)"/>')
    p.append(text((a + b) / 2, y - 88, "over-assigned labels go back to generation", 13, AMBER))
    # the line and its progress
    p.append(f'<line x1="{xs[0]}" y1="{y}" x2="{xs[-1]}" y2="{y}" stroke="{LINE}" stroke-width="4"/>')
    p.append(f'<line class="run" x1="{xs[0]}" y1="{y}" x2="{xs[-1]}" y2="{y}" stroke="{INK}" stroke-width="4"/>')
    for i, (name, figure, unit, who) in enumerate(STATIONS):
        x, col = xs[i], COLOUR[who]
        p.append(f'<circle class="ring" style="animation-delay:{i * STEP}s" cx="{x}" cy="{y}" r="{r + 7}" '
                 f'fill="none" stroke="{INK}" stroke-width="3"/>')
        p.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{col}"/>' + text(x, y + 6, str(i + 1), 18, "#ffffff", 700))
        p.append(text(x, y + 58, name, 15.5, INK, 700))
        p.append(text(x, y + 88, figure, 21, col if who else INK, 700))
        p.append(text(x, y + 107, unit, 12, GREY))
    p.append(legend(28, H - 22, [("large model", AMBER), ("small model", BLUE), ("data and people", GREY)]))
    p.append("</svg>")
    return "".join(p)


ALT = ("The seven steps of the guide on one line, with a figure of the running example under each: 1 labels, 86; "
       "2 generation by a large model, 20,074 sentences; 3 checks, micro F1 of 0.446 for the best generated set; "
       "4 training of the small model, two minutes on one GPU; 5 evaluation, micro F1 of 0.460 on 300 abstracts; "
       "6 application, 0.17 seconds per abstract on CPU; 7 release, a dataset card and a model card. A dashed arrow "
       "from step 6 to step 2 shows that over-assigned labels go back to generation.")


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(workflow_svg())
    print(f"wrote {OUT.relative_to(ROOT)}")
