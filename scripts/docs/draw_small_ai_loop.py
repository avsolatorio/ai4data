"""Animated SVG of the feedback loop in chapter 8 of the cookbook "Practical Guide to Small AI from Synthetic Data".

The targeted arm is drawn as a cycle of four steps (gap report, the agent's plan, generation by a writer and a
labeller, training), and the control arm as a row of five balanced rounds of the same size. Both lead to the
report set, which is scored once at the end. A dot goes once round the cycle per round while the matching round
of the control arm is highlighted. The animation is CSS and stops when the reader has asked for reduced motion.
The scores are in a separate chart (draw_small_ai_results.py). The helpers come from draw_small_ai_workflow.py.

    python scripts/docs/draw_small_ai_loop.py    # writes website/static/img/cookbook/small-ai-from-synthetic-data/loop.svg
"""

import math

from draw_small_ai_workflow import AMBER, BLUE, GREY, INK, LINE, PALE, ROOT, legend, marker, svg_open, text

OUT = ROOT / "website" / "static" / "img" / "cookbook" / "small-ai-from-synthetic-data" / "loop.svg"
ROUNDS, PER_ROUND = 5, 3.0

# cycle steps clockwise from the top: (name, detail, colour)
STEPS = [
    ("Gap report", "on the feedback set", GREY),
    ("Agent plans", "the next round", AMBER),
    ("Writer and labeller", "generate", AMBER),
    ("Train", "to convergence", BLUE),
]


def loop_svg() -> str:
    W, H = 1000, 470
    cx, cy, R, r = 250, 245, 100, 22
    total = PER_ROUND * (ROUNDS + 1)
    share = 100 / (ROUNDS + 1)
    css = (f"@keyframes spin {{ 0% {{ transform: rotate(0deg) }} {share * ROUNDS:.2f}% {{ transform: rotate({360 * ROUNDS}deg) }} "
           f"100% {{ transform: rotate({360 * ROUNDS}deg) }} }}"
           f".dot {{ transform-box: view-box; transform-origin: {cx}px {cy}px; animation: spin {total}s linear infinite; }}"
           f"@keyframes on {{ 0% {{ opacity: 0 }} 3% {{ opacity: 1 }} {share - 1:.2f}% {{ opacity: 1 }} {share + 2:.2f}% {{ opacity: 0 }} 100% {{ opacity: 0 }} }}"
           f".ring {{ opacity: 0; animation: on {total}s linear infinite; }}")
    p = [svg_open(W, H, "Feedback loop with a control arm", ALT, css),
         f'<defs>{marker("ab", INK)}{marker("ag", GREY)}</defs>',
         text(28, 38, "Targeted rounds beside a control arm", 18, INK, 700, "start"),
         text(cx, 80, "Targeted arm", 15, INK, 700), text(760, 80, "Control arm", 15, INK, 700)]
    # the cycle: arcs between the steps, clockwise
    for k in range(4):
        a0, a1 = math.radians(-90 + 90 * k + 17), math.radians(-90 + 90 * (k + 1) - 17)
        x0, y0, x1, y1 = cx + R * math.cos(a0), cy + R * math.sin(a0), cx + R * math.cos(a1), cy + R * math.sin(a1)
        p.append(f'<path d="M{x0:.1f},{y0:.1f} A{R},{R} 0 0 1 {x1:.1f},{y1:.1f}" fill="none" stroke="{LINE}" '
                 f'stroke-width="3" marker-end="url(#ab)"/>')
    p.append(f'<g class="dot"><circle cx="{cx}" cy="{cy - R}" r="{r + 7}" fill="none" stroke="{INK}" stroke-width="3"/></g>')
    places = [(cx + 38, cy - R - 12, "start"), (cx + R + 34, cy - 4, "start"), (cx, cy + R + 44, "middle"), (cx - R - 34, cy - 4, "end")]
    for k, ((name, detail, colour), (lx, ly, anchor)) in enumerate(zip(STEPS, places)):
        a = math.radians(-90 + 90 * k)
        nx, ny = cx + R * math.cos(a), cy + R * math.sin(a)
        p.append(f'<circle cx="{nx:.1f}" cy="{ny:.1f}" r="{r}" fill="{colour}"/>' + text(nx, ny + 6, str(k + 1), 16, "#ffffff", 700))
        p.append(text(lx, ly, name, 14.5, INK, 700, anchor) + text(lx, ly + 18, detail, 12.5, GREY, 400, anchor))
    p.append(text(cx, cy + 2, f"{ROUNDS} rounds", 19, INK, 700) + text(cx, cy + 22, "planned by the agent", 12, GREY))
    # the control arm: five balanced rounds
    bw, gap, bx, by = 60, 14, 582, 150
    for k in range(ROUNDS):
        x = bx + k * (bw + gap)
        p.append(f'<rect class="ring" style="animation-delay:{k * PER_ROUND}s" x="{x - 5}" y="{by - 5}" width="{bw + 10}" '
                 f'height="{bw + 10}" rx="12" fill="none" stroke="{INK}" stroke-width="3"/>')
        p.append(f'<rect x="{x}" y="{by}" width="{bw}" height="{bw}" rx="9" fill="{PALE}" stroke="{GREY}" stroke-width="1.5"/>'
                 + text(x + bw / 2, by + bw / 2 + 6, str(k + 1), 17, GREY, 700))
    p.append(text(760, by + bw + 30, "balanced plans, same number of requests", 12.5, GREY))
    # the report set
    rx, ry, rw, rh = 600, 320, 320, 76
    p.append(f'<rect class="ring" style="animation-delay:{ROUNDS * PER_ROUND}s" x="{rx - 5}" y="{ry - 5}" width="{rw + 10}" '
             f'height="{rh + 10}" rx="15" fill="none" stroke="{INK}" stroke-width="3"/>')
    p.append(f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" rx="12" fill="#ffffff" stroke="{INK}" stroke-width="1.6"/>'
             + text(rx + rw / 2, ry + 32, "Report set", 16, INK, 700)
             + text(rx + rw / 2, ry + 54, "scored once, three seeds per arm", 12.5, GREY))
    p.append(f'<line x1="760" y1="{by + bw + 42}" x2="760" y2="{ry - 9}" stroke="{GREY}" stroke-width="2" marker-end="url(#ag)"/>')
    sx, sy = cx + (R + 30) * math.cos(math.radians(50)), cy + (R + 30) * math.sin(math.radians(50))
    p.append(f'<path d="M{sx:.1f},{sy:.1f} C{sx + 60:.1f},{sy + 50:.1f} {rx - 120},{ry + rh / 2} {rx - 9},{ry + rh / 2}" fill="none" '
             f'stroke="{GREY}" stroke-width="2" marker-end="url(#ag)"/>')
    p.append(text(rx - 120, ry + rh / 2 + 26, "after the last round", 12.5, GREY))
    p.append(legend(28, H - 22, [("large model", AMBER), ("small model", BLUE), ("data and people", GREY)]))
    p.append("</svg>")
    return "".join(p)


ALT = ("The feedback loop. The targeted arm is a cycle of four steps run for five rounds: 1 a gap report on the "
       "feedback set, 2 the agent plans the next round, 3 a writer and a labeller generate examples, 4 the small "
       "model is trained to convergence. The control arm runs five rounds of the same size with balanced plans. "
       "Both arms are scored once on the report set at the end, with three seeds per arm.")


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(loop_svg())
    print(f"wrote {OUT.relative_to(ROOT)}")
