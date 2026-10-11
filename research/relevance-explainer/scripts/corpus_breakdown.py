"""Size and breakdown of the multilingual corpus: raw characters and bytes per language (from raw/*.jsonl), tokens with
a tokenized vocabulary, planned training share, and groups by script and by size.

  python scripts/corpus_breakdown.py data/corpus_multi data/corpus_multi_v12k
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

root, tokroot = Path(sys.argv[1]), Path(sys.argv[2])
plan = json.loads((root / "plan.json").read_text())
stats = {}
for p in sorted((root / "raw").glob("*.jsonl")):
    chars = docs = 0
    with p.open() as f:
        for line in f:
            chars += len(json.loads(line)["t"])
            docs += 1
    tok = tokroot / "tok" / f"{p.stem}.bin"
    stats[p.stem] = {"docs": docs, "chars": chars, "bytes": p.stat().st_size,
                     "tokens": tok.stat().st_size // 2 if tok.exists() else 0, "planned": plan.get(p.stem, {}).get("chars", 0)}
(root / "breakdown.json").write_text(json.dumps(stats, indent=1))
T = {k: sum(s[k] for s in stats.values()) for k in ("docs", "chars", "bytes", "tokens", "planned")}
print(f"{len(stats)} languages: {T['docs'] / 1e6:.1f}M documents, {T['chars'] / 1e9:.2f}B characters, "
      f"{T['bytes'] / 1e9:.1f} GB on disk (JSONL), {T['tokens'] / 1e9:.2f}B tokens ({tokroot.name}); "
      f"planned training share {T['planned'] / 1e9:.1f}B characters (small languages repeated up to 4 times)")
order = sorted(stats, key=lambda k: -stats[k]["chars"])
print("\nlargest 15:")
for k in order[:15]:
    s = stats[k]
    print(f"  {k:10s} {s['chars'] / 1e6:7.0f}M chars  {s['tokens'] / 1e6:6.0f}M tokens  {100 * s['planned'] / T['planned']:4.1f}% of training")
print("smallest 10:")
for k in order[-10:]:
    s = stats[k]
    print(f"  {k:10s} {s['chars'] / 1e6:7.1f}M chars  {s['tokens'] / 1e6:6.1f}M tokens  {100 * s['planned'] / T['planned']:4.2f}% of training "
          f"({s['planned'] / max(1, s['chars']):.1f} passes)")
bands = [(300e6, "300M+"), (100e6, "100-300M"), (50e6, "50-100M"), (20e6, "20-50M"), (0, "under 20M")]
print("\nby size (characters available):")
for lo, name in bands:
    ks = [k for k in stats if stats[k]["chars"] >= lo and not any(stats[k]["chars"] >= l2 for l2, _ in bands if l2 > lo)]
    print(f"  {name:10s} {len(ks):4d} languages  {sum(stats[k]['chars'] for k in ks) / 1e9:6.2f}B chars  "
          f"{100 * sum(stats[k]['planned'] for k in ks) / T['planned']:5.1f}% of training")
sc = defaultdict(lambda: [0, 0, 0])
for k, s in stats.items():
    g = sc[k.split("_")[-1]]
    g[0] += 1; g[1] += s["chars"]; g[2] += s["planned"]
print("\nby script:")
for k, (n, c, pl) in sorted(sc.items(), key=lambda x: -x[1][1]):
    print(f"  {k:5s} {n:4d} languages  {c / 1e9:6.2f}B chars  {100 * pl / T['planned']:5.1f}% of training")
