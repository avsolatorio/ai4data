"""Guide and student similarities for the example training batch of the GISTEmbed video."""
import json
import sys

from sentence_transformers import SentenceTransformer

B = [
    ("Capital cities", "Ottawa", "Maple syrup production in Quebec"),
    ("Which city is the capital of Australia?", "Canberra is the capital city of Australia.", "Sydney is the largest city in Australia."),
    ("Who leads the government of Canada?", "The Prime Minister heads the federal government of Canada.", "What is the capital of Canada?"),
    ("How much did Uganda's economy grow in FY25?", "Real GDP growth accelerated to 6.3 percent in FY25.", "Inflation in Uganda was 3.0 percent in April 2026."),
]


def cosine(a, b):
    """Cosine similarities of normalised vectors, rounded to four decimals."""
    return [[round(float(x), 4) for x in row] for row in (a @ b.T)]


out = {"batch": B}
for name in ["WhereIsAI/UAE-Large-V1", "sentence-transformers/all-MiniLM-L6-v2", "avsolatorio/GIST-all-MiniLM-L6-v2"]:
    m = SentenceTransformer(name)
    q = m.encode([b[0] for b in B], normalize_embeddings=True)
    p = m.encode([b[1] for b in B], normalize_embeddings=True)
    n = m.encode([b[2] for b in B], normalize_embeddings=True)
    out[name] = {"qp": cosine(q, p), "qn": cosine(q, n), "qq": cosine(q, q), "pp": cosine(p, p)}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print(f"{sys.argv[1]}: similarities from {len(out) - 1} models")
