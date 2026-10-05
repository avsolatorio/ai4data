"""Fetch every catalog entry (search row plus full metadata) from a NADA instance into a JSONL file.

  python scripts/fetch_nada.py https://nada-demo.ihsn.org data/nada_demo.jsonl
"""
import json
import sys
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "relevance-explainer research (metadata-schemas test)"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.loads(r.read())


def main(base, out):
    api = base.rstrip("/") + "/index.php/api/catalog"
    rows, page = [], 1
    while True:
        res = get(f"{api}/search?format=json&ps=100&page={page}")["result"]
        rows += res["rows"]
        if not res["rows"] or len(rows) >= int(res["found"]):
            break
        page += 1
    n = 0
    with open(out, "w") as f:
        for r in rows:
            try:
                d = get(f"{api}/{urllib.parse.quote(r['idno'])}")
            except Exception as e:  # keep going; report what failed
                print("skip", r["idno"], e)
                continue
            f.write(json.dumps({"row": r, "detail": d}, ensure_ascii=False) + "\n")
            n += 1
            time.sleep(0.2)
    print(n, "of", len(rows), "written to", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
