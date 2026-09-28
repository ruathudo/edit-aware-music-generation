"""
Embed the listening-test items into the page.

    python scripts/build_listening_page.py

Reads `results/listening/items.json` (from make_listening_set.py) and
`scripts/listening_page.html`, and writes the self-contained page to
`results/listening/listening-test.html`. The page carries only opaque ids and
notes; the answer key stays in `results/listening/key.json`.
"""
import json
from pathlib import Path

items = json.loads(Path("results/listening/items.json").read_text())
page = Path("scripts/listening_page.html").read_text()
assert page.count("/*ITEMS*/null") == 1
out = Path("results/listening/listening-test.html")
out.write_text(page.replace("/*ITEMS*/null", json.dumps(items, separators=(",", ":"))))
print(f"{len(items['items'])} items -> {out} ({out.stat().st_size // 1024} KB)")
