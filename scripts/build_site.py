#!/usr/bin/env python3
"""Build the deployable static site from a saved run.

    python scripts/build_site.py runs/final.json

Writes deploy/public/index.html -- the replay, self-contained. There is nothing
else in the bundle: no keys, no engine, no ability to start a run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dme.export import render

NOT_FOUND = """<!doctype html><meta charset="utf-8"><title>Not here</title>
<style>body{font:15px/1.6 system-ui;margin:12vh auto;max-width:34rem;padding:0 1.5rem;color:#0b0b0b}
a{color:#2a78d6}</style>
<h1>Nothing at this address</h1>
<p>The DME case replay is at <a href="/">the root</a>.</p>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the static site for deployment.")
    ap.add_argument("run", type=Path, help="a run document from run_case.py --json")
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "deploy" / "public")
    args = ap.parse_args()

    document = json.loads(args.run.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.html").write_text(render(document), encoding="utf-8")
    (args.out / "404.html").write_text(NOT_FOUND, encoding="utf-8")

    size = (args.out / "index.html").stat().st_size / 1024
    print(f"  {args.out / 'index.html'}  ({size:.0f} KB)")
    print("  deploy with:  cd deploy && npx wrangler deploy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
