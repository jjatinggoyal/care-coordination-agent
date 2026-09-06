#!/usr/bin/env python3
"""Rebuild the HTML replay from a saved run document, without re-running the case.

    python scripts/make_viewer.py runs/viz.json -o samples/replay.html

Useful because a case costs real money to run and the viewer does not: iterate on
the presentation against a run you already paid for.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dme.export import render

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "viewer" / "template.html"


def main() -> int:
    ap = argparse.ArgumentParser(description="Render a run document as a standalone HTML replay.")
    ap.add_argument("run", type=Path, help="a run document written by run_case.py --json")
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "samples" / "replay.html")
    args = ap.parse_args()

    document = json.loads(args.run.read_text(encoding="utf-8"))
    html = render(document)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    size = args.out.stat().st_size / 1024
    print(f"  {args.out}  ({size:.0f} KB, self-contained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
