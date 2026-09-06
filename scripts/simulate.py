#!/usr/bin/env python3
"""Start the configurable simulator.

    python scripts/simulate.py            # http://127.0.0.1:8800
    python scripts/simulate.py 9000
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dme.web import serve

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8800
    serve(port)
