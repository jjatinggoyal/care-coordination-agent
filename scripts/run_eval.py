#!/usr/bin/env python3
"""Run the case against many worlds and report how the system behaves.

    python scripts/run_eval.py -n 6
    python scripts/run_eval.py -n 12 --shuffle

The personas are the eval set. Because the world is seeded, a bad run is
reproducible: the report prints the seed and clinic behaviour for every case, so
`run_case.py --seed N --clinic X --transcripts` replays exactly what happened.

This is slow and it costs money -- each case is dozens of real model calls. That
is the honest price of evaluating a system whose integration surface is a
conversation.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dme.clock import Clock
from dme.engine import Engine
from dme.llm import LLM, Usage
from dme.loader import load_case
from dme.model import CaseStatus, SupplierStatus
from dme.sim.personas import CLINIC_PERSONAS
from dme.sim.world import World

CLINIC_MIX = ["stalls_once", "prompt", "stalls_once", "black_hole", "miscodes", "stalls_once"]


async def run_one(seed: int, clinic: str, shuffle: bool, usage: Usage) -> dict:
    case, _ = load_case()
    llm = LLM(usage=usage)
    clock = Clock(now=case.opened_at)
    world = World(
        llm=llm, seed=seed, clinic_persona_key=clinic, hcpcs=case.hcpcs,
        patient_name=case.patient.name, pcp_name=case.pcp_name,
        practice=case.pcp_practice, equipment=case.equipment,
    )
    world.assign(list(case.suppliers), shuffle=shuffle)

    engine = Engine(case=case, world=world, llm=llm, clock=clock)
    await engine.run()

    elapsed = clock.now - case.opened_at
    qualified = [s for s in case.suppliers.values() if s.status is SupplierStatus.QUALIFIED]
    contacted = [s for s in case.suppliers.values() if s.attempts]
    return {
        "seed": seed,
        "clinic": clinic,
        "status": case.status.value,
        "escalation": case.escalation.reason if case.escalation else "",
        "days": elapsed.days + elapsed.seconds / 86400,
        "calls": engine.calls_placed,
        "suppliers_touched": len(contacted),
        "calls_to_first_qualified": next(
            (
                i + 1
                for i, s in enumerate(case.suppliers.values())
                if s.status is SupplierStatus.QUALIFIED
            ),
            None,
        ),
        "qualified": len(qualified),
        "vetoed": engine.vetoed_answers,
        "blocked": engine.fabrications_blocked,
        "order": case.order.status.value,
        "coded": case.order.coded_as or "",
    }


async def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep the case across seeded worlds.")
    ap.add_argument("-n", type=int, default=6, help="cases to run")
    ap.add_argument("--shuffle", action="store_true", help="reshuffle supplier personas per case")
    ap.add_argument("--seed0", type=int, default=1)
    args = ap.parse_args()

    usage = Usage()
    rows = []
    print(f"\n  running {args.n} case(s) — each one is dozens of real model calls\n")
    header = (
        f"  {'seed':>4}  {'clinic':<12} {'outcome':<22} {'days':>5} {'calls':>6} "
        f"{'vetoed':>7} {'blocked':>8}"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    for i in range(args.n):
        seed = args.seed0 + i
        clinic = CLINIC_MIX[i % len(CLINIC_MIX)]
        try:
            row = await run_one(seed, clinic, args.shuffle, usage)
        except Exception as exc:
            print(f"  {seed:>4}  {clinic:<12} {'CRASHED: ' + type(exc).__name__:<22}")
            continue
        rows.append(row)
        outcome = row["escalation"] or row["status"]
        print(
            f"  {row['seed']:>4}  {row['clinic']:<12} {outcome:<22} "
            f"{row['days']:>5.1f} {row['calls']:>6} {row['vetoed']:>7} {row['blocked']:>8}"
        )

    if not rows:
        print("\n  no completed runs\n")
        return 1

    resolved = [r for r in rows if r["status"] == "closed_delivered"]
    reasons = Counter(r["escalation"] for r in rows if r["escalation"])

    print()
    print("  ── summary " + "─" * 56)
    print(f"  resolved with no human:   {len(resolved)}/{len(rows)}"
          f"  ({100 * len(resolved) / len(rows):.0f}%)")
    if resolved:
        print(f"  median days to delivery:  {statistics.median(r['days'] for r in resolved):.1f}")
    print(f"  median calls per case:    {statistics.median(r['calls'] for r in rows):.0f}")
    found = [r["calls_to_first_qualified"] for r in rows if r["calls_to_first_qualified"]]
    if found:
        print(f"  suppliers tried to find one that works (median): {statistics.median(found):.0f}")
    print(f"  answers vetoed as ungrounded:  {sum(r['vetoed'] for r in rows)}")
    print(f"  invented identifiers blocked: {sum(r['blocked'] for r in rows)}")
    if reasons:
        print("\n  escalations, by reason:")
        for reason, count in reasons.most_common():
            print(f"    {count:>3}  {reason}")
    print(f"\n  {usage.summary()}")
    print("\n  replay any row:  python scripts/run_case.py --seed N --clinic X --transcripts\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
