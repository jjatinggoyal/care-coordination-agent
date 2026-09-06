#!/usr/bin/env python3
"""Run Eleanor's case end to end and narrate it.

    python scripts/run_case.py
    python scripts/run_case.py --transcripts          # print every phone call
    python scripts/run_case.py --clinic black_hole    # a clinic that never sends it
    python scripts/run_case.py --seed 12              # different luck on the phones

Every line beginning with -> is a decision made by policy.decide(). Every line
under it is something that happened as a result. No model was consulted about
any of the -> lines.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dme import policy
from dme.clock import CENTRAL, Clock
from dme.export import build_run, render
from dme.engine import Engine
from dme.llm import LLM
from dme.loader import load_case
from dme.model import CaseStatus, SupplierStatus
from dme.sim.personas import CLINIC_PERSONAS
from dme.sim.world import World

USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


DIM, BOLD, GREEN, RED, YELLOW, BLUE = "2", "1", "32", "31", "33", "36"


def rule(title: str = "") -> None:
    line = "─" * 78
    print(c(line if not title else f"── {title} " + "─" * (74 - len(title)), DIM))


async def main() -> int:
    ap = argparse.ArgumentParser(description="Run the DME coordination case.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--clinic", default="stalls_once", choices=sorted(CLINIC_PERSONAS))
    ap.add_argument("--transcripts", action="store_true", help="print every phone call in full")
    ap.add_argument("--shuffle", action="store_true", help="reassign supplier personas by seed")
    ap.add_argument("--patient-answers-after", type=int, default=1)
    ap.add_argument("--patient-declines", action="store_true")
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--json", type=Path, help="write the full run document to a file")
    ap.add_argument(
        "--html", type=Path, help="write a self-contained replay you can open in a browser"
    )
    args = ap.parse_args()

    case, assumptions = load_case()
    llm = LLM()
    clock = Clock(now=case.opened_at)
    world = World(
        llm=llm,
        seed=args.seed,
        clinic_persona_key=args.clinic,
        patient_answers_after=args.patient_answers_after,
        patient_accepts_cost=not args.patient_declines,
        hcpcs=case.hcpcs,
        patient_name=case.patient.name,
        pcp_name=case.pcp_name,
        practice=case.pcp_practice,
        equipment=case.equipment,
    )
    world.assign(list(case.suppliers), shuffle=args.shuffle)

    rule("case")
    print(f"  {c(case.patient.name, BOLD)}, {case.patient.age} — {case.patient.coverage}, "
          f"no supplemental plan")
    print(f"  {case.equipment} ({case.hcpcs}) · ordering physician {case.pcp_name}, "
          f"{case.pcp_practice}")
    print(f"  opened {c(clock.stamp(), BOLD)} — everyone this case needs is closed")
    print(f"  {len(case.suppliers)} suppliers in the directory · "
          f"clinic behaves as: {c(CLINIC_PERSONAS[args.clinic].label, DIM)}")
    print()
    for note in assumptions:
        head, _, tail = note.partition(":")
        print(c(f"  assumption · {head}:", YELLOW), c(tail.strip(), DIM))
    print()

    def show_action(action) -> None:
        stamp = clock.stamp()
        if isinstance(action, policy.Wait):
            if action.until > clock.now:
                print(c(f"  {stamp}  ·  wait until {action.until.astimezone(CENTRAL):%a %d %b %H:%M}"
                        f"  ({action.why})", DIM))
            return
        labels = {
            policy.CallSupplier: lambda a: f"call {case.suppliers[a.supplier_id].name}",
            policy.CallClinic: lambda a: f"call {case.pcp_practice} (attempt {a.attempt})",
            policy.FaxClinic: lambda a: f"fax {case.pcp_practice}",
            policy.ContactPatient: lambda a: f"contact {case.patient.name} about {a.topic}",
            policy.SendOrderToSupplier: lambda a: (
                f"send the written order to {case.suppliers[a.supplier_id].name}"
            ),
            policy.ScheduleDelivery: lambda a: (
                f"book delivery with {case.suppliers[a.supplier_id].name}"
            ),
            policy.Escalate: lambda a: f"escalate: {a.reason.value}",
            policy.CloseCase: lambda a: f"close case: {a.outcome}",
        }
        render = labels.get(type(action))
        name = render(action) if render else type(action).__name__
        colour = RED if isinstance(action, policy.Escalate) else BLUE
        print(f"  {c(stamp, DIM)}  {c('→', colour)}  {c(name, colour)}")
        if action.why:
            print(c(f"                       because: {action.why}", DIM))

    def show_event(event) -> None:
        text = event.line()
        colour = ""
        if "QUALIFIED" in text or "SCHEDULED" in text or "closed" in text:
            colour = GREEN
        elif "ESCALATED" in text or "BROKEN" in text:
            colour = RED
        elif "ruled out" in text or "gave up" in text:
            colour = YELLOW
        print(f"                       {c(text, colour) if colour else text}")

    def show_transcript(label, transcript) -> None:
        if not args.transcripts:
            return
        print()
        rule(label)
        for speaker, line in transcript.lines:
            who = "us " if speaker == "agent" else "them"
            print(f"  {c(who, BOLD if speaker == 'agent' else DIM)}  {line}")
        rule()
        print()

    engine = Engine(case=case, world=world, llm=llm, clock=clock)
    engine.on_action = show_action
    engine.on_event = show_event
    engine.on_transcript = show_transcript

    rule("run")
    try:
        await engine.run(max_steps=args.max_steps)
    except KeyboardInterrupt:
        print(c("\n  interrupted", YELLOW))
    print()

    rule("outcome")
    elapsed = clock.now - case.opened_at
    days = elapsed.days + elapsed.seconds / 86400
    if case.status is CaseStatus.CLOSED_DELIVERED:
        supplier = case.suppliers[case.chosen_supplier]
        print(c(f"  RESOLVED with no human involved.", GREEN))
        print(f"  {supplier.name} delivering "
              f"{case.delivery_scheduled_for.astimezone(CENTRAL):%A %d %B}")
        print(f"  written order received, coded {case.order.coded_as}")
    elif case.status is CaseStatus.ESCALATED:
        print(c(f"  ESCALATED — {case.escalation.reason}", RED))
        print(f"  {case.escalation.packet['next_step_for_human']}")
    else:
        print(c(f"  STALLED — {engine.stall_note or 'step budget exhausted'}", YELLOW))

    print()
    print(f"  simulated elapsed:  {days:.1f} days")
    print(f"  phone calls placed: {engine.calls_placed}")
    print(f"  model usage:        {llm.usage.summary()}")

    print()
    rule("supplier ledger")
    for supplier in case.suppliers.values():
        mark = {
            SupplierStatus.QUALIFIED: c("✓", GREEN),
            SupplierStatus.DISQUALIFIED: c("✗", YELLOW),
            SupplierStatus.UNREACHABLE: c("—", DIM),
            SupplierStatus.IN_PROGRESS: c("?", YELLOW),
            SupplierStatus.UNCONTACTED: c("·", DIM),
        }[supplier.status]
        detail = supplier.disqualified_because or ", ".join(
            f"{k.replace('_', ' ')}={v.render()}" for k, v in supplier.facts.items()
        )
        calls = f"{len(supplier.attempts)} call(s)" if supplier.attempts else "not called"
        print(f"  {mark} {supplier.name:<34} {c(calls, DIM)}")
        if detail:
            print(c(f"      {detail}", DIM))

    if case.escalation:
        print()
        rule("handoff packet")
        print(json.dumps(case.escalation.packet, indent=2, default=str))

    if args.json or args.html:
        document = build_run(
            engine, case, clock, assumptions, world_note=(
                f"clinic behaves as: {CLINIC_PERSONAS[args.clinic].label} · seed {args.seed}"
            )
        )
    if args.json:
        args.json.write_text(json.dumps(document, indent=2, default=str), encoding="utf-8")
        print(f"\n  run document written to {args.json}")
    if args.html:
        args.html.write_text(render(document), encoding="utf-8")
        print(f"  replay written to {args.html}  —  open it in a browser")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
