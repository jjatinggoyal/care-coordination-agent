"""Turn a finished run into a document the viewer can render.

State snapshots are produced by replaying the ledger through the real reducer,
not by reimplementing the rules in JavaScript. The viewer is deliberately dumb:
it draws frames, it does not decide what a frame contains. That keeps the demo
honest -- what you see on screen is what the engine actually computed, and it
cannot drift away from the system as the system changes.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .clock import CENTRAL
from .loader import load_case
from .model import Case, CaseStatus
from .reducer import apply

# Which trace entries involved a language model. Everything else is a decision
# the deterministic policy made on its own -- the distinction the viewer draws.
MODEL_TOUCHED = {
    "SupplierCalled",
    "ClinicCalled",
    "SupplierFactsObserved",
    "PatientContacted",
}

# Who is responsible for each thing that happens, and in one line, why.
#
# This exists because "where does the AI sit" is the question this design is an
# answer to, and an answer you have to narrate is weaker than one you can watch.
# Four actors: the policy decides, the model speaks and reads, the engine keeps
# time and performs, the world does things we cannot see or control.
ATTRIBUTION: dict[str, tuple[str, str]] = {
    "SupplierCalled":         ("llm",    "our agent held the call; the other side was a model too"),
    "ClinicCalled":           ("llm",    "our agent held the call; the other side was a model too"),
    "SupplierFactsObserved":  ("llm",    "extractor read the transcript, then Python checked every quote"),
    "PatientContacted":       ("llm",    "our agent rang them; a guard stops it quoting a figure"),
    "SupplierQualified":      ("policy", "policy.gate_verdict: four explicit yeses"),
    "SupplierDisqualified":   ("policy", "policy.gate_verdict: one no is enough"),
    "SupplierUnreachable":    ("policy", "policy.supplier_due_at returned None — no attempts left"),
    "OrderRequested":         ("engine", "performed the action the policy chose"),
    "OrderPromised":          ("policy", "policy.promise_window turned their words into a deadline"),
    "CommitmentMade":         ("policy", "policy.promise_window: their timeframe, capped"),
    "CommitmentFulfilled":    ("world",  "they did what they said they would"),
    "CommitmentBroken":       ("engine", "the verification time passed and nothing arrived"),
    "OrderSentToSupplier":    ("engine", "performed the action the policy chose"),
    "OrderReceived":          ("world",  "it turned up — nobody was on a call when it did"),
    "OrderRefused":           ("world",  "the clinic declined"),
    "ConsentRecorded":        ("world",  "the patient answered"),
    "DeliveryScheduled":      ("policy", "a confirmed slot, verified rather than assumed"),
    "HumanTaskRequested":     ("policy", "one step a person must take; the case carries on"),
    "HumanTaskCompleted":     ("world",  "a person finished their step"),
    "Escalated":              ("policy", "a terminal condition, from a closed set of seven"),
    "CaseClosed":             ("policy", "every gate satisfied"),
    "CaseOpened":             ("engine", "the case was created"),
}


def attribute(kind: str, event_type: str) -> tuple[str, str]:
    """Who did this, and why. Decisions are always the policy, by construction."""
    if kind == "decision":
        return "policy", "policy.decide() — pure function of state and the clock"
    return ATTRIBUTION.get(event_type, ("engine", "performed by the engine"))


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(CENTRAL).isoformat() if value else None


def _supplier_frame(case: Case) -> dict[str, Any]:
    return {
        sid: {
            "status": s.status.value,
            "calls": len(s.attempts),
            "out": s.disqualified_because,
            "facts": {
                k: {
                    "v": f.value.value if hasattr(f.value, "value") else f.value,
                    "src": f.source,
                }
                for k, f in s.facts.items()
            },
        }
        for sid, s in case.suppliers.items()
    }


def _frame(case: Case) -> dict[str, Any]:
    return {
        "case": case.status.value,
        "order": {
            "status": case.order.status.value,
            "coded_as": case.order.coded_as,
            "sent_to": case.order.sent_to_supplier,
            "attempts": len(case.order.attempts),
            "faxed": case.order.faxed_at is not None,
        },
        "patient": {
            "cost_explained": case.patient_track.cost_explained,
            "consent": case.patient_track.consent_given.value,
            "attempts": case.patient_track.contact_attempts,
        },
        "suppliers": _supplier_frame(case),
        "tasks": [
            {
                "kind": t.kind.value,
                "blocks": t.blocks.value if t.blocks else None,
                "open": t.open,
                "note": t.note,
            }
            for t in case.human_tasks.values()
        ],
        "chosen": case.chosen_supplier,
        "delivery_at": _iso(case.delivery_scheduled_for),
        "escalation": case.escalation.reason if case.escalation else None,
    }


def build_run(engine, case: Case, clock, assumptions: list[str], world_note: str = "") -> dict:
    """Everything the viewer needs, in one JSON-serialisable document."""
    replay, _ = load_case()
    events = {e.seq: e for e in engine.ledger.events}

    steps: list[dict] = []
    frames: list[dict] = []
    for entry in engine.trace:
        if entry["kind"] == "event":
            apply(replay, events[entry["seq"]])
        steps.append(
            {
                "kind": entry["kind"],
                "at": _iso(entry["at"]),
                "type": entry["type"],
                "line": entry.get("line", ""),
                "why": entry.get("why", ""),
                "until": _iso(entry.get("until")),
                "call_id": entry.get("call_id"),
                "supplier_id": entry.get("supplier_id"),
                "topic": entry.get("topic"),
                "ask": entry.get("ask", []),
                "reason": entry.get("reason"),
                "model": entry["kind"] == "event" and entry["type"] in MODEL_TOUCHED,
                "by": attribute(entry["kind"], entry["type"])[0],
                "attribution": attribute(entry["kind"], entry["type"])[1],
            }
        )
        frames.append(_frame(replay))

    elapsed = clock.now - case.opened_at
    calls = {
        call_id: {
            "with": engine.call_meta.get(call_id, {}).get("with", ""),
            "blocked": engine.call_meta.get(call_id, {}).get("blocked", []),
            "lines": [{"who": who, "text": text} for who, text in transcript.lines],
            **engine.call_notes.get(call_id, {}),
        }
        for call_id, transcript in engine.transcripts.items()
    }

    return {
        "case": {
            "id": case.case_id,
            "patient": {
                "name": case.patient.name,
                "age": case.patient.age,
                "coverage": case.patient.coverage,
                "supplemental": case.patient.has_supplemental,
                "zip": case.patient.zip_code,
                "zip_assumed": case.patient.zip_is_assumed,
            },
            "equipment": case.equipment,
            "hcpcs": case.hcpcs,
            "pcp": f"{case.pcp_name}, {case.pcp_practice}",
            "opened_at": _iso(case.opened_at),
            "assumptions": assumptions,
            "world": world_note,
        },
        "suppliers": [
            {"id": s.supplier_id, "name": s.name, "phone": s.phone, "address": s.address}
            for s in case.suppliers.values()
        ],
        "steps": steps,
        "frames": frames,
        "calls": calls,
        "messages": [
            {
                "at": _iso(m["at"]),
                "topic": m["topic"],
                "text": m["text"],
                "from_model": m["from_model"],
            }
            for m in engine.messages
        ],
        "outcome": {
            "status": case.status.value,
            "resolved": case.status is CaseStatus.CLOSED_DELIVERED,
            "escalation": case.escalation.reason if case.escalation else None,
            "packet": case.escalation.packet if case.escalation else None,
            "tasks": [
            {
                "kind": t.kind.value,
                "blocks": t.blocks.value if t.blocks else None,
                "open": t.open,
                "note": t.note,
            }
            for t in case.human_tasks.values()
        ],
        "chosen": case.chosen_supplier,
            "chosen_name": (
                case.suppliers[case.chosen_supplier].name if case.chosen_supplier else None
            ),
            "delivery_at": _iso(case.delivery_scheduled_for),
            "order_coded_as": case.order.coded_as,
            "elapsed_days": round(elapsed.days + elapsed.seconds / 86400, 2),
            "calls_placed": engine.calls_placed,
            "vetoed": engine.vetoed_answers,
            "blocked": engine.fabrications_blocked,
            "stall_note": engine.stall_note,
        },
        "usage": {
            "calls": engine.llm.usage.calls,
            "by_role": engine.llm.usage.by_role,
            "input_tokens": engine.llm.usage.input_tokens,
            "output_tokens": engine.llm.usage.output_tokens,
        },
    }


VIEWER = Path(__file__).resolve().parent.parent / "viewer"


def render(document: dict) -> str:
    """Inline the shared stylesheet and renderer into a standalone HTML file.

    The replay and the live simulator draw from the same style.css and
    replay.js; only the shell differs. A demo that disagreed with the artefact
    it produced would be worse than having neither.
    """
    return (
        (VIEWER / "template.html")
        .read_text(encoding="utf-8")
        .replace("__STYLE__", (VIEWER / "style.css").read_text(encoding="utf-8"))
        .replace("__SCRIPT__", (VIEWER / "replay.js").read_text(encoding="utf-8"))
        .replace("__RUN_DATA__", json.dumps(document, default=str))
    )
