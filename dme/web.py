"""A local server that runs a case you configure, and streams it as it happens.

The replay viewer shows a case that already finished. This runs one now: you
fill in the patient, the equipment, the directory and who is on the other end of
each phone, press go, and watch the decisions arrive.

Stdlib only, loopback only. Every model call still happens for real, so a run
here costs the same as a run from the terminal -- there is no shortcut and no
canned data path. The keys stay on this side; the browser never sees one.
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import events as ev
from . import policy
from .clock import CENTRAL, Clock
from .engine import Engine
from .export import MODEL_TOUCHED, _frame, attribute
from .llm import CALLER_MODEL, SIM_MODEL, LLM, SubrequestLimit, load_env
from .loader import default_payload
from .sim.personas import CLINIC_PERSONAS, DEFAULT_CAST, PERSONAS
from .voice import Voice, voice_for

ROOT = Path(__file__).resolve().parent.parent
VIEWER = ROOT / "viewer"
MAX_BODY = 2 * 1024 * 1024


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(CENTRAL).isoformat() if value else None


def _step_from_action(action, now: datetime) -> dict:
    return {
        "kind": "decision",
        "at": _iso(now),
        "type": type(action).__name__,
        "line": "",
        "why": getattr(action, "why", ""),
        "until": _iso(getattr(action, "until", None)),
        "call_id": None,
        "supplier_id": getattr(action, "supplier_id", None),
        "topic": getattr(action, "topic", None),
        "ask": list(getattr(action, "ask", ()) or ()),
        "reason": getattr(getattr(action, "reason", None), "value", None),
        "model": False,
        "by": attribute("decision", type(action).__name__)[0],
        "attribution": attribute("decision", type(action).__name__)[1],
    }


def _step_from_event(event: ev.Event) -> dict:
    return {
        "kind": "event",
        "at": _iso(event.at),
        "type": event.kind,
        "line": event.line(),
        "why": "",
        "until": None,
        "call_id": getattr(event, "call_id", None),
        "supplier_id": getattr(event, "supplier_id", None),
        "topic": getattr(event, "topic", None),
        "ask": [],
        "reason": None,
        "model": event.kind in MODEL_TOUCHED,
        "by": attribute("event", event.kind)[0],
        "attribution": attribute("event", event.kind)[1],
    }


async def run_case_streaming(config: dict, emit, max_steps: int = 80) -> None:
    """Run one configured case, awaiting `emit(dict)` for everything that happens.

    The engine's hooks are synchronous callbacks, so they append to a buffer
    and this loop flushes it after each step. That removes the worker thread
    the first version used -- which is not just tidier, it is the only shape
    that runs on Pyodide, where `threading` imports but does nothing.
    """
    from .loader import build_case  # local import keeps module import cheap

    case, assumptions = build_case(config.get("case") or {})
    if not case.suppliers:
        await emit({"t": "error", "message": "the directory is empty — add at least one supplier"})
        return

    world_config = config.get("world") or {}
    models = config.get("models") or {}
    clinic_key = world_config.get("clinic") or "stalls_once"

    llm = LLM(model=models.get("caller") or None, api_key=config.get("api_key"))
    clock = Clock(now=case.opened_at)

    from .sim.world import World

    world = World(
        llm=llm,
        seed=int(world_config.get("seed") or 7),
        clinic_persona_key=clinic_key,
        patient_answers_after=int(world_config.get("patient_answers_after") or 1),
        patient_accepts_cost=bool(world_config.get("patient_accepts_cost", True)),
        assigned=world_config.get("personas") or None,
        sim_model=models.get("sim") or SIM_MODEL,
        hcpcs=case.hcpcs,
        patient_name=case.patient.name,
        pcp_name=case.pcp_name,
        practice=case.pcp_practice,
        equipment=case.equipment,
    )
    world.assign(list(case.suppliers))

    outbox: list[dict] = []
    await emit({
        "t": "case",
        "case": {
            "patient": {
                "name": case.patient.name, "age": case.patient.age,
                "coverage": case.patient.coverage, "zip": case.patient.zip_code,
            },
            "equipment": case.equipment, "hcpcs": case.hcpcs,
            "pcp": f"{case.pcp_name}, {case.pcp_practice}",
            "opened_at": _iso(case.opened_at),
            "assumptions": assumptions,
            "world": f"clinic behaves as: {CLINIC_PERSONAS[clinic_key].label} · seed {world.seed}",
        },
        "suppliers": [
            {"id": s.supplier_id, "name": s.name, "phone": s.phone, "address": s.address}
            for s in case.suppliers.values()
        ],
    })

    engine = Engine(case=case, world=world, llm=llm, clock=clock)
    engine.on_action = lambda a: outbox.append(
        {"t": "step", "step": _step_from_action(a, clock.now), "frame": _frame(case)}
    )
    engine.on_event = lambda e: outbox.append(
        {"t": "step", "step": _step_from_event(e), "frame": _frame(case)}
    )

    def on_transcript(label, transcript):
        call_id = (label or "").split(" ")[0]
        outbox.append({
            "t": "call",
            "id": call_id or transcript.call_id,
            "call": {
                "with": label.split("—")[-1].strip() if "—" in label else label,
                "lines": [{"who": w, "text": t} for w, t in transcript.lines],
                "blocked": list(transcript.blocked),
            },
        })

    engine.on_transcript = on_transcript

    async def flush() -> None:
        while outbox:
            await emit(outbox.pop(0))

    try:
        for _ in range(max_steps):
            alive = await engine.step()
            await flush()
            if not alive:
                break
    except SubrequestLimit:
        await flush()
        await emit({
            "t": "error",
            "message": (
                "This Worker is on Cloudflare's free plan, which allows 50 outbound "
                "requests per invocation, and every model call spends one. The case got "
                "this far and then ran out of them. Try three suppliers or fewer, or run "
                "it locally where there is no such limit — nothing about the case failed."
            ),
        })
        return
    except Exception as exc:  # a failed run is reportable, never a silent hang
        await flush()
        await emit({"t": "error", "message": f"{type(exc).__name__}: {exc}"})
        traceback.print_exc()
        return

    for call_id, note in engine.call_notes.items():
        await emit({"t": "callnote", "id": call_id, "note": note})
    for message in engine.messages:
        await emit({"t": "message", "message": {**message, "at": _iso(message["at"])}})

    elapsed = clock.now - case.opened_at
    await emit({
        "t": "done",
        "outcome": {
            "status": case.status.value,
            "resolved": case.status.value == "closed_delivered",
            "escalation": case.escalation.reason if case.escalation else None,
            "packet": case.escalation.packet if case.escalation else None,
            "delivery_at": _iso(case.delivery_scheduled_for),
            "elapsed_days": round(elapsed.days + elapsed.seconds / 86400, 2),
            "calls_placed": engine.calls_placed,
            "vetoed": engine.vetoed_answers,
            "blocked": engine.fabrications_blocked,
            "stall_note": engine.stall_note,
        },
        "usage": {
            "calls": llm.usage.calls, "by_role": llm.usage.by_role,
            "input_tokens": llm.usage.input_tokens, "output_tokens": llm.usage.output_tokens,
        },
    })


class Handler(SimpleHTTPRequestHandler):
    voice: Voice

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload, default=str).encode(), "application/json")

    def _file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            return self._json(404, {"error": "not found"})
        self._send(200, path.read_bytes(), content_type)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            raise ValueError("bad body length")
        return json.loads(self.rfile.read(length).decode())

    def do_GET(self):  # noqa: N802
        route = self.path.split("?", 1)[0]
        if route in ("/", "/index.html"):
            return self._file(VIEWER / "app.html", "text/html; charset=utf-8")
        if route == "/style.css":
            return self._file(VIEWER / "style.css", "text/css; charset=utf-8")
        if route == "/replay.js":
            return self._file(VIEWER / "replay.js", "application/javascript; charset=utf-8")
        if route == "/api/defaults":
            payload = default_payload()
            return self._json(200, {
                "case": payload,
                "default_cast": list(DEFAULT_CAST),
                "personas": [{"key": p.key, "label": p.label} for p in PERSONAS],
                "clinic_personas": [
                    {"key": c.key, "label": c.label} for c in CLINIC_PERSONAS.values()
                ],
                "models": {"caller": CALLER_MODEL, "sim": SIM_MODEL},
                "voice_available": self.voice.available,
            })
        return self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        route = self.path.split("?", 1)[0]
        if route == "/api/voice":
            return self._voice()
        if route == "/api/run":
            return self._run()
        return self._json(404, {"error": "not found"})

    def _voice(self) -> None:
        import base64

        try:
            body = self._body()
        except (ValueError, json.JSONDecodeError) as exc:
            return self._json(400, {"error": str(exc)})
        try:
            # Voice.say became a coroutine with the async refactor; this handler
            # is a synchronous BaseHTTPRequestHandler, so it owns the loop.
            audio = asyncio.run(
                self.voice.say(
                    body.get("text", ""),
                    voice_for(body.get("who", "them"), body.get("counterpart", "")),
                )
            )
        except Exception as exc:
            return self._json(200, {"error": str(exc)})
        return self._json(200, {"audio": base64.b64encode(audio).decode()})

    def _run(self) -> None:
        """Stream the run as newline-delimited JSON while it happens."""
        try:
            config = self._body()
        except (ValueError, json.JSONDecodeError) as exc:
            return self._json(400, {"error": str(exc)})

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        async def emit(item: dict) -> None:
            self.wfile.write((json.dumps(item, default=str) + "\n").encode())
            self.wfile.flush()

        try:
            asyncio.run(run_case_streaming(config, emit))
        except (BrokenPipeError, ConnectionResetError):
            return  # the tab went away
        except Exception as exc:
            traceback.print_exc()
            try:
                self.wfile.write(
                    (json.dumps({"t": "error", "message": str(exc)}) + "\n").encode()
                )
            except OSError:
                pass

    def log_message(self, fmt, *args):
        pass


def serve(port: int = 8800, open_browser: bool = True) -> None:
    load_env()
    voice = Voice()
    handler = partial(Handler)
    Handler.voice = voice
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}"
    print(f"\n  DME coordination simulator — {url}")
    print(f"  models: {CALLER_MODEL} (system) · {SIM_MODEL} (world)")
    print(f"  voice:  {'Sarvam, 8 kHz' if voice.available else 'off (no SARVAM_API_KEY)'}")
    print("  loopback only. Ctrl-C to stop.\n")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.\n")
        server.shutdown()
