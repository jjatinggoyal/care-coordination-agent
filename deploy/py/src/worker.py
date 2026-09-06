"""The DME coordination simulator, running on Cloudflare Python Workers.

Same engine as the local server. `dme/policy.py`, `dme/engine.py` and
everything under `dme/agents/` are the files from the repo, copied unmodified by
scripts/build_worker.py -- there is no Workers-specific fork of the system.

What is different here, and only here:

  * outbound HTTP goes through `fetch` rather than a socket, chosen inside
    `dme/llm.py` by asking whether a `js` module exists;
  * the fixture files are inlined, because there is no filesystem;
  * the response is a TransformStream, because the run takes minutes and the
    point is to watch it happen.

Secrets come from Worker bindings. Nothing is baked into the bundle.
"""

from __future__ import annotations

import json
import traceback

from js import TextEncoder, TransformStream, URL
from workers import Response, WorkerEntrypoint

from dme.llm import CALLER_MODEL, SIM_MODEL
from dme.loader import default_payload
from dme.sim.personas import CLINIC_PERSONAS, DEFAULT_CAST, PERSONAS
from dme.voice import Voice, voice_for
from dme.web import run_case_streaming, step_once

JSON_HEADERS = {"Content-Type": "application/json", "Cache-Control": "no-store"}


def json_response(payload: dict, status: int = 200) -> Response:
    # workers-py takes plain dicts for headers; converting them first is what a
    # JS-first instinct does, and it is wrong here.
    return Response(json.dumps(payload, default=str), status=status, headers=dict(JSON_HEADERS))


class Default(WorkerEntrypoint):
    def _secret(self, name: str) -> str | None:
        value = getattr(self.env, name, None)
        return str(value) if value else None

    async def fetch(self, request):
        path = URL.new(request.url).pathname

        if path == "/api/defaults":
            return self.defaults()
        if path == "/api/step":
            return await self.step(request)
        if path == "/api/run":
            return await self.run(request)
        if path == "/api/voice":
            return await self.voice(request)
        # Anything else is the static app: app.html, style.css, replay.js.
        return await self.env.ASSETS.fetch(request)

    # Free plan: 50 subrequests per invocation, and a supplier call costs about
    # ten. Pre-filling all twelve of the brief's suppliers here would guarantee
    # every first run dies two thirds of the way through, so the hosted copy
    # ships a shorter directory and says why.
    # The whole directory. The free plan's 50-request ceiling is per invocation,
    # and the browser now drives the loop a step at a time -- so each step gets
    # its own budget and a case can be any length. Sizing the directory to fit
    # one request was the old workaround.
    HOSTED_SUPPLIERS = 12

    # And it ships a cast that resolves. The first three of the full deck are a
    # closed panel, a backorder and a phone nobody answers -- so out of the box a
    # reviewer pressing Run watched it escalate, which is a true thing about the
    # system and a terrible first impression of it.
    #
    # This tells the whole story in three calls and still ends somewhere:
    #   1. a hard no on the first question
    #   2. the money trap -- has stock, is fast, does not accept assignment, and
    #      only says so if asked
    #   3. a supplier who actually works
    # Every persona is still in the dropdown; this is only what is pre-filled.
    # The last one qualifies, so pressing Run out of the box shows a case that
    # resolves rather than one that escalates. It is the three-week supplier
    # rather than the fast one on purpose: the case still closes, but you can
    # see the system settle for what is actually available once the directory is
    # exhausted, and the delivery date says so. Every other persona is in the
    # dropdown -- this is only what is pre-filled.
    HOSTED_CAST = DEFAULT_CAST

    HOSTED_CLINIC = "stalls_once"

    def defaults(self) -> Response:
        payload = default_payload()
        payload["suppliers"] = payload["suppliers"][: self.HOSTED_SUPPLIERS]
        return json_response(
            {
                "case": payload,
                "platform": {
                    "hosted": True,
                    "note": (
                        "The browser drives this a step at a time, posting the event log back "
                        "with each request, so the case is rebuilt from its own history every "
                        "step and can run as long as it needs to on the free plan. The full "
                        "twelve-row directory takes a couple of minutes."
                    ),
                },
                "default_cast": list(self.HOSTED_CAST),
                "default_clinic": self.HOSTED_CLINIC,
                "personas": [{"key": p.key, "label": p.label} for p in PERSONAS],
                "clinic_personas": [
                    {"key": c.key, "label": c.label} for c in CLINIC_PERSONAS.values()
                ],
                "models": {"caller": CALLER_MODEL, "sim": SIM_MODEL},
                "voice_available": bool(self._secret("SARVAM_API_KEY")),
            }
        )

    async def step(self, request) -> Response:
        """One step per request, so each gets its own subrequest budget.

        The free plan allows 50 outbound requests per invocation and a whole
        case needs several hundred. One step needs about thirteen, and the
        browser holds the ledger between them -- which the event-sourced design
        makes safe, because folding the same events yields the same case.
        """
        api_key = self._secret("GROQ_API_KEY")
        if not api_key:
            return json_response({"error": "GROQ_API_KEY is not set on this Worker"}, 500)
        try:
            payload = json.loads(await request.text())
        except Exception as exc:
            return json_response({"error": f"bad body: {exc}"}, 400)
        payload.setdefault("config", {}).setdefault("models", {})
        payload["config"]["api_key"] = api_key
        try:
            return json_response(await step_once(payload))
        except Exception as exc:
            traceback.print_exc()
            return json_response({"error": f"{type(exc).__name__}: {exc}"})

    async def run(self, request) -> Response:
        """Stream the run as newline-delimited JSON while it happens."""
        api_key = self._secret("GROQ_API_KEY")
        if not api_key:
            return json_response(
                {"t": "error", "message": "GROQ_API_KEY is not set on this Worker"}, 500
            )
        try:
            config = json.loads(await request.text())
        except Exception as exc:
            return json_response({"t": "error", "message": f"bad body: {exc}"}, 400)
        config.setdefault("models", {})
        config["api_key"] = api_key

        stream = TransformStream.new()
        writer = stream.writable.getWriter()
        encoder = TextEncoder.new()

        async def pump() -> None:
            async def emit(item: dict) -> None:
                await writer.write(encoder.encode(json.dumps(item, default=str) + "\n"))

            try:
                await run_case_streaming(config, emit)
            except Exception as exc:
                traceback.print_exc()
                try:
                    await emit({"t": "error", "message": f"{type(exc).__name__}: {exc}"})
                except Exception:
                    pass
            finally:
                await writer.close()

        # waitUntil keeps the run alive for the life of the response body rather
        # than the life of the handler, which returns as soon as the stream exists.
        self.ctx.waitUntil(pump())
        return Response(
            stream.readable,
            headers={"Content-Type": "application/x-ndjson", "Cache-Control": "no-store"},
        )

    async def voice(self, request) -> Response:
        key = self._secret("SARVAM_API_KEY")
        if not key:
            return json_response({"error": "SARVAM_API_KEY is not set on this Worker"})
        try:
            body = json.loads(await request.text())
        except Exception as exc:
            return json_response({"error": f"bad body: {exc}"}, 400)
        try:
            import base64

            audio = await Voice(api_key=key).say(
                body.get("text", ""),
                voice_for(body.get("who", "them"), body.get("counterpart", "")),
            )
            return json_response({"audio": base64.b64encode(audio).decode()})
        except Exception as exc:
            return json_response({"error": str(exc)})
