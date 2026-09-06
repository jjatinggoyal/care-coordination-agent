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
from dme.web import run_case_streaming

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
    HOSTED_SUPPLIERS = 3

    def defaults(self) -> Response:
        payload = default_payload()
        payload["suppliers"] = payload["suppliers"][: self.HOSTED_SUPPLIERS]
        return json_response(
            {
                "case": payload,
                "platform": {
                    "hosted": True,
                    "note": (
                        f"Running on Cloudflare's free plan: 50 outbound requests per run, "
                        f"and one supplier call costs about ten. The directory is trimmed to "
                        f"{self.HOSTED_SUPPLIERS} for that reason. The full twelve-row "
                        f"directory runs locally, or here on a paid plan."
                    ),
                },
                "default_cast": list(DEFAULT_CAST),
                "personas": [{"key": p.key, "label": p.label} for p in PERSONAS],
                "clinic_personas": [
                    {"key": c.key, "label": c.label} for c in CLINIC_PERSONAS.values()
                ],
                "models": {"caller": CALLER_MODEL, "sim": SIM_MODEL},
                "voice_available": bool(self._secret("SARVAM_API_KEY")),
            }
        )

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
