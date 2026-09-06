"""The only file that talks to a model.

Two call sites in the running system (say a turn on a phone call, turn a
finished transcript into typed facts), one for drafting the patient's message,
plus the simulated world, which is scaffolding rather than product.

Provider
--------
Groq's OpenAI-compatible endpoint. Two models, on purpose:

  system under test   openai/gpt-oss-120b   the caller, the extractor, the drafter
  simulated world     qwen/qwen3.8-27b      the suppliers, the clinic front desk

Deliberately a different model family on each side. If one model both hides a
fact and is asked to find it, the two share idiosyncrasies -- phrasings, refusal
habits, ideas about what is obvious -- and the eval quietly flatters itself.

Why raw HTTP rather than the `openai` SDK
-----------------------------------------
The SDK wants sockets. Cloudflare Workers do not have any -- outbound HTTP there
is `fetch`, and Pyodide has no socket layer for httpx to sit on. Since the whole
request is one POST with a JSON body, hand-rolling it costs about forty lines
and buys the system the ability to run in both places from one codebase. The
transport is chosen once, at import, by asking whether we are inside a Worker.

Everything here is async for the same reason: `fetch` is async, and a synchronous
call path cannot be made to await it. Locally the blocking `urllib` call is
pushed to a thread so the shape matches.

Conversational turns run at low reasoning effort. A turn on a phone call is a
short reply under a latency budget -- the person on the other end hears every
millisecond of thinking as dead air -- so depth is the wrong thing to buy there.
Extraction runs higher, because that is where being wrong is expensive and
nobody is waiting.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BASE_URL = os.environ.get("DME_BASE_URL", "https://api.groq.com/openai/v1")
CALLER_MODEL = os.environ.get("DME_MODEL", "openai/gpt-oss-120b")
SIM_MODEL = os.environ.get("DME_SIM_MODEL", "qwen/qwen3.8-27b")

TIMEOUT_S = 90

# urllib announces itself as "Python-urllib/3.x", which Groq's edge blocks with a
# 1010. The SDK set one of these for us; hand-rolling the request means saying
# who we are ourselves.
USER_AGENT = "dme-coordination/1.0"


def in_worker() -> bool:
    """Are we inside a Cloudflare Worker (Pyodide), or on a real machine?"""
    try:
        import js  # noqa: F401  -- only present inside the Workers runtime
        import pyodide.ffi  # noqa: F401
    except ImportError:
        return False
    return True


def load_env(path: Path | None = None) -> None:
    """Read .env without a dependency. Real environment always wins.

    A Worker has no filesystem to speak of and gets its secrets from bindings,
    so this is a no-op there rather than an error.
    """
    try:
        path = path or Path(__file__).resolve().parent.parent / ".env"
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except OSError:
        return


class SubrequestLimit(RuntimeError):
    """The runtime cut us off, not the provider.

    Cloudflare Workers cap outbound requests per invocation -- 50 on the free
    plan, 10,000 on paid -- and every model call spends one. Hitting it is a
    billing setting, not a fault in the case, so it is worth saying so in those
    words rather than surfacing a JsException from three layers down.
    """


class HttpError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"http {status}: {body[:400]}")
        self.status = status
        self.body = body


async def post_json(url: str, headers: dict[str, str], payload: dict) -> dict:
    """One POST, two transports, same contract: JSON in, parsed JSON out."""
    body = json.dumps(payload)
    if in_worker():
        status, text = await _post_via_fetch(url, headers, body)
    else:
        status, text = await _post_via_urllib(url, headers, body)
    if status >= 400:
        raise HttpError(status, text)
    return json.loads(text)


async def _post_via_fetch(url: str, headers: dict[str, str], body: str) -> tuple[int, str]:
    from js import Object
    from pyodide.ffi import to_js
    from js import fetch as js_fetch

    try:
        response = await js_fetch(
            url,
            to_js(
                {"method": "POST", "headers": headers, "body": body},
                dict_converter=Object.fromEntries,
            ),
        )
    except Exception as exc:
        if "Too many subrequests" in str(exc):
            raise SubrequestLimit(str(exc)) from exc
        raise
    text = await response.text()
    return int(response.status), str(text)


async def _post_via_urllib(url: str, headers: dict[str, str], body: str) -> tuple[int, str]:
    import asyncio
    import urllib.error
    import urllib.request

    def blocking() -> tuple[int, str]:
        request = urllib.request.Request(url, data=body.encode(), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                return int(response.status), response.read().decode()
        except urllib.error.HTTPError as exc:
            return int(exc.code), exc.read().decode(errors="replace")

    return await asyncio.to_thread(blocking)


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_role: dict[str, int] = field(default_factory=dict)

    def record(self, role: str, usage: dict | None) -> None:
        usage = usage or {}
        self.calls += 1
        self.input_tokens += usage.get("prompt_tokens") or 0
        self.output_tokens += usage.get("completion_tokens") or 0
        self.by_role[role] = self.by_role.get(role, 0) + 1

    def summary(self) -> str:
        roles = ", ".join(f"{k}={v}" for k, v in sorted(self.by_role.items()))
        return (
            f"{self.calls} model calls ({roles}); "
            f"{self.input_tokens:,} in / {self.output_tokens:,} out tokens"
        )


class ExtractionRefused(RuntimeError):
    """The model declined or returned nothing. Treated as 'we learned nothing', never as a NO."""


class LLM:
    """Thin wrapper. No prompt lives here -- prompts live with the agent that owns them."""

    def __init__(
        self,
        usage: Usage | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        load_env()
        self.api_key = api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "No GROQ_API_KEY found. Put it in .env at the repo root, export it, "
                "or set it as a Worker secret."
            )
        self.base_url = (base_url or BASE_URL).rstrip("/")
        self.model = model or CALLER_MODEL
        self.usage = usage or Usage()

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

    async def _complete(self, payload: dict, role: str) -> dict:
        data = await post_json(f"{self.base_url}/chat/completions", self._headers, payload)
        self.usage.record(role, data.get("usage"))
        return data

    async def say(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        role: str,
        model: str | None = None,
        max_tokens: int = 400,
    ) -> str:
        """One conversational turn. Returns plain text."""
        data = await self._complete(
            {
                "model": model or self.model,
                "max_completion_tokens": max_tokens,
                "reasoning_effort": "low",
                "messages": [{"role": "system", "content": system}, *messages],
            },
            role,
        )
        message = (data.get("choices") or [{}])[0].get("message") or {}
        if message.get("refusal"):
            return "[call ended]"
        return (message.get("content") or "").strip()

    async def extract(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        name: str = "extraction",
        role: str = "extractor",
        model: str | None = None,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        """Transcript in, schema-valid JSON out.

        The schema is enforced twice: once by the provider (strict json_schema)
        and once again in Python by the caller, which checks every value against
        the enum it is allowed to take and every quote against the transcript.
        Belt and braces, because this is the seam where free text becomes
        something the policy will act on, and a plausible-looking wrong value
        here is worse than a refusal.
        """
        payload = {
            "model": model or self.model,
            "max_completion_tokens": max_tokens,
            "reasoning_effort": "medium",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            },
        }
        # Constrained decoding is not a guarantee. Providers do return
        # json_validate_failed -- the model runs out of road mid-string and the
        # object never closes. One retry, then give up and say so.
        #
        # Giving up is cheap here by construction: a failed extraction means the
        # call taught us nothing, every field stays UNKNOWN, and the policy
        # already knows what to do about UNKNOWN. The failure mode of the model
        # layer is a slower case, never a wrong one.
        for attempt in (1, 2):
            try:
                data = await self._complete(payload, role)
            except HttpError as exc:
                if attempt == 2 or "json_validate_failed" not in exc.body:
                    raise ExtractionRefused(f"provider could not produce valid JSON: {exc}") from exc
                continue
            message = (data.get("choices") or [{}])[0].get("message") or {}
            content = message.get("content")
            if message.get("refusal") or not content:
                raise ExtractionRefused(message.get("refusal") or "empty response")
            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:
                if attempt == 2:
                    raise ExtractionRefused(f"unparseable response: {exc}") from exc
        raise ExtractionRefused("extraction exhausted its retries")


def describe_error(exc: Exception) -> str:
    """Retryable and permanent failures are different animals."""
    if isinstance(exc, SubrequestLimit):
        return "hit the Worker's per-invocation subrequest limit"
    if isinstance(exc, HttpError):
        if exc.status == 429:
            return "rate limited"
        if exc.status == 404:
            return "model or endpoint not found"
        return f"api error {exc.status}"
    if isinstance(exc, ExtractionRefused):
        return "extraction refused"
    return f"{type(exc).__name__}: {exc}"
