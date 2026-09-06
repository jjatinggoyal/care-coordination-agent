"""Speak the simulated calls, through Sarvam's TTS.

This is a rendering of a call, not a telephony leg. There is no carrier, no
STT, no turn-taking, no barge-in -- the conversation still happens as text and
the audio is painted on afterwards. Saying so plainly matters, because "you can
hear the calls" invites the assumption that a voice pipeline exists, and it
does not.

What it is good for: a call you can hear is a call you can judge. Reading a
transcript where the agent stacks three questions into one breath looks
tolerable; hearing it does not.

Rendered at 8 kHz on purpose. That is the sample rate a real PSTN call runs at
-- narrowband G.711 territory -- so the demo sounds like the channel this
system would actually live on rather than like a podcast.
"""

from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass, field

from .llm import HttpError, post_json

ENDPOINT = "https://api.sarvam.ai/text-to-speech"
MODEL = "bulbul:v3"
LANGUAGE = "en-IN"
TELEPHONY_RATE = 8000

# bulbul:v3's voices, as the API reports them.
SPEAKERS = (
    "aditya", "ritu", "ashutosh", "priya", "neha", "rahul", "pooja", "rohan", "simran",
    "kavya", "amit", "dev", "ishita", "shreya", "ratan", "varun", "manan", "sumit",
    "roopa", "kabir", "aayan", "shubh", "advait", "anand", "tanya", "tarun", "sunny",
    "mani", "gokul", "vijay", "shruti", "suhani", "mohit", "kavitha", "rehan", "soham",
    "rupali",
)

# Our own caller keeps one voice across every call, the way a person would.
# Everyone else is cast deterministically off their name, so a supplier sounds
# like the same receptionist each time you ring them.
AGENT_SPEAKER = "aditya"
_THEM = tuple(s for s in SPEAKERS if s != AGENT_SPEAKER)


def voice_for(who: str, counterpart: str) -> str:
    if who == "agent":
        return AGENT_SPEAKER
    digest = hashlib.sha1((counterpart or "them").encode()).digest()
    return _THEM[digest[0] % len(_THEM)]


class VoiceUnavailable(RuntimeError):
    """No key, or the provider said no. The demo carries on in silence."""


@dataclass
class Voice:
    api_key: str | None = None
    rate: int = TELEPHONY_RATE
    cache: dict[str, bytes] = field(default_factory=dict, init=False)
    calls: int = field(default=0, init=False)
    characters: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("SARVAM_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    async def say(self, text: str, speaker: str) -> bytes:
        """Return mp3 bytes for one line. Cached -- replaying a call is free."""
        if not self.available:
            raise VoiceUnavailable("SARVAM_API_KEY is not set")
        text = (text or "").strip()
        if not text:
            raise VoiceUnavailable("nothing to say")
        # bulbul:v3 caps at 2500 characters; a phone turn is never near that,
        # but a runaway model turn could be.
        text = text[:2000]
        key = hashlib.sha1(f"{speaker}|{self.rate}|{text}".encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]

        try:
            payload = await post_json(
                ENDPOINT,
                {"api-subscription-key": self.api_key, "Content-Type": "application/json"},
                {
                    "text": text,
                    "language_code": LANGUAGE,
                    "model": MODEL,
                    "speaker": speaker if speaker in SPEAKERS else AGENT_SPEAKER,
                    "speech_sample_rate": self.rate,
                    "output_audio_codec": "mp3",
                },
            )
        except HttpError as exc:
            raise VoiceUnavailable(f"sarvam {exc.status}: {exc.body[:200]}") from exc
        except OSError as exc:
            raise VoiceUnavailable(f"could not reach sarvam: {exc}") from exc

        audios = payload.get("audios") or []
        if not audios:
            raise VoiceUnavailable("sarvam returned no audio")
        audio = base64.b64decode(audios[0])
        self.cache[key] = audio
        self.calls += 1
        self.characters += len(text)
        return audio
