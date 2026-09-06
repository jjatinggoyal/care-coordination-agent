"""Serialise the ledger, so a case can be carried between stateless invocations.

The browser holds the event log and posts it back with each step; the Worker
folds it and decides what happens next. That works because the ledger already
*is* the state -- `reducer.apply` over the same events yields the same case, and
nothing else writes to it.

The same codec is what you would persist a case with in a real deployment. It
exists here because a Cloudflare Worker on the free plan allows 50 outbound
requests per invocation, and a whole case needs more -- but one step of a case
needs about thirteen. Driving the loop a step at a time gives every step its own
budget, and this is what lets the next step pick up where the last one stopped.
"""

from __future__ import annotations

import dataclasses
import enum
import types
import typing
from datetime import datetime

from . import events as ev
from .clock import CENTRAL

# Every event type, by name. Built from the module so a new event cannot be
# added without becoming serialisable -- there is nowhere to forget to register.
EVENT_TYPES: dict[str, type] = {
    name: obj
    for name, obj in vars(ev).items()
    if isinstance(obj, type) and issubclass(obj, ev.Event) and obj is not ev.Event
}


def _plain(value):
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, datetime):
        return value.astimezone(CENTRAL).isoformat()
    return value


def dump_event(event: ev.Event) -> dict:
    out = {"t": type(event).__name__, "seq": event.seq}
    for field in dataclasses.fields(event):
        if field.name == "seq":
            continue
        out[field.name] = _plain(getattr(event, field.name))
    return out


def _revive(annotation, value):
    """Turn a plain JSON value back into whatever the field is declared as."""
    if value is None:
        return None
    origin = typing.get_origin(annotation)
    # `X | None` gives types.UnionType, `Optional[X]` gives typing.Union --
    # both appear in this codebase and both need unwrapping.
    if origin is typing.Union or origin is types.UnionType:
        for arg in typing.get_args(annotation):
            if arg is not type(None):
                return _revive(arg, value)
        return value
    if isinstance(annotation, type):
        if issubclass(annotation, enum.Enum):
            return annotation(value)
        if annotation is datetime:
            return datetime.fromisoformat(value)
    return value


def load_event(payload: dict) -> ev.Event:
    cls = EVENT_TYPES[payload["t"]]
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for field in dataclasses.fields(cls):
        if field.name == "seq" or field.name not in payload:
            continue
        kwargs[field.name] = _revive(hints.get(field.name), payload[field.name])
    event = cls(**kwargs)
    object.__setattr__(event, "seq", payload.get("seq", 0))
    return event


def dump_ledger(events: list[ev.Event]) -> list[dict]:
    return [dump_event(e) for e in events]


def load_ledger(payload: list[dict]) -> list[ev.Event]:
    return [load_event(d) for d in payload]
