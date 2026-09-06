"""The seams between this system and the world outside it.

Everything the engine cannot do by itself is reached through one of these
protocols. `sim.world.World` implements all four, which is how the whole case
runs against simulated humans without the engine knowing it.

In production these split apart and each gets a real implementation -- a carrier
behind PhoneTransport, a fax or Direct gateway behind OrderInbox, SMS or a
patient app behind PatientChannel. Nothing in engine.py or policy.py changes
when they do, which is the point of writing them down now rather than later.

They are Protocols rather than base classes deliberately: structural typing
means the simulator did not have to inherit from anything or know these exist,
and a real adapter will not have to either.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from ..model import CallOutcome


@runtime_checkable
class PhoneTransport(Protocol):
    """Placing a call and exchanging turns on it.

    Note what is *not* here: transcription, endpointing, barge-in. In the
    simulation both sides are text. A real implementation is where streaming
    speech-to-text and turn detection would live, and it would satisfy the same
    interface -- a turn in, a turn out -- because that is genuinely all the
    orchestrator needs to know about a phone call.
    """

    def dial_supplier(self, supplier_id: str) -> CallOutcome: ...

    def dial_clinic(self) -> CallOutcome: ...

    def supplier_opens(self, supplier_id: str, supplier_name: str) -> str: ...

    def supplier_replies(self, supplier_id: str, supplier_name: str, history: list[dict]) -> str: ...

    def clinic_opens(self) -> str: ...

    def clinic_replies(self, history: list[dict]) -> str: ...


@runtime_checkable
class OrderInbox(Protocol):
    """Getting the written order out of a clinic and noticing when it lands.

    Asynchronous by nature: the order arrives hours or days after the call that
    asked for it, through a channel nobody is watching in real time. The engine
    polls this on every step, which is the honest model of a fax tray.
    """

    def clinic_call_finished(self, at: datetime, promised: bool) -> None: ...

    def order_has_arrived(self, now: datetime) -> tuple[bool, str]: ...

    def fax_received(self, at: datetime) -> None: ...


@runtime_checkable
class PatientChannel(Protocol):
    """Reaching Eleanor, and finding out whether she answered."""

    def patient_picks_up(self) -> bool: ...


@runtime_checkable
class BookingLedger(Protocol):
    """Whether a delivery a supplier agreed to on the phone actually exists.

    In the simulation this is the world telling the truth about a ghosting
    supplier. In production there is no such oracle, which is exactly why the
    system verifies bookings on a timer instead of trusting the call.
    """

    def booking_holds(self, supplier_id: str) -> bool: ...
