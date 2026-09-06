"""What happened. Append-only, and the only way case state ever changes.

The engine executes actions against the outside world; the outside world
answers; that answer becomes an event; the reducer folds it into state. No
component writes to the case directly. That means the entire run is replayable
from the ledger, and every field in the final state can be traced to the call
that produced it -- which is what you need when the question is "why did the
system tell this patient she owed money?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .model import CallOutcome, CommitmentKind, HumanTaskKind, Ternary, Track


@dataclass(frozen=True)
class Event:
    at: datetime
    seq: int = field(default=0, compare=False)

    @property
    def kind(self) -> str:
        return type(self).__name__

    def line(self) -> str:
        return self.kind


@dataclass(frozen=True)
class CaseOpened(Event):
    case_id: str = ""
    patient: str = ""

    def line(self) -> str:
        return f"case opened for {self.patient}"


@dataclass(frozen=True)
class SupplierCalled(Event):
    supplier_id: str = ""
    supplier_name: str = ""
    call_id: str = ""
    outcome: CallOutcome = CallOutcome.NO_ANSWER
    turns: int = 0
    summary: str = ""

    def line(self) -> str:
        return f"called {self.supplier_name} -> {self.outcome.value} ({self.turns} turns)"


@dataclass(frozen=True)
class SupplierFactsObserved(Event):
    supplier_id: str = ""
    supplier_name: str = ""
    facts: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def line(self) -> str:
        shown = ", ".join(
            f"{k}={v.value if isinstance(v, Ternary) else v}" for k, v in self.facts.items()
        )
        return f"learned about {self.supplier_name}: {shown or '(nothing)'}"


@dataclass(frozen=True)
class SupplierQualified(Event):
    supplier_id: str = ""
    supplier_name: str = ""
    delivery_days: int | None = None

    def line(self) -> str:
        eta = f", {self.delivery_days}d delivery" if self.delivery_days is not None else ""
        return f"QUALIFIED: {self.supplier_name}{eta}"


@dataclass(frozen=True)
class SupplierDisqualified(Event):
    supplier_id: str = ""
    supplier_name: str = ""
    reason: str = ""

    def line(self) -> str:
        return f"ruled out {self.supplier_name}: {self.reason}"


@dataclass(frozen=True)
class SupplierUnreachable(Event):
    supplier_id: str = ""
    supplier_name: str = ""
    attempts: int = 0

    def line(self) -> str:
        return f"gave up on {self.supplier_name} after {self.attempts} attempts"


@dataclass(frozen=True)
class ClinicCalled(Event):
    call_id: str = ""
    outcome: CallOutcome = CallOutcome.NO_ANSWER
    turns: int = 0
    summary: str = ""

    def line(self) -> str:
        return f"called Dr. Chen's office -> {self.outcome.value} ({self.turns} turns)"


@dataclass(frozen=True)
class OrderRequested(Event):
    channel: str = "phone"

    def line(self) -> str:
        return f"written order requested by {self.channel}"


@dataclass(frozen=True)
class OrderPromised(Event):
    commitment_id: str = ""
    promised_by: datetime | None = None

    def line(self) -> str:
        when = f" by {self.promised_by:%a %d %b}" if self.promised_by else ""
        return f"clinic promised the written order{when} (not believed until it arrives)"


@dataclass(frozen=True)
class OrderReceived(Event):
    coded_as: str = ""

    def line(self) -> str:
        return f"written order received, coded {self.coded_as}"


@dataclass(frozen=True)
class OrderRefused(Event):
    reason: str = ""

    def line(self) -> str:
        return f"clinic will not send the order: {self.reason}"


@dataclass(frozen=True)
class OrderSentToSupplier(Event):
    supplier_id: str = ""
    supplier_name: str = ""
    channel: str = "fax"

    def line(self) -> str:
        return f"written order sent to {self.supplier_name} by {self.channel}"


@dataclass(frozen=True)
class CommitmentMade(Event):
    commitment_id: str = ""
    kind: CommitmentKind = CommitmentKind.CALL_US_BACK
    by_party: str = ""
    promised_by: datetime | None = None
    verify_at: datetime | None = None

    def line(self) -> str:
        return f"{self.by_party} committed to {self.kind.value}"


@dataclass(frozen=True)
class CommitmentBroken(Event):
    commitment_id: str = ""
    by_party: str = ""
    kind: CommitmentKind = CommitmentKind.CALL_US_BACK

    def line(self) -> str:
        return f"BROKEN PROMISE: {self.by_party} did not {self.kind.value}"


@dataclass(frozen=True)
class PatientContacted(Event):
    topic: str = ""
    delivered: bool = True

    def line(self) -> str:
        return f"patient contacted: {self.topic}"


@dataclass(frozen=True)
class ConsentRecorded(Event):
    value: Ternary = Ternary.UNKNOWN
    understood: Ternary = Ternary.UNKNOWN

    def line(self) -> str:
        if self.value is Ternary.YES and self.understood is Ternary.NO:
            return "patient said yes but had not followed the cost — not treated as consent"
        return f"patient cost consent: {self.value.value}"


@dataclass(frozen=True)
class DeliveryScheduled(Event):
    supplier_id: str = ""
    supplier_name: str = ""
    when: datetime | None = None

    def line(self) -> str:
        when = self.when.strftime("%a %d %b") if self.when else "?"
        return f"DELIVERY SCHEDULED with {self.supplier_name} for {when}"


@dataclass(frozen=True)
class HumanTaskRequested(Event):
    task_id: str = ""
    kind: HumanTaskKind = HumanTaskKind.POST_PHYSICAL_REQUEST
    blocks: Track | None = None
    packet: dict[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        held = f"; {self.blocks.value} track waits" if self.blocks else "; nothing waits on it"
        return f"HUMAN ASKED TO {self.kind.value.replace('_', ' ')}{held}"


@dataclass(frozen=True)
class HumanTaskCompleted(Event):
    task_id: str = ""
    note: str = ""

    def line(self) -> str:
        return f"human finished their step{': ' + self.note if self.note else ''}"


@dataclass(frozen=True)
class Escalated(Event):
    reason: str = ""
    packet: dict[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        return f"ESCALATED TO HUMAN: {self.reason}"


@dataclass(frozen=True)
class CaseClosed(Event):
    outcome: str = ""

    def line(self) -> str:
        return f"case closed: {self.outcome}"
