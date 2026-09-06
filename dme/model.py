"""The case: typed state, and facts that carry where they came from.

Two decisions worth defending here.

1. Every qualification fact starts as UNKNOWN and can only become YES or NO
   because somebody said so on a phone call. The supplier directory has three
   columns -- name, phone, address -- so there is nothing to look up. This is
   the actual shape of the problem: the data does not exist until you make the
   call that creates it.

2. There is no confidence score anywhere. A float confidence invites a
   threshold, and a threshold is where model judgement quietly re-enters a
   system you meant to keep deterministic. Instead the extractor is required to
   answer UNKNOWN when the transcript does not settle the question, and UNKNOWN
   is a state the policy knows how to handle -- ask again, or move on.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


class Ternary(str, enum.Enum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"

    def __bool__(self) -> bool:  # pragma: no cover - deliberately unusable
        raise TypeError("Ternary is three-valued; compare explicitly against Ternary.YES")


@dataclass(frozen=True)
class Fact:
    """A value plus the call that produced it. Provenance is not optional."""

    value: Any
    source: str          # "intake" | "directory" | "call:c04"
    observed_at: datetime

    def render(self) -> str:
        v = self.value.value if isinstance(self.value, Ternary) else self.value
        return f"{v} ({self.source})"


class CallOutcome(str, enum.Enum):
    ANSWERED = "answered"
    VOICEMAIL = "voicemail"
    NO_ANSWER = "no_answer"
    BUSY = "busy"
    WRONG_NUMBER = "wrong_number"


@dataclass
class CallAttempt:
    call_id: str
    at: datetime
    outcome: CallOutcome
    turns: int = 0
    summary: str = ""


class SupplierStatus(str, enum.Enum):
    UNCONTACTED = "uncontacted"
    IN_PROGRESS = "in_progress"      # called, some answers still unknown
    QUALIFIED = "qualified"
    DISQUALIFIED = "disqualified"
    UNREACHABLE = "unreachable"      # attempts exhausted, nobody ever picked up


# The five questions a care advocate actually asks on the qualification call.
# Four are gates; the fifth is a tiebreak.
GATE_FIELDS = (
    "accepts_new_medicare_patients",
    "stocks_k0001",
    "accepts_assignment",
    "serves_patient_area",
)
SPEED_FIELD = "earliest_delivery_days"


@dataclass
class Supplier:
    supplier_id: str
    name: str
    phone: str
    address: str
    status: SupplierStatus = SupplierStatus.UNCONTACTED
    facts: dict[str, Fact] = field(default_factory=dict)
    attempts: list[CallAttempt] = field(default_factory=list)
    disqualified_because: str = ""

    def gate(self, name: str) -> Ternary:
        fact = self.facts.get(name)
        return fact.value if fact else Ternary.UNKNOWN

    def unknown_gates(self) -> list[str]:
        return [f for f in GATE_FIELDS if self.gate(f) is Ternary.UNKNOWN]

    def open_questions(self) -> list[str]:
        """What is still worth a phone call: the gates, plus how fast they are.

        Speed is not a gate -- a slow supplier who qualifies is still an option --
        but it decides which qualified supplier the patient actually gets, and
        three days versus three weeks is not a detail to her. Leaving it off the
        call means choosing blind.
        """
        questions = self.unknown_gates()
        if SPEED_FIELD not in self.facts:
            questions.append(SPEED_FIELD)
        return questions

    @property
    def delivery_days(self) -> int | None:
        fact = self.facts.get(SPEED_FIELD)
        return fact.value if fact else None

    @property
    def answered_attempts(self) -> int:
        return sum(1 for a in self.attempts if a.outcome is CallOutcome.ANSWERED)


class CommitmentKind(str, enum.Enum):
    SEND_WRITTEN_ORDER = "send_written_order"
    CALL_US_BACK = "call_us_back"
    HOLD_STOCK = "hold_stock"


@dataclass
class Commitment:
    """Somebody said they would do something.

    A promise is never treated as an outcome. Every commitment carries a
    verify_at, and the case only advances when the thing is independently
    observed -- the order actually arrives, the callback actually happens.
    'Agrees to deliver, then never calls back' is the failure mode the brief
    calls out by name, and this is the whole defence against it.
    """

    commitment_id: str
    kind: CommitmentKind
    by_party: str            # "clinic" | supplier_id
    made_at: datetime
    promised_by: datetime
    verify_at: datetime
    fulfilled: bool = False
    broken: bool = False


class Track(str, enum.Enum):
    """The parallel strands of a case.

    A case is not one queue. The supplier search and the written-order chase run
    independently and join before scheduling, so something that stops one of them
    must not be allowed to stop the others.
    """

    SUPPLIER = "supplier"
    ORDER = "order"
    PATIENT = "patient"


class HumanTaskKind(str, enum.Enum):
    """Work only a person can do, which is not the same as giving up.

    Escalating means the case is no longer ours. A human task means we need a
    hand with one step and will carry on with everything else meanwhile -- post
    a physical request, walk a form across a lobby, make a call that needs
    identifiers we are not permitted to read down a phone line.
    """

    POST_PHYSICAL_REQUEST = "post_physical_request"
    CALL_WITH_IDENTIFIERS = "call_with_identifiers"


@dataclass
class HumanTask:
    task_id: str
    kind: HumanTaskKind
    packet: dict[str, Any]
    requested_at: datetime
    # Which strand this suspends. None means it is queued work that holds
    # nothing up -- the case keeps moving and a person picks it up when they can.
    blocks: Track | None = None
    completed_at: datetime | None = None
    note: str = ""

    @property
    def open(self) -> bool:
        return self.completed_at is None


class OrderStatus(str, enum.Enum):
    VERBAL_ONLY = "verbal_only"          # where Eleanor starts
    REQUESTED = "requested"              # we have asked the clinic
    PROMISED = "promised"                # clinic said they would send it
    RECEIVED = "received"                # signed written order in hand
    REFUSED = "refused"


@dataclass
class OrderTrack:
    status: OrderStatus = OrderStatus.VERBAL_ONLY
    hcpcs: str = "K0001"
    attempts: list[CallAttempt] = field(default_factory=list)
    faxed_at: datetime | None = None
    received_at: datetime | None = None
    coded_as: str | None = None          # what the clinic actually wrote
    # Having the order is not the same as the supplier having it. No supplier
    # will commit to a delivery date against an order they cannot see, and they
    # cannot bill Medicare without it. This is the handoff the brief calls
    # 'match and hand off', and it is a step, not a formality.
    sent_to_supplier: str | None = None
    sent_at: datetime | None = None


@dataclass
class PatientTrack:
    cost_explained: bool = False
    cost_explained_at: datetime | None = None
    consent_required: bool = False       # only when cost exceeds the assigned baseline
    consent_given: Ternary = Ternary.UNKNOWN
    # Agreement from somebody who did not follow what they were agreeing to is
    # not consent. The extractor already reports this; ignoring it would be a
    # choice, and the wrong one.
    understood_cost: Ternary = Ternary.UNKNOWN
    delivery_window_communicated: bool = False
    contact_attempts: int = 0


class CaseStatus(str, enum.Enum):
    OPEN = "open"
    ESCALATED = "escalated"
    CLOSED_DELIVERED = "closed_delivered"


@dataclass
class Escalation:
    reason: str
    at: datetime
    packet: dict[str, Any]


@dataclass
class Patient:
    name: str
    age: int
    coverage: str
    has_supplemental: bool
    zip_code: str
    zip_is_assumed: bool
    phone: str


@dataclass
class Case:
    case_id: str
    patient: Patient
    equipment: str
    hcpcs: str
    pcp_name: str
    pcp_practice: str
    pcp_phone: str
    opened_at: datetime
    suppliers: dict[str, Supplier] = field(default_factory=dict)
    order: OrderTrack = field(default_factory=OrderTrack)
    patient_track: PatientTrack = field(default_factory=PatientTrack)
    commitments: dict[str, Commitment] = field(default_factory=dict)
    human_tasks: dict[str, HumanTask] = field(default_factory=dict)
    status: CaseStatus = CaseStatus.OPEN
    escalation: Escalation | None = None
    chosen_supplier: str | None = None
    delivery_scheduled_for: datetime | None = None
    seq: int = 0

    # --- derived views the policy reads -------------------------------------

    def qualified_suppliers(self) -> list[Supplier]:
        return [s for s in self.suppliers.values() if s.status is SupplierStatus.QUALIFIED]

    def live_suppliers(self) -> list[Supplier]:
        """Still worth calling: never contacted, or contacted with open questions."""
        return [
            s
            for s in self.suppliers.values()
            if s.status in (SupplierStatus.UNCONTACTED, SupplierStatus.IN_PROGRESS)
        ]

    def open_commitments(self) -> list[Commitment]:
        return [c for c in self.commitments.values() if not c.fulfilled and not c.broken]

    def open_tasks(self) -> list[HumanTask]:
        return [t for t in self.human_tasks.values() if t.open]

    def blocked(self, track: Track) -> bool:
        return any(t.blocks is track for t in self.open_tasks())
