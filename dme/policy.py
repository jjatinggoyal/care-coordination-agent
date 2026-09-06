"""What to do next. Pure functions over typed state -- no model runs in here.

This is the file the whole design is arranged around. Every decision that
touches eligibility, money, sequencing, retry, or handing the case to a human
is made by code you can read, on values you can see, in a way that produces the
same answer twice. The language model's job stops at the edge of this file: it
turns a phone call into typed facts, and then it is done.

Read `decide()` top to bottom and you have the entire operating procedure.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .clock import CLINIC_HOURS, PATIENT_HOURS, SUPPLIER_HOURS
from .model import (
    GATE_FIELDS,
    HumanTaskKind,
    Track,
    CallOutcome,
    Case,
    CaseStatus,
    CommitmentKind,
    OrderStatus,
    Supplier,
    SupplierStatus,
    Ternary,
)

# --- knobs ------------------------------------------------------------------
# Deliberately few, deliberately named, deliberately here rather than inline.

MAX_DIAL_ATTEMPTS = 3            # unanswered dials before a supplier is written off
MAX_ANSWERED_ATTEMPTS = 3        # conversations before we accept they will not tell us
CLINIC_MAX_CALL_ATTEMPTS = 3     # phone attempts before we fall back to fax
RETRY_AFTER_NO_ANSWER_BH = 2.0   # business hours
RETRY_AFTER_VOICEMAIL_BH = 4.0
RETRY_AFTER_PARTIAL_BH = 3.0     # they answered but could not answer everything
ORDER_VERIFY_AFTER_DAYS = 1      # fallback when they name no date at all
# How long we are willing to be held to somebody else's timeframe. Believe them
# -- chasing at noon when they said "by Thursday" burns goodwill and teaches you
# nothing -- but not indefinitely: a case cannot afford a fortnight because a
# receptionist said so.
MAX_PROMISE_HONOURED_DAYS = 3
VERIFY_GRACE_BH = 2.0            # let their own deadline pass before chasing it
HUMAN_TASK_STALE_DAYS = 3        # a person has this long before we escalate instead
FAX_VERIFY_AFTER_DAYS = 1        # a fax gets a working day to be read before we give up
PATIENT_MAX_CONTACT_ATTEMPTS = 3
GOOD_ENOUGH_DELIVERY_DAYS = 5    # stop shopping once someone can deliver this fast

# A supplier who would have qualified but wanted an identifier we are not
# permitted to give over the phone. Recorded distinctly, because it is the one
# rejection a human can actually undo.
DISCLOSURE_BLOCK = "demanded identifiers we may not disclose"


class EscalationReason(str, enum.Enum):
    """Every way this system is allowed to give up. There is no 'other'.

    A typed reason means the handoff is routable and countable: you can ask how
    often the directory runs dry versus how often a clinic stonewalls, and the
    answer is a group-by rather than a human reading free text.
    """

    NO_QUALIFIED_SUPPLIER = "no_qualified_supplier"
    ORDER_UNOBTAINABLE = "order_unobtainable"
    ORDER_REFUSED = "order_refused"
    ORDER_CODING_MISMATCH = "order_coding_mismatch"
    PATIENT_UNREACHABLE = "patient_unreachable"
    PATIENT_DECLINED_COST = "patient_declined_cost"
    CONSENT_NOT_INFORMED = "consent_not_informed"
    CALL_SAFETY_STOP = "call_safety_stop"


# --- actions ----------------------------------------------------------------
# Intents, not effects. The policy returns one of these; the engine performs it.


@dataclass(frozen=True)
class Action:
    pass


@dataclass(frozen=True)
class CallSupplier(Action):
    supplier_id: str
    ask: tuple[str, ...]      # exactly which gates are still unknown
    why: str = ""


@dataclass(frozen=True)
class CallClinic(Action):
    attempt: int
    why: str = ""


@dataclass(frozen=True)
class FaxClinic(Action):
    why: str = ""


@dataclass(frozen=True)
class ContactPatient(Action):
    topic: str                # "cost" | "delivery_window"
    why: str = ""


@dataclass(frozen=True)
class SendOrderToSupplier(Action):
    supplier_id: str
    why: str = ""


@dataclass(frozen=True)
class ScheduleDelivery(Action):
    supplier_id: str
    why: str = ""


@dataclass(frozen=True)
class RequestHumanTask(Action):
    """Ask a person for one step, and carry on with everything else.

    Distinct from Escalate on purpose. Escalating says the case is no longer
    ours. This says we need a hand with a single thing -- posting a document,
    making a call that needs identifiers we may not read out -- and the tracks
    that do not depend on it keep running while somebody gets to it.
    """

    kind: HumanTaskKind
    blocks: Track | None = None
    packet: dict[str, Any] = field(default_factory=dict)
    why: str = ""


@dataclass(frozen=True)
class Escalate(Action):
    reason: EscalationReason
    packet: dict[str, Any] = field(default_factory=dict)
    why: str = ""


@dataclass(frozen=True)
class CloseCase(Action):
    outcome: str
    why: str = ""


@dataclass(frozen=True)
class Wait(Action):
    until: datetime
    why: str = ""


# --- qualification ----------------------------------------------------------


def gate_verdict(supplier: Supplier) -> tuple[str, str]:
    """Classify a supplier against the four gates.

    Returns (verdict, reason) where verdict is one of qualified / disqualified /
    incomplete. Note the asymmetry, which is the point: one NO disqualifies,
    but qualifying requires all four to be an explicit YES. Silence is not
    consent -- an UNKNOWN never qualifies anybody.
    """
    for name in GATE_FIELDS:
        if supplier.gate(name) is Ternary.NO:
            return "disqualified", name
    unknown = supplier.unknown_gates()
    if unknown:
        return "incomplete", ",".join(unknown)
    return "qualified", ""


def supplier_due_at(case: Case, supplier: Supplier) -> datetime | None:
    """When this supplier is worth dialling again. None means never.

    Backoff is a function of what happened last time, not a fixed sleep: a busy
    signal is worth retrying inside the hour, a voicemail is not.
    """
    if supplier.status in (
        SupplierStatus.QUALIFIED,
        SupplierStatus.DISQUALIFIED,
        SupplierStatus.UNREACHABLE,
    ):
        return None
    if not supplier.attempts:
        return SUPPLIER_HOURS.next_open(case.opened_at)

    last = supplier.attempts[-1]
    unanswered = sum(1 for a in supplier.attempts if a.outcome is not CallOutcome.ANSWERED)
    if unanswered >= MAX_DIAL_ATTEMPTS and supplier.answered_attempts == 0:
        return None
    if last.outcome in (CallOutcome.NO_ANSWER, CallOutcome.BUSY):
        return SUPPLIER_HOURS.add_business_hours(last.at, RETRY_AFTER_NO_ANSWER_BH)
    if last.outcome is CallOutcome.VOICEMAIL:
        return SUPPLIER_HOURS.add_business_hours(last.at, RETRY_AFTER_VOICEMAIL_BH)
    if last.outcome is CallOutcome.WRONG_NUMBER:
        return None
    # Answered, but we still have open questions -- try again when the person
    # who actually knows the stock position is likely to be at the desk.
    if supplier.answered_attempts >= MAX_ANSWERED_ATTEMPTS:
        return None      # we have spoken to them three times: stop burning calls
    return SUPPLIER_HOURS.add_business_hours(last.at, RETRY_AFTER_PARTIAL_BH)


def next_supplier_to_call(case: Case, now: datetime) -> Supplier | None:
    """Directory order, oldest-due first.

    There is nothing to rank on. The directory has a name, a phone number and an
    address, and the brief is explicit that this is the real shape of the data.
    In production the ranking signal would be our own call history -- who picks
    up, who actually delivers -- which is exactly the asset a system like this
    accrues and a care advocate carries in their head.
    """
    ready = []
    for supplier in case.suppliers.values():
        due = supplier_due_at(case, supplier)
        if due is not None and due <= now:
            ready.append((due, supplier.supplier_id, supplier))
    if not ready:
        return None
    ready.sort(key=lambda row: (row[0], row[1]))
    return ready[0][2]


def clinic_due_at(case: Case) -> datetime | None:
    """When to push the written order along."""
    order = case.order
    if order.status in (OrderStatus.RECEIVED, OrderStatus.REFUSED):
        return None
    if not order.attempts:
        return CLINIC_HOURS.next_open(case.opened_at)

    last = order.attempts[-1]

    # Once it is in writing, the clock is on the fax, not on the last call.
    # Somebody has to open the tray; that takes a working day, not three minutes.
    if order.faxed_at is not None:
        return CLINIC_HOURS.add_business_days(order.faxed_at, FAX_VERIFY_AFTER_DAYS)

    # Nobody picked up. That is a short backoff, not a day -- we have learned
    # nothing, so there is nothing to come back and verify.
    if last.outcome is not CallOutcome.ANSWERED:
        return CLINIC_HOURS.add_business_hours(last.at, RETRY_AFTER_NO_ANSWER_BH)

    # We spoke to somebody. Whether they promised or merely took a message, the
    # next move is the same: come back tomorrow and see what actually happened.
    return CLINIC_HOURS.add_business_days(last.at, ORDER_VERIFY_AFTER_DAYS)


def best_qualified(case: Case) -> Supplier | None:
    """Fastest confirmed delivery wins; unknown speed sorts last."""
    qualified = case.qualified_suppliers()
    if not qualified:
        return None
    return min(
        qualified,
        key=lambda s: (s.delivery_days if s.delivery_days is not None else 10**6, s.supplier_id),
    )


def promise_window(
    hours, now: datetime, promised_days: int | None, default_days: int = ORDER_VERIFY_AFTER_DAYS
) -> tuple[datetime, datetime]:
    """Turn what somebody said into when to hold them to it.

    Returns (promised_by, verify_at). `promised_by` is their deadline, taken from
    their own words where they gave one. `verify_at` is a little after it -- the
    system does not ring back the minute a deadline passes, and it does not wait
    a week either.

    Capped at MAX_PROMISE_HONOURED_DAYS, because "believe what people tell you"
    and "let a case stall on an optimistic receptionist" are different policies
    and only the first one is a good idea.
    """
    days = promised_days if isinstance(promised_days, int) and promised_days >= 0 else default_days
    days = min(days, MAX_PROMISE_HONOURED_DAYS)
    promised_by = (
        hours.add_business_days(now, days) if days else hours.add_business_hours(now, 4.0)
    )
    return promised_by, hours.add_business_hours(promised_by, VERIFY_GRACE_BH)


def pending_booking(case: Case, now: datetime) -> datetime | None:
    """A supplier said 'we'll call you back to confirm'. Returns when to stop believing them.

    This is the 'agrees, then goes silent' failure mode. The supplier is not
    disqualified for saying it -- people say it and mean it -- but the case does
    not advance on the strength of it either, and the clock is running.
    """
    for commitment in case.commitments.values():
        if (
            commitment.kind is CommitmentKind.HOLD_STOCK
            and not commitment.fulfilled
            and not commitment.broken
            and commitment.verify_at > now
        ):
            return commitment.verify_at
    return None


def keep_shopping(case: Case, now: datetime) -> bool:
    """Do we call more suppliers after finding one that works?

    No. One qualified supplier who can deliver inside GOOD_ENOUGH_DELIVERY_DAYS
    ends the search. Shopping for a marginally faster delivery costs real calls
    and real days, and the patient's outcome does not improve.
    """
    best = best_qualified(case)
    if best is None:
        return True
    if not case.live_suppliers():
        return False
    # Not knowing how fast they are is not the same as them being fast. A
    # qualified supplier with no quoted lead time is worth one more phone call,
    # because the alternative is booking a three-week delivery by accident.
    if best.delivery_days is None:
        return True
    return best.delivery_days > GOOD_ENOUGH_DELIVERY_DAYS


# --- escalation -------------------------------------------------------------


def escalation_packet(case: Case, reason: EscalationReason) -> dict[str, Any]:
    """Everything a human needs to pick this up cold, and nothing else."""
    tried = [
        {
            "supplier": s.name,
            "phone": s.phone,
            "status": s.status.value,
            "attempts": len(s.attempts),
            "ruled_out_on": s.disqualified_because,
            "facts": {k: f.render() for k, f in s.facts.items()},
        }
        for s in case.suppliers.values()
        if s.attempts
    ]
    return {
        "reason": reason.value,
        "patient": case.patient.name,
        "equipment": f"{case.equipment} ({case.hcpcs})",
        "written_order": case.order.status.value,
        "order_coded_as": case.order.coded_as,
        "clinic_attempts": len(case.order.attempts),
        "suppliers_called": len(tried),
        "supplier_detail": tried,
        "next_step_for_human": HUMAN_NEXT_STEP[reason],
    }


HUMAN_NEXT_STEP = {
    EscalationReason.NO_QUALIFIED_SUPPLIER: (
        "Directory exhausted. Widen the radius beyond the 12 listed suppliers, or ask the "
        "patient whether she can travel to a pickup location."
    ),
    EscalationReason.ORDER_UNOBTAINABLE: (
        "Dr. Chen's office is not producing the written order through the front desk. "
        "Escalate practice-to-practice or route through the patient's next visit."
    ),
    EscalationReason.ORDER_REFUSED: (
        "The clinic declined to write the order. This is a clinical decision -- it needs a "
        "clinician conversation, not another phone call from us."
    ),
    EscalationReason.ORDER_CODING_MISMATCH: (
        "The written order does not match K0001. Coding corrections change what Medicare pays "
        "and what the patient owes; a human confirms the code before anything is submitted."
    ),
    EscalationReason.PATIENT_UNREACHABLE: (
        "Eleanor has not responded across the allowed attempts. A person should try, and "
        "should check whether there is a caregiver contact on file."
    ),
    EscalationReason.PATIENT_DECLINED_COST: (
        "Eleanor did not accept the out-of-pocket share. Do not proceed. A human should walk "
        "through the cost with her and look at assistance options."
    ),
    EscalationReason.CONSENT_NOT_INFORMED: (
        "They said yes, but the call shows they had not followed what they would owe. "
        "Agreement without understanding is not consent. Someone should call and explain it "
        "again before anything is ordered in their name."
    ),
    EscalationReason.CALL_SAFETY_STOP: (
        "A supplier who could otherwise serve her requires a patient identifier the system is "
        "not permitted to read out over the phone. A person with the chart in front of them can "
        "finish that call -- start with the suppliers listed under blocked_on_disclosure."
    ),
}


def stale_task(case: Case, now: datetime) -> bool:
    """Has a human had long enough, and the thing still is not done?"""
    return any(
        CLINIC_HOURS.add_business_days(t.requested_at, HUMAN_TASK_STALE_DAYS) <= now
        for t in case.open_tasks()
    )


def check_human_help(case: Case, now: datetime) -> RequestHumanTask | None:
    """Is there one step a person could take that would unstick this?

    Asked before check_escalation, because "somebody needs to walk this across a
    lobby" is a smaller thing than "this case is beyond the system", and most of
    what used to escalate is really the former.
    """
    order = case.order
    already = {t.kind for t in case.open_tasks()}

    # The clinic has been called, promised, chased and faxed, and the order still
    # has not come. A person can post it, or walk in. Meanwhile the supplier
    # search has no reason to stop.
    fax_due = clinic_due_at(case)
    if (
        order.status not in (OrderStatus.RECEIVED, OrderStatus.REFUSED)
        and order.faxed_at is not None
        and fax_due is not None
        and fax_due <= now
        and HumanTaskKind.POST_PHYSICAL_REQUEST not in already
        and not any(
            t.kind is HumanTaskKind.POST_PHYSICAL_REQUEST for t in case.human_tasks.values()
        )
    ):
        return RequestHumanTask(
            kind=HumanTaskKind.POST_PHYSICAL_REQUEST,
            blocks=Track.ORDER,
            packet=escalation_packet(case, EscalationReason.ORDER_UNOBTAINABLE),
            why="phone and fax are exhausted; a person can hand this over in the building",
        )

    # A supplier who could serve this patient wants an identifier we are not
    # permitted to read down a phone line. Somebody with the chart can finish
    # that call -- and nothing waits on it, because there are other suppliers.
    blocked_on_disclosure = [
        s.name for s in case.suppliers.values() if s.disqualified_because == DISCLOSURE_BLOCK
    ]
    if blocked_on_disclosure and HumanTaskKind.CALL_WITH_IDENTIFIERS not in already:
        if not any(
            t.kind is HumanTaskKind.CALL_WITH_IDENTIFIERS for t in case.human_tasks.values()
        ):
            return RequestHumanTask(
                kind=HumanTaskKind.CALL_WITH_IDENTIFIERS,
                blocks=None,
                packet={"suppliers": blocked_on_disclosure},
                why="a person with the chart can finish a call we are not allowed to",
            )
    return None


def check_escalation(case: Case, now: datetime) -> Escalate | None:
    """Terminal conditions, checked before anything else is attempted."""
    order = case.order

    if order.status is OrderStatus.REFUSED:
        return Escalate(
            EscalationReason.ORDER_REFUSED,
            escalation_packet(case, EscalationReason.ORDER_REFUSED),
            why="clinic refused to write the order",
        )

    if order.status is OrderStatus.RECEIVED and order.coded_as not in (None, case.hcpcs):
        return Escalate(
            EscalationReason.ORDER_CODING_MISMATCH,
            escalation_packet(case, EscalationReason.ORDER_CODING_MISMATCH),
            why=f"order came back as {order.coded_as}, case is {case.hcpcs}",
        )

    if order.status is not OrderStatus.RECEIVED and clinic_due_at(case) is None:
        return Escalate(
            EscalationReason.ORDER_UNOBTAINABLE,
            escalation_packet(case, EscalationReason.ORDER_UNOBTAINABLE),
            why="no route left to the written order",
        )

    fax_due = clinic_due_at(case)
    if (
        order.faxed_at is not None
        and order.status is not OrderStatus.RECEIVED
        and fax_due is not None
        and fax_due <= now
        # Only once a person has been asked and either finished without result or
        # run out of time. Giving up before asking for help is not giving up, it
        # is not trying.
        and any(
            t.kind is HumanTaskKind.POST_PHYSICAL_REQUEST for t in case.human_tasks.values()
        )
        and (stale_task(case, now) or not case.blocked(Track.ORDER))
    ):
        return Escalate(
            EscalationReason.ORDER_UNOBTAINABLE,
            escalation_packet(case, EscalationReason.ORDER_UNOBTAINABLE),
            why="called, promised, chased and faxed; a working day later still no written order",
        )

    if not case.qualified_suppliers() and not case.live_suppliers():
        # If somebody was turned away only because they wanted an identifier we
        # cannot read out over the phone, that is the more useful thing to tell
        # the human -- it is a call they can finish, not a dead directory.
        blocked = [
            s.name for s in case.suppliers.values() if s.disqualified_because == DISCLOSURE_BLOCK
        ]
        reason = (
            EscalationReason.CALL_SAFETY_STOP if blocked else EscalationReason.NO_QUALIFIED_SUPPLIER
        )
        packet = escalation_packet(case, reason)
        if blocked:
            packet["blocked_on_disclosure"] = blocked
        return Escalate(
            reason,
            packet,
            why=(
                f"directory exhausted; {len(blocked)} supplier(s) blocked only on disclosure"
                if blocked
                else "every supplier in the directory has been ruled out or is unreachable"
            ),
        )

    track = case.patient_track
    if track.consent_required and track.consent_given is Ternary.NO:
        return Escalate(
            EscalationReason.PATIENT_DECLINED_COST,
            escalation_packet(case, EscalationReason.PATIENT_DECLINED_COST),
            why="patient did not accept the cost share",
        )

    if (
        track.consent_required
        and track.understood_cost is Ternary.NO
        and track.contact_attempts >= PATIENT_MAX_CONTACT_ATTEMPTS
    ):
        return Escalate(
            EscalationReason.CONSENT_NOT_INFORMED,
            escalation_packet(case, EscalationReason.CONSENT_NOT_INFORMED),
            why="they agreed, but had not followed what they would owe",
        )

    if (
        track.consent_required
        and track.consent_given is Ternary.UNKNOWN
        and track.contact_attempts >= PATIENT_MAX_CONTACT_ATTEMPTS
    ):
        return Escalate(
            EscalationReason.PATIENT_UNREACHABLE,
            escalation_packet(case, EscalationReason.PATIENT_UNREACHABLE),
            why="patient unreachable and her consent is required to proceed",
        )

    return None


# --- the decision -----------------------------------------------------------


def decide(case: Case, now: datetime) -> Action:
    """The single next thing to do. One action per step keeps the log readable.

    Priority order, and the reasoning behind it:

      1. Escalate on a terminal condition rather than dial anything else.
      2. Chase the written order. It is the longest pole -- a clinic front desk
         moves in days -- and it is the one thing no supplier can do for us.
      3. Work the supplier list. Every fact here costs a phone call.
      4. Tell the patient what she owes, before anything is scheduled in her name.
      5. Join the tracks and book the delivery.
      6. Otherwise sleep until the next thing is due.
    """
    if case.status is not CaseStatus.OPEN:
        return Wait(until=now, why="case is no longer open")

    # Ask a person for one step before concluding the case is beyond us.
    help_needed = check_human_help(case, now)
    if help_needed is not None:
        return help_needed

    escalation = check_escalation(case, now)
    if escalation is not None:
        return escalation

    due: list[datetime] = []

    # 2 -- the written order, unless a person is holding that strand
    order_due = None if case.blocked(Track.ORDER) else clinic_due_at(case)
    if order_due is not None:
        if order_due <= now:
            attempt = len(case.order.attempts) + 1
            if attempt > CLINIC_MAX_CALL_ATTEMPTS and case.order.faxed_at is None:
                return FaxClinic(why="front desk is not producing it by phone; putting it in writing")
            if case.order.status is OrderStatus.PROMISED:
                return CallClinic(attempt, why="they promised it a day ago; verifying it was sent")
            return CallClinic(attempt, why="written order still outstanding")
        due.append(order_due)

    # 3 -- the supplier list. Independent of the order, so a human step on the
    # order track is no reason to stop calling suppliers.
    if not case.blocked(Track.SUPPLIER) and keep_shopping(case, now):
        supplier = next_supplier_to_call(case, now)
        if supplier is not None:
            open_questions = tuple(supplier.open_questions())
            return CallSupplier(
                supplier.supplier_id,
                ask=open_questions or GATE_FIELDS,
                why=f"need: {', '.join(open_questions) or 'first contact'}",
            )
        for candidate in case.suppliers.values():
            candidate_due = supplier_due_at(case, candidate)
            if candidate_due is not None:
                due.append(candidate_due)

    best = best_qualified(case)
    track = case.patient_track

    # 4 -- what they will owe, said out loud, and their yes, before we commit them
    if best is not None and not case.blocked(Track.PATIENT) and track.consent_given is not Ternary.YES:
        if track.contact_attempts < PATIENT_MAX_CONTACT_ATTEMPTS:
            if PATIENT_HOURS.is_open(now):
                return ContactPatient(
                    "cost", why="a supplier is confirmed; she owes 20% and has to agree to it"
                )
            due.append(PATIENT_HOURS.next_open(now))

    # 5 -- the join: supplier, written order, and consent all in hand
    booking_deadline = pending_booking(case, now)
    if booking_deadline is not None:
        due.append(booking_deadline)
    elif (
        best is not None
        and case.order.status is OrderStatus.RECEIVED
        and track.consent_given is Ternary.YES
        and case.chosen_supplier is None
    ):
        # The handoff comes before the booking. Nobody quotes a delivery date
        # against an order they have not seen, and nobody can bill Medicare
        # without it on file.
        if case.order.sent_to_supplier != best.supplier_id:
            # A fax does not need anyone to be awake; send it whenever.
            return SendOrderToSupplier(
                best.supplier_id, why="they cannot schedule against an order they do not have"
            )
        if SUPPLIER_HOURS.is_open(now):
            return ScheduleDelivery(
                best.supplier_id,
                why="qualified supplier, written order in their hands, her agreement",
            )
        due.append(SUPPLIER_HOURS.next_open(now))

    if case.chosen_supplier is not None:
        if not case.patient_track.delivery_window_communicated:
            if PATIENT_HOURS.is_open(now):
                return ContactPatient("delivery_window", why="tell them when it is coming")
            due.append(PATIENT_HOURS.next_open(now))
        else:
            return CloseCase("delivered", why="scheduled, coded, and the patient knows")

    if due:
        return Wait(until=min(due), why="nothing is due yet")
    if case.open_tasks():
        # Everything that could move has moved; the rest is somebody else's turn.
        return Wait(
            until=CLINIC_HOURS.add_business_days(now, 1),
            why="waiting on a person to finish their step",
        )
    return Wait(until=now, why="nothing left to do")
