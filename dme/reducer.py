"""Fold events into case state.

Deterministic and total: every event kind is handled explicitly and an
unrecognised one raises rather than being ignored. Provenance is attached here,
at fold time -- an event carries a raw value, and the reducer is what records
which call it came from and when.
"""

from __future__ import annotations

from . import events as ev
from .model import (
    CallAttempt,
    Ternary,
    Case,
    CaseStatus,
    Commitment,
    CommitmentKind,
    Escalation,
    Fact,
    HumanTask,
    OrderStatus,
    SupplierStatus,
)


def apply(case: Case, event: ev.Event) -> None:
    match event:
        case ev.CaseOpened():
            pass

        case ev.SupplierCalled():
            supplier = case.suppliers[event.supplier_id]
            supplier.attempts.append(
                CallAttempt(
                    call_id=event.call_id,
                    at=event.at,
                    outcome=event.outcome,
                    turns=event.turns,
                    summary=event.summary,
                )
            )
            if supplier.status is SupplierStatus.UNCONTACTED:
                supplier.status = SupplierStatus.IN_PROGRESS

        case ev.SupplierFactsObserved():
            supplier = case.suppliers[event.supplier_id]
            for key, value in event.facts.items():
                supplier.facts[key] = Fact(value=value, source=event.source, observed_at=event.at)

        case ev.SupplierQualified():
            case.suppliers[event.supplier_id].status = SupplierStatus.QUALIFIED
            # From here on we are about to commit her to a purchase with a
            # coinsurance share, so her agreement stops being a courtesy.
            case.patient_track.consent_required = True

        case ev.SupplierDisqualified():
            supplier = case.suppliers[event.supplier_id]
            supplier.status = SupplierStatus.DISQUALIFIED
            supplier.disqualified_because = event.reason

        case ev.SupplierUnreachable():
            case.suppliers[event.supplier_id].status = SupplierStatus.UNREACHABLE

        case ev.ClinicCalled():
            case.order.attempts.append(
                CallAttempt(
                    call_id=event.call_id,
                    at=event.at,
                    outcome=event.outcome,
                    turns=event.turns,
                    summary=event.summary,
                )
            )

        case ev.OrderRequested():
            if case.order.status is OrderStatus.VERBAL_ONLY:
                case.order.status = OrderStatus.REQUESTED
            if event.channel == "fax":
                case.order.faxed_at = event.at

        case ev.OrderPromised():
            case.order.status = OrderStatus.PROMISED

        case ev.OrderReceived():
            case.order.status = OrderStatus.RECEIVED
            case.order.received_at = event.at
            case.order.coded_as = event.coded_as

        case ev.OrderRefused():
            case.order.status = OrderStatus.REFUSED

        case ev.OrderSentToSupplier():
            case.order.sent_to_supplier = event.supplier_id
            case.order.sent_at = event.at

        case ev.CommitmentMade():
            case.commitments[event.commitment_id] = Commitment(
                commitment_id=event.commitment_id,
                kind=event.kind,
                by_party=event.by_party,
                made_at=event.at,
                promised_by=event.promised_by or event.at,
                verify_at=event.verify_at or event.at,
            )

        case ev.CommitmentBroken():
            commitment = case.commitments[event.commitment_id]
            commitment.broken = True
            if (
                commitment.kind is CommitmentKind.SEND_WRITTEN_ORDER
                and case.order.status is OrderStatus.PROMISED
            ):
                # They said they would and they did not. The order is not
                # 'promised' any more -- it is outstanding again, and the retry
                # schedule should treat it that way.
                case.order.status = OrderStatus.REQUESTED

        case ev.PatientContacted():
            track = case.patient_track
            track.contact_attempts += 1
            if event.topic == "cost":
                track.cost_explained = True
                track.cost_explained_at = event.at
            elif event.topic == "delivery_window":
                track.delivery_window_communicated = True

        case ev.ConsentRecorded():
            case.patient_track.understood_cost = event.understood
            case.patient_track.consent_given = (
                Ternary.UNKNOWN
                if event.value is Ternary.YES and event.understood is Ternary.NO
                else event.value
            )

        case ev.DeliveryScheduled():
            case.chosen_supplier = event.supplier_id
            case.delivery_scheduled_for = event.when

        case ev.HumanTaskRequested():
            case.human_tasks[event.task_id] = HumanTask(
                task_id=event.task_id,
                kind=event.kind,
                packet=dict(event.packet),
                requested_at=event.at,
                blocks=event.blocks,
            )

        case ev.HumanTaskCompleted():
            task = case.human_tasks[event.task_id]
            task.completed_at = event.at
            task.note = event.note

        case ev.Escalated():
            case.status = CaseStatus.ESCALATED
            case.escalation = Escalation(reason=event.reason, at=event.at, packet=event.packet)

        case ev.CaseClosed():
            case.status = CaseStatus.CLOSED_DELIVERED

        case _:
            raise ValueError(f"reducer has no case for {type(event).__name__}")


class Ledger:
    """The append-only log. Stamps sequence numbers; nothing else."""

    def __init__(self, case: Case) -> None:
        self.case = case
        self.events: list[ev.Event] = []

    def append(self, event: ev.Event) -> ev.Event:
        self.case.seq += 1
        object.__setattr__(event, "seq", self.case.seq)
        self.events.append(event)
        apply(self.case, event)
        return event

    def replay(self, case: Case) -> Case:
        """Rebuild state from scratch -- proof that the log is the source of truth."""
        for event in self.events:
            apply(case, event)
        return case
