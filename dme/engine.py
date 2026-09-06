"""The loop. Ask the policy what to do, do it, write down what happened.

Asynchronous because the outside world is: a phone call is a network round trip,
and on Cloudflare Workers the only outbound HTTP there is is `fetch`, which
cannot be awaited from synchronous code. Note what did *not* have to change --
`policy.py` is still pure, synchronous functions over typed state, because it
performs no I/O at all. That is the dividend of having put every decision in a
module that never touches the network.

The engine is deliberately dull, and short enough to read in one sitting. It
holds no opinions: it does not decide what to do next, when to give up, or what
a call meant. It performs the action it is handed, converts the result into
events, and appends them. Every interesting judgement in this system lives in
policy.py, and every fact in the ledger arrived through here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable

from . import events as ev
from . import policy
from .agents import caller, extract, patient
from .clock import CENTRAL, Clock, CLINIC_HOURS, SUPPLIER_HOURS
from .llm import LLM, describe_error
from .model import (
    Case,
    HumanTaskKind,
    CaseStatus,
    CallOutcome,
    CommitmentKind,
    OrderStatus,
    SupplierStatus,
    Ternary,
)
from .reducer import Ledger
from .sim.world import World

MAX_CASE_DAYS = 30


@dataclass
class Engine:
    case: Case
    world: World
    llm: LLM
    clock: Clock
    ledger: Ledger = field(init=False)
    transcripts: dict[str, caller.Transcript] = field(default_factory=dict, init=False)
    on_event: Callable[[ev.Event], None] | None = None
    on_action: Callable[[object], None] | None = None
    on_transcript: Callable[[str, caller.Transcript], None] | None = None
    _calls: int = field(default=0, init=False)
    stall_note: str = field(default="", init=False)
    # An ordered interleaving of decisions and their consequences. The ledger
    # alone does not show why anything happened -- a decision that produced no
    # event (a wait, a dial nobody answered) leaves no trace in it. For review
    # and for the demo, the reasoning is the interesting half.
    trace: list[dict] = field(default_factory=list, init=False)
    call_notes: dict[str, dict] = field(default_factory=dict, init=False)
    call_meta: dict[str, dict] = field(default_factory=dict, init=False)
    messages: list[dict] = field(default_factory=list, init=False)
    # How often the groundedness check overruled the model. Worth watching:
    # a rising veto rate is the extractor drifting, visible before it does harm.
    vetoed_answers: int = field(default=0, init=False)
    # Turns where our own agent stated an identifier it was never given, and
    # the guard replaced it before it went down the line.
    fabrications_blocked: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.ledger = Ledger(self.case)

    # --- plumbing ----------------------------------------------------------

    def emit(self, event: ev.Event) -> ev.Event:
        self.ledger.append(event)
        self.trace.append(
            {
                "kind": "event",
                "at": event.at,
                "seq": event.seq,
                "type": event.kind,
                "line": event.line(),
                "call_id": getattr(event, "call_id", None),
                "supplier_id": getattr(event, "supplier_id", None),
            }
        )
        if self.on_event:
            self.on_event(event)
        return event

    def _call_id(self) -> str:
        self._calls += 1
        return f"c{self._calls:02d}"

    @property
    def calls_placed(self) -> int:
        return self._calls

    # --- the loop ----------------------------------------------------------

    async def run(self, max_steps: int = 80) -> Case:
        for _ in range(max_steps):
            if not await self.step():
                break
        return self.case

    async def step(self) -> bool:
        now = self.clock.now
        if now - self.case.opened_at > timedelta(days=MAX_CASE_DAYS):
            self._escalate(
                policy.EscalationReason.ORDER_UNOBTAINABLE,
                f"case has been open {MAX_CASE_DAYS} days",
            )
            return False

        self._collect_arrivals()
        self._expire_commitments()
        if self.case.status is not CaseStatus.OPEN:
            return False

        action = policy.decide(self.case, self.clock.now)
        self.trace.append(
            {
                "kind": "decision",
                "at": self.clock.now,
                "type": type(action).__name__,
                "why": getattr(action, "why", ""),
                "until": getattr(action, "until", None),
                "supplier_id": getattr(action, "supplier_id", None),
                "topic": getattr(action, "topic", None),
                "ask": list(getattr(action, "ask", ()) or ()),
                "reason": getattr(getattr(action, "reason", None), "value", None),
            }
        )
        if self.on_action:
            self.on_action(action)
        return await self._perform(action)

    # --- things that happen without anyone placing a call -------------------

    def _collect_arrivals(self) -> None:
        for task_id, note in self.world.human_tasks_done(self.clock.now):
            if task_id in self.case.human_tasks and self.case.human_tasks[task_id].open:
                self.emit(
                    ev.HumanTaskCompleted(at=self.clock.now, task_id=task_id, note=note)
                )
                if self.case.human_tasks[task_id].kind is HumanTaskKind.POST_PHYSICAL_REQUEST:
                    self.world.human_task_finished_order(self.clock.now)
        arrived, coded = self.world.order_has_arrived(self.clock.now)
        if arrived and self.case.order.status is not OrderStatus.RECEIVED:
            self.emit(ev.OrderReceived(at=self.clock.now, coded_as=coded))
            for commitment in self.case.open_commitments():
                if commitment.kind is CommitmentKind.SEND_WRITTEN_ORDER:
                    commitment.fulfilled = True

    def _expire_commitments(self) -> None:
        now = self.clock.now
        for commitment in list(self.case.open_commitments()):
            if commitment.verify_at > now:
                continue
            self.emit(
                ev.CommitmentBroken(
                    at=now,
                    commitment_id=commitment.commitment_id,
                    by_party=commitment.by_party,
                    kind=commitment.kind,
                )
            )
            if commitment.kind is CommitmentKind.HOLD_STOCK:
                supplier = self.case.suppliers[commitment.by_party]
                # One missed callback is a busy day, not a character flaw. Two
                # is a pattern, and the case cannot afford a third. A real
                # coordinator gives exactly this much rope.
                missed = sum(
                    1
                    for c in self.case.commitments.values()
                    if c.kind is CommitmentKind.HOLD_STOCK
                    and c.by_party == supplier.supplier_id
                    and c.broken
                )
                if missed >= 2:
                    self.emit(
                        ev.SupplierDisqualified(
                            at=now,
                            supplier_id=supplier.supplier_id,
                            supplier_name=supplier.name,
                            reason="agreed to book a slot twice, never confirmed either",
                        )
                    )

    # --- performing an action ----------------------------------------------

    async def _perform(self, action) -> bool:
        match action:
            case policy.CallSupplier():
                return await self._call_supplier(action)
            case policy.CallClinic():
                return await self._call_clinic(action)
            case policy.FaxClinic():
                return await self._fax_clinic()
            case policy.ContactPatient():
                return await self._contact_patient(action)
            case policy.SendOrderToSupplier():
                return await self._send_order(action)
            case policy.ScheduleDelivery():
                return await self._book(action)
            case policy.RequestHumanTask():
                return await self._ask_a_human(action)
            case policy.Escalate():
                self.emit(
                    ev.Escalated(at=self.clock.now, reason=action.reason.value, packet=action.packet)
                )
                return False
            case policy.CloseCase():
                self.emit(ev.CaseClosed(at=self.clock.now, outcome=action.outcome))
                return False
            case policy.Wait():
                if action.until > self.clock.now:
                    self.clock.advance_to(action.until)
                    return True
                self.stall_note = action.why
                return False
            case _:
                raise ValueError(f"engine cannot perform {type(action).__name__}")

    def _escalate(self, reason: policy.EscalationReason, why: str) -> None:
        packet = policy.escalation_packet(self.case, reason)
        packet["trigger"] = why
        self.emit(ev.Escalated(at=self.clock.now, reason=reason.value, packet=packet))

    # --- supplier qualification --------------------------------------------

    async def _call_supplier(self, action: policy.CallSupplier) -> bool:
        supplier = self.case.suppliers[action.supplier_id]
        now = self.clock.now
        call_id = self._call_id()
        outcome = self.world.dial_supplier(supplier.supplier_id)

        if outcome is not CallOutcome.ANSWERED:
            self.emit(
                ev.SupplierCalled(
                    at=now,
                    supplier_id=supplier.supplier_id,
                    supplier_name=supplier.name,
                    call_id=call_id,
                    outcome=outcome,
                )
            )
            self._retire_if_exhausted(supplier)
            self.clock.advance(timedelta(minutes=2))
            return True

        try:
            transcript = await caller.call_supplier(
                self.llm, self.world, self.case, supplier, action.ask, call_id
            )
        except Exception as exc:  # a dropped call is a real outcome, not a crash
            self.emit(
                ev.SupplierCalled(
                    at=now,
                    supplier_id=supplier.supplier_id,
                    supplier_name=supplier.name,
                    call_id=call_id,
                    outcome=CallOutcome.NO_ANSWER,
                    summary=f"call failed: {describe_error(exc)}",
                )
            )
            self.clock.advance(timedelta(minutes=5))
            return True

        self._record_transcript(call_id, transcript, supplier.name)
        found = await extract.extract_supplier(
            self.llm,
            transcript.render(supplier.name),
            their_words=transcript.their_words(),
            item=f"{self.case.equipment.lower()} (HCPCS {self.case.hcpcs})",
        )
        self.vetoed_answers += len(found.dropped)
        self._note_call(call_id, found)

        self.emit(
            ev.SupplierCalled(
                at=now,
                supplier_id=supplier.supplier_id,
                supplier_name=supplier.name,
                call_id=call_id,
                outcome=CallOutcome.ANSWERED,
                turns=transcript.turns,
                summary=found.note,
            )
        )
        if found.facts:
            self.emit(
                ev.SupplierFactsObserved(
                    at=now,
                    supplier_id=supplier.supplier_id,
                    supplier_name=supplier.name,
                    facts=dict(found.facts),
                    source=f"call:{call_id}",
                )
            )

        self.clock.advance(timedelta(minutes=6 + 2 * transcript.turns))

        if found.safety_stop:
            self.emit(
                ev.SupplierDisqualified(
                    at=self.clock.now,
                    supplier_id=supplier.supplier_id,
                    supplier_name=supplier.name,
                    reason=policy.DISCLOSURE_BLOCK,
                )
            )
            return True

        verdict, reason = policy.gate_verdict(supplier)
        if verdict == "qualified":
            self.emit(
                ev.SupplierQualified(
                    at=self.clock.now,
                    supplier_id=supplier.supplier_id,
                    supplier_name=supplier.name,
                    delivery_days=supplier.delivery_days,
                )
            )
        elif verdict == "disqualified":
            self.emit(
                ev.SupplierDisqualified(
                    at=self.clock.now,
                    supplier_id=supplier.supplier_id,
                    supplier_name=supplier.name,
                    reason=reason,
                )
            )
        else:
            self._retire_if_exhausted(supplier)
        return True

    def _retire_if_exhausted(self, supplier) -> None:
        if supplier.status in (SupplierStatus.QUALIFIED, SupplierStatus.DISQUALIFIED):
            return
        if policy.supplier_due_at(self.case, supplier) is None:
            self.emit(
                ev.SupplierUnreachable(
                    at=self.clock.now,
                    supplier_id=supplier.supplier_id,
                    supplier_name=supplier.name,
                    attempts=len(supplier.attempts),
                )
            )

    # --- the written order --------------------------------------------------

    async def _call_clinic(self, action: policy.CallClinic) -> bool:
        now = self.clock.now
        call_id = self._call_id()
        outcome = self.world.dial_clinic()

        if outcome is not CallOutcome.ANSWERED:
            self.emit(ev.ClinicCalled(at=now, call_id=call_id, outcome=outcome))
            self.clock.advance(timedelta(minutes=2))
            return True

        nudging = self.case.order.status is OrderStatus.PROMISED or len(self.case.order.attempts) > 0
        try:
            transcript = await caller.call_clinic(self.llm, self.world, self.case, call_id, nudging)
        except Exception as exc:
            self.emit(
                ev.ClinicCalled(
                    at=now,
                    call_id=call_id,
                    outcome=CallOutcome.NO_ANSWER,
                    summary=f"call failed: {describe_error(exc)}",
                )
            )
            self.clock.advance(timedelta(minutes=5))
            return True

        self._record_transcript(call_id, transcript, "Front desk")
        found = await extract.extract_clinic(
            self.llm, transcript.render("Front desk"), their_words=transcript.their_words()
        )
        self.vetoed_answers += len(found.dropped)
        self._note_call(call_id, found)
        self.emit(
            ev.ClinicCalled(
                at=now,
                call_id=call_id,
                outcome=CallOutcome.ANSWERED,
                turns=transcript.turns,
            )
        )
        self.emit(ev.OrderRequested(at=now, channel="phone"))
        self.clock.advance(timedelta(minutes=5 + 2 * transcript.turns))
        now = self.clock.now

        if found.facts.get("refuses_to_provide"):
            self.emit(ev.OrderRefused(at=now, reason="clinic declined"))
            return True

        promised = found.facts.get("will_send_written_order") is Ternary.YES
        if promised:
            promised_by, verify_at = policy.promise_window(
                CLINIC_HOURS, now, found.facts.get("promised_within_business_days")
            )
            commitment_id = f"cm{len(self.case.commitments) + 1:02d}"
            self.emit(ev.OrderPromised(at=now, commitment_id=commitment_id, promised_by=promised_by))
            self.emit(
                ev.CommitmentMade(
                    at=now,
                    commitment_id=commitment_id,
                    kind=CommitmentKind.SEND_WRITTEN_ORDER,
                    by_party="clinic",
                    promised_by=promised_by,
                    verify_at=verify_at,
                )
            )
        self.world.clinic_call_finished(now, promised=promised)
        return True

    async def _fax_clinic(self) -> bool:
        now = self.clock.now
        self.emit(ev.OrderRequested(at=now, channel="fax"))
        self.world.fax_received(now)
        self.clock.advance(timedelta(minutes=3))
        return True

    # --- the patient --------------------------------------------------------

    async def _contact_patient(self, action: policy.ContactPatient) -> bool:
        """Ring the patient. Consent is something they say, not something we assume."""
        now = self.clock.now
        call_id = self._call_id()
        supplier = self.case.suppliers.get(self.case.chosen_supplier or "")

        outcome = self.world.dial_patient()
        if outcome is not CallOutcome.ANSWERED:
            self.emit(ev.PatientContacted(at=now, topic=action.topic, delivered=False))
            self.clock.advance(timedelta(hours=3))
            return True

        try:
            transcript = await caller.call_patient(
                self.llm, self.world, self.case, call_id, action.topic,
                supplier_name=supplier.name if supplier else "",
                when=self.case.delivery_scheduled_for,
            )
        except Exception as exc:
            self.emit(
                ev.PatientContacted(at=now, topic=action.topic, delivered=False)
            )
            self.clock.advance(timedelta(minutes=10))
            return True

        self._record_transcript(call_id, transcript, self.case.patient.name)
        self.emit(ev.PatientContacted(at=now, topic=action.topic, delivered=True))
        self.clock.advance(timedelta(minutes=5 + 2 * transcript.turns))

        if action.topic == "cost":
            found = await extract.extract_patient(
                self.llm, transcript.render(self.case.patient.name),
                their_words=transcript.their_words(),
            )
            self.vetoed_answers += len(found.dropped)
            self._note_call(call_id, found)
            answer = found.facts.get("agrees_to_proceed", Ternary.UNKNOWN)
            understood = found.facts.get("understood_the_cost", Ternary.UNKNOWN)
            if answer is not Ternary.UNKNOWN:
                self.emit(
                    ev.ConsentRecorded(at=self.clock.now, value=answer, understood=understood)
                )
        return True

    async def _book(self, action: policy.ScheduleDelivery) -> bool:
        supplier = self.case.suppliers[action.supplier_id]
        now = self.clock.now
        call_id = self._call_id()

        outcome = self.world.dial_supplier(supplier.supplier_id)
        if outcome is not CallOutcome.ANSWERED:
            self.emit(
                ev.SupplierCalled(
                    at=now,
                    supplier_id=supplier.supplier_id,
                    supplier_name=supplier.name,
                    call_id=call_id,
                    outcome=outcome,
                    summary="booking attempt",
                )
            )
            self.clock.advance(timedelta(hours=2))
            return True

        transcript = await caller.call_to_book(self.llm, self.world, self.case, supplier, call_id)
        self._record_transcript(call_id, transcript, supplier.name)
        found = await extract.extract_booking(
            self.llm, transcript.render(supplier.name), their_words=transcript.their_words()
        )
        self.vetoed_answers += len(found.dropped)
        self._note_call(call_id, found)
        self.emit(
            ev.SupplierCalled(
                at=now,
                supplier_id=supplier.supplier_id,
                supplier_name=supplier.name,
                call_id=call_id,
                outcome=CallOutcome.ANSWERED,
                turns=transcript.turns,
                summary="booking attempt",
            )
        )
        self.clock.advance(timedelta(minutes=6 + 2 * transcript.turns))
        now = self.clock.now

        confirmed = found.facts.get("slot_confirmed")
        days = found.facts.get("delivery_in_business_days", supplier.delivery_days or 5)

        # The world gets the final say on whether a booking is real. A supplier
        # who agrees on the phone and then does nothing sounds identical to one
        # who agrees and delivers -- which is exactly why a promise is verified
        # rather than believed.
        if confirmed is Ternary.YES and self.world.booking_holds(supplier.supplier_id):
            self.emit(
                ev.DeliveryScheduled(
                    at=now,
                    supplier_id=supplier.supplier_id,
                    supplier_name=supplier.name,
                    when=SUPPLIER_HOURS.add_business_hours(now, max(1, int(days)) * 8.0),
                )
            )
        elif confirmed is Ternary.NO:
            self.emit(
                ev.SupplierDisqualified(
                    at=now,
                    supplier_id=supplier.supplier_id,
                    supplier_name=supplier.name,
                    reason="declined the booking",
                )
            )
        else:
            # They would not commit to a slot, but they may have said when they
            # would come back to us. Hold them to their own timeframe.
            promised_by, verify_at = policy.promise_window(
                SUPPLIER_HOURS, now, found.facts.get("promised_within_business_days")
            )
            commitment_id = f"cm{len(self.case.commitments) + 1:02d}"
            self.emit(
                ev.CommitmentMade(
                    at=now,
                    commitment_id=commitment_id,
                    kind=CommitmentKind.HOLD_STOCK,
                    by_party=supplier.supplier_id,
                    promised_by=promised_by,
                    verify_at=verify_at,
                )
            )
        return True

    async def _ask_a_human(self, action: policy.RequestHumanTask) -> bool:
        """Queue one step for a person. The case does not stop.

        The difference from escalating is the whole point: this suspends at most
        one track, and everything independent of it keeps running while somebody
        gets to the thing only a person can do.
        """
        task_id = f"h{len(self.case.human_tasks) + 1:02d}"
        self.emit(
            ev.HumanTaskRequested(
                at=self.clock.now,
                task_id=task_id,
                kind=action.kind,
                blocks=action.blocks,
                packet=action.packet,
            )
        )
        self.world.human_task_queued(task_id, self.clock.now)
        self.clock.advance(timedelta(minutes=5))
        return True

    async def _send_order(self, action: policy.SendOrderToSupplier) -> bool:
        """Fax the signed order across. Mocked -- a function call, not a modem."""
        supplier = self.case.suppliers[action.supplier_id]
        self.emit(
            ev.OrderSentToSupplier(
                at=self.clock.now,
                supplier_id=supplier.supplier_id,
                supplier_name=supplier.name,
                channel="fax",
            )
        )
        self.clock.advance(timedelta(minutes=15))
        return True

    def _note_call(self, call_id: str, found) -> None:
        self.call_notes[call_id] = {
            "facts": {
                k: (v.value if hasattr(v, "value") else v) for k, v in found.facts.items()
            },
            "dropped": list(found.dropped),
            "safety_stop": found.safety_stop,
            "note": found.note,
        }

    def _record_transcript(self, call_id: str, transcript: caller.Transcript, them: str) -> None:
        self.transcripts[call_id] = transcript
        self.call_meta[call_id] = {"with": them, "blocked": list(transcript.blocked)}
        self.fabrications_blocked += len(transcript.blocked)
        if self.on_transcript:
            self.on_transcript(f"{call_id} — {them}", transcript)
