"""The simulated outside world. Everything here is mocked; nothing here is product.

The world holds facts the system is not allowed to see: which supplier is
actually out of stock, whether the clinic really put the order in the fax queue,
whether Eleanor picks up. The system can only learn any of it by placing a call
and listening to the answer -- which is the honest shape of a problem where the
integration surface is a telephone.

The world holds no mutable state. Whether a phone is answered is a hash of the
seed, who is being rung, and which attempt this is -- so it is stable for a given
seed but carries nothing forward, and everything else it needs to know it reads
out of the case.

That is what lets a case run across many stateless requests: the browser holds
the ledger, and both the case and the world can be rebuilt exactly from it. The
same seed gives the same world twice, so a run that goes wrong can be re-run and
watched.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..clock import CLINIC_HOURS
from ..llm import LLM, SIM_MODEL
from ..model import CallOutcome
from .personas import (
    CLINIC_PERSONAS,
    DEFAULT_CAST,
    PATIENT_PERSONAS,
    PERSONAS_BY_KEY,
    ClinicPersona,
    PatientPersona,
    SupplierPersona,
)

MAX_TURNS = 12  # a qualification call that needs more than this has gone wrong


@dataclass
class World:
    llm: LLM
    seed: int = 7
    clinic_persona_key: str = "stalls_once"
    cast: tuple[str, ...] = DEFAULT_CAST
    patient_answers_after: int = 1     # she picks up on the Nth attempt
    patient_accepts_cost: bool = True
    patient_persona_key: str = "agreeable"
    # How long a person takes to do the thing we asked for, in working days.
    # Set it past the policy's staleness window to watch the case give up on
    # waiting and escalate properly.
    human_turnaround_days: int = 1
    # Explicit casting, when the caller wants to choose who plays whom.
    assigned: dict[str, str] | None = None
    sim_model: str = SIM_MODEL      # who plays the humans; overridable per run
    hcpcs: str = "K0001"            # what this case actually asked for
    # Who the simulated humans think they are talking about. Hard-coding the
    # brief's patient here made every other case incoherent -- the front desk
    # cheerfully discussed Eleanor Martinez no matter whose case it was.
    patient_name: str = "the patient"
    pcp_name: str = "the physician"
    practice: str = "the practice"
    equipment: str = "the equipment"

    supplier_personas: dict[str, SupplierPersona] = field(default_factory=dict, init=False)
    # The case this world is simulating around. Set by the engine; everything the
    # world needs to remember, it reads from here rather than holding itself.
    case: object | None = field(default=None, init=False, repr=False)

    def luck(self, *key) -> float:
        """A stable number in [0, 1) for this seed and this question.

        Replaces a random.Random whose internal state would have had to be
        carried between requests. Same inputs, same answer, no memory.
        """
        raw = f"{self.seed}|" + "|".join(str(k) for k in key)
        digest = hashlib.sha256(raw.encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64

    # --- casting -----------------------------------------------------------

    def assign(self, supplier_ids: list[str], shuffle: bool = False) -> None:
        keys = list(self.cast)
        if shuffle:
            keys.sort(key=lambda k: self.luck("cast", k))
        for i, supplier_id in enumerate(supplier_ids):
            chosen = (self.assigned or {}).get(supplier_id)
            if chosen in (None, "", "random"):
                chosen = keys[i % len(keys)] if keys else "good"
            self.supplier_personas[supplier_id] = PERSONAS_BY_KEY[chosen]

    @property
    def clinic(self) -> ClinicPersona:
        return CLINIC_PERSONAS[self.clinic_persona_key]

    @property
    def patient(self) -> PatientPersona:
        return PATIENT_PERSONAS[self.patient_persona_key]

    # --- the phone ---------------------------------------------------------

    def dial_supplier(self, supplier_id: str) -> CallOutcome:
        persona = self.supplier_personas[supplier_id]
        attempt = len(self.case.suppliers[supplier_id].attempts) if self.case else 0
        if self.luck("dial", supplier_id, attempt) < persona.pickup_rate:
            return CallOutcome.ANSWERED
        if self.luck("vm", supplier_id, attempt) < persona.voicemail_rate:
            return CallOutcome.VOICEMAIL
        return CallOutcome.NO_ANSWER

    def dial_clinic(self) -> CallOutcome:
        attempt = len(self.case.order.attempts) if self.case else 0
        if self.luck("dial", "clinic", attempt) < self.clinic.pickup_rate:
            return CallOutcome.ANSWERED
        return CallOutcome.NO_ANSWER

    def dial_patient(self) -> CallOutcome:
        attempt = self.case.patient_track.contact_attempts if self.case else 0
        if attempt + 1 < self.patient_answers_after:
            return CallOutcome.NO_ANSWER
        if self.luck("dial", "patient", attempt) < self.patient.pickup_rate:
            return CallOutcome.ANSWERED
        return CallOutcome.NO_ANSWER

    async def supplier_opens(self, supplier_id: str, supplier_name: str) -> str:
        return await self._speak(
            system=self._supplier_system(self.supplier_personas[supplier_id], supplier_name),
            history=[],
            opener=True,
            role="sim_supplier",
        )

    async def supplier_replies(self, supplier_id: str, supplier_name: str, history: list[dict]) -> str:
        return await self._speak(
            system=self._supplier_system(self.supplier_personas[supplier_id], supplier_name),
            history=history,
            role="sim_supplier",
        )

    async def clinic_opens(self) -> str:
        return await self._speak(system=self._clinic_system(), history=[], opener=True, role="sim_clinic")

    async def clinic_replies(self, history: list[dict]) -> str:
        return await self._speak(system=self._clinic_system(), history=history, role="sim_clinic")

    async def patient_opens(self) -> str:
        return await self._speak(
            system=self._patient_system(), history=[], opener=True, role="sim_patient"
        )

    async def patient_replies(self, history: list[dict]) -> str:
        return await self._speak(system=self._patient_system(), history=history, role="sim_patient")

    async def _speak(self, *, system: str, history: list[dict], role: str, opener: bool = False) -> str:
        messages = list(history)
        if opener:
            messages = [{"role": "user", "content": "[the phone rings and you pick it up]"}]
        # Generous budget: the world model reasons before it speaks, and a tight
        # cap spends the whole allowance on thinking and returns an empty turn --
        # which reads as a silent line and derails the call.
        line = await self.llm.say(
            system=system, messages=messages, role=role, model=self.sim_model, max_tokens=700
        )
        if not line.strip():
            return "Sorry, could you say that again?"
        return line

    # --- prompts for the simulated humans -----------------------------------

    _SIM_RULES = (
        "\n\nHow to play this:\n"
        "- You are on a phone call. Speak only your own words -- no narration, no stage "
        "directions, no speaker labels.\n"
        "- One to three sentences per turn. Real phone speech, contractions, interruptions.\n"
        "- You do NOT know you are in a simulation and you never mention one.\n"
        "- Stay inside what you actually know. If you don't know something, say you don't "
        "know -- do not invent a fact to be helpful.\n"
        "- When the call has reached its natural end, say goodbye normally."
    )

    def _supplier_system(self, persona: SupplierPersona, supplier_name: str) -> str:
        truth = "\n".join(f"  - {k}: {v}" for k, v in persona.truth.items())
        extra = ""
        if persona.ghosts:
            extra = (
                "\nYou agree to things you will not follow through on. You never say no.\n"
            )
        if persona.demands_out_of_scope:
            extra = (
                "\nYou will not proceed without the patient's Medicare Beneficiary Identifier "
                "and date of birth, and you ask for them repeatedly.\n"
            )
        withheld = ""
        if persona.withholds:
            withheld = (
                "\nThese are things you genuinely do not have in front of you: "
                + ", ".join(persona.withholds)
                + ". You must not confirm or deny any of them, however many times you are "
                "asked and however much easier a yes would make the call. Say you cannot "
                "see it from where you are sitting and offer the person who can. Guessing "
                "to be helpful is the one thing you never do.\n"
            )
        return (
            f"You answer the phone at {supplier_name}, a durable medical equipment supplier "
            f"in the Chicago area.\n\n{persona.style}\n{extra}{withheld}\n"
            f"The item being asked about is {self.equipment.lower()}, HCPCS {self.hcpcs}. Where "
            f"the notes below say 'stocks_k0001', they mean that item.\n\n"
            f"The true state of your business, which shapes your answers:\n{truth}\n"
            f"('accepts_assignment' means you accept Medicare's approved amount as full "
            f"payment and bill Medicare directly.)"
            + self._SIM_RULES
        )

    def _patient_system(self) -> str:
        persona = self.patient
        stance = (
            "You are willing to go ahead once the cost is clear."
            if persona.accepts
            else "You are not willing to commit to the cost on this call."
        )
        return (
            f"You are {self.patient_name}, at home, and your phone rings. It is the care team "
            f"helping you get {self.equipment.lower()} that your doctor ordered.\n\n"
            f"{persona.style}\n\n{stance}\n\n"
            f"You are on Original Medicare with no supplemental plan. You do not know the "
            f"jargon and you should not use it -- you talk like a person, not a policy "
            f"document."
            + self._SIM_RULES
        )

    def _clinic_system(self) -> str:
        clinic = self.clinic
        return (
            f"You answer the phone at {self.practice} in Chicago. {self.pcp_name} is one of the "
            f"physicians here.\n\n"
            f"{clinic.style}\n\n"
            f"What is true in your records: {self.pcp_name} saw {self.patient_name} a few days "
            f"ago. There is a verbal order noted in the chart for {self.equipment.lower()}. No "
            f"written order has been signed or sent yet."
            + self._SIM_RULES
        )

    # --- things that happen later, without anyone on the phone ---------------

    def clinic_call_finished(self, at: datetime, promised: bool) -> None:
        """Nothing to record -- when the order moves is derived, not remembered."""

    def _order_arrival(self) -> tuple[datetime | None, str]:
        """When the written order lands, read out of the case rather than stored.

        Three ways it can move, in the order they are tried: a person was asked
        to hand it over; it was faxed; or somebody promised it on a call. The
        stalling front desk means the promise only counts from the second time
        they answered the phone.
        """
        clinic = self.clinic
        coded = clinic.miscodes_as or self.hcpcs
        if self.case is None or not clinic.sends_order:
            return None, coded

        for task in self.case.human_tasks.values():
            if task.completed_at is not None:
                return CLINIC_HOURS.add_business_hours(task.completed_at, 4.0), coded

        if self.case.order.faxed_at is not None:
            return CLINIC_HOURS.add_business_hours(self.case.order.faxed_at, 8.0), coded

        answered = [a for a in self.case.order.attempts if a.outcome is CallOutcome.ANSWERED]
        needed = 2 if clinic.promises_but_stalls else 1
        if len(answered) >= needed:
            return (
                CLINIC_HOURS.add_business_hours(
                    answered[needed - 1].at, max(1.0, clinic.business_days_to_send * 8.0)
                ),
                coded,
            )
        return None, coded

    def order_has_arrived(self, now: datetime) -> tuple[bool, str]:
        arrives_at, coded = self._order_arrival()
        if arrives_at is not None and now >= arrives_at:
            return True, coded
        return False, ""

    def fax_received(self, at: datetime) -> None:
        """Nothing to record; _order_arrival reads order.faxed_at off the case."""

    def human_task_queued(self, task_id: str, at: datetime) -> None:
        """Nothing to record; when a person gets to it is derived from requested_at."""

    def human_tasks_done(self, now: datetime) -> list[tuple[str, str]]:
        """Which asked-for steps a person has finished by now."""
        if self.case is None:
            return []
        finished = []
        for task in self.case.human_tasks.values():
            if not task.open:
                continue
            due = CLINIC_HOURS.add_business_days(
                task.requested_at, max(0, self.human_turnaround_days)
            )
            if now >= due:
                finished.append((task.task_id, "handled by the care team"))
        return finished

    def human_task_finished_order(self, now: datetime) -> None:
        """Nothing to record; _order_arrival sees the completed task."""

    def patient_picks_up(self) -> bool:
        attempt = self.case.patient_track.contact_attempts if self.case else 0
        return attempt + 1 >= self.patient_answers_after

    def booking_holds(self, supplier_id: str) -> bool:
        """Does a supplier who agreed to a delivery slot actually book it?"""
        return not self.supplier_personas[supplier_id].ghosts
