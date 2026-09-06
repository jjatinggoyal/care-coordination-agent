"""The simulated outside world. Everything here is mocked; nothing here is product.

The world holds facts the system is not allowed to see: which supplier is
actually out of stock, whether the clinic really put the order in the fax queue,
whether Eleanor picks up. The system can only learn any of it by placing a call
and listening to the answer -- which is the honest shape of a problem where the
integration surface is a telephone.

Randomness is seeded. The same seed gives the same world twice, so a run that
goes wrong can be re-run and watched.
"""

from __future__ import annotations

import random
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

    rng: random.Random = field(init=False)
    supplier_personas: dict[str, SupplierPersona] = field(default_factory=dict, init=False)
    clinic_calls: int = field(default=0, init=False)
    order_arrives_at: datetime | None = field(default=None, init=False)
    order_coded_as: str = field(default="", init=False)
    patient_contacts: int = field(default=0, init=False)
    # Human steps we have been asked for: task_id -> when a person gets to it.
    human_queue: dict[str, datetime] = field(default_factory=dict, init=False)
    human_done: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    # --- casting -----------------------------------------------------------

    def assign(self, supplier_ids: list[str], shuffle: bool = False) -> None:
        keys = list(self.cast)
        if shuffle:
            self.rng.shuffle(keys)
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
        if self.rng.random() < persona.pickup_rate:
            return CallOutcome.ANSWERED
        if self.rng.random() < persona.voicemail_rate:
            return CallOutcome.VOICEMAIL
        return CallOutcome.NO_ANSWER

    def dial_clinic(self) -> CallOutcome:
        if self.rng.random() < self.clinic.pickup_rate:
            return CallOutcome.ANSWERED
        return CallOutcome.NO_ANSWER

    def dial_patient(self) -> CallOutcome:
        self.patient_contacts += 1
        if self.patient_contacts < self.patient_answers_after:
            return CallOutcome.NO_ANSWER
        if self.rng.random() < self.patient.pickup_rate:
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
        """Decide -- out of the system's sight -- whether the order actually moves."""
        self.clinic_calls += 1
        clinic = self.clinic
        if not clinic.sends_order or not promised:
            return
        if clinic.promises_but_stalls and self.clinic_calls == 1:
            return  # they meant it at the time. It still did not happen.
        if self.order_arrives_at is None:
            self.order_arrives_at = CLINIC_HOURS.add_business_hours(
                at, max(1.0, clinic.business_days_to_send * 8.0)
            )
            self.order_coded_as = clinic.miscodes_as or self.hcpcs

    def order_has_arrived(self, now: datetime) -> tuple[bool, str]:
        if self.order_arrives_at is not None and now >= self.order_arrives_at:
            return True, self.order_coded_as
        return False, ""

    def fax_received(self, at: datetime) -> None:
        """A fax gets read by whoever opens the tray. Some practices never do."""
        clinic = self.clinic
        if clinic.sends_order and self.order_arrives_at is None:
            self.order_arrives_at = CLINIC_HOURS.add_business_hours(at, 8.0)
            self.order_coded_as = clinic.miscodes_as or self.hcpcs

    def human_task_queued(self, task_id: str, at: datetime) -> None:
        """A person has been asked. They are not instant, and they are not the
        system -- so this lives in the world, not in the engine."""
        self.human_queue[task_id] = CLINIC_HOURS.add_business_days(
            at, max(0, self.human_turnaround_days)
        )

    def human_tasks_done(self, now: datetime) -> list[tuple[str, str]]:
        """Which asked-for steps a person has finished by now."""
        finished = []
        for task_id, when in self.human_queue.items():
            if task_id not in self.human_done and now >= when:
                self.human_done.add(task_id)
                finished.append((task_id, "handled by the care team"))
        return finished

    def human_task_finished_order(self, now: datetime) -> None:
        """A posted request eventually produces the written order."""
        if self.order_arrives_at is None and self.clinic.sends_order:
            self.order_arrives_at = CLINIC_HOURS.add_business_hours(now, 4.0)
            self.order_coded_as = self.clinic.miscodes_as or self.hcpcs

    def patient_picks_up(self) -> bool:
        self.patient_contacts += 1
        return self.patient_contacts >= self.patient_answers_after

    def booking_holds(self, supplier_id: str) -> bool:
        """Does a supplier who agreed to a delivery slot actually book it?"""
        return not self.supplier_personas[supplier_id].ghosts
