"""Turn a finished phone call into typed facts -- or into nothing.

This is the seam. On one side is a transcript, which is free text and can say
anything. On the other side is the policy, which acts on money and eligibility.
Everything that crosses has to be a value from a closed set, and it has to be
traceable to something the person on the phone actually said.

Three defences, in order:

  1. The API is given a JSON schema, so the shape and the enums are enforced
     before the response is returned.
  2. The model must quote the line that justifies each answer.
  3. Python checks that the quote is really in the transcript. If it is not,
     the answer is discarded and the field goes back to UNKNOWN.

Defence 3 is the one that matters. A model that confabulates a plausible "yes,
they take new patients" fails a string search, and a supplier does not get
qualified on a sentence nobody said.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..llm import LLM, ExtractionRefused
from ..model import GATE_FIELDS, Ternary

TERNARY = {"type": "string", "enum": ["yes", "no", "unknown"]}
EVIDENCE_RECALL_FLOOR = 0.7   # share of the quoted words that must appear in the transcript


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def _words(text: str) -> list[str]:
    return [w for w in _normalise(text).split() if w]


def grounded(quote: str, transcript_text: str) -> bool:
    """Is this quote really in the call?

    Exact-substring first, then a word-recall floor so an accurate paraphrase is
    not thrown away over a dropped 'um'. A fabrication clears neither.
    """
    if not quote.strip():
        return False
    haystack = _normalise(transcript_text)
    needle = _normalise(quote)
    if needle and needle in haystack:
        return True
    quote_words = _words(quote)
    if len(quote_words) < 3:
        return False
    present = sum(1 for w in quote_words if w in haystack.split())
    return present / len(quote_words) >= EVIDENCE_RECALL_FLOOR


@dataclass
class Extraction:
    facts: dict[str, Any] = field(default_factory=dict)
    safety_stop: bool = False
    dropped: list[str] = field(default_factory=list)   # answers vetoed as ungrounded
    note: str = ""


SUPPLIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "accepts_new_medicare_patients": TERNARY,
        "stocks_k0001": TERNARY,
        "accepts_assignment": TERNARY,
        "serves_patient_area": TERNARY,
        "earliest_delivery_days": {
            "type": ["integer", "null"],
            "description": "Business days until they could deliver. null if not stated.",
        },
        "requested_identifiers_we_cannot_share": {
            "type": "boolean",
            "description": "True if they demanded a Medicare ID, DOB, SSN or similar as a "
            "precondition for helping.",
        },
        "evidence": {
            "type": "object",
            "properties": {
                "accepts_new_medicare_patients": {"type": "string"},
                "stocks_k0001": {"type": "string"},
                "accepts_assignment": {"type": "string"},
                "serves_patient_area": {"type": "string"},
                "earliest_delivery_days": {"type": "string"},
            },
            "required": [
                "accepts_new_medicare_patients",
                "stocks_k0001",
                "accepts_assignment",
                "serves_patient_area",
                "earliest_delivery_days",
            ],
            "additionalProperties": False,
        },
    },
    "required": [
        "accepts_new_medicare_patients",
        "stocks_k0001",
        "accepts_assignment",
        "serves_patient_area",
        "earliest_delivery_days",
        "requested_identifiers_we_cannot_share",
        "evidence",
    ],
    "additionalProperties": False,
}

SUPPLIER_SYSTEM = """You read one recorded phone call between a patient's care team and a \
durable medical equipment supplier, and you record only what the supplier actually said.

The item this case is about is {item}. Wherever a field below says "stocks_k0001", it means \
that item.

Answer each question yes / no / unknown:

- accepts_new_medicare_patients: are they taking on new Medicare patients right now?
- stocks_k0001: can they supply the item named above?
- accepts_assignment: do they accept Medicare assignment -- taking Medicare's approved \
amount as full payment and billing Medicare directly?
- serves_patient_area: will they deliver to the patient's address?
- earliest_delivery_days: business days until delivery, as an integer, or null.

UNKNOWN is the correct and expected answer whenever the call did not settle the question. \
It is not a failure. A receptionist saying "I'd have to check" is unknown, not no. A supplier \
being friendly is not evidence of anything. Never infer one answer from another -- having \
stock says nothing about assignment.

For every answer that is not unknown, quote the supplier's own words from the transcript in \
the matching evidence field, verbatim. If you cannot quote it, the answer is unknown and the \
evidence is an empty string.

Set requested_identifiers_we_cannot_share to true only if they made a patient identifier -- \
Medicare number, date of birth, SSN -- a precondition of helping."""


async def extract_supplier(
    llm: LLM,
    transcript_text: str,
    their_words: str | None = None,
    item: str = "a standard manual wheelchair (HCPCS K0001)",
) -> Extraction:
    try:
        raw = await llm.extract(
            system=SUPPLIER_SYSTEM.format(item=item),
            user=f"<call>\n{transcript_text}\n</call>",
            schema=SUPPLIER_SCHEMA,
        )
    except ExtractionRefused:
        return Extraction(note="extraction refused; nothing learned from this call")

    evidence = raw.get("evidence") or {}
    haystack = their_words if their_words is not None else transcript_text
    out = Extraction(safety_stop=bool(raw.get("requested_identifiers_we_cannot_share")))

    for name in GATE_FIELDS:
        value = raw.get(name, "unknown")
        if value not in ("yes", "no"):
            continue
        if not grounded(evidence.get(name, ""), haystack):
            out.dropped.append(name)
            continue
        out.facts[name] = Ternary(value)

    days = raw.get("earliest_delivery_days")
    if isinstance(days, int) and days >= 0:
        if grounded(evidence.get("earliest_delivery_days", ""), haystack):
            out.facts["earliest_delivery_days"] = days
        else:
            out.dropped.append("earliest_delivery_days")
    return out


CLINIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "will_send_written_order": TERNARY,
        "already_sent": TERNARY,
        "refuses_to_provide": {"type": "boolean"},
        "promised_within_business_days": {
            "type": ["integer", "null"],
            "description": "If they said when it would go out -- 'today', 'by Friday' -- that "
            "many business days. null if they committed to no timeframe.",
        },
        "evidence": {"type": "string"},
    },
    "required": [
        "will_send_written_order", "already_sent", "refuses_to_provide",
        "promised_within_business_days", "evidence",
    ],
    "additionalProperties": False,
}

CLINIC_SYSTEM = """You read one recorded phone call between a patient's care team and a \
physician's office about a written equipment order, and record only what the office said.

- will_send_written_order: did they commit to sending the signed written order?
- already_sent: did they state it has already gone out?
- refuses_to_provide: true only if they declined outright to produce the order.

"We'll take a message" and "I'll pass it along" are not commitments -- those are unknown. \
Quote the office's own words in evidence for any answer that is not unknown."""


async def extract_clinic(llm: LLM, transcript_text: str, their_words: str | None = None) -> Extraction:
    try:
        raw = await llm.extract(
            system=CLINIC_SYSTEM,
            user=f"<call>\n{transcript_text}\n</call>",
            schema=CLINIC_SCHEMA,
        )
    except ExtractionRefused:
        return Extraction(note="extraction refused; nothing learned from this call")

    out = Extraction()
    quote = raw.get("evidence", "")
    is_grounded = grounded(quote, their_words if their_words is not None else transcript_text)
    for name in ("will_send_written_order", "already_sent"):
        value = raw.get(name, "unknown")
        if value in ("yes", "no"):
            if is_grounded:
                out.facts[name] = Ternary(value)
            else:
                out.dropped.append(name)
    out.facts["refuses_to_provide"] = bool(raw.get("refuses_to_provide"))
    promised = raw.get("promised_within_business_days")
    if isinstance(promised, int) and promised >= 0:
        out.facts["promised_within_business_days"] = promised
    return out


BOOKING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "slot_confirmed": TERNARY,
        "delivery_in_business_days": {"type": ["integer", "null"]},
        "promised_within_business_days": {
            "type": ["integer", "null"],
            "description": "If they said when they would come back to us -- 'two days', "
            "'end of the week' -- that many business days. null if they gave no timeframe.",
        },
        "evidence": {"type": "string"},
    },
    "required": [
        "slot_confirmed", "delivery_in_business_days",
        "promised_within_business_days", "evidence",
    ],
    "additionalProperties": False,
}

BOOKING_SYSTEM = """You read one recorded phone call in which a patient's care team tried to \
book a delivery slot with a supplier.

- slot_confirmed: yes if the supplier committed to delivering, no if they declined, unknown \
if they left it open.
- promised_within_business_days: if they said when they would get back to you -- "I'll call \
you tomorrow", "give us a couple of days" -- record that as business days. Say null if they \
named no timeframe at all. This is when THEY said they would act, not how fast they deliver.

A commitment does not have to be a calendar date. This industry quotes lead times, and "we \
can have it to her in about two days" or "three business days once we have the order" is a \
supplier committing to deliver -- that is yes. What is NOT a commitment: deferring to \
somebody else ("the coordinator will call you back", "I'd have to check", "let me take your \
number"), or committing only to start paperwork ("I'll get that going on my end") with no \
timeframe at all. Those are unknown.

- delivery_in_business_days: the lead time they quoted, as an integer, or null.

Quote the supplier's own words in evidence."""


async def extract_booking(llm: LLM, transcript_text: str, their_words: str | None = None) -> Extraction:
    try:
        raw = await llm.extract(
            system=BOOKING_SYSTEM,
            user=f"<call>\n{transcript_text}\n</call>",
            schema=BOOKING_SCHEMA,
        )
    except ExtractionRefused:
        return Extraction(note="extraction refused; nothing learned from this call")

    out = Extraction()
    is_grounded = grounded(
        raw.get("evidence", ""), their_words if their_words is not None else transcript_text
    )
    value = raw.get("slot_confirmed", "unknown")
    if value in ("yes", "no") and is_grounded:
        out.facts["slot_confirmed"] = Ternary(value)
    elif value in ("yes", "no"):
        out.dropped.append("slot_confirmed")
    days = raw.get("delivery_in_business_days")
    if isinstance(days, int) and days >= 0:
        out.facts["delivery_in_business_days"] = days
    promised = raw.get("promised_within_business_days")
    if isinstance(promised, int) and promised >= 0:
        out.facts["promised_within_business_days"] = promised
    return out


PATIENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "understood_the_cost": TERNARY,
        "agrees_to_proceed": TERNARY,
        "evidence": {"type": "string"},
    },
    "required": ["understood_the_cost", "agrees_to_proceed", "evidence"],
    "additionalProperties": False,
}

PATIENT_SYSTEM = """You read one recorded phone call between a care team and an elderly \
patient about equipment their doctor has ordered, and record only what the patient said.

- understood_the_cost: did they follow what they will be responsible for?
- agrees_to_proceed: did they clearly agree to go ahead?

A clear "yes, go ahead" is yes. A clear refusal, or "not until I've spoken to my daughter", \
is no. Anything else -- politeness, thanks, a question left hanging, the call ending without \
an answer -- is unknown. Do not read agreement into good manners.

Quote the patient's own words in evidence for any answer that is not unknown."""


async def extract_patient(
    llm: LLM, transcript_text: str, their_words: str | None = None
) -> Extraction:
    """What the patient actually agreed to.

    Consent is the one fact in this system with a person's money behind it, so
    it is held to the same bar as everything else: it has to be quotable, and
    the quote has to be theirs.
    """
    try:
        raw = await llm.extract(
            system=PATIENT_SYSTEM,
            user=f"<call>\n{transcript_text}\n</call>",
            schema=PATIENT_SCHEMA,
            name="patient_consent",
        )
    except ExtractionRefused:
        return Extraction(note="extraction refused; nothing learned from this call")

    out = Extraction()
    is_grounded = grounded(
        raw.get("evidence", ""), their_words if their_words is not None else transcript_text
    )
    for name in ("understood_the_cost", "agrees_to_proceed"):
        value = raw.get(name, "unknown")
        if value in ("yes", "no"):
            if is_grounded:
                out.facts[name] = Ternary(value)
            else:
                out.dropped.append(name)
    return out
