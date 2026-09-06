"""Our voice on the phone.

The agent's job is bounded on purpose: get through a conversation with a human
and come back with what was said. It does not decide whether a supplier
qualifies, whether to try the next one, what the patient owes, or when to give
up. Those are decisions, and decisions live in policy.py.

The prompts below are written as constraints rather than goals. An agent told to
"get the wheelchair delivered" will start negotiating, agreeing to things, and
answering questions it should refuse. An agent told exactly which four things to
ask, what it may say about the patient, and when to hang up, stays inside its
lane -- and when it does step outside, the extractor's groundedness check and
the policy's typed gates catch it downstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..llm import LLM

# Patterns for identifiers the agent does not have and must never invent.
# A model asked for a field it was never given will produce a plausible one --
# this is the same failure as the "[Your Name]" placeholder, except a fabricated
# street address gets confirmed by the person on the other end and ends up on a
# delivery. Detected in code, because a prompt rule is a request, not a control.
STREET = re.compile(
    r"\b\d{1,5}\s+[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,3}\s+"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Way|Terrace)\b\.?",
    re.I,
)
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
MBI = re.compile(r"\b[1-9][A-Z][0-9A-Z]\d[A-Z][0-9A-Z]\d[A-Z]{2}\d{2}\b")
DOB = re.compile(r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b")
# A figure the agent cannot possibly know. There is no fee schedule in this
# system and no way to know how much of a deductible has been met, so any
# amount spoken aloud is invented -- and the person on the other end will
# plan around it. Percentages are fine; money is not.
MONEY = re.compile(r"[$£€]\s?\d|\b\d[\d,]*(?:\.\d{2})?\s?(?:dollars|bucks|usd)\b", re.I)

REFUSAL = (
    "The delivery address is on the written order the doctor's office sent over -- "
    "I'm not able to read patient details back over the phone, but it should match "
    "what you have on file."
)
PATIENT_REFUSAL = (
    "Sorry — I shouldn't be reading personal details back to you over the phone. The "
    "delivery details come from the order your doctor's office sent, so we have what we need."
)
NO_FIGURE = (
    "I can't give you an exact amount -- that depends on Medicare's approved rate and "
    "how much of your deductible you've already met this year. What I can tell you is "
    "the shares: Medicare pays 80%, and the remaining 20% is yours."
)

END = "[END CALL]"
MAX_TURNS = 10

# The agent needs a name. Without one it introduces itself as "[Your Name]",
# which is both absurd on a phone call and a nice reminder that a model will
# happily emit a placeholder into production if you leave a hole for it.
CALLER_NAME = "Sam Rivera"

# What the agent is permitted to disclose. This is the minimum-necessary rule
# made concrete: qualifying a supplier needs no patient identifiers at all, so
# the agent is not given any to leak.
DISCLOSURE_RULES = """
What you may say about the patient: that they are a Medicare Part B patient in Chicago, what
equipment they need, and their ordering physician's name and practice. Their neighbourhood or
ZIP if asked whether they are in the delivery area.

You do NOT have their street address, their date of birth, their Medicare Beneficiary
Identifier or their Social Security number. None of them are in front of you. If a
conversation seems to need one, the answer is that you do not have it -- not a
plausible-looking guess. Never state a street address; the supplier has the patient's details
on the written order and can read them there.

What you must never say, to anyone, for any reason: their Medicare Beneficiary Identifier,
their date of birth, or their Social Security number. You do not have these and you must not
guess or improvise them. If somebody makes one a condition of helping you, say plainly that you cannot
provide it over the phone and that a member of the care team will follow up, then end the
call. Do not argue.

Once a delivery is actually being booked you may confirm the patient's name and the ZIP you
were given, and you may point the supplier at the written order for the rest. That is not obstruction --
the order is the record, and it is already in their hands.
"""

STANCE = f"""
You are {CALLER_NAME}. Introduce yourself by that name -- never with a placeholder.

You never commit the patient to anything. You do not agree to a purchase, quote a price,
accept a delivery slot on their behalf, or say what they will owe. If asked to decide something,
say you will confirm and come back.

Produce ONLY your own next turn, then stop. Never write the other person's reply -- do not
continue past your own words, do not put answers in their mouth, do not script the rest of the
call. You say one thing and then you wait, exactly as you would on a real phone.

Keep it to one or two sentences per turn -- this is a phone call, not an email. No narration,
no stage directions, no speaker labels. Be warm and quick; the person you are calling is at
work and has other lines ringing.

Ask ONE question per turn and wait for the answer. Stacking three questions into a single
breath is not how people talk on the phone, and it reliably gets you a partial answer to the
first one and silence on the rest. Work down your list in order, one at a time, and do not
move on until the current one has actually been answered.

If somebody genuinely cannot answer, take that as an answer and stop pushing -- two attempts
at the same question is the limit. Do not fill silence with questions you were not sent to
ask, and do not end the call while items on your list are still unasked and the person is
still willing to talk.

End the call ONLY when every question on your list has either been answered or clearly
refused, or the person has made it plain they cannot help further. Then thank them, say
goodbye, and put {END} on its own final line -- goodbye before that line, not after.
"""

def field_questions(case) -> dict[str, str]:
    """What to ask, phrased for the item this case is actually about.

    The internal key stays `stocks_k0001` because that is the gate's name in the
    policy; what goes down the phone line is whatever the case asked for. A
    hospital bed case that asks a supplier about wheelchairs is not a smaller
    problem than a crash -- it just fails more quietly.
    """
    item = f"{case.equipment.lower()} (HCPCS {case.hcpcs})"
    return {
        "accepts_new_medicare_patients": "whether they are accepting new Medicare patients",
        "stocks_k0001": f"whether they have {item} in stock",
        "accepts_assignment": (
            "whether they accept Medicare assignment -- that is, bill Medicare directly and "
            "take the approved amount as payment in full"
        ),
        "serves_patient_area": "whether they deliver to the patient's neighbourhood",
        "earliest_delivery_days": "how soon they could deliver once a written order is in hand",
    }


@dataclass
class Transcript:
    call_id: str
    lines: list[tuple[str, str]] = field(default_factory=list)   # (speaker, text)
    blocked: list[str] = field(default_factory=list)             # identifiers we refused to invent

    def add(self, speaker: str, text: str) -> None:
        self.lines.append((speaker, text))

    @property
    def turns(self) -> int:
        return sum(1 for speaker, _ in self.lines if speaker == "agent")

    def render(self, them: str = "Supplier") -> str:
        label = {"agent": "Care team", "them": them}
        return "\n".join(f"{label[s]}: {t}" for s, t in self.lines)

    def their_words(self) -> str:
        """Only what the other party actually said.

        Evidence for a claim about the supplier has to come out of the
        supplier's own mouth. Grounding against the whole transcript looks
        equivalent and is not: our agent has, in testing, written the other
        side's replies into its own turn -- asking a question and answering it
        in the same breath -- and every one of those invented 'yes'es is
        present in the full transcript for a quote to match against. Restricting
        the haystack to their turns makes that failure inert instead of
        catastrophic.
        """
        return "\n".join(text for speaker, text in self.lines if speaker == "them")

    def as_agent_messages(self) -> list[dict]:
        return [
            {"role": "assistant" if s == "agent" else "user", "content": t} for s, t in self.lines
        ]

    def as_them_messages(self) -> list[dict]:
        messages: list[dict] = [{"role": "user", "content": "[the phone rings and you pick it up]"}]
        for speaker, text in self.lines:
            messages.append(
                {"role": "assistant" if speaker == "them" else "user", "content": text}
            )
        return messages


def supplier_system(case, supplier_name: str, ask: tuple[str, ...]) -> str:
    questions = field_questions(case)
    wanted = "\n".join(f"  {i}. Ask {questions[f]}." for i, f in enumerate(ask, 1))
    n = len(ask)
    return (
        f"You are a care coordinator calling {supplier_name}, a durable medical equipment "
        f"supplier, on behalf of a patient's care team. You are not the patient and you say so "
        f"early.\n\n"
        f"You are qualifying them for {case.equipment.lower()} (HCPCS {case.hcpcs}) for a patient "
        f"on {case.patient.coverage}. Their physician is {case.pcp_name} at "
        f"{case.pcp_practice}. They live in Chicago, ZIP {case.patient.zip_code}.\n\n"
        f"You have exactly {n} question(s) to get through on this call:\n{wanted}\n"
        f"Work down that list in order, one question per turn. Before you say goodbye, check "
        f"the list: if any of the {n} still has no answer and the person is still on the line, "
        f"ask the next one instead of closing. Getting {n - 1} of {n} means somebody has to "
        f"call this supplier back tomorrow, so finish the list.\n\n"
        f"The exception: any clear NO ends the call. If they are not taking new Medicare "
        f"patients, cannot supply the chair, do not accept assignment, or will not deliver to "
        f"her, then this supplier is out and the rest of your list no longer matters. Thank "
        f"them and finish -- do not spend their time on questions whose answers you will never "
        f"use.\n"
        f"{DISCLOSURE_RULES}{STANCE}"
    )


def clinic_system(case, nudging: bool) -> str:
    context = (
        "You have called before about this and were told it would be sent. It has not arrived. "
        "You are following up -- friendly, not accusatory -- and you want to know whether it "
        "actually went out and when."
        if nudging
        else
        "This is your first call about it. Confirm the visit and the verbal order are in the "
        "chart, and ask them to have the written order signed and sent."
    )
    return (
        f"You are a care coordinator calling {case.pcp_practice}, {case.pcp_name}'s office, "
        f"about a written equipment order.\n\n"
        f"The patient is {case.patient.name}, who saw {case.pcp_name} a few days ago, and there "
        f"is a verbal order noted in their chart for {case.equipment.lower()} (HCPCS "
        f"{case.hcpcs}). Medicare will not pay without a signed written order.\n\n"
        f"{context}\n\n"
        f"Ask for a specific commitment -- who is sending it and when. 'We'll pass it along' is "
        f"not something you can work with, so ask once for something firmer. Do not dictate the "
        f"clinical content of the order and do not suggest a different equipment code; that is "
        f"the physician's call, not yours."
        f"{DISCLOSURE_RULES}{STANCE}"
    )


def booking_system(case, supplier_name: str) -> str:
    return (
        f"You are a care coordinator calling {supplier_name} to book a delivery.\n\n"
        f"They have already told you they take new Medicare patients, accept assignment, have "
        f"{case.equipment.lower()} (HCPCS {case.hcpcs}) in stock, and deliver to this "
        f"patient's area. The signed written order from {case.pcp_name} is in hand.\n\n"
        f"The patient is {case.patient.name}, in Chicago ZIP {case.patient.zip_code}, and has "
        f"agreed to proceed.\n\n"
        f"The signed written order has ALREADY been faxed to them -- say so plainly if they ask "
        f"for it, and offer to resend it rather than refusing. It is their own paperwork and "
        f"you are permitted to send it.\n\n"
        f"Get one thing: a commitment to deliver, with a timeframe. A lead time is a perfectly "
        f"good answer -- 'two days once we have the order' is a booking, and you should accept "
        f"it and confirm it back to them rather than pushing for a calendar date. What is not a "
        f"booking is being handed to somebody else: if they will only take your number and have "
        f"a coordinator call back, accept that politely and end the call."
        f"{DISCLOSURE_RULES}{STANCE}"
    )


def invented_identifier(text: str) -> str:
    """Name the identifier our agent just made up, or an empty string."""
    for label, pattern in (
        ("street address", STREET),
        ("Social Security number", SSN),
        ("Medicare Beneficiary Identifier", MBI),
        ("date of birth", DOB),
    ):
        if pattern.search(text):
            return label
    return ""


def quotes_a_figure(text: str) -> bool:
    """Did the agent just say a number that would land as a bill?"""
    return bool(MONEY.search(text))


def patient_system(case, topic: str, supplier_name: str = "", when=None) -> str:
    who = f"{case.patient.name}, {case.patient.age}" if case.patient.age else case.patient.name
    if topic == "cost":
        objective = (
            "Explain what Medicare will and will not cover, and get a clear yes or no about "
            "going ahead.\n\n"
            "Cover, in plain words: Medicare Part B pays 80% of its approved amount for this "
            "equipment; the remaining 20% is theirs; that sits on top of any part of the "
            "annual Part B deductible they have not met; and the supplier bills Medicare "
            "directly, so there is no large payment up front.\n\n"
            "Never say a dollar amount. You do not have the fee schedule and you do not know "
            "how much of their deductible is met, so any figure you give would be invented and "
            "they would plan around it. Say what the shares are, not what they come to. If "
            "they press for a number, say plainly that you cannot give one and why.\n\n"
            "You cannot look anything up. You have no access to their deductible balance, "
            "their claims history, or any account. If they ask what they have already paid "
            "this year, say plainly that you cannot see that and that Medicare or their "
            "statements will show it. NEVER offer to go and check, and never promise to call "
            "back with a figure -- you would not be able to, and they would wait for it.\n\n"
            "End by asking clearly whether they are happy for you to go ahead."
        )
    else:
        objective = (
            f"Tell them the delivery is booked with {supplier_name}"
            + (f" for {when:%A %d %B}" if when else "")
            + ". Say the supplier will call before they come, and give them a way to reach the "
            "care team. Keep it short and warm. No dollar amounts."
        )
    return (
        f"You are a care coordinator ringing {who} at home about "
        f"{case.equipment.lower()} their doctor has ordered.\n\n{objective}\n\n"
        f"They are elderly and this is their own health and money, so go at their pace, use no "
        f"jargon, and check they have followed you before moving on.\n\n"
        f"Do NOT ask them to confirm their address, their date of birth, or any other detail "
        f"about themselves. You are not verifying anything and you do not need it -- the "
        f"delivery details come from the written order. Asking makes you sound like a scam "
        f"call, and it is the one thing an older person on the phone should be wary of."
        f"{STANCE}"
    )


async def _run(
    llm: LLM, world, transcript: Transcript, agent_prompt: str, them_reply,
    refusal: str = REFUSAL,
) -> Transcript:
    """Drive the two-sided conversation. Our agent always speaks second."""
    for _ in range(MAX_TURNS):
        ours = await llm.say(
            system=agent_prompt,
            messages=transcript.as_agent_messages(),
            role="caller",
            max_tokens=180,   # a phone turn is short; a long one is the model monologuing
        )
        finished = END in ours
        spoken = ours.replace(END, "").strip()

        # The guard. A turn that states an identifier we were never given does
        # not go down the line -- it is replaced by what a careful coordinator
        # would actually say. Recorded, because the rate is worth watching.
        made_up = invented_identifier(spoken)
        if made_up:
            transcript.blocked.append(made_up)
            spoken = refusal
        elif quotes_a_figure(spoken):
            transcript.blocked.append("a dollar amount it cannot know")
            spoken = NO_FIGURE

        if spoken:
            transcript.add("agent", spoken)
        if finished:
            break
        theirs = await them_reply(transcript)
        transcript.add("them", theirs)
    return transcript


async def call_supplier(llm: LLM, world, case, supplier, ask: tuple[str, ...], call_id: str) -> Transcript:
    transcript = Transcript(call_id)
    transcript.add("them", await world.supplier_opens(supplier.supplier_id, supplier.name))
    return await _run(
        llm,
        world,
        transcript,
        supplier_system(case, supplier.name, ask),
        lambda t: world.supplier_replies(supplier.supplier_id, supplier.name, t.as_them_messages()),
    )


async def call_clinic(llm: LLM, world, case, call_id: str, nudging: bool) -> Transcript:
    transcript = Transcript(call_id)
    transcript.add("them", await world.clinic_opens())
    return await _run(
        llm,
        world,
        transcript,
        clinic_system(case, nudging),
        lambda t: world.clinic_replies(t.as_them_messages()),
    )


async def call_patient(llm: LLM, world, case, call_id: str, topic: str,
                       supplier_name: str = "", when=None) -> Transcript:
    transcript = Transcript(call_id)
    transcript.add("them", await world.patient_opens())
    return await _run(
        llm,
        world,
        transcript,
        patient_system(case, topic, supplier_name, when),
        lambda t: world.patient_replies(t.as_them_messages()),
        refusal=PATIENT_REFUSAL,
    )


async def call_to_book(llm: LLM, world, case, supplier, call_id: str) -> Transcript:
    transcript = Transcript(call_id)
    transcript.add("them", await world.supplier_opens(supplier.supplier_id, supplier.name))
    return await _run(
        llm,
        world,
        transcript,
        booking_system(case, supplier.name),
        lambda t: world.supplier_replies(supplier.supplier_id, supplier.name, t.as_them_messages()),
    )
