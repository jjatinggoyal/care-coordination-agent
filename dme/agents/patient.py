"""What the care team is allowed to say to the patient about money.

Contacting the patient used to mean drafting a message. It is a phone call now,
like every other party in this workflow -- which means consent is something
somebody said out loud and we can quote, rather than a flag the simulation set.

What survives is the rule, and it moved somewhere better. The draft used to be
checked before sending; the check now sits on every turn the agent speaks
(`caller.quotes_a_figure`), because on a live call there is no draft to inspect.

The rule itself is unchanged: no dollar amounts. There is no fee schedule in
this system and no way to know how much of a Part B deductible has been met, so
any figure would be invented -- and an elderly person on a fixed income would
plan around it. Percentages are knowable and get said; totals are not and do not.
"""

from __future__ import annotations

import re

MONEY = re.compile(r"[$£€]\s?\d|\b\d[\d,]*(?:\.\d+)?\s?(?:dollars|usd)\b", re.I)


def passes_cost_check(text: str) -> bool:
    """Does an explanation of cost say the shares without inventing a total?"""
    if MONEY.search(text):
        return False
    if "20" not in text:
        return False
    return "deductible" in text.lower()
