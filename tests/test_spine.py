"""Tests for the deterministic half.

These run with no API key and make no network calls, which is the whole claim:
if the sequencing, the eligibility gates, the retry schedule and the escalation
rules can be tested without a model, then a model is not making those decisions.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dme import events as ev
from dme import policy
from dme.agents.extract import grounded
from dme.agents.patient import passes_cost_check
from dme.clock import CENTRAL, CLINIC_HOURS, SUPPLIER_HOURS
from dme.loader import load_case
from dme.model import (
    GATE_FIELDS,
    HumanTaskKind,
    Track,
    Case,
    CaseStatus,
    CommitmentKind,
    OrderStatus,
    Patient,
    SupplierStatus,
    Ternary,
)
from dme.reducer import Ledger, apply

THU_2AM = datetime(2026, 9, 3, 2, 14, tzinfo=CENTRAL)


def fresh() -> Case:
    case, _ = load_case()
    return case


def observe(case: Case, ledger: Ledger, supplier_id: str, **facts) -> None:
    ledger.append(
        ev.SupplierFactsObserved(
            at=case.opened_at,
            supplier_id=supplier_id,
            supplier_name=case.suppliers[supplier_id].name,
            facts=facts,
            source="call:test",
        )
    )


class TestClock(unittest.TestCase):
    def test_nothing_is_open_at_2am(self):
        self.assertFalse(SUPPLIER_HOURS.is_open(THU_2AM))
        self.assertFalse(CLINIC_HOURS.is_open(THU_2AM))

    def test_work_queues_to_opening(self):
        self.assertEqual(SUPPLIER_HOURS.next_open(THU_2AM).hour, 9)
        self.assertEqual(CLINIC_HOURS.next_open(THU_2AM).hour, 8)

    def test_business_hours_skip_the_weekend(self):
        friday_3pm = datetime(2026, 9, 4, 15, 0, tzinfo=CENTRAL)
        landed = SUPPLIER_HOURS.add_business_hours(friday_3pm, 4)
        self.assertEqual(landed.strftime("%a"), "Mon")

    def test_chasing_it_tomorrow_means_monday_when_today_is_friday(self):
        friday_2pm = datetime(2026, 9, 4, 14, 0, tzinfo=CENTRAL)
        landed = CLINIC_HOURS.add_business_days(friday_2pm, policy.ORDER_VERIFY_AFTER_DAYS)
        self.assertEqual(landed.strftime("%a"), "Mon")
        self.assertEqual(landed.hour, CLINIC_HOURS.open_hour)


class TestQualification(unittest.TestCase):
    def test_one_no_disqualifies(self):
        case = fresh()
        ledger = Ledger(case)
        observe(case, ledger, "s01", accepts_new_medicare_patients=Ternary.NO)
        verdict, reason = policy.gate_verdict(case.suppliers["s01"])
        self.assertEqual(verdict, "disqualified")
        self.assertEqual(reason, "accepts_new_medicare_patients")

    def test_silence_never_qualifies_anybody(self):
        case = fresh()
        ledger = Ledger(case)
        observe(
            case, ledger, "s01",
            accepts_new_medicare_patients=Ternary.YES,
            stocks_k0001=Ternary.YES,
            accepts_assignment=Ternary.YES,
        )
        verdict, missing = policy.gate_verdict(case.suppliers["s01"])
        self.assertEqual(verdict, "incomplete")
        self.assertEqual(missing, "serves_patient_area")

    def test_four_yeses_qualify(self):
        case = fresh()
        ledger = Ledger(case)
        observe(
            case, ledger, "s01",
            accepts_new_medicare_patients=Ternary.YES,
            stocks_k0001=Ternary.YES,
            accepts_assignment=Ternary.YES,
            serves_patient_area=Ternary.YES,
        )
        self.assertEqual(policy.gate_verdict(case.suppliers["s01"])[0], "qualified")

    def test_no_assignment_is_a_hard_stop(self):
        """She has no Medigap. A non-assigned supplier means she pays up front."""
        case = fresh()
        ledger = Ledger(case)
        observe(
            case, ledger, "s01",
            accepts_new_medicare_patients=Ternary.YES,
            stocks_k0001=Ternary.YES,
            accepts_assignment=Ternary.NO,
            serves_patient_area=Ternary.YES,
        )
        self.assertEqual(policy.gate_verdict(case.suppliers["s01"])[0], "disqualified")

    def test_facts_carry_where_they_came_from(self):
        case = fresh()
        ledger = Ledger(case)
        observe(case, ledger, "s01", stocks_k0001=Ternary.YES)
        self.assertEqual(case.suppliers["s01"].facts["stocks_k0001"].source, "call:test")


class TestRetrySchedule(unittest.TestCase):
    def test_first_call_waits_for_opening(self):
        case = fresh()
        due = policy.supplier_due_at(case, case.suppliers["s01"])
        self.assertEqual(due.hour, 9)

    def test_three_dead_dials_and_we_stop(self):
        case = fresh()
        ledger = Ledger(case)
        at = SUPPLIER_HOURS.next_open(case.opened_at)
        for i in range(policy.MAX_DIAL_ATTEMPTS):
            ledger.append(
                ev.SupplierCalled(
                    at=at + timedelta(hours=2 * i),
                    supplier_id="s01",
                    supplier_name=case.suppliers["s01"].name,
                    call_id=f"c{i}",
                    outcome=ev.CallOutcome.NO_ANSWER,
                )
            )
        self.assertIsNone(policy.supplier_due_at(case, case.suppliers["s01"]))

    def test_voicemail_backs_off_further_than_a_busy_signal(self):
        case = fresh()
        at = SUPPLIER_HOURS.next_open(case.opened_at)
        busy = Ledger(fresh())
        case_b = fresh()
        lb = Ledger(case_b)
        lb.append(ev.SupplierCalled(at=at, supplier_id="s01", supplier_name="x",
                                    call_id="c1", outcome=ev.CallOutcome.BUSY))
        case_v = fresh()
        lv = Ledger(case_v)
        lv.append(ev.SupplierCalled(at=at, supplier_id="s01", supplier_name="x",
                                    call_id="c1", outcome=ev.CallOutcome.VOICEMAIL))
        self.assertLess(
            policy.supplier_due_at(case_b, case_b.suppliers["s01"]),
            policy.supplier_due_at(case_v, case_v.suppliers["s01"]),
        )


class TestSequencing(unittest.TestCase):
    def test_at_2am_the_only_move_is_to_wait(self):
        case = fresh()
        action = policy.decide(case, case.opened_at)
        self.assertIsInstance(action, policy.Wait)
        self.assertGreater(action.until, case.opened_at)

    def test_the_clinic_is_called_before_the_suppliers(self):
        """The written order is the longest pole and nobody else can produce it."""
        case = fresh()
        action = policy.decide(case, CLINIC_HOURS.next_open(case.opened_at))
        self.assertIsInstance(action, policy.CallClinic)

    def test_nothing_is_booked_without_her_agreement(self):
        case = fresh()
        ledger = Ledger(case)
        now = SUPPLIER_HOURS.next_open(case.opened_at)
        observe(
            case, ledger, "s01",
            accepts_new_medicare_patients=Ternary.YES, stocks_k0001=Ternary.YES,
            accepts_assignment=Ternary.YES, serves_patient_area=Ternary.YES,
            earliest_delivery_days=3,
        )
        ledger.append(ev.SupplierQualified(at=now, supplier_id="s01", supplier_name="x"))
        ledger.append(ev.OrderReceived(at=now, coded_as="K0001"))
        action = policy.decide(case, now)
        self.assertIsInstance(action, policy.ContactPatient)
        self.assertEqual(action.topic, "cost")

    def _ready_to_book(self, case, ledger, now):
        observe(
            case, ledger, "s01",
            accepts_new_medicare_patients=Ternary.YES, stocks_k0001=Ternary.YES,
            accepts_assignment=Ternary.YES, serves_patient_area=Ternary.YES,
            earliest_delivery_days=3,
        )
        ledger.append(ev.SupplierQualified(at=now, supplier_id="s01", supplier_name="x"))
        ledger.append(ev.OrderReceived(at=now, coded_as="K0001"))
        ledger.append(ev.PatientContacted(at=now, topic="cost"))
        ledger.append(ev.ConsentRecorded(at=now, value=Ternary.YES))

    def test_the_join_hands_the_order_over_before_it_books(self):
        """No supplier quotes a date against an order they have not seen."""
        case = fresh()
        ledger = Ledger(case)
        now = SUPPLIER_HOURS.next_open(case.opened_at)
        self._ready_to_book(case, ledger, now)

        first = policy.decide(case, now)
        self.assertIsInstance(first, policy.SendOrderToSupplier)
        self.assertEqual(first.supplier_id, "s01")

        ledger.append(
            ev.OrderSentToSupplier(at=now, supplier_id="s01", supplier_name="x", channel="fax")
        )
        second = policy.decide(case, now)
        self.assertIsInstance(second, policy.ScheduleDelivery)
        self.assertEqual(second.supplier_id, "s01")

    def test_the_order_is_handed_to_whoever_actually_qualified(self):
        """Sending it to one supplier does not count as sending it to another."""
        case = fresh()
        ledger = Ledger(case)
        now = SUPPLIER_HOURS.next_open(case.opened_at)
        self._ready_to_book(case, ledger, now)
        ledger.append(
            ev.OrderSentToSupplier(at=now, supplier_id="s07", supplier_name="other", channel="fax")
        )
        action = policy.decide(case, now)
        self.assertIsInstance(action, policy.SendOrderToSupplier)
        self.assertEqual(action.supplier_id, "s01")

    def test_an_unconfirmed_booking_blocks_a_second_attempt(self):
        case = fresh()
        ledger = Ledger(case)
        now = SUPPLIER_HOURS.next_open(case.opened_at)
        observe(
            case, ledger, "s01",
            accepts_new_medicare_patients=Ternary.YES, stocks_k0001=Ternary.YES,
            accepts_assignment=Ternary.YES, serves_patient_area=Ternary.YES,
            earliest_delivery_days=3,
        )
        ledger.append(ev.SupplierQualified(at=now, supplier_id="s01", supplier_name="x"))
        ledger.append(ev.OrderReceived(at=now, coded_as="K0001"))
        ledger.append(ev.PatientContacted(at=now, topic="cost"))
        ledger.append(ev.ConsentRecorded(at=now, value=Ternary.YES))
        ledger.append(
            ev.OrderSentToSupplier(at=now, supplier_id="s01", supplier_name="x", channel="fax")
        )
        ledger.append(
            ev.CommitmentMade(
                at=now, commitment_id="cm01", kind=CommitmentKind.HOLD_STOCK,
                by_party="s01", promised_by=now + timedelta(hours=8),
                verify_at=now + timedelta(hours=8),
            )
        )
        self.assertNotIsInstance(policy.decide(case, now), policy.ScheduleDelivery)


class TestChasingTheWrittenOrder(unittest.TestCase):
    """The longest pole, and the one nobody else can produce for us."""

    def _called(self, case, ledger, at, outcome=ev.CallOutcome.ANSWERED):
        ledger.append(ev.ClinicCalled(at=at, call_id="c", outcome=outcome))
        ledger.append(ev.OrderRequested(at=at, channel="phone"))

    def test_a_broken_promise_puts_the_order_back_in_the_queue(self):
        case = fresh()
        ledger = Ledger(case)
        now = CLINIC_HOURS.next_open(case.opened_at)
        self._called(case, ledger, now)
        ledger.append(ev.OrderPromised(at=now, commitment_id="cm01"))
        ledger.append(
            ev.CommitmentMade(
                at=now, commitment_id="cm01", kind=CommitmentKind.SEND_WRITTEN_ORDER,
                by_party="clinic", promised_by=now, verify_at=now,
            )
        )
        self.assertEqual(case.order.status, OrderStatus.PROMISED)
        ledger.append(
            ev.CommitmentBroken(
                at=now, commitment_id="cm01", by_party="clinic",
                kind=CommitmentKind.SEND_WRITTEN_ORDER,
            )
        )
        self.assertEqual(case.order.status, OrderStatus.REQUESTED)

    def test_nobody_picking_up_backs_off_in_hours_not_days(self):
        """A dead line teaches us nothing, so there is nothing to verify tomorrow."""
        case = fresh()
        ledger = Ledger(case)
        now = CLINIC_HOURS.next_open(case.opened_at)
        self._called(case, ledger, now, outcome=ev.CallOutcome.NO_ANSWER)
        due = policy.clinic_due_at(case)
        self.assertEqual(due.date(), now.date())
        self.assertLess(due - now, timedelta(hours=4))

    def test_a_fax_gets_a_working_day_before_anyone_is_asked(self):
        case = fresh()
        ledger = Ledger(case)
        now = CLINIC_HOURS.next_open(case.opened_at)
        self._called(case, ledger, now)
        ledger.append(ev.OrderRequested(at=now, channel="fax"))
        self.assertIsNotNone(case.order.faxed_at)
        self.assertNotIsInstance(policy.decide(case, now), policy.RequestHumanTask)
        due = policy.clinic_due_at(case)
        self.assertGreater(due, now + timedelta(hours=8))

        # Phone and fax exhausted: ask a person to hand it over, do not give up.
        action = policy.decide(case, due)
        self.assertIsInstance(action, policy.RequestHumanTask)
        self.assertEqual(action.kind, HumanTaskKind.POST_PHYSICAL_REQUEST)
        self.assertEqual(action.blocks, Track.ORDER)

    def test_the_supplier_search_carries_on_while_a_person_posts_the_order(self):
        """The whole point of a task that blocks a track instead of the case."""
        case = fresh()
        ledger = Ledger(case)
        now = CLINIC_HOURS.next_open(case.opened_at)
        self._called(case, ledger, now)
        ledger.append(ev.OrderRequested(at=now, channel="fax"))
        ledger.append(
            ev.HumanTaskRequested(
                at=now, task_id="h01", kind=HumanTaskKind.POST_PHYSICAL_REQUEST,
                blocks=Track.ORDER, packet={},
            )
        )
        self.assertTrue(case.blocked(Track.ORDER))
        self.assertFalse(case.blocked(Track.SUPPLIER))

        action = policy.decide(case, SUPPLIER_HOURS.next_open(now))
        self.assertIsInstance(action, policy.CallSupplier)

    def test_the_order_track_resumes_when_the_person_is_done(self):
        case = fresh()
        ledger = Ledger(case)
        now = CLINIC_HOURS.next_open(case.opened_at)
        self._called(case, ledger, now)
        ledger.append(ev.OrderRequested(at=now, channel="fax"))
        ledger.append(
            ev.HumanTaskRequested(
                at=now, task_id="h01", kind=HumanTaskKind.POST_PHYSICAL_REQUEST,
                blocks=Track.ORDER, packet={},
            )
        )
        ledger.append(ev.HumanTaskCompleted(at=now, task_id="h01", note="walked it over"))
        self.assertFalse(case.blocked(Track.ORDER))

    def test_it_only_gives_up_after_a_person_has_had_their_turn(self):
        case = fresh()
        ledger = Ledger(case)
        now = CLINIC_HOURS.next_open(case.opened_at)
        self._called(case, ledger, now)
        ledger.append(ev.OrderRequested(at=now, channel="fax"))
        ledger.append(
            ev.HumanTaskRequested(
                at=now, task_id="h01", kind=HumanTaskKind.POST_PHYSICAL_REQUEST,
                blocks=Track.ORDER, packet={},
            )
        )
        ledger.append(ev.HumanTaskCompleted(at=now, task_id="h01", note="posted"))
        due = policy.clinic_due_at(case)
        action = policy.decide(case, due)
        self.assertIsInstance(action, policy.Escalate)
        self.assertEqual(action.reason, policy.EscalationReason.ORDER_UNOBTAINABLE)


class TestEscalation(unittest.TestCase):
    def _rule_out_everyone(self, case: Case, ledger: Ledger) -> None:
        for supplier_id, supplier in case.suppliers.items():
            ledger.append(
                ev.SupplierDisqualified(
                    at=case.opened_at, supplier_id=supplier_id,
                    supplier_name=supplier.name, reason="stocks_k0001",
                )
            )

    def test_empty_directory_escalates(self):
        case = fresh()
        ledger = Ledger(case)
        ledger.append(ev.OrderReceived(at=case.opened_at, coded_as="K0001"))
        self._rule_out_everyone(case, ledger)
        action = policy.decide(case, case.opened_at)
        self.assertIsInstance(action, policy.Escalate)
        self.assertEqual(action.reason, policy.EscalationReason.NO_QUALIFIED_SUPPLIER)

    def _disclosure_blocked(self):
        case = fresh()
        ledger = Ledger(case)
        ledger.append(ev.OrderReceived(at=case.opened_at, coded_as="K0001"))
        self._rule_out_everyone(case, ledger)
        ledger.append(
            ev.SupplierDisqualified(
                at=case.opened_at, supplier_id="s03",
                supplier_name=case.suppliers["s03"].name, reason=policy.DISCLOSURE_BLOCK,
            )
        )
        return case, ledger

    def test_a_disclosure_block_asks_a_person_before_giving_up(self):
        """A call we may not make is not a dead end — somebody else can make it."""
        case, _ = self._disclosure_blocked()
        action = policy.decide(case, case.opened_at)
        self.assertIsInstance(action, policy.RequestHumanTask)
        self.assertEqual(action.kind, HumanTaskKind.CALL_WITH_IDENTIFIERS)
        self.assertIsNone(action.blocks, "nothing should wait on this — call other suppliers")
        self.assertIn(case.suppliers["s03"].name, action.packet["suppliers"])

    def test_and_escalates_with_that_reason_once_they_have(self):
        case, ledger = self._disclosure_blocked()
        ledger.append(
            ev.HumanTaskRequested(
                at=case.opened_at, task_id="h01",
                kind=HumanTaskKind.CALL_WITH_IDENTIFIERS, blocks=None, packet={},
            )
        )
        ledger.append(ev.HumanTaskCompleted(at=case.opened_at, task_id="h01", note="tried"))
        action = policy.decide(case, case.opened_at)
        self.assertEqual(action.reason, policy.EscalationReason.CALL_SAFETY_STOP)
        self.assertIn(case.suppliers["s03"].name, action.packet["blocked_on_disclosure"])

    def test_the_wrong_billing_code_stops_the_case(self):
        case = fresh()
        ledger = Ledger(case)
        ledger.append(ev.OrderReceived(at=case.opened_at, coded_as="K0003"))
        action = policy.decide(case, case.opened_at)
        self.assertIsInstance(action, policy.Escalate)
        self.assertEqual(action.reason, policy.EscalationReason.ORDER_CODING_MISMATCH)

    def test_a_refusal_is_a_clinical_decision_not_a_retry(self):
        case = fresh()
        ledger = Ledger(case)
        ledger.append(ev.OrderRefused(at=case.opened_at, reason="declined"))
        action = policy.decide(case, case.opened_at)
        self.assertEqual(action.reason, policy.EscalationReason.ORDER_REFUSED)

    def test_declining_the_cost_stops_the_case(self):
        case = fresh()
        ledger = Ledger(case)
        now = case.opened_at
        observe(
            case, ledger, "s01",
            accepts_new_medicare_patients=Ternary.YES, stocks_k0001=Ternary.YES,
            accepts_assignment=Ternary.YES, serves_patient_area=Ternary.YES,
        )
        ledger.append(ev.SupplierQualified(at=now, supplier_id="s01", supplier_name="x"))
        ledger.append(ev.ConsentRecorded(at=now, value=Ternary.NO))
        action = policy.decide(case, now)
        self.assertEqual(action.reason, policy.EscalationReason.PATIENT_DECLINED_COST)

    def test_every_escalation_reason_tells_a_human_what_to_do(self):
        for reason in policy.EscalationReason:
            self.assertIn(reason, policy.HUMAN_NEXT_STEP)
            self.assertGreater(len(policy.HUMAN_NEXT_STEP[reason]), 40)


class TestLedger(unittest.TestCase):
    def test_state_is_reconstructible_from_the_log(self):
        case = fresh()
        ledger = Ledger(case)
        now = case.opened_at
        observe(
            case, ledger, "s01",
            accepts_new_medicare_patients=Ternary.YES, stocks_k0001=Ternary.YES,
            accepts_assignment=Ternary.YES, serves_patient_area=Ternary.YES,
        )
        ledger.append(ev.SupplierQualified(at=now, supplier_id="s01", supplier_name="x"))
        ledger.append(ev.OrderReceived(at=now, coded_as="K0001"))

        rebuilt, _ = load_case()
        ledger.replay(rebuilt)
        self.assertEqual(rebuilt.suppliers["s01"].status, SupplierStatus.QUALIFIED)
        self.assertEqual(rebuilt.order.status, OrderStatus.RECEIVED)
        self.assertEqual(
            rebuilt.suppliers["s01"].facts["stocks_k0001"].source,
            case.suppliers["s01"].facts["stocks_k0001"].source,
        )

    def test_an_unhandled_event_raises_rather_than_being_ignored(self):
        class Rogue(ev.Event):
            pass

        with self.assertRaises(ValueError):
            apply(fresh(), Rogue(at=THU_2AM))


class TestModelOutputGates(unittest.TestCase):
    def test_a_quote_nobody_said_is_rejected(self):
        transcript = "Supplier: We take new Medicare patients, sure."
        self.assertTrue(grounded("we take new Medicare patients", transcript))
        self.assertFalse(grounded("we accept Medicare assignment", transcript))

    def test_the_patient_message_may_not_carry_a_figure(self):
        self.assertTrue(passes_cost_check(
            "Medicare pays 80%, you owe 20% coinsurance plus any unmet Part B deductible."
        ))
        self.assertFalse(passes_cost_check(
            "Medicare pays 80%; your 20% share is about $47 once the deductible is met."
        ))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestSeams(unittest.TestCase):
    """The simulator is swappable because it satisfies protocols, not a base class."""

    def test_the_world_implements_every_adapter(self):
        from dme.adapters import BookingLedger, OrderInbox, PatientChannel, PhoneTransport
        from dme.llm import Usage
        from dme.sim.world import World

        world = World.__new__(World)  # no client, no key -- structure only
        for protocol in (PhoneTransport, OrderInbox, PatientChannel, BookingLedger):
            self.assertIsInstance(world, protocol, f"World does not satisfy {protocol.__name__}")


class TestBookingHours(unittest.TestCase):
    def test_a_booking_call_is_not_placed_before_the_supplier_opens(self):
        case = fresh()
        ledger = Ledger(case)
        now = CLINIC_HOURS.next_open(case.opened_at)   # 08:00 — clinic open, suppliers not
        observe(
            case, ledger, "s01",
            accepts_new_medicare_patients=Ternary.YES, stocks_k0001=Ternary.YES,
            accepts_assignment=Ternary.YES, serves_patient_area=Ternary.YES,
            earliest_delivery_days=3,
        )
        ledger.append(ev.SupplierQualified(at=now, supplier_id="s01", supplier_name="x"))
        ledger.append(ev.OrderReceived(at=now, coded_as="K0001"))
        ledger.append(ev.PatientContacted(at=now, topic="cost"))
        ledger.append(ev.ConsentRecorded(at=now, value=Ternary.YES))
        ledger.append(
            ev.OrderSentToSupplier(at=now, supplier_id="s01", supplier_name="x", channel="fax")
        )
        self.assertFalse(SUPPLIER_HOURS.is_open(now))
        action = policy.decide(case, now)
        self.assertIsInstance(action, policy.Wait)
        self.assertEqual(action.until.hour, SUPPLIER_HOURS.open_hour)


class TestShoppingStopsForTheRightReason(unittest.TestCase):
    def _qualify(self, case, ledger, supplier_id, days):
        facts = {f: Ternary.YES for f in GATE_FIELDS}
        if days is not None:
            facts["earliest_delivery_days"] = days
        observe(case, ledger, supplier_id, **facts)
        ledger.append(
            ev.SupplierQualified(at=case.opened_at, supplier_id=supplier_id, supplier_name="x")
        )

    def test_speed_is_on_the_call_list(self):
        case = fresh()
        action = policy.decide(case, SUPPLIER_HOURS.next_open(case.opened_at))
        # the clinic goes first at 08:00; at 09:00 suppliers are due
        while not isinstance(action, policy.CallSupplier):
            case.order.status = OrderStatus.RECEIVED
            action = policy.decide(case, SUPPLIER_HOURS.next_open(case.opened_at))
        self.assertIn("earliest_delivery_days", action.ask)

    def test_a_fast_confirmed_supplier_ends_the_search(self):
        case = fresh()
        ledger = Ledger(case)
        self._qualify(case, ledger, "s01", days=3)
        self.assertFalse(policy.keep_shopping(case, case.opened_at))

    def test_an_unknown_lead_time_does_not_end_the_search(self):
        """Otherwise the first supplier to qualify wins, however slow they are."""
        case = fresh()
        ledger = Ledger(case)
        self._qualify(case, ledger, "s01", days=None)
        self.assertTrue(policy.keep_shopping(case, case.opened_at))

    def test_a_three_week_supplier_does_not_end_the_search(self):
        case = fresh()
        ledger = Ledger(case)
        self._qualify(case, ledger, "s01", days=21)
        self.assertTrue(policy.keep_shopping(case, case.opened_at))

    def test_the_fastest_qualified_supplier_wins(self):
        case = fresh()
        ledger = Ledger(case)
        self._qualify(case, ledger, "s01", days=21)
        self._qualify(case, ledger, "s02", days=3)
        self.assertEqual(policy.best_qualified(case).supplier_id, "s02")


class TestExtractionFailureIsHarmless(unittest.TestCase):
    """A model that cannot answer must slow the case down, never corrupt it."""

    def test_a_refused_extraction_teaches_us_nothing_and_says_no_to_nothing(self):
        import asyncio

        from dme.agents import extract
        from dme.llm import ExtractionRefused

        class Broken:
            async def extract(self, **_):
                raise ExtractionRefused("provider could not produce valid JSON")

        found = asyncio.run(
            extract.extract_supplier(Broken(), "Care team: hello\nSupplier: hi")
        )
        self.assertEqual(found.facts, {})
        self.assertFalse(found.safety_stop)
        self.assertIn("nothing", found.note)

    def test_and_the_supplier_stays_qualifiable(self):
        """The gates must read UNKNOWN, not NO -- a failure is not a rejection."""
        case = fresh()
        for name in GATE_FIELDS:
            self.assertIs(case.suppliers["s01"].gate(name), Ternary.UNKNOWN)
        self.assertEqual(policy.gate_verdict(case.suppliers["s01"])[0], "incomplete")


class TestTheSystemCannotBeFooledByItsOwnVoice(unittest.TestCase):
    """A real failure, caught in testing and fixed structurally.

    The caller model asked a question and answered it in the same turn --
    writing the supplier's replies itself -- and the extractor read those
    invented 'yes'es as facts. Three of one supplier's four qualification gates
    came from words nobody on the other end ever said.

    A prompt telling the model not to do that is worth having and is not a
    defence. The defence is that evidence for a claim about the supplier is only
    ever matched against the supplier's own turns.
    """

    def _fabricated(self):
        from dme.agents.caller import Transcript

        return Transcript(
            "c01",
            [
                ("them", "Northside CarePlus Equipment, this is Dana. How can I help you?"),
                (
                    "agent",
                    "Do you accept Medicare assignment, billing Medicare directly?"
                    "Yes, we do accept Medicare assignment and bill directly."
                    "Do you deliver to the Chicago 60614 area?"
                    "Yes, we deliver to that area.",
                ),
            ],
        )

    def test_the_fabrication_really_is_in_the_transcript(self):
        """Proving the old check would have passed it -- this is not a straw man."""
        transcript = self._fabricated()
        self.assertTrue(
            grounded("we do accept Medicare assignment and bill directly", transcript.render())
        )

    def test_but_it_is_not_in_anything_they_said(self):
        transcript = self._fabricated()
        self.assertFalse(
            grounded("we do accept Medicare assignment and bill directly", transcript.their_words())
        )
        self.assertFalse(grounded("we deliver to that area", transcript.their_words()))

    def test_their_words_holds_only_their_turns(self):
        transcript = self._fabricated()
        self.assertIn("this is Dana", transcript.their_words())
        self.assertNotIn("Do you accept", transcript.their_words())


class TestTheAgentCannotInventIdentifiers(unittest.TestCase):
    """The second fabrication the eval caught, and the guard that stops it.

    Relaxing the disclosure rule so the agent would stop refusing to give a
    delivery address created a hole, and the model filled it: it read out
    "1234 West Maple Street" to a supplier, who confirmed it matched the order.
    Eleanor has no street address anywhere in this system -- only an assumed ZIP.

    A prompt rule is a request. This is the control.
    """

    def test_a_made_up_street_address_is_caught(self):
        from dme.agents.caller import invented_identifier

        self.assertEqual(
            invented_identifier("The delivery address is 1234 West Maple Street, Chicago, IL"),
            "street address",
        )

    def test_identifiers_we_never_hold_are_caught(self):
        from dme.agents.caller import invented_identifier

        self.assertEqual(invented_identifier("Her DOB is 03/14/1954."), "date of birth")
        self.assertEqual(invented_identifier("SSN 123-45-6789"), "Social Security number")

    def test_legitimate_turns_pass_through(self):
        from dme.agents.caller import invented_identifier

        for line in (
            "Her ZIP is 60614 and the ordering physician is Dr. Sarah Chen.",
            "Could you deliver to her neighbourhood within three business days?",
            "I'm calling on behalf of a Medicare Part B patient's care team.",
        ):
            self.assertEqual(invented_identifier(line), "", line)


class TestArbitraryCases(unittest.TestCase):
    """Eleanor is the fixture the brief supplies, not a special case in the code."""

    def test_a_different_patient_and_item_builds(self):
        from dme.loader import build_case, default_payload

        payload = default_payload()
        payload["patient"]["name"] = "Harold Byrne"
        payload["equipment"] = "Hospital bed, semi-electric"
        payload["hcpcs"] = "E0260"
        payload["suppliers"] = payload["suppliers"][:3]
        case, _ = build_case(payload)
        self.assertEqual(case.patient.name, "Harold Byrne")
        self.assertEqual(case.hcpcs, "E0260")
        self.assertEqual(case.order.hcpcs, "E0260")
        self.assertEqual(len(case.suppliers), 3)

    def test_a_naive_form_time_is_read_as_chicago(self):
        """A browser datetime-local field carries no zone; the clock needs one."""
        from dme.loader import parse_opened_at

        naive = parse_opened_at("2026-09-04T16:40:00")
        self.assertIsNotNone(naive.tzinfo)
        self.assertEqual(naive.hour, 16)
        aware = parse_opened_at("2026-09-03T02:14:00-05:00")
        self.assertEqual(aware.astimezone(CENTRAL).hour, 2)

    def test_an_empty_directory_is_rejected_not_crashed(self):
        from dme.loader import build_case, default_payload

        payload = default_payload()
        payload["suppliers"] = []
        case, _ = build_case(payload)
        self.assertEqual(len(case.suppliers), 0)
        # policy must still be able to answer; it simply has nobody to call
        self.assertIsNotNone(policy.decide(case, case.opened_at))


class TestTheWorldFollowsTheCase(unittest.TestCase):
    """Caught by running a non-wheelchair case through the configurable simulator.

    The clinic personas wrote 'K0001' whatever the case asked for, so a hospital
    bed came back miscoded and escalated every time. The system under test was
    right; the simulated world was wrong -- which is its own kind of bug, and one
    a demo fixed to a single case can never surface.
    """

    def test_an_honest_clinic_writes_the_code_that_was_asked_for(self):
        from dme.sim.personas import CLINIC_PERSONAS

        self.assertEqual(CLINIC_PERSONAS["prompt"].miscodes_as, "")
        self.assertEqual(CLINIC_PERSONAS["stalls_once"].miscodes_as, "")

    def test_only_the_miscoding_clinic_miscodes(self):
        from dme.sim.personas import CLINIC_PERSONAS

        self.assertTrue(CLINIC_PERSONAS["miscodes"].miscodes_as)
        wrong = {k: c.miscodes_as for k, c in CLINIC_PERSONAS.items() if c.miscodes_as}
        self.assertEqual(list(wrong), ["miscodes"])


class TestNoCaseLeaksIntoPrompts(unittest.TestCase):
    """The configurable simulator's real value: it exposed what was welded shut.

    Running a hospital-bed case revealed that the prompts still asked suppliers
    about wheelchairs, the simulated front desk still discussed Eleanor
    Martinez, and the clinic still wrote K0001 whatever the case said. All three
    were invisible while the only case anyone ran was the brief's.
    """

    def _other_case(self):
        from dme.loader import build_case, default_payload

        payload = default_payload()
        payload["patient"].update(name="Harold Byrne", age=78, zip_code="60640")
        payload.update(
            equipment="Hospital bed, semi-electric", hcpcs="E0260",
            pcp_name="Dr. Nadia Okonjo", pcp_practice="Uptown Internal Medicine",
        )
        return build_case(payload)[0]

    BRIEF = ("Eleanor", "Sarah Chen", "Sunrise", "wheelchair", "K0001")

    def test_the_supplier_call_asks_about_this_case(self):
        from dme.agents import caller

        case = self._other_case()
        prompt = caller.supplier_system(case, "Acme DME", ("stocks_k0001", "earliest_delivery_days"))
        for token in self.BRIEF:
            self.assertNotIn(token, prompt, f"{token!r} leaked from the brief's case")
        self.assertIn("E0260", prompt)

    def test_the_clinic_call_asks_about_this_case(self):
        from dme.agents import caller

        case = self._other_case()
        prompt = caller.clinic_system(case, nudging=False)
        for token in self.BRIEF:
            self.assertNotIn(token, prompt, f"{token!r} leaked from the brief's case")
        self.assertIn("Harold Byrne", prompt)

    def test_the_simulated_clinic_talks_about_this_case(self):
        from dme.sim.world import World

        case = self._other_case()
        world = World.__new__(World)
        world.patient_name, world.pcp_name = case.patient.name, case.pcp_name
        world.practice, world.equipment = case.pcp_practice, case.equipment
        world.hcpcs, world.clinic_persona_key = case.hcpcs, "prompt"
        prompt = world._clinic_system()
        for token in self.BRIEF:
            self.assertNotIn(token, prompt, f"{token!r} leaked from the brief's case")
        self.assertIn("Harold Byrne", prompt)
        self.assertIn("Dr. Nadia Okonjo", prompt)


class TestTheDecisionsStayedSynchronous(unittest.TestCase):
    """The async refactor was supposed to touch the edges and nothing else.

    Cloudflare Workers have no sockets: outbound HTTP is `fetch`, and `fetch`
    cannot be awaited from synchronous code, so the call path had to become
    async all the way down. `policy.py` did not, because it performs no I/O at
    all — and this test is what keeps it that way.
    """

    def test_policy_has_no_coroutines(self):
        import inspect

        from dme import policy

        coroutines = [
            name for name, fn in vars(policy).items()
            if inspect.iscoroutinefunction(fn)
        ]
        self.assertEqual(coroutines, [], "a decision started doing I/O")

    def test_policy_imports_nothing_that_talks_to_a_network(self):
        import dme.policy as policy

        source = Path(policy.__file__).read_text(encoding="utf-8")
        for banned in ("import openai", "urllib", "from .llm", "import js", "fetch("):
            self.assertNotIn(banned, source, f"policy.py reached for {banned!r}")

    def test_the_engine_edges_are_coroutines(self):
        import inspect

        from dme.engine import Engine

        for name in ("run", "step", "_call_supplier", "_call_clinic", "_book"):
            self.assertTrue(
                inspect.iscoroutinefunction(getattr(Engine, name)),
                f"Engine.{name} should be async",
            )


class TestPlatformLimitsAreTyped(unittest.TestCase):
    """A runtime cutting us off is not the same as a case failing.

    Cloudflare caps outbound requests per invocation. Hitting that is a billing
    setting, and reporting it as a JsException from three layers down tells
    whoever is watching nothing they can act on.
    """

    def test_the_subrequest_limit_has_its_own_type(self):
        from dme.llm import SubrequestLimit, describe_error

        self.assertIn("subrequest", describe_error(SubrequestLimit("Too many subrequests")))

    def test_it_is_not_confused_with_a_provider_error(self):
        from dme.llm import HttpError, describe_error

        self.assertEqual(describe_error(HttpError(429, "slow down")), "rate limited")
        self.assertEqual(describe_error(HttpError(404, "nope")), "model or endpoint not found")


class TestPromisesAreHonouredOnTheirOwnTerms(unittest.TestCase):
    """"We'll confirm in two days" should mean two days, not the default.

    The commitment machinery was always there — a promise carries a verification
    time and the case advances only on something observed. What was missing is
    that the time was a constant: whatever anyone said, the system checked back
    tomorrow. It now takes the timeframe from their own words, within a cap.
    """

    def test_a_stated_timeframe_sets_the_deadline(self):
        now = SUPPLIER_HOURS.next_open(THU_2AM)          # Thursday 09:00
        promised_by, verify_at = policy.promise_window(SUPPLIER_HOURS, now, 2)
        self.assertEqual(promised_by.strftime("%a"), "Mon")   # Thu + 2 working days
        self.assertGreater(verify_at, promised_by)            # a grace margin after it

    def test_no_timeframe_falls_back_to_the_default(self):
        now = SUPPLIER_HOURS.next_open(THU_2AM)
        default_by, _ = policy.promise_window(SUPPLIER_HOURS, now, None)
        stated_by, _ = policy.promise_window(SUPPLIER_HOURS, now, policy.ORDER_VERIFY_AFTER_DAYS)
        self.assertEqual(default_by, stated_by)

    def test_today_means_later_today_not_tomorrow(self):
        now = SUPPLIER_HOURS.next_open(THU_2AM)
        promised_by, _ = policy.promise_window(SUPPLIER_HOURS, now, 0)
        self.assertEqual(promised_by.date(), now.date())

    def test_an_optimistic_promise_is_capped(self):
        """Believe them, but do not let a case stall for a fortnight on a maybe."""
        now = SUPPLIER_HOURS.next_open(THU_2AM)
        far, _ = policy.promise_window(SUPPLIER_HOURS, now, 30)
        capped, _ = policy.promise_window(SUPPLIER_HOURS, now, policy.MAX_PROMISE_HONOURED_DAYS)
        self.assertEqual(far, capped)

    def test_the_extractor_is_allowed_to_report_one(self):
        from dme.agents.extract import BOOKING_SCHEMA, CLINIC_SCHEMA

        for schema in (BOOKING_SCHEMA, CLINIC_SCHEMA):
            self.assertIn("promised_within_business_days", schema["properties"])
            self.assertIn("promised_within_business_days", schema["required"])


class TestVoiceIsStillWiredUp(unittest.TestCase):
    """Caught after the async refactor: Voice.say became a coroutine and the
    local server's handler kept calling it synchronously, so the audio endpoint
    returned nothing at all until somebody clicked a play button."""

    def test_say_is_a_coroutine(self):
        import inspect

        from dme.voice import Voice

        self.assertTrue(inspect.iscoroutinefunction(Voice.say))

    def test_the_handler_awaits_it(self):
        source = Path("dme/web.py").read_text(encoding="utf-8")
        voice_call = source[source.index("def _voice"):source.index("def _run")]
        self.assertIn("asyncio.run(", voice_call, "the voice handler must drive the coroutine")

    def test_speakers_are_cast_deterministically(self):
        from dme.voice import voice_for

        self.assertEqual(voice_for("them", "Roosevelt"), voice_for("them", "Roosevelt"))
        self.assertNotEqual(voice_for("agent", ""), voice_for("them", "Roosevelt"))


class TestThePatientGetsAPhoneCall(unittest.TestCase):
    """Every other party in this workflow is reached by telephone. The patient
    was the exception — a drafted message, and consent set by a simulation flag.
    Now she is rung like anyone else, and her consent has to be quotable."""

    def test_the_agent_cannot_say_a_figure_out_loud(self):
        from dme.agents.caller import quotes_a_figure

        self.assertTrue(quotes_a_figure("Your share is about $47 after the deductible."))
        self.assertTrue(quotes_a_figure("It comes to 120 dollars."))
        self.assertFalse(quotes_a_figure("Medicare pays 80% and the remaining 20% is yours."))

    def test_the_briefing_forbids_a_figure_and_names_the_shares(self):
        from dme.agents import caller
        from dme.loader import load_case

        case, _ = load_case()
        prompt = caller.patient_system(case, "cost")
        self.assertIn("80%", prompt)
        self.assertIn("deductible", prompt)
        self.assertIn("Never say a dollar amount", prompt)

    def test_good_manners_are_not_consent(self):
        from dme.agents.extract import PATIENT_SYSTEM

        self.assertIn("Do not read agreement into good manners", PATIENT_SYSTEM)
        self.assertIn("agrees_to_proceed", PATIENT_SYSTEM)

    def test_consent_must_be_quotable(self):
        from dme.agents.extract import PATIENT_SCHEMA

        self.assertIn("evidence", PATIENT_SCHEMA["required"])
        self.assertEqual(
            PATIENT_SCHEMA["properties"]["agrees_to_proceed"]["enum"], ["yes", "no", "unknown"]
        )

    def test_she_can_decline_or_never_answer(self):
        from dme.sim.personas import PATIENT_PERSONAS

        self.assertFalse(PATIENT_PERSONAS["declines"].accepts)
        self.assertEqual(PATIENT_PERSONAS["never_answers"].pickup_rate, 0.0)


class TestConsentMustBeInformed(unittest.TestCase):
    """Found by listening to a real call. The agent explained the 80/20 split,
    the patient said "I think I'm mostly following… I don't know what you mean
    by the deductible", and then said yes. The extractor recorded
    understood_the_cost=no — and the system proceeded anyway.

    Agreement from somebody who did not follow what they were agreeing to is not
    consent, and the signal was already there to act on."""

    def _qualified(self):
        case = fresh()
        ledger = Ledger(case)
        now = SUPPLIER_HOURS.next_open(case.opened_at)
        observe(
            case, ledger, "s01",
            accepts_new_medicare_patients=Ternary.YES, stocks_k0001=Ternary.YES,
            accepts_assignment=Ternary.YES, serves_patient_area=Ternary.YES,
            earliest_delivery_days=3,
        )
        ledger.append(ev.SupplierQualified(at=now, supplier_id="s01", supplier_name="x"))
        ledger.append(ev.OrderReceived(at=now, coded_as="K0001"))
        return case, ledger, now

    def test_yes_without_understanding_is_not_recorded_as_consent(self):
        case, ledger, now = self._qualified()
        ledger.append(
            ev.ConsentRecorded(at=now, value=Ternary.YES, understood=Ternary.NO)
        )
        self.assertIs(case.patient_track.consent_given, Ternary.UNKNOWN)
        self.assertIs(case.patient_track.understood_cost, Ternary.NO)

    def test_and_nothing_is_booked_on_it(self):
        case, ledger, now = self._qualified()
        ledger.append(ev.PatientContacted(at=now, topic="cost"))
        ledger.append(ev.ConsentRecorded(at=now, value=Ternary.YES, understood=Ternary.NO))
        action = policy.decide(case, now)
        self.assertNotIsInstance(action, policy.ScheduleDelivery)
        self.assertNotIsInstance(action, policy.SendOrderToSupplier)

    def test_understood_yes_and_agreed_yes_proceeds(self):
        case, ledger, now = self._qualified()
        ledger.append(ev.PatientContacted(at=now, topic="cost"))
        ledger.append(ev.ConsentRecorded(at=now, value=Ternary.YES, understood=Ternary.YES))
        self.assertIs(case.patient_track.consent_given, Ternary.YES)
        self.assertIsInstance(policy.decide(case, now), policy.SendOrderToSupplier)

    def test_it_escalates_rather_than_proceeding_after_repeated_attempts(self):
        case, ledger, now = self._qualified()
        for _ in range(policy.PATIENT_MAX_CONTACT_ATTEMPTS):
            ledger.append(ev.PatientContacted(at=now, topic="cost"))
        ledger.append(ev.ConsentRecorded(at=now, value=Ternary.YES, understood=Ternary.NO))
        action = policy.decide(case, now)
        self.assertIsInstance(action, policy.Escalate)
        self.assertEqual(action.reason, policy.EscalationReason.CONSENT_NOT_INFORMED)

    def test_the_agent_is_told_it_cannot_look_anything_up(self):
        """It offered to 'check her deductible and get back to her'. It cannot."""
        from dme.agents import caller
        from dme.loader import load_case

        prompt = caller.patient_system(load_case()[0], "cost")
        self.assertIn("cannot look anything up", prompt)
        self.assertIn("never promise to call back with a figure", prompt.lower())


class TestTheGuardFitsTheConversation(unittest.TestCase):
    """A real call went wrong twice. The agent asked an elderly patient to read
    her address back, and when the guard blocked its reply it substituted text
    written for a supplier — "the written order the doctor's office sent over" —
    three times running, which is incoherent to a patient and sounds broken."""

    def test_the_patient_call_forbids_fishing_for_details(self):
        from dme.agents import caller
        from dme.loader import load_case

        prompt = caller.patient_system(load_case()[0], "cost")
        self.assertIn("Do NOT ask them to confirm their address", prompt)
        self.assertIn("scam call", prompt)

    def test_there_is_a_refusal_written_for_each_audience(self):
        from dme.agents.caller import PATIENT_REFUSAL, REFUSAL

        self.assertNotEqual(PATIENT_REFUSAL, REFUSAL)
        # the supplier line points at paperwork they hold; the patient line does not
        self.assertIn("we have what we need", PATIENT_REFUSAL)
        self.assertIn("on file", REFUSAL)

    def test_run_takes_the_refusal_as_a_parameter(self):
        import inspect

        from dme.agents import caller

        self.assertIn("refusal", inspect.signature(caller._run).parameters)


class TestNothingChangesStateOutsideTheLedger(unittest.TestCase):
    """The claim is that the ledger is the state. This is the test of it.

    Found when the browser started carrying the log between stateless requests:
    the engine marked a fulfilled commitment by setting a flag rather than
    emitting an event, so the fulfilment vanished on rebuild and the promise was
    later reported broken -- after the order had already arrived."""

    def test_a_fulfilled_commitment_survives_a_replay(self):
        case = fresh()
        ledger = Ledger(case)
        now = CLINIC_HOURS.next_open(case.opened_at)
        ledger.append(
            ev.CommitmentMade(
                at=now, commitment_id="cm01", kind=CommitmentKind.SEND_WRITTEN_ORDER,
                by_party="clinic", promised_by=now, verify_at=now,
            )
        )
        ledger.append(
            ev.CommitmentFulfilled(
                at=now, commitment_id="cm01", by_party="clinic",
                kind=CommitmentKind.SEND_WRITTEN_ORDER,
            )
        )
        self.assertEqual(case.open_commitments(), [])

        rebuilt, _ = load_case()
        ledger.replay(rebuilt)
        self.assertEqual(rebuilt.open_commitments(), [], "fulfilment lost on rebuild")

    def test_the_engine_mutates_no_case_state_directly(self):
        import re

        source = Path("dme/engine.py").read_text(encoding="utf-8")
        stray = [
            line.strip() for line in source.splitlines()
            if re.match(r"\s+(commitment|supplier|task)\.[a-z_]+ = ", line)
        ]
        self.assertEqual(stray, [], "state changed outside an event")


class TestEveryCallIsReachableFromTheFeed(unittest.TestCase):
    """The patient's call was recorded but not linked. Its event carried no
    call_id -- left over from when she was reached by message rather than by
    phone -- so the row in the feed had nothing to open."""

    def test_every_event_that_follows_a_call_carries_its_id(self):
        import dataclasses

        for cls in (ev.SupplierCalled, ev.ClinicCalled, ev.PatientContacted):
            names = [f.name for f in dataclasses.fields(cls)]
            self.assertIn("call_id", names, f"{cls.__name__} cannot be linked to its transcript")
