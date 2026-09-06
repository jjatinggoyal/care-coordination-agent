"""Virtual clock and business-hours calendar.

The case opens at 02:14 — the care advocate is asleep, and so is every supplier
in the directory. Almost all of the interesting sequencing in this problem is a
consequence of that: what can be done at 2am (assemble the packet, draft the
order request, rank the directory) and what has to be queued until somebody
picks up a phone at 09:00.

Time is virtual so the whole case can be replayed in a second. Nothing in the
system calls datetime.now() -- the clock is passed in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

CENTRAL = ZoneInfo("America/Chicago")


@dataclass(frozen=True)
class Hours:
    """When an organisation answers its phone. Weekday indices are Mon=0."""

    open_hour: int = 9
    close_hour: int = 17
    weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)

    def is_open(self, at: datetime) -> bool:
        local = at.astimezone(CENTRAL)
        return local.weekday() in self.weekdays and self.open_hour <= local.hour < self.close_hour

    def next_open(self, at: datetime) -> datetime:
        """The first instant at or after `at` when this org is reachable."""
        local = at.astimezone(CENTRAL)
        for _ in range(14):  # a fortnight is plenty; guards against an empty weekday set
            if local.weekday() in self.weekdays:
                if local.hour < self.open_hour:
                    return local.replace(hour=self.open_hour, minute=0, second=0, microsecond=0)
                if local.hour < self.close_hour:
                    return local
            local = (local + timedelta(days=1)).replace(
                hour=self.open_hour, minute=0, second=0, microsecond=0
            )
        raise ValueError("no open window found within 14 days")

    def add_business_hours(self, at: datetime, hours: float) -> datetime:
        """`at` plus `hours` of open time, skipping nights and weekends.

        This is how every follow-up is scheduled. 'Call them back in two hours'
        must not mean 2am, and 'nudge the clinic tomorrow' must not mean Sunday.
        """
        cursor = self.next_open(at)
        remaining = timedelta(hours=hours)
        while remaining > timedelta(0):
            local = cursor.astimezone(CENTRAL)
            closes = local.replace(hour=self.close_hour, minute=0, second=0, microsecond=0)
            available = closes - local
            if available >= remaining:
                return local + remaining
            remaining -= available
            cursor = self.next_open(closes + timedelta(minutes=1))
        return cursor


    def add_business_days(self, at: datetime, days: int) -> datetime:
        """`at` plus `days` working days, landing at opening time.

        Distinct from add_business_hours on purpose. A nine-hour working day
        means '24 business hours' is nearly three days away -- which is not what
        anybody means by 'chase it tomorrow'. Follow-ups that a person would
        describe in days are scheduled in days.
        """
        cursor = self.next_open(at)
        for _ in range(max(0, days)):
            local = cursor.astimezone(CENTRAL)
            close = local.replace(hour=self.close_hour, minute=0, second=0, microsecond=0)
            cursor = self.next_open(close + timedelta(minutes=1))
        return cursor


SUPPLIER_HOURS = Hours(open_hour=9, close_hour=17)
CLINIC_HOURS = Hours(open_hour=8, close_hour=17)
# The patient has no business hours, but we still do not text a 72-year-old at 3am.
PATIENT_HOURS = Hours(open_hour=9, close_hour=20, weekdays=(0, 1, 2, 3, 4, 5, 6))


@dataclass
class Clock:
    """A clock the engine advances explicitly. Never wall time."""

    now: datetime
    _elapsed: timedelta = field(default=timedelta(0), init=False)

    def advance_to(self, when: datetime) -> None:
        if when < self.now:
            raise ValueError(f"clock cannot run backwards: {when} < {self.now}")
        self._elapsed += when - self.now
        self.now = when

    def advance(self, delta: timedelta) -> None:
        self.advance_to(self.now + delta)

    @property
    def elapsed(self) -> timedelta:
        return self._elapsed

    def stamp(self) -> str:
        return self.now.astimezone(CENTRAL).strftime("%a %d %b %H:%M")
