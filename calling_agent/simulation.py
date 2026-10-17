"""Invented demo availability. Never connects to a center or reserves a real slot."""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo


PRAGUE = ZoneInfo("Europe/Prague")
WEEKDAYS = ("pondělí", "úterý", "středa", "čtvrtek", "pátek", "sobota", "neděle")
MONTHS = (
    "ledna",
    "února",
    "března",
    "dubna",
    "května",
    "června",
    "července",
    "srpna",
    "září",
    "října",
    "listopadu",
    "prosince",
)


def demo_slots(
    center_id: int,
    day_period: str = "either",
    not_before: str = "",
    on_date: str = "",
    *,
    today: date | None = None,
) -> list[dict]:
    today = today or datetime.now(PRAGUE).date()
    exact = date.fromisoformat(on_date) if on_date else None
    if exact and (exact <= today or exact.weekday() >= 5):
        return []
    first = max(
        today + timedelta(days=1),
        date.fromisoformat(not_before) if not_before else today,
    )
    if exact:
        if exact < first:
            return []
        first = exact
    hours = {
        "morning": ("09:00", "10:30"),
        "afternoon": ("14:00", "15:30"),
        "either": ("09:00", "14:00"),
    }[day_period]
    slots = []
    day = first
    while len(slots) < (2 if exact else 6):
        if day.weekday() < 5:
            for hour in hours:
                start = datetime.combine(day, time.fromisoformat(hour), PRAGUE)
                slots.append(
                    {
                        "id": f"demo:{center_id}:{day.isoformat()}T{hour}",
                        "center_id": center_id,
                        "starts_at": start.isoformat(),
                        "label": f"{WEEKDAYS[day.weekday()]} {day.day}. {day.month}. {day.year} v {hour}",
                        "spoken_label": f"{WEEKDAYS[day.weekday()]} {day.day}. {MONTHS[day.month - 1]}"
                        + (f" {day.year}" if day.year != today.year else "")
                        + f" v {hour}",
                        "day_period": "morning" if hour < "12:00" else "afternoon",
                        "simulated": True,
                    }
                )
        day += timedelta(days=1)
    return slots
