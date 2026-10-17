import unittest
from datetime import date, datetime

from calling_agent.simulation import demo_slots


class DemoAvailabilityTests(unittest.TestCase):
    def test_exact_date_offers_only_the_requested_day_and_spoken_time(self):
        slots = demo_slots(1, "morning", on_date="2026-11-13", today=date(2026, 10, 17))
        self.assertEqual(
            [s["starts_at"][:10] for s in slots], ["2026-11-13", "2026-11-13"]
        )
        self.assertEqual([s["starts_at"][11:16] for s in slots], ["09:00", "10:30"])
        self.assertEqual(slots[1]["spoken_label"], "pátek 13. listopadu v 10:30")

    def test_past_or_weekend_exact_date_is_not_silently_moved(self):
        for requested in ("2026-10-13", "2026-11-14"):
            self.assertEqual(
                demo_slots(1, "morning", on_date=requested, today=date(2026, 10, 17)),
                [],
            )

    def test_slots_are_future_weekdays_in_prague_and_match_period(self):
        for period in ("morning", "afternoon"):
            slots = demo_slots(1, period, today=date(2026, 10, 23))
            self.assertEqual(len(slots), 6)
            for slot in slots:
                start = datetime.fromisoformat(slot["starts_at"])
                self.assertGreater(start.date(), date(2026, 10, 23))
                self.assertLess(start.weekday(), 5)
                self.assertEqual(start.utcoffset().total_seconds(), 3600)
                self.assertEqual(start.hour < 12, period == "morning")
                self.assertTrue(slot["simulated"])
            self.assertEqual(len({s["id"] for s in slots}), 6)

    def test_requested_future_period_is_respected(self):
        slots = demo_slots(4, "either", "2026-11-10", today=date(2026, 10, 17))
        self.assertEqual(slots[0]["starts_at"][:10], "2026-11-10")
        self.assertEqual({s["day_period"] for s in slots}, {"morning", "afternoon"})
        with self.assertRaises(ValueError):
            demo_slots(4, not_before="tomorrow")
