import json
import tempfile
import unittest
from pathlib import Path

from calling_agent.centers import Directory, parse_directory, safe_url
from calling_agent.store import Store
from calling_agent.tests.helpers import directory_fixture
from calling_agent.workflow import Workflow


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        directory_fixture(self.folder)
        self.store = Store(self.folder / "calls.sqlite3")
        self.directory = Directory(self.folder / "centers.json")
        self.workflow = Workflow(self.store, self.directory)
        self.sid = self.store.new("browser", "test")["id"]
        self.counter = 0

    async def execute(self, name, args, call_id=None):
        self.counter += 1
        return await self.workflow.execute(
            self.sid, call_id or str(self.counter), name, json.dumps(args)
        )

    async def consent(self):
        return await self.execute(
            "record_consent",
            {
                "consent": "granted",
                "user_quote": "Ano, souhlasím s rozhovorem a uložením odpovědí.",
            },
        )

    async def prepare(self):
        await self.consent()
        await self.execute(
            "save_interview",
            {
                "attendance": "not_attended",
                "reason_category": "declined_to_answer",
                "reason_quote": "",
                "municipality": "Brno",
                "travel_preferences": "MHD",
            },
        )
        await self.execute("search_centers", {"query": "Brno"})
        await self.execute(
            "select_center", {"center_id": 1, "user_quote": "První centrum."}
        )
        await self.execute(
            "save_preferences",
            {
                "preferred_dates": "listopad",
                "preferred_times": "odpoledne",
                "accessibility_needs": "",
            },
        )

    async def test_no_collection_before_consent(self):
        result = await self.execute(
            "save_preferences",
            {
                "preferred_dates": "listopad",
                "preferred_times": "odpoledne",
                "accessibility_needs": "",
            },
        )
        self.assertFalse(result["ok"])
        self.assertEqual(self.store.get(self.sid)["payload"], {})

    async def test_public_search_without_saving_caller_information(self):
        result = await self.execute("search_centers", {"query": "Brno"})
        self.assertTrue(result["ok"])
        row = self.store.get(self.sid)
        self.assertEqual(row["payload"], {})
        self.assertEqual(row["consent"], "unknown")

    async def test_demo_booking_requires_offered_future_slot_and_confirmation(self):
        args = {
            "center_id": 1,
            "day_period": "afternoon",
            "not_before": "",
            "on_date": "",
        }
        self.assertFalse((await self.execute("get_demo_slots", args))["ok"])
        await self.execute("search_centers", {"query": "Brno"})
        result = await self.execute("get_demo_slots", args)
        self.assertTrue(result["simulated"])
        self.assertTrue(all(s["day_period"] == "afternoon" for s in result["slots"]))
        self.assertEqual(self.store.get(self.sid)["payload"], {})
        confirm = {
            "slot_id": result["slots"][0]["id"],
            "user_confirmed": True,
            "confirmation_quote": "Ve dvě v tom prvním centru.",
        }
        self.assertFalse(
            (
                await self.execute(
                    "confirm_demo_booking", {**confirm, "slot_id": "invented"}
                )
            )["ok"]
        )
        self.assertFalse(
            (
                await self.execute(
                    "confirm_demo_booking", {**confirm, "user_confirmed": False}
                )
            )["ok"]
        )
        result = await self.execute(
            "confirm_demo_booking", confirm, call_id="confirm_demo"
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["booking_status"], "simulated")
        self.assertEqual(
            (
                await self.execute(
                    "confirm_demo_booking", confirm, call_id="confirm_demo"
                )
            ),
            result,
        )
        booking = self.store.get(self.sid)["payload"]["booking"]
        self.assertTrue(booking["simulated"])
        self.assertEqual(booking["status"], "demo_confirmed")
        await self.execute(
            "request_handoff", {"reason": "symptoms", "note": "Synthetic symptom"}
        )
        self.assertNotIn("booking", self.store.get(self.sid)["payload"])
        self.assertFalse((await self.execute("confirm_demo_booking", confirm))["ok"])

    def scheduling_request(self, **changes):
        return {
            "center_id": 1,
            "preferred_dates": "listopad",
            "preferred_times": "odpoledne",
            "accessibility_needs": "",
            "permission_to_save": True,
            "confirmation_quote": "Ano, poznamenejte si to.",
            **changes,
        }

    async def test_confirmed_scheduling_needs_no_interview(self):
        await self.execute("search_centers", {"query": "Brno"})
        result = await self.execute(
            "save_scheduling_request", self.scheduling_request()
        )
        self.assertTrue(result["ok"])
        row = self.store.get(self.sid)
        self.assertEqual(row["consent"], "granted")
        self.assertEqual(row["payload"]["request"]["booking_status"], "not_booked")
        self.assertNotIn("interview", row["payload"])

    async def test_scheduling_requires_permission_and_offered_center(self):
        for changes in ({}, {"center_id": 9999}):
            result = await self.execute(
                "save_scheduling_request", self.scheduling_request(**changes)
            )
            self.assertFalse(result["ok"])
        await self.execute("search_centers", {"query": "Brno"})
        result = await self.execute(
            "save_scheduling_request", self.scheduling_request(permission_to_save=False)
        )
        self.assertFalse(result["ok"])
        self.assertEqual(self.store.get(self.sid)["payload"], {})

    async def test_scheduling_cannot_override_withdrawal_or_handoff(self):
        await self.execute("search_centers", {"query": "Brno"})
        await self.execute(
            "record_consent", {"consent": "withdrawn", "user_quote": "Neukládejte to."}
        )
        self.assertFalse(
            (await self.execute("save_scheduling_request", self.scheduling_request()))[
                "ok"
            ]
        )
        self.assertEqual(self.store.get(self.sid)["consent"], "withdrawn")
        await self.execute("request_handoff", {"reason": "human_requested", "note": ""})
        self.assertFalse(
            (await self.execute("save_scheduling_request", self.scheduling_request()))[
                "ok"
            ]
        )

    async def test_request_is_never_a_booking_and_optional_reason_can_be_skipped(self):
        await self.prepare()
        result = await self.execute(
            "confirm_request",
            {"user_confirmed": True, "confirmation_quote": "Ano, uložte požadavek."},
        )
        self.assertTrue(result["ok"])
        request = self.store.get(self.sid)["payload"]["request"]
        self.assertEqual(request["status"], "pending_manual_booking")
        self.assertEqual(request["booking_status"], "not_booked")

    async def test_rejects_unknown_or_not_offered_center(self):
        await self.consent()
        for center_id in (1, 9999):
            result = await self.execute(
                "select_center", {"center_id": center_id, "user_quote": "Toto."}
            )
            self.assertFalse(result["ok"])

    async def test_cannot_confirm_incomplete_request(self):
        await self.consent()
        result = await self.execute(
            "confirm_request", {"user_confirmed": True, "confirmation_quote": "Ano."}
        )
        self.assertFalse(result["ok"])

    async def test_changes_invalidate_confirmation(self):
        await self.prepare()
        await self.execute(
            "confirm_request",
            {"user_confirmed": True, "confirmation_quote": "Uložte prosím."},
        )
        await self.execute(
            "save_preferences",
            {
                "preferred_dates": "prosinec",
                "preferred_times": "ráno",
                "accessibility_needs": "",
            },
        )
        self.assertNotIn("request", self.store.get(self.sid)["payload"])

    async def test_replayed_tool_is_idempotent(self):
        await self.prepare()
        args = {"user_confirmed": True, "confirmation_quote": "Ano."}
        first = await self.execute("confirm_request", args, call_id="same")
        second = await self.execute("confirm_request", args, call_id="same")
        self.assertEqual(first, second)

    async def test_symptoms_block_routine_request(self):
        await self.prepare()
        await self.execute(
            "request_handoff", {"reason": "symptoms", "note": "Žena uvádí nové potíže."}
        )
        result = await self.execute(
            "confirm_request", {"user_confirmed": True, "confirmation_quote": "Ano."}
        )
        self.assertFalse(result["ok"])
        self.assertNotIn("request", self.store.get(self.sid)["payload"])

    async def test_opt_out_suppresses_phone_and_blocks_future_collection(self):
        phone = "+420555000111"
        self.store.update(self.sid, phone_hash=self.store.phone_hash(phone))
        await self.prepare()
        await self.execute("opt_out", {"user_quote": "Už mi nevolejte."})
        self.assertTrue(self.store.is_suppressed(phone))
        self.assertEqual(
            self.store.get(self.sid)["payload"], {"outcome": "do_not_contact"}
        )
        self.assertFalse(
            (await self.execute("search_centers", {"query": "Brno"}))["ok"]
        )
        self.assertTrue(Store(self.folder / "calls.sqlite3").is_suppressed(phone))

    async def test_withdrawal_discards_answers_and_prior_tool_cache(self):
        await self.prepare()
        await self.execute(
            "record_consent",
            {"consent": "withdrawn", "user_quote": "Odvolávám souhlas."},
        )
        self.assertEqual(
            self.store.get(self.sid)["payload"], {"outcome": "consent_withdrawn"}
        )
        self.assertFalse(self.store.cached_tool(self.sid, "1")["ok"])
        await self.execute(
            "record_consent",
            {"consent": "granted", "user_quote": "Stará citace souhlasu."},
            call_id="1",
        )
        self.assertEqual(self.store.get(self.sid)["consent"], "withdrawn")
        self.assertFalse(
            (await self.execute("search_centers", {"query": "Brno"}))["ok"]
        )

    async def test_handoff_before_consent_does_not_save_symptoms(self):
        await self.execute(
            "request_handoff",
            {"reason": "symptoms", "note": "Unsolicited clinical detail"},
        )
        self.assertNotIn("Unsolicited", json.dumps(self.store.get(self.sid)["payload"]))

    def test_diacritic_search_and_wait_semantics(self):
        result = self.directory.search("trebic")
        self.assertEqual(result["centers"][0]["city"], "Třebíč")
        self.assertEqual(result["availability"], "unknown")
        self.assertEqual(self.directory.search("Nonexistent")["centers"], [])

    def test_bad_directory_and_unsafe_links_fail_closed(self):
        with self.assertRaises(ValueError):
            parse_directory("<html>Changed layout</html>")
        self.assertEqual(safe_url("javascript:alert(1)"), "")

    def test_recovery_never_redials(self):
        self.assertTrue(self.store.claim_start(self.sid))
        self.assertFalse(self.store.claim_start(self.sid))
        self.store.recover()
        self.assertEqual(self.store.get(self.sid)["state"], "interrupted")
        self.assertFalse(self.store.claim_start(self.sid))
