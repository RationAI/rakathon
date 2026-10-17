import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

import httpx

from calling_agent.centers import Directory
from calling_agent.config import Settings
from calling_agent.live import LiveService, voice_tool_result
from calling_agent.store import Store
from calling_agent.tests.helpers import directory_fixture
from calling_agent.workflow import Workflow


class LiveProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_compact_results_keep_bookable_choice_and_complete_spoken_confirmation(
        self,
    ):
        search = await self.workflow.execute(
            self.sid, "lookup", "search_centers", '{"query":"Brno"}'
        )
        centers = voice_tool_result("search_centers", search)
        center_id = centers["centers"][0]["id"]
        slots = await self.workflow.execute(
            self.sid,
            "slots",
            "get_demo_slots",
            json.dumps(
                {
                    "center_id": center_id,
                    "day_period": "morning",
                    "not_before": "",
                    "on_date": "",
                }
            ),
        )
        offer = voice_tool_result("get_demo_slots", slots)
        self.assertEqual(len(offer["recommended_slots"]), 2)
        chosen = offer["recommended_slots"][1]
        confirmed = await self.workflow.execute(
            self.sid,
            "chosen",
            "confirm_demo_booking",
            json.dumps(
                {
                    "slot_id": chosen["id"],
                    "user_confirmed": True,
                    "confirmation_quote": "V půl jedenácté v tom prvním centru.",
                }
            ),
        )
        spoken = voice_tool_result("confirm_demo_booking", confirmed)
        self.assertTrue(spoken["ok"])
        self.assertIn(chosen["spoken_label"], spoken["spoken_summary"])
        self.assertIn(centers["centers"][0]["name"], spoken["spoken_summary"])
        self.assertLess(len(json.dumps(offer)), len(json.dumps(slots)))
        self.assertEqual(
            self.store.get(self.sid)["payload"]["booking"]["slot"]["id"], chosen["id"]
        )

    async def test_voice_and_tone_apply_to_live_and_separate_greeting_cache(self):
        warm = self.live.session_config(voice="shimmer", tone="gentle")
        direct = self.live.session_config(voice="cedar", tone="business")
        self.assertEqual(warm["audio"]["output"]["voice"], "shimmer")
        self.assertNotEqual(warm["instructions"], direct["instructions"])
        self.live.http.post = AsyncMock(
            return_value=httpx.Response(
                200,
                content=b"synthetic mp3",
                request=httpx.Request("POST", "https://api.openai.com/v1/audio/speech"),
            )
        )
        first = await self.live.greeting_audio(voice="shimmer", tone="gentle")
        second = await self.live.greeting_audio(voice="cedar", tone="business")
        self.assertNotEqual(first, second)
        specs = [call.kwargs["json"] for call in self.live.http.post.await_args_list]
        self.assertEqual([s["voice"] for s in specs], ["shimmer", "cedar"])
        self.assertNotEqual(specs[0]["instructions"], specs[1]["instructions"])

    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        folder = Path(self.temp.name)
        directory_fixture(folder)
        self.store = Store(folder / "calls.sqlite3")
        config = Settings(
            _env_file=None, data_dir=folder, openai_api_key="synthetic-key"
        )
        self.workflow = Workflow(self.store, Directory(folder / "centers.json"))
        self.live = LiveService(config, self.store, self.workflow)
        self.addAsyncCleanup(self.live.http.aclose)
        self.sid = self.store.new("browser", "test")["id"]

    async def test_nested_tool_items_are_returned_then_response_continues(self):
        ws = AsyncMock()
        pending, seen = {}, set()
        output = {
            "type": "response.event",
            "event_id": "event1",
            "delegation_id": "d1",
            "event": {
                "type": "response.output_item.done",
                "response_id": "resp1",
                "item": {
                    "type": "function_call",
                    "call_id": "call1",
                    "name": "record_consent",
                    "arguments": json.dumps(
                        {"consent": "granted", "user_quote": "Souhlasím."}
                    ),
                },
            },
        }
        done = {
            "type": "response.event",
            "event_id": "event2",
            "delegation_id": "d1",
            "event": {
                "type": "response.completed",
                "response": {"id": "resp1", "output": []},
            },
        }
        await self.live.handle_event(self.sid, output, ws, pending, seen)
        ws.send.assert_not_called()
        await self.live.handle_event(self.sid, done, ws, pending, seen)
        sent = [json.loads(call.args[0]) for call in ws.send.await_args_list]
        self.assertEqual(
            [e["type"] for e in sent], ["response.item.create", "response.create"]
        )
        self.assertEqual(sent[0]["item"]["call_id"], "call1")
        await self.live.handle_event(self.sid, output, ws, pending, seen)
        await self.live.handle_event(self.sid, done, ws, pending, seen)
        self.assertEqual(ws.send.await_count, 2)

    async def test_cancelled_response_does_not_execute_pending_tools(self):
        ws = AsyncMock()
        pending = {"r1": {"call1": {"name": "record_consent"}}}
        event = {
            "type": "response.event",
            "event": {"type": "response.cancelled", "response": {"id": "r1"}},
        }
        await self.live.handle_event(self.sid, event, ws, pending, set())
        self.assertEqual(pending, {})
        self.assertEqual(self.store.get(self.sid)["consent"], "unknown")

    async def test_live_output_without_response_id_continues_correctly(self):
        ws = AsyncMock()
        pending, seen = {}, set()
        events = [
            {"type": "response.created", "response": {"id": "response_actual"}},
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "call_id": "lookup_actual",
                    "name": "search_centers",
                    "arguments": '{"query":"Brno"}',
                },
            },
            {
                "type": "response.completed",
                "response": {"id": "response_actual", "output": []},
            },
        ]
        for index, event in enumerate(events):
            await self.live.handle_event(
                self.sid,
                {
                    "type": "response.event",
                    "event_id": f"actual_{index}",
                    "delegation_id": "delegation_actual",
                    "event": event,
                },
                ws,
                pending,
                seen,
            )
        sent = [json.loads(call.args[0]) for call in ws.send.await_args_list]
        self.assertEqual(
            [event["type"] for event in sent],
            ["response.item.create", "response.create"],
        )
        self.assertTrue(json.loads(sent[0]["item"]["output"])["ok"])
        self.assertEqual(pending, {})

    async def test_live_tools_do_not_require_a_questionnaire(self):
        config = self.live.session_config()
        names = {tool["name"] for tool in config["delegation"]["responses"]["tools"]}
        self.assertEqual(
            names,
            {
                "search_centers",
                "get_demo_slots",
                "confirm_demo_booking",
                "request_handoff",
                "opt_out",
            },
        )
        self.assertIn("Interruption policy:", config["instructions"])

    async def test_browser_greeting_is_played_by_app_without_duplicate_voice_intro(
        self,
    ):
        config = self.live.session_config(app_greeting=True)
        self.assertEqual(config["input"][0]["role"], "assistant")
        self.assertIn("Úvod přehraje aplikace", config["instructions"])
        self.assertEqual(
            config["delegation"]["responses"]["reasoning"], {"effort": "none"}
        )

    async def test_public_greeting_is_cached_and_rendered_only_once(self):
        self.live.http.post = AsyncMock(
            return_value=httpx.Response(
                200,
                content=b"synthetic mp3",
                request=httpx.Request("POST", "https://api.openai.com/v1/audio/speech"),
            )
        )
        first = await self.live.greeting_audio()
        second = await self.live.greeting_audio()
        self.assertEqual(first, second)
        self.assertEqual(first.read_bytes(), b"synthetic mp3")
        self.assertEqual(self.live.http.post.await_count, 1)

    async def test_ambiguous_timeout_is_not_retried(self):
        self.store.claim_start(self.sid)
        self.live.http.post = AsyncMock(
            side_effect=httpx.ReadTimeout("synthetic timeout")
        )
        with self.assertRaises(ValueError):
            await self.live.create(self.sid, {"type": "sip"})
        self.assertEqual(self.live.http.post.await_count, 1)
        self.assertEqual(self.store.get(self.sid)["state"], "unknown")

    async def test_failed_hangup_is_not_marked_closed(self):
        self.store.update(self.sid, remote_id="live_synthetic", state="active")
        self.live.http.post = AsyncMock(return_value=httpx.Response(502))
        await self.live.hangup(self.sid)
        self.assertEqual(self.store.get(self.sid)["state"], "unknown")

    async def test_phone_session_start_is_not_an_answered_call(self):
        sid = self.store.new("phone", "test")["id"]
        self.store.claim_start(sid)
        ws = AsyncMock()
        await self.live.handle_event(sid, {"type": "session.started"}, ws, {}, set())
        self.assertEqual(self.store.get(sid)["state"], "connecting")
        await self.live.handle_event(sid, {"type": "transport.answered"}, ws, {}, set())
        self.assertEqual(self.store.get(sid)["state"], "active")

    async def test_voice_configuration_keeps_secrets_out_and_disables_recording(self):
        config = self.live.session_config()
        self.assertFalse(config["store"])
        self.assertIn("češtinou", config["instructions"])
        self.assertEqual(config["delegation"]["type"], "responses")
        self.assertNotIn("synthetic-key", json.dumps(config))
