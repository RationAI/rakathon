"""GPT-Live WebRTC/SIP transport and server-owned Responses tool execution."""

import asyncio
import hashlib
import json
from collections import defaultdict, deque
from contextlib import suppress
from urllib.parse import quote

import httpx
from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from calling_agent.config import Settings
from calling_agent.prompts import (
    BACKEND_PROMPT,
    OPENING,
    voice_instructions,
)
from calling_agent.speech import Tone, Voice, speech_style
from calling_agent.store import Store
from calling_agent.workflow import Workflow, tool_definitions


def voice_tool_result(name: str, result: dict) -> dict:
    """Keep model-facing tool results small without losing decision evidence."""
    if not result.get("ok"):
        return result
    if name == "search_centers":
        fields = {
            "id",
            "name",
            "city",
            "address",
            "phone",
            "email",
            "website",
            "opening_hours",
            "online_booking_url",
            "reported_wait_days",
            "wait_updated",
            "distance_km",
            "source_url",
        }
        return {
            "ok": True,
            "centers": [
                {key: value for key, value in center.items() if key in fields}
                for center in result["centers"]
            ],
            "ranking": result["ranking"],
            "needs_clarification": result.get("needs_clarification", False),
            "location_candidates": [
                {
                    key: place[key]
                    for key in ("name", "district", "region")
                    if key in place
                }
                for place in result.get("location_candidates", [])
            ],
            "availability": "unknown",
            "distance_basis": "Approximate straight-line distance from the locality center, never driving time.",
        }
    if name == "get_demo_slots":

        def slot_summary(slot):
            return {key: slot[key] for key in ("id", "starts_at", "spoken_label")}

        return {
            "ok": True,
            "center": result["center"],
            "recommended_slots": [slot_summary(s) for s in result["recommended_slots"]],
            "alternatives": [slot_summary(s) for s in result["slots"][2:]],
            "date_available": result["date_available"],
            "requested_date": result["requested_date"],
        }
    if name == "confirm_demo_booking":
        return {
            key: result[key]
            for key in ("ok", "booking_status", "center", "spoken_summary")
        }
    return result


class LiveService:
    def __init__(self, settings: Settings, store: Store, workflow: Workflow):
        self.settings = settings
        self.store = store
        self.workflow = workflow
        self.http = httpx.AsyncClient(base_url="https://api.openai.com/v1/", timeout=45)
        self.tasks: dict[str, asyncio.Task] = {}
        self.events = defaultdict(lambda: deque(maxlen=120))
        self.event_trace = defaultdict(lambda: deque(maxlen=80))
        self.ready: dict[str, asyncio.Event] = {}
        self.response_ids = defaultdict(dict)
        self.greeting_lock = asyncio.Lock()

    async def greeting_audio(
        self, *, voice: Voice | None = None, tone: Tone | None = None
    ):
        spec = {
            "model": self.settings.greeting_model,
            "voice": voice or self.settings.voice,
            "input": OPENING,
            "instructions": speech_style(tone or self.settings.tone),
            "response_format": "mp3",
            "speed": self.settings.greeting_speed,
        }
        digest = hashlib.sha256(
            json.dumps(spec, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()[:24]
        path = self.settings.data_dir / f"greeting-{digest}.mp3"
        async with self.greeting_lock:
            if not path.exists():
                response = await self.http.post(
                    "audio/speech", headers=self.headers, json=spec
                )
                response.raise_for_status()
                if not response.content:
                    raise ValueError("Hlasový úvod se nepodařilo připravit.")
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(response.content)
                temporary.replace(path)
        return path

    @property
    def headers(self) -> dict:
        key = self.settings.openai_api_key.get_secret_value()
        if not key:
            raise ValueError("Na serveru chybí OPENAI_API_KEY.")
        return {"Authorization": f"Bearer {key}"}

    def session_config(
        self,
        *,
        app_greeting: bool = False,
        voice: Voice | None = None,
        tone: Tone | None = None,
    ) -> dict:
        config = {
            "model": self.settings.voice_model,
            "store": False,
            "instructions": voice_instructions(
                self.settings.organization_name,
                app_greeting=app_greeting,
                tone=tone or self.settings.tone,
            ),
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Začni ihned česky přesně tímto krátkým úvodem, bez dalších otázek, a potom poslouchej: {OPENING}",
                        }
                    ],
                }
            ],
            "audio": {"output": {"voice": voice or self.settings.voice}},
            "delegation": {
                "type": "responses",
                "responses": {
                    "model": self.settings.backend_model,
                    "reasoning": {"effort": self.settings.backend_reasoning_effort},
                    "max_output_tokens": 700,
                    "instructions": BACKEND_PROMPT,
                    "tools": tool_definitions(for_live=True),
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                },
            },
        }
        if app_greeting:
            config["input"] = [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": OPENING}],
                }
            ]
        return config

    async def create(self, session_id: str, transport: dict) -> dict:
        # Deliberately no automatic retries: a failed response may have placed a call.
        try:
            preferences = self.store.get(session_id)
            response = await self.http.post(
                "live/sessions",
                headers=self.headers,
                json={
                    "session": self.session_config(
                        app_greeting=transport.get("type") == "webrtc",
                        voice=preferences.get("voice"),
                        tone=preferences.get("tone"),
                    ),
                    "transport": transport,
                },
            )
            if response.is_error:
                code = "provider_error"
                detail = ""
                with suppress(ValueError, AttributeError):
                    upstream = response.json().get("error", {})
                    code = upstream.get("code", code)
                    detail = str(upstream.get("message", ""))[:600]
                self.store.update(
                    session_id,
                    state="failed",
                    error=f"OpenAI HTTP {response.status_code}: {code}. {detail}",
                )
                raise ValueError(
                    f"Hlasová služba odmítla spojení (HTTP {response.status_code}, {code}). {detail}"
                )
            result = response.json()
            remote_id = result["session"]["id"]
            self.store.update(session_id, remote_id=remote_id)
            self.ready[session_id] = asyncio.Event()
            task = asyncio.create_task(self.monitor(session_id, remote_id))
            self.tasks[session_id] = task
            await asyncio.wait_for(self.ready[session_id].wait(), timeout=10)
            if task.done():
                raise ValueError("Nepodařilo se připojit serverové řízení hovoru.")
            return {"session": {"id": remote_id}, "transport": result["transport"]}
        except (httpx.TransportError, TimeoutError) as error:
            self.store.update(
                session_id,
                state="unknown",
                error="Nejasný stav spojení. Neopakovat volání automaticky; ověřte stav u poskytovatele.",
            )
            if self.store.get(session_id)["remote_id"]:
                await self.hangup(session_id)
            raise ValueError(
                "Stav spojení není známý. Před dalším pokusem ověřte stav u poskytovatele."
            ) from error

    async def dial(self, session_id: str, phone: str):
        config = self.settings
        try:
            await self.create(
                session_id,
                {
                    "type": "sip",
                    "destination": phone,
                    "trunk": {
                        "provider_url": config.sip_provider_url,
                        "auth": {
                            "type": "digest",
                            "username": config.sip_username,
                            "password": config.sip_password.get_secret_value(),
                        },
                        "caller_number": config.sip_caller_number,
                    },
                },
            )
        except ValueError:
            session = self.store.get(session_id)
            if session["remote_id"]:
                await self.hangup(session_id)
            elif session["state"] == "connecting":
                self.store.update(
                    session_id, state="failed", error="Hovor se nepodařilo zahájit."
                )

    def remember(self, session_id: str, event: dict):
        event_type = event.get("type", "")
        if "audio" not in event_type and "transcript" not in event_type:
            nested = event.get("event") or {}
            self.event_trace[session_id].append(
                {
                    "type": event_type,
                    "keys": sorted(event.keys()),
                    "nested_type": nested.get("type")
                    if isinstance(nested, dict)
                    else None,
                    "item_type": nested.get("item", {}).get("type")
                    if isinstance(nested, dict)
                    else None,
                }
            )
        if event_type in {
            "session.input_transcript.delta",
            "session.output_transcript.delta",
        }:
            # Bounded, memory-only captions. Never print or write raw events to logs.
            self.events[session_id].append(
                {
                    "type": event_type,
                    "delta": event.get("delta", ""),
                    "start_ms": event.get("start_ms"),
                }
            )
        elif event_type == "error":
            self.events[session_id].append(
                {
                    "type": "error",
                    "code": event.get("error", {}).get("code", "voice_error"),
                }
            )

    async def handle_event(
        self, session_id: str, event: dict, ws, pending: dict, seen: set
    ):
        event_id = event.get("event_id")
        if event_id:
            if event_id in seen:
                return
            seen.add(event_id)
        self.remember(session_id, event)
        kind = event.get("type")
        if kind in {"transport.ringing", "transport.answered", "session.started"}:
            if (
                kind == "session.started"
                and self.store.get(session_id)["mode"] == "phone"
            ):
                return
            state = "ringing" if kind == "transport.ringing" else "active"
            self.store.update(session_id, state=state)
        elif kind == "transport.failed":
            self.store.update(
                session_id, state="failed", error="Telefonní spojení se nezdařilo."
            )
        elif kind == "error":
            code = str(event.get("error", {}).get("code", "voice_error"))
            self.store.update(session_id, error=f"Hlasová služba: {code}")
            if code == "responses_validation_failed":
                await ws.send(json.dumps({"type": "session.close"}))
        elif kind == "session.closed":
            session = self.store.get(session_id)
            state = "failed" if session["state"] == "failed" else "closed"
            self.store.update(session_id, state=state)
        elif kind == "response.event":
            nested = event.get("event", {})
            delegation_id = event.get("delegation_id")
            lifecycle_id = nested.get("response", {}).get("id")
            if (
                nested.get("type") == "response.created"
                and delegation_id
                and lifecycle_id
            ):
                self.response_ids[session_id][delegation_id] = lifecycle_id
            # output_item.done often has no response_id. Bind it to the ID from
            # response.created; using delegation_id here and response.id at
            # completion stranded every pending tool result in real calls.
            response_id = (
                nested.get("response_id")
                or lifecycle_id
                or self.response_ids[session_id].get(delegation_id)
                or delegation_id
            )
            if nested.get("type") == "response.output_item.done":
                item = nested.get("item", {})
                if item.get("type") == "function_call":
                    pending.setdefault(response_id, {})[item["call_id"]] = item
            elif nested.get("type") == "response.completed":
                calls = pending.pop(response_id, {})
                if not calls and delegation_id:
                    calls = pending.pop(delegation_id, {})
                for item in calls.values():
                    result = await self.workflow.execute(
                        session_id, item["call_id"], item["name"], item["arguments"]
                    )
                    await ws.send(
                        json.dumps(
                            {
                                "type": "response.item.create",
                                "item": {
                                    "type": "function_call_output",
                                    "call_id": item["call_id"],
                                    "output": json.dumps(
                                        voice_tool_result(item["name"], result),
                                        ensure_ascii=False,
                                    ),
                                },
                            }
                        )
                    )
                if calls:
                    await ws.send(json.dumps({"type": "response.create"}))
            elif nested.get("type") in {
                "response.failed",
                "response.cancelled",
                "response.incomplete",
            }:
                pending.pop(response_id, None)
                if delegation_id:
                    pending.pop(delegation_id, None)

    async def monitor(self, session_id: str, remote_id: str):
        pending, seen = {}, set()
        try:
            async with asyncio.timeout(self.settings.max_call_seconds):
                async with connect(
                    f"wss://api.openai.com/v1/live/sessions/{quote(remote_id, safe='')}/attach",
                    additional_headers=self.headers,
                    max_size=2**22,
                ) as ws:
                    self.ready[session_id].set()
                    async for raw in ws:
                        event = json.loads(raw)
                        await self.handle_event(session_id, event, ws, pending, seen)
                        if event.get("type") == "session.closed":
                            return
        except TimeoutError:
            self.store.update(session_id, error="Dosažen časový limit hovoru.")
        except asyncio.CancelledError:
            raise
        except (WebSocketException, OSError, ValueError, KeyError):
            # Connection exceptions may include credentials or transcript data.
            self.store.update(
                session_id,
                error="Ztraceno serverové řízení hovoru; spojení se ukončuje.",
            )
        finally:
            self.ready[session_id].set()
            if self.store.get(session_id)["state"] not in {"closed", "failed"}:
                await self.hangup(session_id)
                self.store.update(
                    session_id,
                    state="unknown",
                    error="Chybí konečné potvrzení session.closed. Ověřte ukončení u poskytovatele.",
                )

    async def hangup(self, session_id: str):
        session = self.store.get(session_id)
        if not session["remote_id"]:
            self.store.update(session_id, state="closed")
            return
        try:
            response = await self.http.post(
                f"live/sessions/{quote(session['remote_id'], safe='')}/hangup",
                headers=self.headers,
            )
            if response.is_error:
                self.store.update(
                    session_id,
                    state="unknown",
                    error="Ukončení vzdáleného hovoru nebylo potvrzeno.",
                )
            elif session["state"] != "failed":
                # The monitor records closed once final session.closed arrives.
                self.store.update(session_id, state="closing")
        except (httpx.TransportError, ValueError):
            self.store.update(
                session_id,
                state="unknown",
                error="Ukončení vzdáleného hovoru nebylo potvrzeno.",
            )

    async def shutdown(self):
        for session_id, task in list(self.tasks.items()):
            if not task.done():
                await self.hangup(session_id)
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        await self.http.aclose()
