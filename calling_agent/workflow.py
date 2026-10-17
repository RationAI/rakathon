import asyncio
from collections import defaultdict
from datetime import datetime

from pydantic import ValidationError

from calling_agent.centers import Directory
from calling_agent.schemas import (
    ConfirmRequest,
    Consent,
    DemoBooking,
    DemoSlots,
    Handoff,
    Interview,
    OptOut,
    Preferences,
    SchedulingRequest,
    SearchCenters,
    SelectCenter,
)
from calling_agent.simulation import PRAGUE, demo_slots
from calling_agent.store import Store, now


TOOL_MODELS = {
    "get_demo_slots": (
        DemoSlots,
        "Return demo appointments for ONE chosen center. on_date is an exact requested YYYY-MM-DD or empty; not_before is the earliest date for flexible requests or empty. Use recommended_slots, never mix centers or invent a date. No real availability is checked.",
    ),
    "confirm_demo_booking": (
        DemoBooking,
        "Save a simulated appointment when the caller explicitly chooses a previously offered slot in a known center. That choice counts as confirmation; do not demand a second yes or permission-to-note stage. Quote the choice. Never books with the real center.",
    ),
    "save_scheduling_request": (
        SchedulingRequest,
        "Save one scheduling request after the caller explicitly permits saving the read-back center and preferences. No questionnaire or prior consent tool is needed. Never reserves a slot.",
    ),
    "record_consent": (
        Consent,
        "Record the caller's explicit voluntary consent, refusal or withdrawal. Quote her words.",
    ),
    "save_interview": (
        Interview,
        "Save only caller-reported attendance, optional reason, municipality and travel preferences; use empty strings for unanswered optional fields.",
    ),
    "search_centers": (
        SearchCenters,
        "Search mamo.cz by a mentioned locality or center name. Known Czech places return nearest centers by approximate straight-line distance from the place center. Ambiguous names require district clarification. Never returns appointment slots or driving times.",
    ),
    "select_center": (
        SelectCenter,
        "Record a center explicitly chosen by the caller from previously retrieved results.",
    ),
    "save_preferences": (
        Preferences,
        "Record requested dates/period and time of day, not confirmed availability.",
    ),
    "confirm_request": (
        ConfirmRequest,
        "After reading back center and preferences, save an explicitly confirmed request for manual follow-up. This NEVER books an appointment.",
    ),
    "request_handoff": (
        Handoff,
        "Flag a need for human follow-up, particularly new symptoms; no live transfer is available.",
    ),
    "opt_out": (
        OptOut,
        "Record do-not-contact immediately and stop further data collection.",
    ),
}


def tool_definitions(*, for_live: bool = False) -> list[dict]:
    definitions = []
    for name, (model, description) in TOOL_MODELS.items():
        if for_live and name not in {
            "search_centers",
            "get_demo_slots",
            "confirm_demo_booking",
            "request_handoff",
            "opt_out",
        }:
            continue
        schema = model.model_json_schema()
        definitions.append(
            {
                "type": "function",
                "name": name,
                "description": description,
                "parameters": schema,
                "strict": True,
            }
        )
    return definitions


class Workflow:
    def __init__(self, store: Store, directory: Directory):
        self.store = store
        self.directory = directory
        self.locks = defaultdict(asyncio.Lock)
        self.offered_centers = defaultdict(set)
        self.offered_slots = defaultdict(dict)

    async def execute(
        self, session_id: str, call_id: str, name: str, arguments: str
    ) -> dict:
        async with self.locks[session_id]:
            cached = self.store.cached_tool(session_id, call_id)
            if cached is not None:
                return cached
            try:
                if name not in TOOL_MODELS:
                    raise ValueError("Unknown tool")
                args = TOOL_MODELS[name][0].model_validate_json(arguments)
                result = self.apply(session_id, name, args)
            except (ValueError, ValidationError) as error:
                # Do not echo input values, which can contain personal information.
                result = {
                    "ok": False,
                    "error": str(error)
                    if not isinstance(error, ValidationError)
                    else "Invalid tool arguments",
                    "booking_status": "not_booked",
                }
            self.store.cache_tool(session_id, call_id, result)
            return result

    def apply(self, session_id: str, name: str, args) -> dict:
        session = self.store.get(session_id)
        payload = session["payload"]
        if session["opted_out"]:
            raise ValueError("Volající odmítla další kontakt. Ukončete rozhovor.")
        if name == "opt_out":
            self.store.clear_tool_cache(session_id)
            self.store.update(
                session_id,
                opted_out=1,
                consent="withdrawn",
                payload={"outcome": "do_not_contact"},
            )
            self.store.suppress(session_id)
            return {
                "ok": True,
                "outcome": "do_not_contact",
                "booking_status": "not_booked",
            }
        if name == "record_consent":
            if args.consent != "granted":
                self.store.clear_tool_cache(session_id)
                payload = {
                    "outcome": "consent_declined"
                    if args.consent == "declined"
                    else "consent_withdrawn"
                }
            else:
                payload["consent_quote"] = args.user_quote
            self.store.update(session_id, consent=args.consent, payload=payload)
            return {"ok": True, "consent": args.consent, "booking_status": "not_booked"}
        if name == "request_handoff":
            # No clinical details retained before consent.
            if session["consent"] == "granted":
                payload["handoff"] = args.model_dump()
            else:
                payload["handoff"] = {"reason": "human_requested", "note": ""}
            payload.pop("request", None)
            payload.pop("booking", None)
            payload["outcome"] = "human_followup_required"
            self.store.update(session_id, payload=payload)
            return {
                "ok": True,
                "outcome": "human_followup_required",
                "live_transfer_available": False,
                "booking_status": "not_booked",
            }
        if name in {"get_demo_slots", "confirm_demo_booking"}:
            if session["consent"] in {"declined", "withdrawn"} or payload.get(
                "handoff"
            ):
                raise ValueError(
                    "Rozhovor nebo ukládání bylo odmítnuto, nebo je potřeba člověk."
                )
            if name == "get_demo_slots":
                if args.center_id not in self.offered_centers[session_id] | set(
                    payload.get("offered_center_ids", [])
                ):
                    raise ValueError("Nejdříve vyhledejte vybrané centrum.")
                center = self.directory.get(args.center_id)
                slots = demo_slots(
                    args.center_id, args.day_period, args.not_before, args.on_date
                )
                self.offered_slots[session_id].update((s["id"], s) for s in slots)
                return {
                    "ok": True,
                    "simulated": True,
                    "center": {
                        "id": center["id"],
                        "name": center["name"],
                        "spoken_name": center["name"].split("\N{EN DASH}")[0].strip(),
                    },
                    "slots": slots,
                    "recommended_slots": slots[:2],
                    "requested_date": args.on_date or None,
                    "date_available": bool(slots),
                    "notice": "Operator-only simulation metadata; never read it aloud in the presented conversation. Offer recommended_slots without repeating a demo disclaimer. If empty, ask for another date; never silently move an exact requested day.",
                }
            slot = self.offered_slots[session_id].get(args.slot_id)
            if not slot or datetime.fromisoformat(slot["starts_at"]) <= datetime.now(
                PRAGUE
            ):
                raise ValueError("Vyberte budoucí ukázkový termín z nabídky.")
            center = self.directory.get(slot["center_id"])
            booking = {
                "id": f"demo_booking_{session_id}",
                "status": "demo_confirmed",
                "booking_status": "simulated",
                "simulated": True,
                "slot": slot,
                "confirmed_at": now(),
                "confirmation_quote": args.confirmation_quote,
            }
            payload.pop("request", None)
            payload.update(
                selected_center=center,
                booking=booking,
                consent_quote=args.confirmation_quote,
                outcome="demo_confirmed",
            )
            self.store.update(session_id, consent="granted", payload=payload)
            return {
                "ok": True,
                "simulated": True,
                "booking_status": "simulated",
                "center": center["name"],
                "slot": slot,
                "spoken_summary": f"Máme to domluvené: {center['name'].split('\N{EN DASH}')[0].strip()}, {slot['spoken_label']}.",
                "notice": "Operator-only simulation metadata. Do not speak a disclaimer or ask for another confirmation. State the center, date AND time once.",
            }
        if name == "search_centers":
            if session["consent"] in {"declined", "withdrawn"}:
                raise ValueError(
                    "Volající rozhovor odmítla nebo odvolala souhlas. Ukončete rozhovor."
                )
            if payload.get("handoff"):
                raise ValueError("Je potřeba lidské dořešení.")
            result = self.directory.search(args.query)
            self.offered_centers[session_id].update(c["id"] for c in result["centers"])
            if session["consent"] == "granted":
                payload["offered_center_ids"] = sorted(
                    self.offered_centers[session_id]
                    | set(payload.get("offered_center_ids", []))
                )
                self.store.update(session_id, payload=payload)
            return {"ok": True, **result}
        if name == "save_scheduling_request":
            if session["consent"] in {"declined", "withdrawn"}:
                raise ValueError("Volající odmítla ukládání nebo odvolala souhlas.")
            if payload.get("handoff"):
                raise ValueError(
                    "Je potřeba lidské dořešení. Rutinní požadavek nelze uložit."
                )
            offered = self.offered_centers[session_id] | set(
                payload.get("offered_center_ids", [])
            )
            if args.center_id not in offered:
                raise ValueError("Nejdříve vyhledejte a ověřte vybrané centrum.")
            center = self.directory.get(args.center_id)
            preferences = {
                key: getattr(args, key)
                for key in ("preferred_dates", "preferred_times", "accessibility_needs")
            }
            payload.update(
                {
                    "consent_quote": args.confirmation_quote,
                    "selected_center": center,
                    "preferences": preferences,
                    "outcome": "pending_manual_booking",
                    "request": {
                        "id": f"request_{session_id}",
                        "status": "pending_manual_booking",
                        "booking_status": "not_booked",
                        "confirmed_at": now(),
                        "confirmation_quote": args.confirmation_quote,
                        "center_id": center["id"],
                        "preferences": preferences,
                    },
                }
            )
            self.store.update(session_id, consent="granted", payload=payload)
            return {"ok": True, "booking_status": "not_booked", "payload": payload}
        if session["consent"] != "granted":
            raise ValueError(
                "Nejdříve je nutný výslovný souhlas s rozhovorem a uložením odpovědí."
            )
        if payload.get("handoff"):
            raise ValueError(
                "Je potřeba lidské dořešení. Rutinní požadavek nelze uložit."
            )
        if name == "save_interview":
            payload["interview"] = args.model_dump()
            payload.pop("request", None)
        elif name == "select_center":
            if args.center_id not in payload.get("offered_center_ids", []):
                raise ValueError("Nejdříve vyhledejte centrum a ověřte volbu ženy.")
            payload["selected_center"] = self.directory.get(args.center_id)
            payload["center_choice_quote"] = args.user_quote
            payload.pop("request", None)
        elif name == "save_preferences":
            payload["preferences"] = args.model_dump()
            payload.pop("request", None)
        elif name == "confirm_request":
            if not payload.get("selected_center") or not payload.get("preferences"):
                raise ValueError("Chybí zvolené centrum nebo časové preference.")
            if not payload.get("interview", {}).get("municipality"):
                raise ValueError("Chybí potvrzená obec, kde si žena přeje vyšetření.")
            if "request" not in payload:
                payload["request"] = {
                    "id": f"request_{session_id}",
                    "status": "pending_manual_booking",
                    "booking_status": "not_booked",
                    "confirmed_at": now(),
                    "confirmation_quote": args.confirmation_quote,
                    "center_id": payload["selected_center"]["id"],
                    "preferences": payload["preferences"],
                }
        else:
            raise ValueError("Unknown tool")
        payload["outcome"] = (
            "pending_manual_booking"
            if payload.get("request")
            else "interview_in_progress"
        )
        self.store.update(session_id, payload=payload)
        return {"ok": True, "booking_status": "not_booked", "payload": payload}
