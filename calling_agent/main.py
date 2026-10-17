import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from calling_agent.centers import Directory
from calling_agent.config import Settings
from calling_agent.live import LiveService
from calling_agent.prompts import OPENING
from calling_agent.schemas import Dial, NewSession, Offer, ResolveCall
from calling_agent.speech import TONES, VOICES, Tone, Voice
from calling_agent.store import Store
from calling_agent.workflow import Workflow


STATIC = Path(__file__).parent / "static"
ACTIVE = {"connecting", "ringing", "active", "closing", "unknown", "interrupted"}


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = Store(config.data_dir / "calls.sqlite3")
        store.recover()
        directory = Directory(config.data_dir / "centers.json")
        workflow = Workflow(store, directory)
        app.state.store = store
        app.state.directory = directory
        app.state.workflow = workflow
        app.state.live = LiveService(config, store, workflow)
        app.state.dial_tasks = set()
        try:
            yield
        finally:
            if app.state.dial_tasks:
                await asyncio.gather(*app.state.dial_tasks, return_exceptions=True)
            await app.state.live.shutdown()

    app = FastAPI(
        title="Mamo · Česká hlasová asistentka",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(
            {
                "localhost",
                "127.0.0.1",
                "[::1]",
                urlsplit(config.public_origin).hostname or "localhost",
            }
        ),
    )
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def headers(request: Request, call_next):
        if (
            request.headers.get("content-length", "").isdigit()
            and int(request.headers["content-length"]) > 70000
        ):
            return JSONResponse(
                status_code=413, content={"detail": "Požadavek je příliš velký."}
            )
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; media-src 'self' blob:; object-src 'none'; frame-ancestors 'none'"
        )
        response.headers["Permissions-Policy"] = (
            "microphone=(self), camera=(), geolocation=()"
        )
        return response

    async def local_access(request: Request):
        # Local console: no operator token. Reject cross-site browser access,
        # including GETs that can render paid speech or return private records.
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != config.public_origin.rstrip("/"):
            raise HTTPException(403, "Nepovolený původ požadavku.")
        if request.headers.get("sec-fetch-site") == "cross-site":
            raise HTTPException(403, "Nepovolený původ požadavku.")

    local_api = [Depends(local_access)]

    def get_session(session_id: str):
        try:
            return app.state.store.get(session_id)
        except KeyError as error:
            raise HTTPException(404, "Hovor nebyl nalezen.") from error

    def ensure_capacity():
        running = sum(
            row["state"] in {"connecting", "ringing", "active", "closing"}
            for row in app.state.store.list()
        )
        if running >= 3:
            raise HTTPException(429, "Nejvýše tři současné hovory.")

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/api/config", dependencies=local_api)
    async def configuration():
        snapshot = app.state.directory.snapshot()
        return {
            "voice_model": config.voice_model,
            "voice": config.voice,
            "tone": config.tone,
            "voices": VOICES,
            "tones": TONES,
            "backend_model": config.backend_model,
            "voice_ready": bool(config.openai_api_key.get_secret_value()),
            "outbound_enabled": config.enable_outbound_calls,
            "organization": config.organization_name,
            "directory_count": len(snapshot["centers"]),
            "directory_fetched_at": snapshot["fetched_at"],
            "booking_status": "simulated",
            "booking_mode": "demo",
            "opening": OPENING,
        }

    @app.get("/api/centers", dependencies=local_api)
    async def centers(q: str = ""):
        if len(q.strip()) < 2:
            return {"centers": [], "matches": 0}
        return app.state.directory.search(q[:150])

    @app.get("/api/greeting", dependencies=local_api)
    async def greeting(voice: Voice | None = None, tone: Tone | None = None):
        try:
            path = await app.state.live.greeting_audio(voice=voice, tone=tone)
            return FileResponse(path, media_type="audio/mpeg")
        except (ValueError, httpx.HTTPError):
            raise HTTPException(
                503,
                "Nepodařilo se připravit hlasový úvod. Ověřte API klíč a přístup k modelu pro řeč.",
            ) from None

    @app.get("/api/sessions", dependencies=local_api)
    async def sessions():
        return app.state.store.list()

    @app.post("/api/sessions", dependencies=local_api, status_code=201)
    async def new_session(body: NewSession):
        return app.state.store.new(
            body.mode,
            body.contact_label,
            voice=body.voice or config.voice,
            tone=body.tone or config.tone,
        )

    @app.get("/api/sessions/{session_id}", dependencies=local_api)
    async def session(session_id: str):
        return {
            **get_session(session_id),
            "captions": list(app.state.live.events[session_id]),
            "event_trace": list(app.state.live.event_trace[session_id]),
        }

    @app.post(
        "/api/sessions/{session_id}/offer", dependencies=local_api, status_code=201
    )
    async def offer(session_id: str, body: Offer):
        row = get_session(session_id)
        if row["mode"] != "browser":
            raise HTTPException(409, "Tento hovor není pro prohlížeč.")
        if not config.openai_api_key.get_secret_value():
            raise HTTPException(503, "Na serveru chybí OPENAI_API_KEY.")
        ensure_capacity()
        if not app.state.store.claim_start(session_id):
            raise HTTPException(
                409, "Tento hovor už byl zahájen; neopakujte požadavek."
            )
        try:
            return await app.state.live.create(
                session_id, {"type": "webrtc", "sdp": body.sdp}
            )
        except ValueError as error:
            if app.state.store.get(session_id)["remote_id"]:
                await app.state.live.hangup(session_id)
            raise HTTPException(502, str(error)) from error

    @app.post(
        "/api/sessions/{session_id}/dial", dependencies=local_api, status_code=202
    )
    async def dial(session_id: str, body: Dial):
        row = get_session(session_id)
        if row["mode"] != "phone":
            raise HTTPException(409, "Tento hovor není telefonní.")
        if not config.enable_outbound_calls:
            raise HTTPException(403, "Odchozí volání není zapnuto.")
        if body.phone not in config.allowed_numbers:
            raise HTTPException(403, "Číslo není v OUTBOUND_ALLOWLIST.")
        if app.state.store.is_suppressed(body.phone):
            raise HTTPException(409, "Pro toto číslo je uložen zákaz dalšího kontaktu.")
        if not all(
            (
                config.sip_provider_url.startswith("sips:"),
                config.sip_username,
                config.sip_password.get_secret_value(),
                config.sip_caller_number,
                config.openai_api_key.get_secret_value(),
            )
        ):
            raise HTTPException(503, "Doplňte OPENAI_API_KEY a konfiguraci SIP.")
        ensure_capacity()
        if not app.state.store.claim_start(session_id):
            raise HTTPException(
                409, "Hovor již byl zahájen. Stav ověřte před dalším pokusem."
            )
        app.state.store.update(
            session_id, phone_hash=app.state.store.phone_hash(body.phone)
        )
        task = asyncio.create_task(app.state.live.dial(session_id, body.phone))
        app.state.dial_tasks.add(task)
        task.add_done_callback(app.state.dial_tasks.discard)
        return {"status": "connecting", "session_id": session_id}

    @app.post("/api/sessions/{session_id}/stop", dependencies=local_api)
    async def stop(session_id: str):
        get_session(session_id)
        await app.state.live.hangup(session_id)
        return {"status": "stop_requested"}

    @app.delete("/api/sessions/{session_id}", dependencies=local_api)
    async def delete(session_id: str):
        if get_session(session_id)["state"] in ACTIVE:
            raise HTTPException(409, "Nejdříve ukončete hovor a ověřte vzdálený stav.")
        app.state.store.delete(session_id)
        app.state.live.events.pop(session_id, None)
        return {
            "deleted": True,
            "note": "Případný zákaz kontaktování zůstává zachován.",
        }

    @app.post("/api/sessions/{session_id}/resolve", dependencies=local_api)
    async def resolve(session_id: str, body: ResolveCall):
        row = get_session(session_id)
        if row["state"] not in {"unknown", "interrupted"}:
            raise HTTPException(
                409, "Ruční ověření je určeno pro nejasný nebo přerušený hovor."
            )
        app.state.store.update(
            session_id,
            state="closed",
            error="Operátor potvrdil ukončení hovoru u poskytovatele; úplná finalizace API nebyla ověřena.",
        )
        return app.state.store.get(session_id)

    @app.post("/api/demo", dependencies=local_api, status_code=201)
    async def demo():
        # A deterministic synthetic rehearsal; never invokes any paid API or phone provider.
        directory_result = app.state.directory.search("Brno")
        if not directory_result["centers"]:
            raise HTTPException(
                503, "Nejprve načtěte adresář: python -m calling_agent.centers"
            )
        row = app.state.store.new("demo", "Syntetická ukázka")
        sid = row["id"]
        search_result = await app.state.workflow.execute(
            sid, "demo_search", "search_centers", '{"query":"Brno"}'
        )
        if not search_result.get("ok"):
            raise HTTPException(500, "Ukázkový průchod nebyl dokončen.")
        slots = await app.state.workflow.execute(
            sid,
            "demo_slots",
            "get_demo_slots",
            json.dumps(
                {
                    "center_id": directory_result["centers"][0]["id"],
                    "day_period": "afternoon",
                    "not_before": "",
                    "on_date": "",
                }
            ),
        )
        if not slots.get("ok"):
            raise HTTPException(500, "Ukázkové termíny nejsou připravené.")
        result = await app.state.workflow.execute(
            sid,
            "demo_confirm",
            "confirm_demo_booking",
            json.dumps(
                {
                    "slot_id": slots["slots"][0]["id"],
                    "user_confirmed": True,
                    "confirmation_quote": "Ano, potvrďte a poznamenejte si tento ukázkový termín.",
                },
                ensure_ascii=False,
            ),
        )
        if not result.get("ok"):
            raise HTTPException(500, "Ukázkové potvrzení nebylo dokončeno.")
        app.state.store.update(sid, state="closed")
        return app.state.store.get(sid)

    return app


app = create_app()
