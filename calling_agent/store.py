"""Transactional local state. No audio or transcript is persisted."""

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


def now() -> str:
    return datetime.now(UTC).isoformat()


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        key_path = path.parent / "phone-hash.key"
        if not key_path.exists():
            try:
                descriptor = os.open(
                    key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                )
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(secrets.token_bytes(32))
            except FileExistsError:
                pass
        self.hash_key = key_path.read_bytes()
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    state TEXT NOT NULL, mode TEXT NOT NULL, contact_label TEXT NOT NULL,
                    remote_id TEXT, phone_hash TEXT, consent TEXT NOT NULL DEFAULT 'unknown',
                    opted_out INTEGER NOT NULL DEFAULT 0,
                    payload TEXT NOT NULL DEFAULT '{}', error TEXT
                );
                CREATE TABLE IF NOT EXISTS tool_results (
                    session_id TEXT NOT NULL, call_id TEXT NOT NULL, result TEXT NOT NULL,
                    PRIMARY KEY (session_id, call_id)
                );
                CREATE TABLE IF NOT EXISTS suppression (
                    phone_hash TEXT PRIMARY KEY, created_at TEXT NOT NULL
                );
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(sessions)")}
            for column in ("voice", "tone"):
                if column not in columns:
                    db.execute(f"ALTER TABLE sessions ADD COLUMN {column} TEXT")
        os.chmod(path, 0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def phone_hash(self, phone: str) -> str:
        return hmac.new(self.hash_key, phone.encode(), hashlib.sha256).hexdigest()

    def new(
        self,
        mode: str,
        label: str,
        *,
        voice: str | None = None,
        tone: str | None = None,
    ) -> dict:
        session_id = uuid.uuid4().hex
        with self.connect() as db:
            db.execute(
                "INSERT INTO sessions (id,created_at,updated_at,state,mode,contact_label,voice,tone) VALUES (?,?,?,?,?,?,?,?)",
                (session_id, now(), now(), "created", mode, label, voice, tone),
            )
        return self.get(session_id)

    def get(self, session_id: str) -> dict:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
        if row is None:
            raise KeyError(session_id)
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        result["opted_out"] = bool(result["opted_out"])
        return result

    def list(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT id FROM sessions ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
        return [self.get(row["id"]) for row in rows]

    def update(self, session_id: str, **fields):
        permitted = {
            "state",
            "remote_id",
            "phone_hash",
            "consent",
            "opted_out",
            "payload",
            "error",
        }
        if not fields.keys() <= permitted:
            raise ValueError("Unknown state field")
        if "payload" in fields:
            fields["payload"] = json.dumps(fields["payload"], ensure_ascii=False)
        fields["updated_at"] = now()
        with self.connect() as db:
            db.execute(
                f"UPDATE sessions SET {','.join(f'{key}=?' for key in fields)} WHERE id=?",
                (*fields.values(), session_id),
            )

    def claim_start(self, session_id: str) -> bool:
        with self.connect() as db:
            return (
                db.execute(
                    "UPDATE sessions SET state='connecting',updated_at=? WHERE id=? AND state='created'",
                    (now(), session_id),
                ).rowcount
                == 1
            )

    def suppress(self, session_id: str):
        session = self.get(session_id)
        if session["phone_hash"]:
            with self.connect() as db:
                db.execute(
                    "INSERT OR IGNORE INTO suppression VALUES (?,?)",
                    (session["phone_hash"], now()),
                )

    def is_suppressed(self, phone: str) -> bool:
        with self.connect() as db:
            return (
                db.execute(
                    "SELECT 1 FROM suppression WHERE phone_hash=?",
                    (self.phone_hash(phone),),
                ).fetchone()
                is not None
            )

    def recover(self):
        # A process restart must never dial again automatically.
        with self.connect() as db:
            db.execute(
                "UPDATE sessions SET state='interrupted',error='Server restarted; remote call state requires verification' "
                "WHERE state IN ('connecting','ringing','active')"
            )

    def cached_tool(self, session_id: str, call_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT result FROM tool_results WHERE session_id=? AND call_id=?",
                (session_id, call_id),
            ).fetchone()
        return json.loads(row["result"]) if row else None

    def cache_tool(self, session_id: str, call_id: str, result: dict):
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO tool_results VALUES (?,?,?)",
                (session_id, call_id, json.dumps(result, ensure_ascii=False)),
            )

    def clear_tool_cache(self, session_id: str):
        # Keep call IDs as tombstones so replay cannot restore withdrawn consent.
        with self.connect() as db:
            db.execute(
                "UPDATE tool_results SET result=? WHERE session_id=?",
                (
                    json.dumps(
                        {
                            "ok": False,
                            "error": "Předchozí výsledek byl odstraněn po odvolání souhlasu.",
                            "booking_status": "not_booked",
                        }
                    ),
                    session_id,
                ),
            )

    def delete(self, session_id: str):
        with self.connect() as db:
            db.execute("DELETE FROM tool_results WHERE session_id=?", (session_id,))
            db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
