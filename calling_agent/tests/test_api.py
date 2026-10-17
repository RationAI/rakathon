import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from calling_agent.config import Settings
from calling_agent.main import create_app
from calling_agent.schemas import Offer
from calling_agent.tests.helpers import directory_fixture


class ApiTests(unittest.TestCase):
    def test_selected_voice_and_tone_are_saved_and_invalid_choices_rejected(self):
        response = self.client.post(
            "/api/sessions", json={"voice": "shimmer", "tone": "gentle"}
        )
        self.assertEqual(response.status_code, 201)
        row = self.app.state.store.get(response.json()["id"])
        self.assertEqual((row["voice"], row["tone"]), ("shimmer", "gentle"))
        for data in ({"voice": "unknown"}, {"tone": "unknown"}):
            self.assertEqual(
                self.client.post("/api/sessions", json=data).status_code, 422
            )
        self.assertEqual(
            self.client.get("/api/greeting?voice=unknown").status_code, 422
        )

    def test_sdp_offer_preserves_final_crlf(self):
        sdp = "v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n"
        self.assertEqual(Offer(sdp=sdp).sdp, sdp)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        directory_fixture(self.folder)
        self.config = Settings(_env_file=None, data_dir=self.folder)
        self.app = create_app(self.config)
        self.client = self.enterContext(
            TestClient(self.app, base_url="http://localhost:8080")
        )
        self.headers = {}

    def test_local_console_works_without_app_token(self):
        self.assertEqual(self.client.get("/healthz").status_code, 200)
        self.assertEqual(self.client.get("/api/sessions").status_code, 200)
        self.assertEqual(self.client.get("/api/config").status_code, 200)
        self.assertEqual(self.client.post("/api/sessions", json={}).status_code, 201)
        page = self.client.get("/").text
        self.assertNotIn('id="login"', page)
        self.assertNotIn("APP_TOKEN", page)

    def test_cross_site_reads_and_rebound_hosts_are_rejected(self):
        for endpoint in ("/api/config", "/api/sessions", "/api/greeting"):
            for headers in (
                {"Origin": "https://evil.example"},
                {"Sec-Fetch-Site": "cross-site"},
            ):
                self.assertEqual(
                    self.client.get(endpoint, headers=headers).status_code, 403
                )
        self.assertEqual(
            self.client.get(
                "/api/config", headers={"Host": "evil.example"}
            ).status_code,
            400,
        )

    def test_same_origin_writes_are_allowed(self):
        self.assertEqual(
            self.client.post(
                "/api/sessions", json={}, headers={"Origin": "http://localhost:8080"}
            ).status_code,
            201,
        )

    def test_cross_origin_mutation_is_rejected(self):
        response = self.client.post(
            "/api/sessions",
            headers={**self.headers, "Origin": "https://evil.example"},
            json={},
        )
        self.assertEqual(response.status_code, 403)

    def test_demo_has_no_provider_calls_and_never_books(self):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as provider:
            response = self.client.post("/api/demo", headers=self.headers)
        self.assertEqual(response.status_code, 201)
        provider.assert_not_called()
        row = response.json()
        self.assertEqual(row["mode"], "demo")
        self.assertEqual(row["payload"]["booking"]["booking_status"], "simulated")
        self.assertIsNone(row["remote_id"])

    def test_outbound_is_disabled_by_default(self):
        row = self.client.post(
            "/api/sessions", json={"mode": "phone"}, headers=self.headers
        ).json()
        response = self.client.post(
            f"/api/sessions/{row['id']}/dial",
            headers=self.headers,
            json={"phone": "+420555000111", "contact_permission_confirmed": True},
        )
        self.assertEqual(response.status_code, 403)

    def test_missing_openai_key_does_not_claim_browser_session(self):
        row = self.client.post(
            "/api/sessions", json={"mode": "browser"}, headers=self.headers
        ).json()
        response = self.client.post(
            f"/api/sessions/{row['id']}/offer",
            headers=self.headers,
            json={"sdp": "v=0\nsynthetic-offer"},
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.app.state.store.get(row["id"])["state"], "created")

    def test_deletion_is_blocked_for_active_calls(self):
        row = self.client.post("/api/sessions", json={}, headers=self.headers).json()
        self.app.state.store.claim_start(row["id"])
        self.assertEqual(
            self.client.delete(
                f"/api/sessions/{row['id']}", headers=self.headers
            ).status_code,
            409,
        )
        self.app.state.store.update(row["id"], state="closed")
        self.assertEqual(
            self.client.delete(
                f"/api/sessions/{row['id']}", headers=self.headers
            ).status_code,
            200,
        )
        self.assertEqual(
            self.client.get(
                f"/api/sessions/{row['id']}", headers=self.headers
            ).status_code,
            404,
        )

    def test_request_size_limit(self):
        response = self.client.post(
            "/api/sessions", headers=self.headers, content=" " * 71000
        )
        self.assertEqual(response.status_code, 413)

    def test_operator_can_resolve_only_uncertain_calls(self):
        row = self.client.post("/api/sessions", json={}, headers=self.headers).json()
        endpoint = f"/api/sessions/{row['id']}/resolve"
        body = {"remote_ended_confirmed": True}
        self.assertEqual(
            self.client.post(endpoint, json=body, headers=self.headers).status_code, 409
        )
        self.app.state.store.update(row["id"], state="interrupted")
        self.assertEqual(
            self.client.post(
                endpoint, json={"remote_ended_confirmed": False}, headers=self.headers
            ).status_code,
            422,
        )
        result = self.client.post(endpoint, json=body, headers=self.headers)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["state"], "closed")


class DialGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        folder = Path(self.temp.name)
        directory_fixture(folder)
        config = Settings(
            _env_file=None,
            data_dir=folder,
            openai_api_key="synthetic-key",
            enable_outbound_calls=True,
            outbound_allowlist="+420555000111",
            sip_provider_url="sips:example.com:5061",
            sip_username="test",
            sip_password="synthetic-password",
            sip_caller_number="+420555000222",
        )
        self.app = create_app(config)
        self.client = self.enterContext(
            TestClient(self.app, base_url="http://localhost:8080")
        )
        self.headers = {}

    def new_phone(self):
        return self.client.post(
            "/api/sessions", json={"mode": "phone"}, headers=self.headers
        ).json()["id"]

    def test_allowlist_and_permission(self):
        sid = self.new_phone()
        endpoint = f"/api/sessions/{sid}/dial"
        self.assertEqual(
            self.client.post(
                endpoint,
                headers=self.headers,
                json={"phone": "+420555000333", "contact_permission_confirmed": True},
            ).status_code,
            403,
        )
        self.assertEqual(
            self.client.post(
                endpoint,
                headers=self.headers,
                json={"phone": "+420555000111", "contact_permission_confirmed": False},
            ).status_code,
            422,
        )

    def test_duplicate_dial_is_rejected(self):
        sid = self.new_phone()
        with patch.object(self.app.state.live, "dial", new_callable=AsyncMock) as dial:
            args = {"phone": "+420555000111", "contact_permission_confirmed": True}
            first = self.client.post(
                f"/api/sessions/{sid}/dial", headers=self.headers, json=args
            )
            second = self.client.post(
                f"/api/sessions/{sid}/dial", headers=self.headers, json=args
            )
            self.assertEqual(first.status_code, 202)
            self.assertEqual(second.status_code, 409)
            self.assertEqual(dial.call_count, 1)

    def test_suppression_blocks_new_session(self):
        sid = self.new_phone()
        self.app.state.store.update(
            sid, phone_hash=self.app.state.store.phone_hash("+420555000111")
        )
        self.app.state.store.suppress(sid)
        other = self.new_phone()
        response = self.client.post(
            f"/api/sessions/{other}/dial",
            headers=self.headers,
            json={"phone": "+420555000111", "contact_permission_confirmed": True},
        )
        self.assertEqual(response.status_code, 409)
