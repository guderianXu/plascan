import http.client
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEV_DIRECTORY = ROOT / "scripts" / "dev"
sys.path.insert(0, str(DEV_DIRECTORY))
SCRIPT_PATH = DEV_DIRECTORY / "browser_debug_server.py"
SPEC = importlib.util.spec_from_file_location("plascan_browser_debug_server", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SERVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVER)


class FakeContext:
    token = "test-debug-hub-token-123456789"

    def __init__(self, web_root: Path):
        self.web_root = web_root
        self.calls = []

    def bridge_call(self, method, parameters=None):
        self.calls.append((method, parameters))
        if method == "ping":
            return {"schema_version": 1, "pid": 42}
        if method == "snapshot":
            return {"project": {"open": False}, "windows": [], "logs": []}
        return {"method": method, "parameters": parameters or {}}

    def session(self):
        return {"run_id": "run-1", "novnc_url": "http://127.0.0.1:6081/vnc.html"}

    def agent_view(self, view, query="", offset=0, limit=25):
        return {
            "schema_version": 1,
            "revision": "abc123",
            "view": view,
            "data": {"query": query, "offset": offset, "limit": limit},
        }


class BrowserDebugServerTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.web_root = Path(self.temporary_directory.name)
        (self.web_root / "index.html").write_text("<h1>Debug Hub</h1>", encoding="utf-8")
        self.context = FakeContext(self.web_root)
        self.server = SERVER.DebugHubServer(("127.0.0.1", 0), self.context)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.server_address

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temporary_directory.cleanup()

    def request(self, method, path, body=None, authenticated=False):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=3)
        headers = {}
        payload = None
        if authenticated:
            headers["X-PlaScan-Debug-Token"] = self.context.token
        if body is not None:
            headers["Content-Type"] = "application/json"
            payload = json.dumps(body).encode("utf-8")
        connection.request(method, path, payload, headers)
        response = connection.getresponse()
        content = response.read()
        connection.close()
        return response.status, response.headers, content

    def test_static_page_and_health_are_loopback_bootstrap_surfaces(self):
        status, headers, content = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Debug Hub", content)
        self.assertEqual(headers["Cache-Control"], "no-store")

        status, _, content = self.request("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(content)["bridge"]["schema_version"], 1)

    def test_structured_endpoints_require_session_token(self):
        status, _, _ = self.request("GET", "/api/session")
        self.assertEqual(status, 401)
        status, _, content = self.request("GET", "/api/session", authenticated=True)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(content)["result"]["run_id"], "run-1")

    def test_interact_forwards_only_allowlisted_fields(self):
        status, _, content = self.request(
            "POST",
            "/api/interact",
            {
                "object_name": "actionAbout",
                "operation": "activate",
                "value": True,
                "arbitrary_method": "deleteEverything",
            },
            authenticated=True,
        )
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(content)["ok"])
        method, parameters = self.context.calls[-1]
        self.assertEqual(method, "interact")
        self.assertEqual(set(parameters), {"object_name", "operation", "value"})

    def test_agent_views_are_tokenized_compact_and_paginated(self):
        status, _, content = self.request("GET", "/api/agent/summary")
        self.assertEqual(401, status)
        status, _, content = self.request(
            "GET", "/api/agent/controls?query=about&offset=5&limit=10", authenticated=True
        )
        self.assertEqual(200, status)
        result = json.loads(content)["result"]
        self.assertEqual("controls", result["view"])
        self.assertEqual({"query": "about", "offset": 5, "limit": 10}, result["data"])

    def test_production_hub_keeps_stable_accessible_debug_controls(self):
        html = (DEV_DIRECTORY / "browser_gui_web/index.html").read_text(encoding="utf-8")
        script = (DEV_DIRECTORY / "browser_gui_web/app.js").read_text(encoding="utf-8")
        for element_id in (
            "bridgeHealth",
            "globalRefresh",
            "vncFrame",
            "projectState",
            "refreshSnapshot",
            "refreshUi",
            "closeDialog",
            "captureScreenshot",
        ):
            with self.subTest(element_id=element_id):
                self.assertIn(f'id="{element_id}"', html)
                self.assertIn(f'byId("{element_id}")', script)
        self.assertIn('aria-label="调试视图"', html)
        self.assertIn('href="/api/agent/capabilities"', html)
        self.assertIn('data-agent-api="/api/agent/capabilities"', html)
        self.assertIn("renderSummary(message.data || message)", script)
        self.assertNotIn("<style", html)
        self.assertNotIn("<script>", html)


if __name__ == "__main__":
    unittest.main()
