import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
    import server
except ImportError:  # pragma: no cover - permits stdlib checks before dependencies are installed.
    TestClient = None
    server = None


@unittest.skipIf(TestClient is None, "FastAPI server dependencies are not installed")
class ServerTests(unittest.TestCase):
    def setUp(self):
        def fake_load_model():
            server.GENERATOR = object()
            server.TOKENIZER = object()
            server.ACTIVE_MODEL = "test-model"

        self.loader = patch("server.load_model", side_effect=fake_load_model)
        self.generator = patch("server.generate_response", return_value="test response")
        self.loader.start()
        self.generator.start()
        self.client_context = TestClient(server.app)
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.generator.stop()
        self.loader.stop()

    def test_health_metadata_and_chat(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        metadata = self.client.get("/api/model")
        self.assertEqual(metadata.status_code, 200)
        self.assertEqual(metadata.json()["active_model"], "test-model")
        response = self.client.post("/api/chat", json={"prompt": "Hello"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"response": "test response"})

    def test_invalid_prompt_is_rejected(self):
        self.assertEqual(self.client.post("/api/chat", json={"prompt": ""}).status_code, 422)
        self.assertEqual(self.client.post("/api/chat", json={"prompt": "x" * (server.SETTINGS.max_prompt_characters + 1)}).status_code, 422)
