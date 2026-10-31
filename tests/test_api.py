import json
import httpx
import pytest
from fastapi.testclient import TestClient
from tests.synth import phrase_template, instance
import server.app as appmod
from server import engines

client = TestClient(appmod.app)

class FakeResp:
    def __init__(self, code, data): self.status_code, self._d = code, data
    def json(self): return self._d
    @property
    def text(self): return json.dumps(self._d)

@pytest.fixture(autouse=True)
def reset(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "SARVAM_API_KEY", "MAUN_COLAB_URL"):
        monkeypatch.delenv(var, raising=False)  # never hit real APIs with keys from .env
    appmod.memory.clear()
    appmod.settings.update({"mode": "smart", "vsr_engine": "local", "vsr_fallback": True, "colab_url": "",
                            "cleanup_engine": "openai", "cleanup_fallback": True, "voice_engine": "sarvam",
                            "voice_fallback": True, "anthropic_api_key": "", "openai_api_key": "", "sarvam_api_key": "",
                            "save_training_clips": True})
    yield

def test_index_and_static():
    r = client.get("/"); assert r.status_code == 200 and "Shhhhhhh AI" in r.text
    assert client.get("/static/app.js").status_code == 200

def test_settings_masking_and_validation():
    r = client.post("/api/settings", json={"openai_api_key": "sk-test-123456"}).json()
    assert r["openai_api_key"] == "••••3456" and r["has_openai_api_key"]
    # echoing the masked value back must not overwrite the real key
    client.post("/api/settings", json={"openai_api_key": "••••3456"})
    assert appmod.settings["openai_api_key"] == "sk-test-123456"
    assert client.post("/api/settings", json={"vsr_engine": "banana"}).status_code == 400
    assert client.post("/api/settings", json={"memory_strictness": 7}).json()["memory_strictness"] == 1.0

def test_memory_flow_and_recognize_memory():
    client.post("/api/memory/starter")
    phrases = client.get("/api/memory").json()["phrases"]
    assert len(phrases) == 20 and sum(p["lang"] == "hi" for p in phrases) == 10
    hi = next(p for p in phrases if p["text"] == "मुझे पानी चाहिए, अभी के अभी")
    en = next(p for p in phrases if p["text"] == "Thank you, that really helped.")
    for pid, seed in ((hi["id"], 1), (en["id"], 2)):
        tm = phrase_template(seed)
        for k in range(5):
            r = client.post(f"/api/memory/phrases/{pid}/samples", json={"features": instance(tm, seed * 100 + k),
                            "frames": ["/9j/AA=="], "keypoints": [[[1, 2], [3, 4], [5, 6], [7, 8]]], "timestamps": [0]})
            assert r.status_code == 200
    r = client.post("/api/recognize", json={"features": instance(phrase_template(1), 777), "mode": "memory"}).json()
    assert r["source"] == "memory" and r["text"] == "मुझे पानी चाहिए, अभी के अभी" and r["lang"] == "hi"
    ev = client.get("/api/memory/evaluate").json()
    assert ev["overall"] == 1.0
    # learn from a confirmed live result
    r = client.post("/api/memory/learn", json={"text": "Open the door.", "features": instance(phrase_template(9), 1)}).json()
    assert r["samples"] == 1
    assert client.delete(f"/api/memory/phrases/{r['id']}/samples/last").json()["samples"] == 0

def test_memory_mode_unknown_returns_not_sure():
    client.post("/api/memory/learn", json={"text": "a", "features": instance(phrase_template(1), 1)})
    client.post("/api/memory/learn", json={"text": "a", "features": instance(phrase_template(1), 2)})
    client.post("/api/memory/learn", json={"text": "b", "features": instance(phrase_template(2), 1)})
    client.post("/api/memory/learn", json={"text": "b", "features": instance(phrase_template(2), 2)})
    r = client.post("/api/recognize", json={"features": instance(phrase_template(55), 3), "mode": "memory"}).json()
    assert r["source"] == "none" and r["text"] == ""

def _clip():
    return {"frames": ["x"] * 30, "keypoints": [[[0, 0]] * 4] * 30, "timestamps": list(range(0, 1200, 40)),
            "features": instance(phrase_template(3), 3)}

def test_live_path_local_then_cleanup_fallback(monkeypatch):
    monkeypatch.setattr(appmod.local_vsr, "transcribe", lambda clip, beam: {"text": "I NEED SOME WOTER", "seconds": 0.5, "device": "cpu"})
    appmod.settings.update({"openai_api_key": "sk-x", "sarvam_api_key": "sv-x"})
    calls = []
    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(url)
        if "openai" in url:
            return FakeResp(500, {"error": "boom"})
        assert headers["api-subscription-key"] == "sv-x" and json["model"] == "sarvam-105b"
        return FakeResp(200, {"choices": [{"message": {"content": '{"text": "I need some water.", "changes": "wo->wa"}'}}]})
    monkeypatch.setattr(httpx, "post", fake_post)
    r = client.post("/api/recognize", json={**_clip(), "mode": "live"}).json()
    assert r["source"] == "live" and r["raw"] == "I NEED SOME WOTER"
    assert r["text"] == "I need some water." and r["cleanup_engine"] == "sarvam"
    assert any("openai" in f for f in r["fallbacks"]) and len(calls) == 2

def test_claude_cleanup(monkeypatch):
    sent = {}
    class FakeMessages:
        def create(self, **kw):
            sent.update(kw)
            block = type("B", (), {"type": "text", "text": '{"text": "I need some water."}'})()
            return type("R", (), {"stop_reason": "end_turn", "content": [block]})()
    class FakeClient:
        def __init__(self, **kw): self.beta = type("Beta", (), {"messages": FakeMessages()})()
    monkeypatch.setattr(engines.anthropic, "Anthropic", FakeClient)
    appmod.settings.update({"cleanup_engine": "claude", "anthropic_api_key": "sk-ant-x"})
    r = engines.run_cleanup("I NEED SOME WOTER", appmod.settings, [])
    assert r["engine"] == "claude" and r["text"] == "I need some water."
    assert sent["model"] == "claude-opus-5-5" and "WOTER" in sent["messages"][0]["content"]

def test_live_falls_back_to_colab_and_basic_cleanup(monkeypatch):
    def broken(clip, beam): raise engines.EngineError("local model is still loading")
    monkeypatch.setattr(appmod.local_vsr, "transcribe", broken)
    appmod.settings.update({"colab_url": "https://demo.trycloudflare.com"})
    def fake_post(url, json=None, timeout=None, headers=None):
        assert url == "https://demo.trycloudflare.com/transcribe" and json["beam_size"] == appmod.settings["vsr_beam"]
        return FakeResp(200, {"text": "WHAT TIME IS IT", "seconds": 0.2, "device": "Tesla T4"})
    monkeypatch.setattr(httpx, "post", fake_post)
    r = client.post("/api/recognize", json={**_clip(), "mode": "live"}).json()
    assert r["vsr_engine"] == "colab" and r["device"] == "Tesla T4"
    assert r["text"] == "What time is it?" and r["cleanup_engine"] == "basic"

def test_no_engine_and_fallback_off(monkeypatch):
    def broken(clip, beam): raise engines.EngineError("not installed")
    monkeypatch.setattr(appmod.local_vsr, "transcribe", broken)
    appmod.settings.update({"vsr_fallback": False})
    r = client.post("/api/recognize", json={**_clip(), "mode": "live"})
    assert r.status_code == 503 and "not installed" in r.json()["detail"]

def test_speak_fallbacks(monkeypatch):
    # no Sarvam key, not on a Mac -> browser voice
    r = client.post("/api/speak", json={"text": "नमस्ते"}).json()
    assert r["engine"] == "browser" and len(r["notes"]) == 2
    appmod.settings.update({"sarvam_api_key": "sv-x"})
    seen = []
    def fake_post(url, headers=None, json=None, timeout=None):
        seen.append(json)
        if "target_language_code" in json:
            return FakeResp(422, {"detail": "unknown field"})
        return FakeResp(200, {"audios": ["UklGRg=="]})
    monkeypatch.setattr(httpx, "post", fake_post)
    r = client.post("/api/speak", json={"text": "नमस्ते"}).json()
    assert r["engine"] == "sarvam" and r["audio_b64"] == "UklGRg=="
    assert seen[1]["language_code"] == "hi-IN" and seen[1]["model"] == "bulbul:v3"

def test_status_shape():
    s = client.get("/api/status").json()
    assert set(s) >= {"local", "colab", "claude", "openai", "sarvam", "mac_voice", "memory"}
    assert s["local"]["state"] in ("missing", "idle", "loading", "error", "ready")

def test_basic_cleanup():
    assert engines.basic_cleanup("HOW ARE YOU") == "How are you?"
    assert engines.basic_cleanup("i am fine") == "I am fine."
    assert engines._extract_json('```json\n{"text": "Hi."}\n```')["text"] == "Hi."

def test_invisible_h_greeting():
    no_keys = {"cleanup_engine": "openai", "cleanup_fallback": True, "openai_api_key": "", "sarvam_api_key": ""}
    assert engines.run_cleanup("I", no_keys, [])["text"] == "Hi."
    assert engines.run_cleanup("I THERE", no_keys, [])["text"] == "Hi there."
    assert engines.run_cleanup("I NEED WATER", no_keys, [])["text"] == "I need water."


def test_claude_gets_nbest_and_lip_pieces(monkeypatch):
    sent = {}
    class FakeMessages:
        def create(self, **kw):
            sent.update(kw)
            block = type("B", (), {"type": "text", "text": '{"text": "I need some water."}'})()
            return type("R", (), {"stop_reason": "end_turn", "content": [block]})()
    class FakeClient:
        def __init__(self, **kw): self.beta = type("Beta", (), {"messages": FakeMessages()})()
    monkeypatch.setattr(engines.anthropic, "Anthropic", FakeClient)
    monkeypatch.setattr(appmod.local_vsr, "transcribe", lambda clip, beam: {
        "text": "I NEED SOME MOTOR", "seconds": 0.5, "device": "cpu",
        "nbest": [{"text": "I NEED SOME MOTOR", "score": -3.1}, {"text": "I NEED SOME WATER", "score": -3.4}],
        "ctc_raw": "I NEE SOM WAT ER"})
    appmod.settings.update({"cleanup_engine": "claude", "anthropic_api_key": "sk-ant-x", "mode": "live"})
    r = client.post("/api/recognize", json={"frames": ["x"], "keypoints": [[[0, 0]] * 4]}).json()
    prompt = sent["messages"][0]["content"]
    assert "Important words (spell exactly like this): hackathon, Amazon, AWS" in prompt
    assert "2. I NEED SOME WATER" in prompt and "Raw lip pieces (no language model): I NEE SOM WAT ER" in prompt
    assert r["text"] == "I need some water." and r["ctc_raw"] == "I NEE SOM WAT ER" and len(r["nbest"]) == 2


def test_decode_keeps_nbest_and_collapses_ctc():
    import torch
    from server.vsr_core import VSRCore
    tokens = ["<blank>", "▁I", "▁NEE", "D", "▁WATER", "▁MOTOR", "<eos>"]
    H = lambda ids, score: type("H", (), {"yseq": torch.tensor([6] + ids + [6]), "score": score})()
    class FakeCTC:
        def argmax(self, enc): return torch.tensor([[1, 1, 0, 2, 2, 3, 0, 0, 4, 4]])
    class FakeE2E:
        ctc = FakeCTC()
        def encode(self, x): return x
    core = VSRCore.__new__(VSRCore)
    core.model = type("M", (), {"device": "cpu", "token_list": tokens, "model": FakeE2E(),
                                "beam_search": lambda self, enc: [H([1, 2, 3, 5], -1.0), H([1, 2, 3, 5], -1.2),
                                                                  H([1, 2, 3, 4], -1.5)]})()
    nbest, ctc = core._decode(torch.zeros(1))
    assert [h["text"] for h in nbest] == ["I NEED MOTOR", "I NEED WATER"]  # duplicate dropped
    assert ctc == "I NEED WATER"
