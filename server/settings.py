"""Settings store: every engine + every fallback switch lives here (data/settings.json)."""
from __future__ import annotations

import json
import os
import threading

DEFAULTS = {
    # Lip reading (open-vocabulary English model)
    "vsr_engine": "colab" if os.environ.get("VERCEL") else "local",  # local | colab (no local model on Vercel)
    "vsr_fallback": True,           # if the chosen engine fails, try the other one
    "colab_url": "",
    "vsr_beam": 20,                 # 5 = fastest, 40 = most accurate
    "vsr_device": "cpu",            # cpu | mps (Apple GPU, experimental)

    # How a clip is understood
    "mode": "smart",                # smart (memory first, then live) | live | memory
    "memory_strictness": 0.5,

    # AI correction of the raw lip-read text
    "cleanup_engine": "claude",     # claude | openai | sarvam | off
    "cleanup_fallback": True,
    "claude_model": "claude-opus-5-5",
    "openai_model": "gpt-4.1-mini",
    "sarvam_model": "sarvam-105b",
    "context": "The user is silently dictating short everyday messages and commands.",
    # Names and terms the speaker uses a lot. The lip model has never seen most of them,
    # so it writes look-alike English words instead; the AI maps those back.
    "vocabulary": "hackathon, Amazon, AWS, Airo, Sarvam, Aakansha, Shaurya, Hirak",

    # Speaking the result out loud
    "voice_engine": "sarvam",       # sarvam | mac | browser | off
    "voice_fallback": True,
    "speak_automatically": True,
    "sarvam_speaker": "shubh",

    # Capture
    "capture_mode": "hold",         # hold (Space / button) | auto (hands-free)
    "save_training_clips": not os.environ.get("VERCEL"),  # keeps video clips so you can fine-tune a model later

    # Secrets (never sent back to the browser in full)
    "anthropic_api_key": "",
    "openai_api_key": "",
    "sarvam_api_key": "",
}

SECRET_KEYS = ("anthropic_api_key", "openai_api_key", "sarvam_api_key")
ENV_KEYS = {"anthropic_api_key": "ANTHROPIC_API_KEY", "openai_api_key": "OPENAI_API_KEY", "sarvam_api_key": "SARVAM_API_KEY",
            "colab_url": "MAUN_COLAB_URL"}

CHOICES = {
    "vsr_engine": {"local", "colab"},
    "vsr_device": {"cpu", "mps"},
    "mode": {"smart", "live", "memory"},
    "cleanup_engine": {"claude", "openai", "sarvam", "off"},
    "voice_engine": {"sarvam", "mac", "browser", "off"},
    "capture_mode": {"hold", "auto"},
}


def load_dotenv(path: str):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class Settings:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        self.values = dict(DEFAULTS)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                stored = json.load(f)
            self.values.update({k: v for k, v in stored.items() if k in DEFAULTS})

    def get(self, key):
        v = self.values.get(key, DEFAULTS.get(key))
        if key in ENV_KEYS and not v:
            v = os.environ.get(ENV_KEYS[key], "")
        return v

    def __getitem__(self, key):
        return self.get(key)

    def update(self, patch: dict) -> dict:
        with self.lock:
            for k, v in patch.items():
                if k not in DEFAULTS:
                    continue
                if k in SECRET_KEYS and isinstance(v, str) and v.startswith("••"):
                    continue  # masked value echoed back: keep the stored one
                if k in CHOICES and v not in CHOICES[k]:
                    raise ValueError(f"{k} must be one of {sorted(CHOICES[k])}")
                default = DEFAULTS[k]
                if isinstance(default, bool):
                    v = bool(v)
                elif isinstance(default, int):
                    v = int(v)
                elif isinstance(default, float):
                    v = max(0.0, min(1.0, float(v)))
                elif isinstance(default, str):
                    v = str(v).strip()
                if k == "colab_url":
                    v = v.rstrip("/")
                self.values[k] = v
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.values, f, indent=2)
        return self.public()

    def public(self) -> dict:
        out = {k: self.get(k) for k in DEFAULTS}
        for k in SECRET_KEYS:
            val = out[k]
            out[k] = ("••••" + val[-4:]) if val else ""
            out["has_" + k] = bool(val)
        return out
