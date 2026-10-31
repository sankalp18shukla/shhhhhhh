"""
All swappable engines + their fallbacks.

  Lip reading : LocalVSR (your Mac)      <->  ColabVSR (free Google Colab GPU)
  Cleanup     : Claude  ->  OpenAI  ->  Sarvam  ->  basic tidy-up (no AI)
  Voice       : Sarvam (Bulbul)  ->  Mac `say`  ->  browser voice
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from typing import Dict, List, Optional

import anthropic
import httpx

from .memory import is_devanagari

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPLIN_DIR = os.environ.get("MAUN_CHAPLIN_DIR", os.path.join(ROOT, "vendor", "chaplin"))


class EngineError(Exception):
    pass


# =========================================================== lip reading
class LocalVSR:
    """Loads the Auto-AVSR model in the background so the UI opens instantly."""

    def __init__(self):
        self.core = None
        self.state = "idle"     # idle | missing | loading | ready | error
        self.error = ""
        self.device = None
        self._lock = threading.Lock()

    def installed(self) -> bool:
        p = os.path.join(CHAPLIN_DIR, "benchmarks", "LRS3", "models", "LRS3_V_WER19.1", "model.pth")
        return os.path.isfile(p)

    def ensure_loading(self, device: str = "cpu", beam: int = 20):
        with self._lock:
            if self.state in ("loading", "ready"):
                return
            if not self.installed():
                self.state, self.error = "missing", "Model not downloaded - run ./setup.sh"
                return
            self.state, self.error = "loading", ""
        threading.Thread(target=self._load, args=(device, beam), daemon=True).start()

    def _load(self, device, beam):
        try:
            from .vsr_core import VSRCore
            core = VSRCore(CHAPLIN_DIR, device=device, beam_size=beam)
            self.core, self.device, self.state = core, core.device, "ready"
        except Exception as e:  # pragma: no cover - depends on the Mac install
            self.state, self.error = "error", f"{type(e).__name__}: {e}"

    def reload(self, device: str, beam: int):
        with self._lock:
            if self.state == "loading":
                return
            self.core, self.state = None, "idle"
        self.ensure_loading(device, beam)

    def status(self) -> Dict:
        return {"state": self.state, "error": self.error, "device": self.device,
                "installed": self.installed()}

    def transcribe(self, clip: Dict, beam: int) -> Dict:
        if self.state != "ready" or self.core is None:
            raise EngineError({"missing": "local model not installed (run ./setup.sh)",
                               "loading": "local model is still loading",
                               "error": f"local model failed to load: {self.error}"}.get(self.state, "local model not ready"))
        return self.core.transcribe(clip["frames"], clip["keypoints"], clip.get("timestamps"), beam_size=beam)


class ColabVSR:
    def __init__(self):
        self._health = {"ok": False, "checked": 0.0, "detail": "not checked"}

    def health(self, url: str, force: bool = False) -> Dict:
        if not url:
            return {"ok": False, "detail": "no Colab URL set"}
        if not force and time.time() - self._health["checked"] < 15 and self._health.get("url") == url:
            return self._health
        try:
            r = httpx.get(url + "/health", timeout=6)
            j = r.json()
            h = {"ok": bool(j.get("ready")), "detail": j.get("state", "unknown"), "gpu": j.get("device")}
        except Exception as e:
            h = {"ok": False, "detail": f"unreachable ({type(e).__name__})"}
        h.update(checked=time.time(), url=url)
        self._health = h
        return h

    def transcribe(self, url: str, clip: Dict, beam: int) -> Dict:
        if not url:
            raise EngineError("no Colab URL set (Settings -> Lip reading)")
        try:
            r = httpx.post(url + "/transcribe", json={**clip, "beam_size": beam}, timeout=90)
        except Exception as e:
            raise EngineError(f"Colab unreachable ({type(e).__name__})")
        if r.status_code != 200:
            raise EngineError(f"Colab error {r.status_code}: {r.text[:160]}")
        return r.json()


# ============================================================== cleanup
SYSTEM_PROMPT = """You turn the raw output of a LIP-READING model into the sentence the speaker silently mouthed. The model only sees lips, so treat its output as noisy evidence about mouth shapes, not as text to tidy up.

You get up to three kinds of evidence:
1. "Candidate readings": the model's top guesses, best first, with log scores (closer to 0 = more confident). They come from a beam search that leans on a small built-in English language model, so they can be fluent but wrong. Their disagreements show you where the model is unsure.
2. "Raw lip pieces": what the lip frames alone produced, with no language model. These are word fragments in ALL CAPS and are often broken or merged ("I NEE SOM WAT ER"). This is the most faithful record of the mouth shapes and sounds, so use it to check and repair the candidates.
3. Optional context: the speaker's situation, their "important words", their personal phrases, and a Voice Memory hint.

The lip model only knows common English words. When the speaker says a name or term it has never seen (Indian names, brands, tech terms), it writes something else in its place: a fragment that is not a real word ("AOR", "HAKA", "KAJ"), a different name or place ("ARJUNA", "IDAHO", "SYLVIA"), or a run of words that makes no sense in the sentence. Only in those spots, play a sound-alike game: check whether the spot looks and sounds like one of the important words (same number of syllables, same lip groups, allowing for the missing "h" below). If it clearly does, use the important word with its exact spelling.

How to reconstruct:
- Sounds in the same group look identical on the lips and get swapped freely: p/b/m | f/v | th/dh | t/d/n/l | s/z | sh/ch/j/zh | k/g/ng | w/r/oo | ee/i | a/ah/uh. Swap within a group whenever it gives a more sensible sentence.
- "h" is made in the throat and is INVISIBLE on the lips, so a word may have lost a leading h (e.g. "OW ARE YOU" = "How are you", "ELP ME" = "Help me"). But "I" followed by a normal sentence ("I NEED WATER") is just "I" - never turn it into "Hi".
- Keep the rhythm: about the same number of syllables and words as the evidence. Do not add new ideas or pad the sentence.
- The default is the plain reading. Ordinary English words that make sense in the sentence are what the speaker said: never replace them with an important word.
- Every important word you use needs its own spot in the evidence. Do not add names that have no spot, and do not use an important word just because it is on the list. Most sentences contain none of them.
- Prefer the most natural everyday sentence that fits the mouth shapes. If the speaker's personal phrases list contains a sentence that clearly matches, use it exactly.
- Use normal capitalisation and correct punctuation. End with . ? or !
- Return ONLY JSON: {"text": "<the sentence>", "changes": "<short note on what you inferred>"}"""


# Whole-clip readings that only make sense as a greeting whose "h" the camera can't see.
_INVISIBLE_H = {"I": "HI", "I I": "HI HI", "A": "HEY", "I THERE": "HI THERE",
                "I EVERYONE": "HI EVERYONE", "I ALL": "HI ALL", "I GUYS": "HI GUYS",
                "A THERE": "HEY THERE", "ELLO": "HELLO"}


def fix_invisible_h(raw: str) -> str:
    key = re.sub(r"[^A-Z ]", "", re.sub(r"\s+", " ", (raw or "").upper())).strip()
    return _INVISIBLE_H.get(key, raw)


def basic_cleanup(raw: str) -> str:
    t = re.sub(r"\s+", " ", (raw or "")).strip()
    if not t:
        return ""
    if t.isupper():
        t = t.lower()
    t = t[0].upper() + t[1:]
    t = re.sub(r"\bi\b", "I", t)
    if t[-1] not in ".?!।":
        t += "?" if t.split()[0].lower() in {"what", "where", "when", "why", "who", "how", "can", "could", "is", "are", "do", "does", "will", "would"} else "."
    return t


def _extract_json(text: str) -> Dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"text": text.strip().strip('"')}


def _user_prompt(raw: str, context: str, phrases: List[str], memory_hint: Optional[str],
                 evidence: Optional[Dict] = None, vocabulary: str = "") -> str:
    evidence = evidence or {}
    parts = [f"Context: {context or 'everyday conversation'}"]
    if vocabulary and vocabulary.strip():
        parts.append(f"Important words (spell exactly like this): {vocabulary.strip()}")
    if phrases:
        parts.append("Speaker's personal phrases: " + " | ".join(phrases[:40]))
    if memory_hint:
        parts.append(f"Voice memory thinks it may be: \"{memory_hint}\" (only use if it fits)")
    nbest = [h for h in evidence.get("nbest") or [] if h.get("text")]
    if len(nbest) > 1:
        parts.append("Candidate readings (best first):\n" + "\n".join(
            f"{i}. {h['text']}  (score {h.get('score', '?')})" for i, h in enumerate(nbest, 1)))
    else:
        parts.append(f"Candidate reading: {raw}")
    if evidence.get("ctc_raw"):
        parts.append(f"Raw lip pieces (no language model): {evidence['ctc_raw']}")
    return "\n".join(parts)


def _chat(url: str, headers: Dict, body: Dict, timeout: float = 25) -> str:
    try:
        r = httpx.post(url, headers=headers, json=body, timeout=timeout)
    except Exception as e:
        raise EngineError(f"network error ({type(e).__name__})")
    if r.status_code != 200:
        raise EngineError(f"HTTP {r.status_code}: {r.text[:200]}")
    try:
        return r.json()["choices"][0]["message"]["content"] or ""
    except Exception:
        raise EngineError("unexpected response format")


def cleanup_claude(raw, settings, phrases, hint, evidence=None) -> str:
    key = settings.get("anthropic_api_key")
    if not key:
        raise EngineError("no Claude API key")
    client = anthropic.Anthropic(api_key=key, timeout=25, max_retries=1)
    try:
        r = client.beta.messages.create(
            model=settings.get("claude_model") or "claude-opus-5-5",
            max_tokens=2000,
            output_config={"effort": "low"},
            # if Claude declines, the API retries on a fallback model in the same call
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _user_prompt(raw, settings.get("context"), phrases, hint, evidence, settings.get("vocabulary"))}])
    except anthropic.APIStatusError as e:
        raise EngineError(f"HTTP {e.status_code}: {str(e.message)[:200]}")
    except anthropic.APIConnectionError as e:
        raise EngineError(f"network error ({type(e).__name__})")
    if r.stop_reason == "refusal":
        raise EngineError("Claude declined")
    content = "".join(b.text for b in r.content if b.type == "text")
    return _extract_json(content).get("text", "").strip()


def cleanup_openai(raw, settings, phrases, hint, evidence=None) -> str:
    key = settings["openai_api_key"]
    if not key:
        raise EngineError("no OpenAI API key")
    content = _chat("https://api.openai.com/v1/chat/completions",
                    {"Authorization": f"Bearer {key}"},
                    {"model": settings["openai_model"] or "gpt-4.1-mini", "temperature": 0.2,
                     "response_format": {"type": "json_object"},
                     "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user", "content": _user_prompt(raw, settings["context"], phrases, hint, evidence, settings.get("vocabulary"))}]})
    return _extract_json(content).get("text", "").strip()


def cleanup_sarvam(raw, settings, phrases, hint, evidence=None) -> str:
    key = settings["sarvam_api_key"]
    if not key:
        raise EngineError("no Sarvam API key")
    content = _chat("https://api.sarvam.ai/v1/chat/completions",
                    {"api-subscription-key": key},
                    {"model": settings["sarvam_model"] or "sarvam-105b", "temperature": 0.2,
                     "reasoning_effort": None, "max_tokens": 400,
                     "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user", "content": _user_prompt(raw, settings["context"], phrases, hint, evidence, settings.get("vocabulary"))}]})
    return _extract_json(content).get("text", "").strip()


CLEANERS = {"claude": cleanup_claude, "openai": cleanup_openai, "sarvam": cleanup_sarvam}


def run_cleanup(raw: str, settings, phrases: List[str], hint: Optional[str] = None,
                evidence: Optional[Dict] = None) -> Dict:
    raw = fix_invisible_h(raw)
    chosen = settings["cleanup_engine"]
    order = [chosen] if chosen != "off" else []
    if chosen != "off" and settings["cleanup_fallback"]:
        order += [e for e in ("claude", "openai", "sarvam") if e != chosen]
    notes = []
    for name in order:
        try:
            t0 = time.time()
            text = CLEANERS[name](raw, settings, phrases, hint, evidence)
            if text:
                if text.isupper():  # some models echo the raw ALL CAPS style back
                    text = basic_cleanup(text)
                return {"text": text, "engine": name, "seconds": round(time.time() - t0, 3), "notes": notes}
            notes.append(f"{name}: empty reply")
        except EngineError as e:
            notes.append(f"{name}: {e}")
    return {"text": basic_cleanup(raw), "engine": "basic", "seconds": 0.0, "notes": notes}


# ================================================================ voice
_MAC_VOICES: Optional[List[str]] = None


def mac_available() -> bool:
    return platform.system() == "Darwin" and shutil.which("say") is not None


def _mac_voice_for(text: str) -> Optional[str]:
    global _MAC_VOICES
    if _MAC_VOICES is None:
        try:
            out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=5).stdout
            _MAC_VOICES = out.splitlines()
        except Exception:
            _MAC_VOICES = []
    want = "hi_IN" if is_devanagari(text) else "en_IN"
    for line in _MAC_VOICES:
        if want in line:
            return re.split(r"\s{2,}", line.strip())[0]
    return None


def speak_mac(text: str) -> Dict:
    if not mac_available():
        raise EngineError("Mac voice only works on macOS")
    cmd = ["say"]
    v = _mac_voice_for(text)
    if v:
        cmd += ["-v", v]
    subprocess.Popen(cmd + [text])
    return {"engine": "mac", "played_on_server": True, "voice": v}


def speak_sarvam(text: str, settings) -> Dict:
    key = settings["sarvam_api_key"]
    if not key:
        raise EngineError("no Sarvam API key")
    lang = "hi-IN" if is_devanagari(text) else "en-IN"
    base = {"text": text[:2400], "model": "bulbul:v3", "speaker": settings["sarvam_speaker"] or "shubh"}
    last = ""
    # Sarvam docs show both field names in different places, so try both.
    for lang_field in ("target_language_code", "language_code"):
        try:
            r = httpx.post("https://api.sarvam.ai/text-to-speech",
                           headers={"api-subscription-key": key},
                           json={**base, lang_field: lang}, timeout=25)
        except Exception as e:
            raise EngineError(f"network error ({type(e).__name__})")
        if r.status_code == 200:
            audios = r.json().get("audios") or []
            if audios:
                return {"engine": "sarvam", "audio_b64": audios[0], "mime": "audio/wav"}
            raise EngineError("Sarvam returned no audio")
        last = f"HTTP {r.status_code}: {r.text[:160]}"
        if r.status_code not in (400, 422):
            break
    raise EngineError(last)


def run_speak(text: str, settings, engine: Optional[str] = None) -> Dict:
    chosen = engine or settings["voice_engine"]
    if chosen == "off" or not text:
        return {"engine": "off"}
    order = [chosen]
    if settings["voice_fallback"]:
        order += [e for e in ("sarvam", "mac", "browser") if e != chosen]
    notes = []
    for name in order:
        try:
            if name == "sarvam":
                res = speak_sarvam(text, settings)
            elif name == "mac":
                res = speak_mac(text)
            else:
                res = {"engine": "browser"}  # the page speaks it with the Web Speech API
            res["notes"] = notes
            return res
        except EngineError as e:
            notes.append(f"{name}: {e}")
    return {"engine": "off", "notes": notes}


# ============================================================ connectivity tests
def test_claude(settings) -> Dict:
    t0 = time.time()
    text = cleanup_claude("I NEED SOME WOTER PLEASE", settings, [], None)
    return {"ok": True, "detail": f"“{text}”", "seconds": round(time.time() - t0, 2)}


def test_openai(settings) -> Dict:
    t0 = time.time()
    text = cleanup_openai("I NEED SOME WOTER PLEASE", settings, [], None)
    return {"ok": True, "detail": f"“{text}”", "seconds": round(time.time() - t0, 2)}


def test_sarvam_chat(settings) -> Dict:
    t0 = time.time()
    text = cleanup_sarvam("I NEED SOME WOTER PLEASE", settings, [], None)
    return {"ok": True, "detail": f"“{text}”", "seconds": round(time.time() - t0, 2)}
