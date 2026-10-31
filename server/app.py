"""Shhhhhhh AI - silent speech from your Mac camera.  Run:  ./run.sh  (or: python -m server.app)"""
from __future__ import annotations

import base64
import os
import threading
import time
import uuid
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import engines
from .engines import EngineError, LocalVSR, ColabVSR, run_cleanup, run_speak
from .memory import VoiceMemory
from .settings import Settings, load_dotenv

ROOT = engines.ROOT
DATA = os.environ.get("MAUN_DATA_DIR", os.path.join(ROOT, "data"))
load_dotenv(os.path.join(ROOT, ".env"))

settings = Settings(os.path.join(DATA, "settings.json"))
memory = VoiceMemory(os.path.join(DATA, "memory.json"))
local_vsr = LocalVSR()
colab_vsr = ColabVSR()

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(_app):
    # start loading the local lip-reading model in the background
    if settings["vsr_engine"] == "local" or settings["vsr_fallback"]:
        local_vsr.ensure_loading(settings["vsr_device"], settings["vsr_beam"])
    threading.Thread(target=memory.warm, daemon=True).start()
    yield


app = FastAPI(title="Shhhhhhh AI", lifespan=lifespan)


# ------------------------------------------------------------------ models
class Clip(BaseModel):
    frames: List[str] = []                          # base64 JPEG crops around the face
    keypoints: List[Optional[List[List[float]]]] = []  # per frame: 4 x [x, y] (R eye, L eye, nose, mouth)
    timestamps: List[float] = []                    # ms
    features: List[List[float]] = []                # per-frame mouth features (Voice Memory)
    mode: Optional[str] = None                      # override settings.mode


class SampleIn(BaseModel):
    features: List[List[float]]
    frames: List[str] = []
    keypoints: List[Optional[List[List[float]]]] = []
    timestamps: List[float] = []


class PhraseIn(BaseModel):
    text: str
    hint: str = ""


class LearnIn(BaseModel):
    text: str
    features: List[List[float]]


class SpeakIn(BaseModel):
    text: str
    engine: Optional[str] = None


STARTER = [
    ("Read my lips: we need Diet Coke.", ""), ("I’m not ignoring you, I’m on mute.", ""),
    ("Plot twist: I never said a word.", ""), ("Can you turn the lights on?", ""),
    ("I’ll be there in five minutes.", ""), ("Let’s grab some food after this.", ""),
    ("Thank you, that really helped.", ""), ("Zero words spoken, full sentence delivered.", ""),
    ("Can someone explain this to my mom?", ""), ("Judges, first prize looks good on me.", ""),
    ("मुझे पानी चाहिए, अभी के अभी", "mujhe paani chahiye, abhi ke abhi"),
    ("डॉक्टर साहब, दर्द यहाँ है", "doctor saahab, dard yahaan hai"),
    ("प्लीज़ मेरी मदद करो", "please meri madad karo"), ("चाय मिलेगी क्या?", "chai milegi kya?"),
    ("मैं ठीक हूँ, बस भूख लगी है", "main theek hoon, bas bhookh lagi hai"),
    ("थोड़ा धीरे चलो", "thoda dheere chalo"), ("मम्मी को मत बताना", "mummy ko mat batana"),
    ("बिल आप भरोगे", "bill aap bharoge"), ("एक सेल्फी हो जाए?", "ek selfie ho jaaye?"),
    ("बहुत बढ़िया, ताली बजाओ!", "bahut badhiya, taali bajao!"),
]


def _save_clip(text: str, clip) -> Optional[str]:
    """Keep labelled clips (frames + face points) -> your own dataset for fine-tuning later."""
    if not settings["save_training_clips"] or not clip.frames:
        return None
    try:
        d = os.path.join(DATA, "clips")
        os.makedirs(d, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:6]}.npz"
        np.savez_compressed(
            os.path.join(d, name), text=text,
            frames=np.array([base64.b64decode(f.split(",", 1)[-1]) for f in clip.frames], dtype=object),
            keypoints=np.array([k if k is not None else [[np.nan, np.nan]] * 4 for k in clip.keypoints], dtype=np.float32),
            timestamps=np.array(clip.timestamps, dtype=np.float32))
        return name
    except Exception:
        return None


# ------------------------------------------------------------------- pages
STATIC = os.path.join(ROOT, "static")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"), headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------- settings
@app.get("/api/settings")
def get_settings():
    return settings.public()


@app.post("/api/settings")
def post_settings(patch: dict):
    before = (settings["vsr_device"], settings["vsr_beam"])
    try:
        out = settings.update(patch)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if "vsr_device" in patch and settings["vsr_device"] != before[0]:
        local_vsr.reload(settings["vsr_device"], settings["vsr_beam"])
    elif settings["vsr_engine"] == "local" or settings["vsr_fallback"]:
        local_vsr.ensure_loading(settings["vsr_device"], settings["vsr_beam"])
    return out


@app.get("/api/status")
def status(force: bool = False):
    s = settings
    return {
        "local": local_vsr.status(),
        "colab": colab_vsr.health(s["colab_url"], force=force),
        "claude": {"ok": bool(s["anthropic_api_key"]), "detail": "key set" if s["anthropic_api_key"] else "no key"},
        "openai": {"ok": bool(s["openai_api_key"]), "detail": "key set" if s["openai_api_key"] else "no key"},
        "sarvam": {"ok": bool(s["sarvam_api_key"]), "detail": "key set" if s["sarvam_api_key"] else "no key"},
        "mac_voice": {"ok": engines.mac_available()},
        "memory": memory.summary()["total_samples"],
    }


@app.post("/api/test/{component}")
def test_component(component: str):
    try:
        if component == "colab":
            h = colab_vsr.health(settings["colab_url"], force=True)
            return {"ok": h["ok"], "detail": h.get("detail")}
        if component == "local":
            local_vsr.ensure_loading(settings["vsr_device"], settings["vsr_beam"])
            st = local_vsr.status()
            return {"ok": st["state"] == "ready", "detail": st["error"] or st["state"]}
        if component == "claude":
            return engines.test_claude(settings)
        if component == "openai":
            return engines.test_openai(settings)
        if component == "sarvam":
            return engines.test_sarvam_chat(settings)
        if component == "voice":
            res = run_speak("Hello! Shhhhhhh AI voice is working.", settings)
            return {"ok": res["engine"] != "off", "detail": res["engine"], **res}
    except EngineError as e:
        return {"ok": False, "detail": str(e)}
    raise HTTPException(404, "unknown component")


# --------------------------------------------------------------- recognise
def _run_vsr(clip: Clip):
    """Chosen lip-reading engine, falling back to the other one if allowed."""
    payload = {"frames": clip.frames, "keypoints": clip.keypoints, "timestamps": clip.timestamps}
    chosen = settings["vsr_engine"]
    order = [chosen] + ([e for e in ("local", "colab") if e != chosen] if settings["vsr_fallback"] else [])
    notes = []
    for name in order:
        try:
            if name == "local":
                res = local_vsr.transcribe(payload, settings["vsr_beam"])
            else:
                res = colab_vsr.transcribe(settings["colab_url"], payload, settings["vsr_beam"])
            return name, res, notes
        except EngineError as e:
            notes.append(f"{name}: {e}")
        except ValueError as e:  # bad clip (no face etc.) - same on every engine
            raise HTTPException(422, str(e))
        except Exception as e:
            notes.append(f"{name}: {type(e).__name__}: {e}")
    raise HTTPException(503, "No lip-reading engine available. " + " | ".join(notes))


@app.post("/api/recognize")
def recognize(clip: Clip):
    t0 = time.time()
    mode = clip.mode or settings["mode"]
    out = {"mode": mode, "fallbacks": []}

    # 1) Personal Voice Memory (works in any language)
    mem = None
    if mode in ("smart", "memory") and clip.features:
        try:
            mem = memory.match(clip.features, settings["memory_strictness"])
        except ValueError as e:
            mem = {"accepted": False, "reason": str(e)}
        out["memory"] = mem
        print(f"[memory] accepted={mem.get('accepted')} best={mem.get('text')!r} dist={mem.get('distance')} "
              f"limit={mem.get('limit')} margin={mem.get('margin')} runner_up={mem.get('runner_up')!r} "
              f"reason={mem.get('reason')} frames={len(clip.features)}", flush=True)
        if mem.get("accepted"):
            out.update(text=mem["text"], raw=None, source="memory", lang=mem.get("lang", "en"),
                       confidence=mem.get("confidence"), seconds=round(time.time() - t0, 3))
            return out
        if mode == "memory":
            out.update(text="", source="none", seconds=round(time.time() - t0, 3),
                       message="Not sure - try again or train this phrase")
            return out

    # 2) Live lip reading (open vocabulary, English)
    if not clip.frames:
        raise HTTPException(400, "no video frames in clip")
    engine, vsr, notes = _run_vsr(clip)
    out["fallbacks"] += notes
    raw = vsr.get("text", "")
    evidence = {"nbest": vsr.get("nbest") or [], "ctc_raw": vsr.get("ctc_raw", "")}
    out.update(vsr_engine=engine, vsr_seconds=vsr.get("seconds"), raw=raw, device=vsr.get("device"), **evidence)
    print(f"[vsr] best={raw!r} ctc={evidence['ctc_raw']!r} nbest={[h['text'] for h in evidence['nbest']]}", flush=True)
    if not raw:
        out.update(text="", source="none", seconds=round(time.time() - t0, 3),
                   message="Didn't catch that - mouth a little bigger and try again")
        return out

    # 3) AI correction (Claude / OpenAI / Sarvam / basic) with fallback
    phrases = [p["text"] for p in memory.summary()["phrases"] if p["lang"] == "en"]
    hint = mem.get("text") if mem and mem.get("confidence", 0) > 0.25 else None
    cl = run_cleanup(raw, settings, phrases, hint, evidence)
    out["fallbacks"] += cl["notes"]
    print(f"[cleanup] {cl['engine']} ({cl['seconds']}s) -> {cl['text']!r} notes={cl['notes']}", flush=True)
    out.update(text=cl["text"], cleanup_engine=cl["engine"], cleanup_seconds=cl["seconds"],
               source="live", lang="en", seconds=round(time.time() - t0, 3))
    return out


@app.post("/api/speak")
def speak(req: SpeakIn):
    return run_speak(req.text.strip(), settings, req.engine)


# ------------------------------------------------------------------ memory
@app.get("/api/memory")
def memory_list():
    return memory.summary()


@app.post("/api/memory/phrases")
def memory_add(p: PhraseIn):
    try:
        ph = memory.add_phrase(p.text, p.hint)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": ph["id"], "text": ph["text"], "lang": ph["lang"], "samples": len(ph["samples"])}


@app.post("/api/memory/starter")
def memory_starter():
    for text, hint in STARTER:
        memory.add_phrase(text, hint)
    return memory.summary()


@app.delete("/api/memory/phrases/{pid}")
def memory_delete(pid: str):
    memory.delete_phrase(pid)
    return {"ok": True}


@app.post("/api/memory/phrases/{pid}/samples")
def memory_sample(pid: str, s: SampleIn):
    try:
        n = memory.add_sample(pid, s.features, source="train")
        text = memory.get(pid)["text"]
    except KeyError:
        raise HTTPException(404, "phrase not found")
    except ValueError as e:
        raise HTTPException(422, str(e))
    _save_clip(text, s)
    threading.Thread(target=memory.warm, daemon=True).start()
    return {"samples": n}


@app.delete("/api/memory/phrases/{pid}/samples/last")
def memory_undo(pid: str):
    try:
        return {"samples": memory.delete_last_sample(pid)}
    except KeyError:
        raise HTTPException(404, "phrase not found")


@app.post("/api/memory/learn")
def memory_learn(req: LearnIn):
    try:
        out = memory.learn(req.text, req.features)
        threading.Thread(target=memory.warm, daemon=True).start()
        return out
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/memory/evaluate")
def memory_evaluate():
    return memory.evaluate(settings["memory_strictness"])


@app.delete("/api/memory")
def memory_clear():
    memory.clear()
    return {"ok": True}


@app.exception_handler(Exception)
async def _unhandled(request, exc):  # keep the UI alive whatever happens
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("MAUN_PORT", 8765)))
