"""Shhhhhhh AI lip-reading server for a free Google Colab GPU (used by Shhhhhhh AI's "Colab GPU" engine)."""
import os
import threading
import time

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from vsr_core import VSRCore

CHAPLIN_DIR = os.environ.get("MAUN_CHAPLIN_DIR", "/content/chaplin")
STATE = {"state": "loading", "core": None, "error": "", "device": None}
app = FastAPI(title="Shhhhhhh AI Colab GPU")


def _load():
    try:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        core = VSRCore(CHAPLIN_DIR, device=dev, beam_size=40)
        STATE.update(core=core, state="ready", device=(torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"))
    except Exception as e:
        STATE.update(state="error", error=f"{type(e).__name__}: {e}")


threading.Thread(target=_load, daemon=True).start()


class Clip(BaseModel):
    frames: List[str]
    keypoints: List[Optional[List[List[float]]]]
    timestamps: List[float] = []
    beam_size: Optional[int] = None


@app.get("/health")
def health():
    return {"ready": STATE["state"] == "ready", "state": STATE["state"], "error": STATE["error"], "device": STATE["device"]}


@app.post("/transcribe")
def transcribe(clip: Clip):
    if STATE["state"] != "ready":
        raise HTTPException(503, f"model {STATE['state']} {STATE['error']}")
    t0 = time.time()
    try:
        out = STATE["core"].transcribe(clip.frames, clip.keypoints, clip.timestamps or None, beam_size=clip.beam_size)
    except ValueError as e:
        raise HTTPException(422, str(e))
    out["server_seconds"] = round(time.time() - t0, 3)
    out["device"] = STATE["device"]
    return out
