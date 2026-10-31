"""
Shhhhhhh AI - shared lip-reading core.

Runs the open-source Auto-AVSR visual speech recognition model (LRS3, ~19% WER)
that Chaplin packages. Used by BOTH the local Mac engine and the Colab GPU server,
so the two always behave identically.

Why this file exists instead of calling Chaplin directly:
  * Chaplin's own face detector uses `mediapipe.solutions`, which no longer exists
    in current MediaPipe releases (fresh installs crash).
  * Chaplin reads video with `torchvision.io.read_video`, which newer torchvision
    versions dropped.
  We avoid both: the browser already tracks the face, so it sends us frames plus the
  4 alignment points the model needs (right eye, left eye, nose, mouth centre).
"""
from __future__ import annotations

import base64
import configparser
import os
import sys
import threading
import time
from typing import List, Optional, Sequence

import numpy as np

MODEL_FPS = 25.0
CONFIG_NAME = "LRS3_V_WER19.1.ini"
NBEST = 5


_STRTOBOOL = '''def strtobool(val):  # maun patch: distutils was removed in Python 3.12
    val = str(val).lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return 1
    if val in ("n", "no", "f", "false", "off", "0"):
        return 0
    raise ValueError("invalid truth value %r" % (val,))'''

_LOOSEVERSION = '''try:  # maun patch: distutils was removed in Python 3.12
    from packaging.version import Version as LooseVersion
except ImportError:
    import re as _re
    def LooseVersion(v):
        return tuple(int(x) for x in _re.findall(r"\\d+", str(v))[:3])'''

_PATCHES = [
    ("from distutils.util import strtobool as dist_strtobool", _STRTOBOOL + "\ndist_strtobool = strtobool"),
    ("from distutils.util import strtobool", _STRTOBOOL),
    ("from distutils.version import LooseVersion", _LOOSEVERSION),
    ("dtype=np.int)", "dtype=int)"),  # numpy >= 1.24 removed np.int
]


def patch_chaplin(chaplin_dir: str) -> int:
    """Make the bundled espnet code run on modern Python/NumPy. Idempotent."""
    changed = 0
    for base in ("espnet", "pipelines"):
        for dirpath, _, files in os.walk(os.path.join(chaplin_dir, base)):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fn)
                with open(path, encoding="utf-8") as f:
                    src = f.read()
                new = src
                for old, rep in _PATCHES:
                    if old in new:
                        new = new.replace(old, rep)
                if new != src:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(new)
                    changed += 1
    return changed


def decode_frames(frames_b64: Sequence[str]) -> List[np.ndarray]:
    """Base64 JPEG/PNG (optionally data-URL) -> list of RGB uint8 arrays."""
    import cv2

    out = []
    for item in frames_b64:
        if "," in item[:64]:
            item = item.split(",", 1)[1]
        buf = np.frombuffer(base64.b64decode(item), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("could not decode a video frame")
        out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return out


def resample_indices(timestamps_ms: Sequence[float], fps: float = MODEL_FPS) -> List[int]:
    """Pick frame indices so the clip plays at exactly `fps` (nearest-frame)."""
    ts = np.asarray(timestamps_ms, dtype=np.float64)
    if len(ts) == 0:
        return []
    if len(ts) == 1:
        return [0]
    step = 1000.0 / fps
    targets = np.arange(ts[0], ts[-1] + step * 0.5, step)
    idx = np.searchsorted(ts, targets)
    idx = np.clip(idx, 1, len(ts) - 1)
    left, right = ts[idx - 1], ts[idx]
    idx = idx - ((targets - left) < (right - targets))
    return idx.astype(int).tolist()


def prepare_clip(frames: List[np.ndarray], keypoints: Sequence[Optional[Sequence]],
                 timestamps_ms: Optional[Sequence[float]] = None):
    """Align lengths, resample to 25 fps, return (video[T,H,W,3], landmarks list)."""
    n = min(len(frames), len(keypoints))
    if n == 0:
        raise ValueError("empty clip")
    frames, keypoints = frames[:n], list(keypoints[:n])
    if timestamps_ms is not None and len(timestamps_ms) >= n:
        idx = resample_indices(list(timestamps_ms)[:n])
        frames = [frames[i] for i in idx]
        keypoints = [keypoints[i] for i in idx]
    lms = [None if k is None else np.asarray(k, dtype=np.float64).reshape(4, 2) for k in keypoints]
    if all(l is None for l in lms):
        raise ValueError("no face found in this clip - move closer / add light")
    return np.stack(frames), lms


def to_model_tensor(mouth_seq: np.ndarray):
    """(T,96,96) uint8 mouth crops -> (1,T,88,88) normalised tensor (same as Auto-AVSR VideoTransform)."""
    import torch

    x = torch.as_tensor(mouth_seq, dtype=torch.float32)  # T,96,96
    x = x.unsqueeze(0) / 255.0  # 1,T,96,96
    h, w = x.shape[-2:]
    top, left = int(round((h - 88) / 2.0)), int(round((w - 88) / 2.0))
    x = x[..., top:top + 88, left:left + 88]
    return (x - 0.421) / 0.165


class VSRCore:
    """Loads the Auto-AVSR model once and transcribes clips. Thread-safe."""

    def __init__(self, chaplin_dir: str, device: str = "cpu", beam_size: int = 20):
        import torch

        chaplin_dir = os.path.abspath(chaplin_dir)
        patch_chaplin(chaplin_dir)
        # model.pth files are plain state dicts from the Chaplin/Auto-AVSR release
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        if chaplin_dir not in sys.path:
            sys.path.insert(0, chaplin_dir)
        from pipelines.model import AVSR  # noqa: E402  (from the Chaplin / Auto-AVSR code)
        from pipelines.detectors.mediapipe.video_process import VideoProcess  # noqa: E402

        cfg_path = os.path.join(chaplin_dir, "configs", CONFIG_NAME)
        cfg = configparser.ConfigParser()
        cfg.read(cfg_path)
        rel = lambda k: os.path.join(chaplin_dir, cfg.get("model", k))  # noqa: E731
        for k in ("model_path", "model_conf", "rnnlm", "rnnlm_conf"):
            if not os.path.isfile(rel(k)):
                raise FileNotFoundError(f"missing model file: {rel(k)} (run ./setup.sh)")

        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.lock = threading.Lock()
        self.video_process = VideoProcess(convert_gray=True)
        self.model = AVSR(
            "video", rel("model_path"), rel("model_conf"), rel("rnnlm"), rel("rnnlm_conf"),
            cfg.getfloat("decode", "penalty"), cfg.getfloat("decode", "ctc_weight"),
            cfg.getfloat("decode", "lm_weight"), int(beam_size), device,
        )
        self.beam_size = int(beam_size)

    def set_beam_size(self, beam: int):
        beam = max(1, int(beam))
        if beam == self.beam_size:
            return
        bs = self.model.beam_search
        bs.beam_size = beam
        bs.pre_beam_size = int(1.5 * beam)
        self.beam_size = beam

    def _pieces_to_text(self, ids) -> str:
        toks = [self.model.token_list[i] for i in ids]
        toks = [t for t in toks if t not in ("<blank>", "<eos>", "<unk>")]
        return "".join(toks).replace("▁", " ").strip()

    def _decode(self, x, n: int = NBEST):
        """Like AVSR.infer, but keeps the N-best list and the raw CTC reading.

        The beam search leans on a small English language model, so its #1 guess can be
        fluent but wrong. The alternatives plus the lip-only CTC reading give the LLM the
        actual evidence to rebuild the sentence from.
        """
        import torch

        m = self.model
        with torch.no_grad():
            enc = m.model.encode(x.to(m.device))
            hyps = m.beam_search(enc)
            ctc_ids = m.model.ctc.argmax(enc.unsqueeze(0))[0].tolist()

        nbest, seen = [], set()
        for h in hyps:
            text = self._pieces_to_text(h.yseq[1:].tolist())  # drop <sos>
            if not text or text in seen:
                continue
            seen.add(text)
            nbest.append({"text": text, "score": round(float(h.score), 2)})
            if len(nbest) >= n:
                break

        # CTC greedy: collapse repeats, drop blanks (id 0)
        collapsed = [t for i, t in enumerate(ctc_ids) if t != 0 and (i == 0 or t != ctc_ids[i - 1])]
        return nbest, self._pieces_to_text(collapsed)

    def transcribe(self, frames_b64, keypoints, timestamps_ms=None, beam_size=None) -> dict:
        t0 = time.time()
        frames = decode_frames(frames_b64)
        video, lms = prepare_clip(frames, keypoints, timestamps_ms)
        mouth = self.video_process(video, lms)
        if mouth is None or len(mouth) == 0:
            raise ValueError("could not crop the mouth region")
        x = to_model_tensor(mouth)
        with self.lock:
            if beam_size:
                self.set_beam_size(beam_size)
            nbest, ctc_raw = self._decode(x)
        return {
            "text": nbest[0]["text"] if nbest else "",
            "nbest": nbest,          # top beam-search readings, best first
            "ctc_raw": ctc_raw,      # lip-only reading: no language model, no decoder
            "frames": int(len(mouth)),
            "seconds": round(time.time() - t0, 3),
            "device": self.device,
        }
