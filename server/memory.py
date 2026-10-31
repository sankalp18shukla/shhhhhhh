"""
Personal Voice Memory.

Learns how *your* lips and jaw move for the sentences you say most, in any
language (English, Hindi, Hinglish...). Each utterance is a sequence of per-frame
mouth features computed in the browser (lip contour, inner lip, jaw line, mouth
opening and MediaPipe mouth blendshapes). Matching uses Dynamic Time Warping so
speaking a bit faster or slower still matches.

Every sample can come from the Train screen or from a confirmed live result
("Save to memory"), so the memory genuinely improves as you use it.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Dict, List, Optional

import numpy as np

# Feature layout sent by the browser (see static/app.js -> FEATURE_LAYOUT)
GROUPS = [("outer_lip", 40), ("inner_lip", 40), ("jaw", 30), ("scalars", 3)]
GEOM_DIMS = sum(n for _, n in GROUPS)  # 113; the rest are blendshapes
RESAMPLE_LEN = 40
BAND = 7
DELTA_WEIGHT = 0.6
DURATION_WEIGHT = 0.35


def is_devanagari(text: str) -> bool:
    return any("ऀ" <= ch <= "ॿ" for ch in text)


def _resample(seq: np.ndarray, length: int = RESAMPLE_LEN) -> np.ndarray:
    t = seq.shape[0]
    if t == 1:
        return np.repeat(seq, length, axis=0)
    src = np.linspace(0, 1, t)
    dst = np.linspace(0, 1, length)
    return np.stack([np.interp(dst, src, seq[:, d]) for d in range(seq.shape[1])], axis=1)


OPEN_IDX = GEOM_DIMS - 3  # "mouth opening" scalar (static/app.js -> iOpen)
TRIM_PAD = 3


def _trim(seq: np.ndarray) -> np.ndarray:
    """Drop closed-mouth idle frames before/after the sentence so clips line up."""
    if seq.shape[0] < 8 or seq.shape[1] <= OPEN_IDX:
        return seq
    op = seq[:, OPEN_IDX]
    lo, hi = float(op.min()), float(op.max())
    if hi - lo < 1e-6:
        return seq
    active = np.where(op > lo + 0.2 * (hi - lo))[0]
    a, b = max(0, active[0] - TRIM_PAD), min(len(seq), active[-1] + 1 + TRIM_PAD)
    return seq[a:b] if b - a >= 5 else seq


def _group_slices(dim: int):
    out, start = [], 0
    for _, n in GROUPS:
        out.append(slice(start, min(start + n, dim)))
        start += n
    if dim > start:
        out.append(slice(start, dim))
    return [s for s in out if s.stop > s.start]


def dtw(a: np.ndarray, b: np.ndarray, band: int = BAND) -> float:
    """Banded DTW (vectorised over anti-diagonals). Returns cost normalised by (n + m)."""
    n, m = len(a), len(b)
    cost = np.sqrt(np.maximum(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1), 0.0))
    w = max(band, abs(n - m))
    ii, jj = np.indices((n, m))
    cost = np.where(np.abs(ii - jj) <= w, cost, np.inf)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for d in range(2, n + m + 1):
        i = np.arange(max(1, d - m), min(n, d - 1) + 1)
        j = d - i
        best = np.minimum(np.minimum(acc[i - 1, j - 1], acc[i - 1, j]), acc[i, j - 1])
        acc[i, j] = best + cost[i - 1, j - 1]
    return float(acc[n, m] / (n + m)) if np.isfinite(acc[n, m]) else float("inf")


class VoiceMemory:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.RLock()
        self.data: Dict = {"phrases": []}
        self._norm = None  # (mean, std) cache
        self._cache: Dict[str, np.ndarray] = {}
        self._lens: Dict[str, int] = {}
        self._intra: Optional[float] = None
        self.load()

    # ------------------------------------------------------------------ storage
    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        self.data.setdefault("phrases", [])
        self._invalidate()

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    def _invalidate(self):
        self._norm = None
        self._cache = {}
        self._lens = {}
        self._intra = None

    # ---------------------------------------------------------------- phrases
    def summary(self) -> Dict:
        with self.lock:
            phrases = [{
                "id": p["id"], "text": p["text"], "hint": p.get("hint", ""),
                "lang": p.get("lang", "en"), "samples": len(p["samples"]),
                "created": p.get("created"), "active": p.get("active", True),
            } for p in self.data["phrases"]]
            return {"phrases": phrases,
                    "total_samples": sum(p["samples"] for p in phrases)}

    def find(self, text: str) -> Optional[Dict]:
        key = text.strip().lower()
        for p in self.data["phrases"]:
            if p["text"].strip().lower() == key:
                return p
        return None

    def add_phrase(self, text: str, hint: str = "") -> Dict:
        text = text.strip()
        if not text:
            raise ValueError("phrase text is empty")
        with self.lock:
            existing = self.find(text)
            if existing:
                if hint and not existing.get("hint"):
                    existing["hint"] = hint
                    self.save()
                return existing
            p = {"id": uuid.uuid4().hex[:10], "text": text, "hint": hint,
                 "lang": "hi" if is_devanagari(text) else "en",
                 "samples": [], "created": time.time()}
            self.data["phrases"].append(p)
            self.save()
            return p

    def get(self, pid: str) -> Dict:
        for p in self.data["phrases"]:
            if p["id"] == pid:
                return p
        raise KeyError(pid)

    def delete_phrase(self, pid: str):
        with self.lock:
            self.data["phrases"] = [p for p in self.data["phrases"] if p["id"] != pid]
            self._invalidate()
            self.save()

    def set_active(self, pid: str, active: bool):
        """Switched-off phrases keep their samples but are never matched."""
        with self.lock:
            self.get(pid)["active"] = bool(active)
            self.save()

    def add_sample(self, pid: str, features: List[List[float]], source: str = "train") -> int:
        arr = self._validate(features)
        with self.lock:
            p = self.get(pid)
            p["samples"].append({"id": uuid.uuid4().hex[:10], "t": time.time(),
                                 "source": source, "len": int(arr.shape[0]),
                                 "f": np.round(arr, 4).tolist()})
            self._invalidate()
            self.save()
            return len(p["samples"])

    def delete_last_sample(self, pid: str) -> int:
        with self.lock:
            p = self.get(pid)
            if p["samples"]:
                p["samples"].pop()
                self._invalidate()
                self.save()
            return len(p["samples"])

    def learn(self, text: str, features, source: str = "confirmed") -> Dict:
        p = self.add_phrase(text)
        n = self.add_sample(p["id"], features, source=source)
        return {"id": p["id"], "samples": n}

    def clear(self):
        with self.lock:
            self.data = {"phrases": []}
            self._invalidate()
            self.save()

    # -------------------------------------------------------------- features
    @staticmethod
    def _validate(features) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 5 or arr.shape[1] < GEOM_DIMS:
            raise ValueError("clip too short or no face tracked - try again")
        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr)
        return arr

    def _raw_processed(self, arr: np.ndarray) -> np.ndarray:
        seq = _resample(_trim(arr))
        geom = seq[:, :GEOM_DIMS] - seq[:, :GEOM_DIMS].mean(axis=0, keepdims=True)
        return np.concatenate([geom, seq[:, GEOM_DIMS:]], axis=1)

    def _all_samples(self):
        for p in self.data["phrases"]:
            for s in p["samples"]:
                yield p, s

    def _active_samples(self):
        for p, s in self._all_samples():
            if p.get("active", True):
                yield p, s

    def _normaliser(self):
        if self._norm is None:
            rows = [self._raw_processed(np.asarray(s["f"])) for _, s in self._all_samples()]
            dim = rows[0].shape[1] if rows else None
            if rows and all(r.shape[1] == dim for r in rows):
                allr = np.concatenate(rows, axis=0)
                mean, std = allr.mean(axis=0), allr.std(axis=0)
                std = np.maximum(std, np.percentile(std, 25) + 1e-3)
            else:
                mean, std = 0.0, 1.0
            self._norm = (mean, std)
        return self._norm

    def _embed(self, arr: np.ndarray) -> np.ndarray:
        mean, std = self._normaliser()
        x = (self._raw_processed(arr) - mean) / std
        weighted = np.concatenate([x[:, s] / np.sqrt(s.stop - s.start) for s in _group_slices(x.shape[1])], axis=1)
        delta = np.diff(weighted, axis=0, prepend=weighted[:1]) * DELTA_WEIGHT * 3.0
        return np.concatenate([weighted, delta], axis=1)

    def _sample_embed(self, s) -> np.ndarray:
        if s["id"] not in self._cache:
            f = np.asarray(s["f"])
            self._cache[s["id"]] = self._embed(f)
            self._lens[s["id"]] = len(_trim(f))
        return self._cache[s["id"]]

    def _sample_len(self, s) -> int:
        self._sample_embed(s)
        return self._lens[s["id"]]

    def _distance(self, ea, la, eb, lb) -> float:
        if ea.shape[1] != eb.shape[1]:
            return float("inf")
        return dtw(ea, eb) + DURATION_WEIGHT * abs(np.log(max(la, 1) / max(lb, 1)))

    def warm(self):
        """Pre-compute embeddings + calibration so the next match is instant."""
        with self.lock:
            for _, s in self._all_samples():
                self._sample_embed(s)
            self._intra_scale()

    # --------------------------------------------------------------- matching
    def _phrase_scores(self, emb, length, exclude_sample: Optional[str] = None, top_k: int = 36):
        """Cheap linear pre-filter over every sample, then DTW on the closest `top_k`."""
        items = [(p, s, self._sample_embed(s)) for p, s in self._active_samples() if s["id"] != exclude_sample]
        items = [it for it in items if it[2].shape == emb.shape]
        if not items:
            return []
        stack = np.stack([it[2] for it in items])
        linear = np.sqrt(((stack - emb[None]) ** 2).sum(-1)).mean(-1)
        keep = np.argsort(linear)[:top_k]
        per: Dict[str, list] = {}
        phrase_of: Dict[str, Dict] = {}
        for k in keep:
            p, s, e = items[k]
            per.setdefault(p["id"], []).append(self._distance(emb, length, e, self._sample_len(s)))
            phrase_of[p["id"]] = p
        scores = []
        for pid, ds in per.items():
            ds.sort()
            # average of the 2 closest samples is more stable than the single best
            scores.append((float(np.mean(ds[:2])) if len(ds) >= 3 else ds[0], phrase_of[pid]))
        scores.sort(key=lambda x: x[0])
        return scores

    def _intra_scale(self) -> float:
        """Typical distance between two samples of the SAME phrase (self-calibration)."""
        if self._intra is not None:
            return self._intra
        vals = []
        for p in self.data["phrases"]:
            ss = p["samples"][:6]
            for i in range(len(ss)):
                for j in range(i + 1, len(ss)):
                    vals.append(self._distance(self._sample_embed(ss[i]), self._sample_len(ss[i]),
                                               self._sample_embed(ss[j]), self._sample_len(ss[j])))
        vals = [v for v in vals if np.isfinite(v)]
        self._intra = float(np.median(vals)) if vals else 1.0
        return self._intra

    def match(self, features, strictness: float = 0.5) -> Dict:
        """strictness 0..1  (0 = accept almost anything, 1 = only very sure matches)."""
        with self.lock:
            if not any(True for _ in self._active_samples()):
                return {"accepted": False, "reason": "memory is empty"}
            arr = self._validate(features)
            emb = self._embed(arr)
            scores = self._phrase_scores(emb, len(_trim(arr)))
            if not scores:
                return {"accepted": False, "reason": "memory is empty"}
            d1, best = scores[0]
            d2 = scores[1][0] if len(scores) > 1 else None
            margin = 1.0 - d1 / d2 if d2 and d2 > 0 else None
            intra = self._intra_scale()
            abs_limit = intra * (2.4 - 1.2 * strictness)
            margin_need = 0.04 + 0.22 * strictness
            ok_abs = d1 <= abs_limit
            ok_margin = margin is None or margin >= margin_need
            closeness = max(0.0, 1.0 - d1 / (abs_limit * 1.25))
            conf = closeness if margin is None else min(1.0, 0.5 * closeness + 0.5 * min(1.0, margin / 0.35))
            return {
                "accepted": bool(ok_abs and ok_margin),
                "id": best["id"], "text": best["text"], "lang": best.get("lang", "en"),
                "distance": round(d1, 4), "runner_up": scores[1][1]["text"] if len(scores) > 1 else None,
                "margin": None if margin is None else round(margin, 4),
                "confidence": round(float(conf), 3),
                "limit": round(abs_limit, 4),
                "reason": None if ok_abs and ok_margin else ("too different from memory" if not ok_abs else "too close to another phrase"),
            }

    def evaluate(self, strictness: float = 0.5) -> Dict:
        """Leave-one-out test: replay every stored sample against the rest of memory."""
        with self.lock:
            per, total, correct, rejected = [], 0, 0, 0
            for p in self.data["phrases"]:
                n_ok = n = n_rej = 0
                if not p.get("active", True):
                    per.append({"id": p["id"], "text": p["text"], "samples": len(p["samples"]),
                                "accuracy": None, "note": "switched off"})
                    continue
                if len(p["samples"]) < 2:
                    per.append({"id": p["id"], "text": p["text"], "samples": len(p["samples"]),
                                "accuracy": None, "note": "needs 2+ samples"})
                    continue
                for s in p["samples"]:
                    emb = self._sample_embed(s)
                    scores = self._phrase_scores(emb, self._sample_len(s), exclude_sample=s["id"])
                    if not scores:
                        continue
                    n += 1
                    if scores[0][1]["id"] == p["id"]:
                        n_ok += 1
                per.append({"id": p["id"], "text": p["text"], "samples": len(p["samples"]),
                            "accuracy": round(n_ok / n, 3) if n else None, "confused_with": None})
                total += n
                correct += n_ok
                rejected += n_rej
            return {"overall": round(correct / total, 3) if total else None,
                    "tested": total, "phrases": per}
