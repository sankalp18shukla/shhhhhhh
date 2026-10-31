"""Synthetic mouth-feature sequences: each phrase = its own smooth trajectory."""
import numpy as np
D = 113 + 28

def phrase_template(seed):
    r = np.random.default_rng(seed)
    T = int(r.integers(35, 95))
    t = np.linspace(0, 1, T)[:, None]
    freqs = r.uniform(1, 6, (1, D)); phase = r.uniform(0, 6.28, (1, D)); amp = r.uniform(0.02, 0.12, (1, D))
    return np.sin(2 * np.pi * freqs * t + phase) * amp

def instance(template, seed, noise=0.012):
    r = np.random.default_rng(seed)
    speed = r.uniform(0.82, 1.18)
    T = max(10, int(len(template) * speed))
    src = np.linspace(0, 1, len(template)); dst = np.linspace(0, 1, T) ** r.uniform(0.9, 1.1)
    seq = np.stack([np.interp(dst, src, template[:, d]) for d in range(D)], 1)
    seq += r.normal(0, noise, seq.shape) + r.normal(0, 0.01, (1, D))  # jitter + small identity offset
    return seq.tolist()
