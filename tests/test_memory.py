import os, tempfile, numpy as np
from server.memory import VoiceMemory, dtw
from tests.synth import phrase_template, instance, D

def make_memory(n_phr=12, n_samp=6):
    m = VoiceMemory(os.path.join(tempfile.mkdtemp(), "mem.json"))
    temps = {}
    for k in range(n_phr):
        text = f"phrase {k}" if k % 2 else f"वाक्य {k}"
        p = m.add_phrase(text)
        temps[p["id"]] = (text, phrase_template(k))
        for s in range(n_samp):
            m.add_sample(p["id"], instance(temps[p["id"]][1], 1000 * k + s))
    return m, temps

def test_dtw_identity_and_order():
    a = np.random.rand(30, 4)
    assert dtw(a, a) == 0
    assert dtw(a, a + 1) > dtw(a, a + 0.1)

def test_match_and_reject():
    m, temps = make_memory()
    correct = accepted = 0
    for k, (pid, (text, tmpl)) in enumerate(temps.items()):
        for s in range(3):
            r = m.match(instance(tmpl, 99999 + 10 * k + s), 0.5)
            correct += r["text"] == text
            accepted += r["accepted"] and r["text"] == text
    n = len(temps) * 3
    print("top1", correct / n, "accepted-correct", accepted / n)
    assert correct / n >= 0.9
    assert accepted / n >= 0.75
    # an unseen sentence should mostly be rejected
    rej = sum(not m.match(instance(phrase_template(500 + i), 7 + i), 0.5)["accepted"] for i in range(10))
    print("rejected unknown", rej, "/10")
    assert rej >= 7

def test_evaluate_and_persistence():
    m, _ = make_memory(6, 4)
    ev = m.evaluate()
    assert ev["overall"] >= 0.9 and ev["tested"] == 24
    m2 = VoiceMemory(m.path)
    assert m2.summary()["total_samples"] == 24
    hi = [p for p in m2.summary()["phrases"] if p["lang"] == "hi"]
    assert hi and hi[0]["text"].startswith("वाक्य")

def test_learn_and_undo_and_validation():
    m, temps = make_memory(3, 3)
    pid, (text, tmpl) = next(iter(temps.items()))
    assert m.learn(text, instance(tmpl, 5))["samples"] == 4
    assert m.delete_last_sample(pid) == 3
    try:
        m.add_sample(pid, [[0.0] * D] * 3); assert False
    except ValueError:
        pass

def test_empty_memory():
    m = VoiceMemory(os.path.join(tempfile.mkdtemp(), "x.json"))
    assert m.match([[0.0] * D] * 20)["accepted"] is False

def test_switched_off_phrase_is_never_matched():
    m, temps = make_memory(4, 4)
    pid, (text, tmpl) = next(iter(temps.items()))
    assert m.match(instance(tmpl, 4242), 0.5)["text"] == text
    m.set_active(pid, False)
    assert m.match(instance(tmpl, 4242), 0.5)["text"] != text
    assert not [p for p in VoiceMemory(m.path).summary()["phrases"] if p["id"] == pid][0]["active"]
    for other in temps:
        m.set_active(other, False)
    assert m.match(instance(tmpl, 4242))["reason"] == "memory is empty"
