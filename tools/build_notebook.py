"""Builds colab/Shhhhhhh_AI_GPU_Server.ipynb from server/vsr_core.py + colab/colab_server.py."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n").splitlines(True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}


core = open(os.path.join(ROOT, "server", "vsr_core.py"), encoding="utf-8").read()
server = open(os.path.join(ROOT, "colab", "colab_server.py"), encoding="utf-8").read()

cells = [
    md("""
# Shhhhhhh AI · free Colab GPU for lip reading

1. **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**.
2. Wait for the big green link at the bottom (≈3–4 min the first time).
3. Paste it into **Shhhhhhh AI → Settings → Lip reading model → Colab URL** and press **Test**.

Keep this tab open while you demo. If Colab disconnects, Shhhhhhh AI switches back to your Mac automatically.
"""),
    code("""
!nvidia-smi -L || echo "No GPU - switch Runtime > Change runtime type > T4 GPU"
"""),
    code("""
%%bash
set -e
cd /content
[ -d chaplin ] || git clone -q --depth 1 https://github.com/amanvirparhar/chaplin
cd chaplin
dl() { [ -s "$2" ] || { mkdir -p "$(dirname "$2")"; wget -q -O "$2" "$1"; }; }
dl https://huggingface.co/Amanvir/LRS3_V_WER19.1/resolve/main/model.json benchmarks/LRS3/models/LRS3_V_WER19.1/model.json
dl https://huggingface.co/Amanvir/LRS3_V_WER19.1/resolve/main/model.pth  benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth
dl https://huggingface.co/Amanvir/lm_en_subword/resolve/main/model.json  benchmarks/LRS3/language_models/lm_en_subword/model.json
dl https://huggingface.co/Amanvir/lm_en_subword/resolve/main/model.pth   benchmarks/LRS3/language_models/lm_en_subword/model.pth
pip -q install fastapi uvicorn six scikit-image opencv-python-headless packaging
cd /content
[ -x cloudflared ] || { wget -q -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x cloudflared; }
echo "setup done"
"""),
    code("%%writefile /content/vsr_core.py\n" + core),
    code("%%writefile /content/colab_server.py\n" + server),
    code("""
import os, re, subprocess, sys, threading, time
sys.path.insert(0, "/content"); os.chdir("/content")
import uvicorn
import colab_server

server = uvicorn.Server(uvicorn.Config(colab_server.app, host="0.0.0.0", port=8000, log_level="warning"))
threading.Thread(target=server.run, daemon=True).start()

log = open("/content/tunnel.log", "w")
tunnel = subprocess.Popen(["/content/cloudflared", "tunnel", "--url", "http://localhost:8000", "--no-autoupdate"], stdout=log, stderr=log)
url = None
for _ in range(60):
    time.sleep(1)
    m = re.search(r"https://[-a-z0-9]+\\.trycloudflare\\.com", open("/content/tunnel.log").read())
    if m:
        url = m.group(0); break

while colab_server.STATE["state"] == "loading":
    time.sleep(1)
print("Model:", colab_server.STATE["state"], colab_server.STATE["device"] or "", colab_server.STATE["error"])

from IPython.display import HTML, display
display(HTML(f'<div style="font:600 20px -apple-system,sans-serif;padding:18px;border-radius:14px;background:#e8f7ee;color:#0a7a3b">Paste into Shhhhhhh AI → Settings → Colab URL:<br><code style="font-size:18px">{url}</code></div>'))
"""),
    md("Leave this running. To stop: **Runtime → Disconnect and delete runtime**."),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(ROOT, "colab", "Shhhhhhh_AI_GPU_Server.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out)
