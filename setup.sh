#!/usr/bin/env bash
# Shhhhhhh AI one-time setup for macOS (Apple Silicon). Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"
say_step() { printf "\n\033[1;34m==>\033[0m \033[1m%s\033[0m\n" "$1"; }
dl() {  # dl URL DEST  (skips files that already exist)
  if [ -s "$2" ]; then echo "  ✓ $(basename "$2") already here"; return; fi
  mkdir -p "$(dirname "$2")"
  echo "  ↓ $(basename "$2")"
  curl -L --fail --retry 3 --progress-bar -o "$2.part" "$1" && mv "$2.part" "$2"
}

say_step "1/5  Python environment (uv + Python 3.11)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
[ -d .venv ] || uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-local.txt

say_step "2/5  Lip-reading model code (Auto-AVSR, packaged by Chaplin)"
mkdir -p vendor
[ -d vendor/chaplin ] || git clone --depth 1 https://github.com/amanvirparhar/chaplin vendor/chaplin

say_step "3/5  Lip-reading model weights (~1 GB, one time)"
HF=https://huggingface.co/Amanvir
M=vendor/chaplin/benchmarks/LRS3
dl "$HF/LRS3_V_WER19.1/resolve/main/model.json" "$M/models/LRS3_V_WER19.1/model.json"
dl "$HF/LRS3_V_WER19.1/resolve/main/model.pth"  "$M/models/LRS3_V_WER19.1/model.pth"
dl "$HF/lm_en_subword/resolve/main/model.json"  "$M/language_models/lm_en_subword/model.json"
dl "$HF/lm_en_subword/resolve/main/model.pth"   "$M/language_models/lm_en_subword/model.pth"
.venv/bin/python -c "from server.vsr_core import patch_chaplin; n=patch_chaplin('vendor/chaplin'); print(f'  ✓ compatibility patches applied ({n} files)')"

say_step "4/5  Face tracker, stored offline (works without Wi-Fi at the venue)"
MP=https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.0
V=static/vendor/mediapipe
dl "$MP/vision_bundle.mjs" "$V/vision_bundle.mjs"
for f in vision_wasm_internal.js vision_wasm_internal.wasm vision_wasm_nosimd_internal.js vision_wasm_nosimd_internal.wasm; do
  dl "$MP/wasm/$f" "$V/wasm/$f"
done
dl https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task static/models/face_landmarker.task

say_step "5/5  Config"
[ -f .env ] || cp .env.example .env
echo "  ✓ add your API keys to .env (or paste them in Settings inside the app)"

printf "\n\033[1;32mAll set.\033[0m  Start Shhhhhhh AI with:  \033[1m./run.sh\033[0m\n"
