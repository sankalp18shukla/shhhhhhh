# Shhhhhhh AI — silent speech from your Mac camera

You mouth a sentence without making a sound, and Shhhhhhh AI reads your lips, shows the text, and says it out loud in English or Hindi.

| Part | What it does | Runs on |
|---|---|---|
| **Live lip reading** | Reads any English sentence using the open-source Auto-AVSR model (about 19% word error rate on the LRS3 benchmark) | Your Mac **or** a free Colab GPU (switch in Settings) |
| **AI correction** | Fixes words that look the same on the lips (p/b/m, f/v…) | OpenAI **or** Sarvam **or** off |
| **Personal Voice Memory** | Learns how *you* say your frequent sentences, in English, Hindi, or both, and gets better each time you confirm a result | Your Mac (instant, works offline) |
| **Voice** | Speaks the result aloud | Sarvam Bulbul **or** Mac voices **or** browser **or** off |

Every engine has a switch in **Settings**, and each one can fall back automatically: if Colab drops, it uses your Mac; if OpenAI fails, it uses Sarvam, then a basic tidy-up; if Sarvam's voice fails, it uses the Mac voice, then the browser voice. A toast tells you whenever a fallback happens.

---

## 1. Setup (once, on Wi-Fi — about 10 minutes)

```bash
cd maun
chmod +x setup.sh run.sh
./setup.sh
```
The script installs Python 3.11 in a private folder, downloads the lip-reading model (about 1 GB), and saves the face tracker locally so the app works without Wi-Fi at the venue.

Add your keys either in **Settings** inside the app or in `.env`:
- **Sarvam:** log in at https://indus.sarvam.ai with your hackathon email and copy your API key.
- **OpenAI:** optional, if you have a key.

## 2. Run

```bash
./run.sh
```
Chrome opens at **http://127.0.0.1:8765**. Allow camera access when asked.
- **Live:** hold **Space** (or the red button) and mouth a sentence, then release.
- **Hands-free:** Shhhhhhh AI starts and stops by itself when it sees your lips move.
- **Presentation mode** (bottom left of the sidebar): full-screen, large captions for the judges.

## 3. Do this first: the speed test

Before building anything else, set **Settings → Understanding → Mode → Live read**, then mouth 10 sentences.
- Look at the timing chip under each result (for example `1.8 s`).
- If it takes more than about 3 s, try **Speed vs accuracy → Fast**, or **Processor → Apple GPU**.
- If it's still slow, use the Colab GPU (next section).

## 4. Free Colab GPU (switch + backup)

1. Open `colab/Shhhhhhh_AI_GPU_Server.ipynb` in Google Colab (File → Upload notebook).
2. **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**.
3. Copy the green `https://….trycloudflare.com` link into **Settings → Colab URL** and press **Test**.
4. Switch **Engine → Colab GPU**. Leave **Auto-switch on failure** on, so a Colab disconnect falls back to your Mac.

## 5. Training Personal Voice Memory

**Voice Memory → Add starter phrases** adds 10 English and 10 Hindi phrases, chosen to look different on the lips. Click **Train** on a phrase, then press **Record** and mouth it; recording stops by itself when your lips go still. Do this 6 times. Then press **Check memory health**: it replays every sample against the rest of your memory and shows the accuracy for each phrase.

Tips:
- Train in the same light you'll demo in, and record 2 or 3 extra samples at the venue.
- Keep sentences different in length and in mouth shape.
- Mouth each sentence the same way every time.
- When a Live result is right (or after you fix it with **Edit**), press **Save to Voice Memory**. Your memory grows as you use it.

**Mode** decides how a clip is handled:
- **Smart:** your memory first, then live reading.
- **Live read:** open-vocabulary model only.
- **Memory:** your trained phrases only.

## 6. Building your own Hindi model (the roadmap)

With **Settings → Keep training clips** on, every training sample is saved to `data/clips/*.npz`. Each file holds the face frames, the 4 alignment points per frame, and the sentence. That is a labelled, personal Hindi and English lip-reading dataset in exactly the format Auto-AVSR's preprocessing expects. Fine-tuning the model on it, starting with Hindi phrases, is the next step, and the honest answer when a judge asks "how does this scale to Hindi?"

## Honest limits (say these in the pitch — judges respect it)

- A normal camera reads **visible** mouthing. Glasses with muscle sensors can catch almost-invisible movement; a camera can't.
- Free-form lip reading still gets some words wrong. AI correction fixes many of them, but not all.
- Lighting matters most. Bring a small desk lamp for even light on your face.
- The LRS3-trained model weights are for **non-commercial research** use. They're fine for a hackathon; check the licence before building a product.

## Files

```
server/app.py        API + recognition pipeline (memory → lip model → AI correction)
server/vsr_core.py   lip-model wrapper (shared by Mac + Colab)
server/memory.py     Personal Voice Memory (DTW over lip/jaw movement)
server/engines.py    every engine and its fallback
static/              the interface (index.html, app.css, app.js)
colab/               free GPU server notebook
tests/               pytest suite (python -m pytest -q)
```
Credits: Auto-AVSR (Ma et al., Imperial College London), packaged by Chaplin (MIT); MediaPipe Face Landmarker; Sarvam Bulbul.
