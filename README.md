# Deutsch TTS — Local German Text-to-Speech UI

Free, local, no API keys. Powered by **Chatterbox Multilingual** (Resemble AI, MIT) —
open-source TTS that beats ElevenLabs in blind tests, with **native German** and
voice cloning from a short reference clip.

Voices are real reference clips in [`voices/`](voices/), cloned at synthesis time —
so a "preset" is a named German speaker sample, not a canned model voice.

## Setup (one time)

Requires **Python 3.11** (Chatterbox/PyTorch don't support newer yet). The steps
below use [`uv`](https://github.com/astral-sh/uv) to keep it isolated from your
system Python.

### 1. Create the environment + install deps
```bash
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements.txt
```
This pulls PyTorch and the Chatterbox model code (~a few GB).

### 2. Start the server

Quickest — one command that sets up the venv on first run, starts the server, and
opens the browser when ready:
```bash
./run.sh
```

Or manually:
```bash
source .venv/bin/activate
python server.py
```
- First launch downloads the multilingual checkpoint (~2 GB) automatically.
- The model is hosted **in-process** — there is no separate engine to run.
- On boot it warms the model in a background thread; the header status dot turns
  green once it's ready.

### 3. Open the UI
http://localhost:5050

---

## Apple Silicon (M-series)

The server auto-selects the best device: **MPS** (Apple GPU) → CUDA → CPU, with an
automatic CPU fallback if MPS fails to load. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set
for you so unimplemented MPS ops fall back to CPU instead of crashing.

Generation is **slow** on a Mac — measured ~7× real-time on an M3 via MPS (a 7s clip
took ~52s; model load adds ~11s once at boot). Fine for generating clips you wait on;
not for live/interactive use. Try `TTS_DEVICE=cpu` too — with MPS op-fallback churn,
clean CPU is sometimes comparable or faster.

Override the device if needed:
```bash
TTS_DEVICE=cpu python server.py     # force CPU
TTS_DEVICE=mps python server.py     # force MPS
```

## Stopping

Stop the server with **Ctrl+C** (or close the terminal). The generation runs in a
worker subprocess that's reaped on exit; if the server is force-killed (`kill -9`),
the orphaned worker self-terminates within ~5s so it never lingers holding the
model in memory. To check for strays: `pgrep -fl spawn_main`.

## German voices

| Voice ID      | Name   | Style              | Source clip                   |
|---------------|--------|--------------------|-------------------------------|
| helmut        | Helmut | Expressive (default) | `voices/helmut_expressive.wav` |
| helmut_calm   | Helmut | Calm               | `voices/helmut_calm.wav`      |

Both are cloned from the same ElevenLabs "Helmut" recording — the default uses an
animated stretch (~320s) for a livelier voice; `helmut_calm` uses the calmer intro.

### Adding a voice
1. Drop a **clean, ~10–15s, mono WAV** of the speaker into [`voices/`](voices/).
   (To convert/trim: `ffmpeg -ss START -t 13 -i input.mp3 -ac 1 -ar 24000 voices/name.wav`)
2. Add an entry to the `VOICES` list in [`server.py`](server.py).
3. Restart the server. It appears in the voice grid automatically.

## Tips
- **Ausdruck (expressiveness) ~0.5** is neutral. Lower = flatter/steadier, higher =
  more dramatic. For language learning, keep it moderate.
- Audio is generated as WAV and downloads as WAV.
- **Long text** is auto-split into ≤400-char chunks (sentence boundaries) so it
  won't exhaust memory; the UI shows a per-chunk progress bar ("Teil 3 / 11").
  Tune the chunk size with `TTS_MAX_CHARS=300 python server.py` if you hit OOM.
- **Abbrechen** hard-cancels a running synthesis: it kills the generation worker
  process (instantly freeing the GPU), then respawns it — so the next generation
  waits ~10s for the model to reload. Use it to stop a runaway/too-long generation.
- A refresh during generation **reconnects** to the in-flight job (the job id is
  kept in localStorage); the progress bar resumes where it left off.
- History panel lets you replay recent generations (cleared on refresh).
- The green dot in the header = model loaded and ready (amber/neutral = still loading).
