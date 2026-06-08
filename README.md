# Deutsch TTS — Local German Text-to-Speech UI

Free, local, no API keys. German voice **cloned** from a short reference clip in
[`voices/`](voices/), so a "preset" is a named German speaker sample, not a canned voice.

**Two engines** (select with `TTS_ENGINE`):
- **XTTS-v2** (coqui-tts) — **default**. Steady and ~real-time on CPU; best for
  long-form German. License: CPML (**non-commercial**).
- **Chatterbox Multilingual** (Resemble AI, MIT) — more expressive, but ~4–5× slower
  on CPU and prone to occasional artifacts (drones/skips) on long text.

## Setup & run

Requires **Python 3.11** and [`uv`](https://github.com/astral-sh/uv). One command —
it creates the engine's isolated venv on first run, starts the server, and opens the
browser:
```bash
./run.sh                         # XTTS (default)  -> .venv-xtts
TTS_ENGINE=chatterbox ./run.sh   # Chatterbox      -> .venv
```
First launch downloads the model (~1.8 GB XTTS / ~2 GB Chatterbox) automatically.
The model loads in a background worker; the header status dot turns green when ready.

UI: **http://localhost:5050**

---

## Device selection (Apple Silicon)

**XTTS** runs on **CPU** on Apple Silicon (its MPS support is unreliable) at ~1× real
time — fast enough that this is fine. The notes below are mainly about **Chatterbox**,
which is heavier. The server auto-selects: CUDA → MPS (only if the machine has enough
RAM) → CPU.

**Why the RAM check:** generating this model on **MPS peaks ~24 GB** of unified
memory. On a Mac with less than ~32 GB that overflows into swap, which both *crawls*
(swap thrash) and once **crashed the machine** with an out-of-memory restart. So on
≤16–24 GB Macs the server picks **CPU**, which has a flat ~6 GB footprint and is
actually faster in practice (no paging). Measured on a 16 GB M3:

| Device | Peak RAM | Speed | Notes |
|--------|----------|-------|-------|
| MPS    | ~24 GB   | crawls under ~13 GB swap | unusable / crashed the machine |
| CPU    | ~6 GB    | ~4–5× real-time, steady  | the safe default here |

So generation is **not** instant — budget ~4–5× the clip length on CPU (a 20s clip
≈ ~100s), plus a one-time ~8s model load. Fine for clips you wait on; not live use.

Override the auto-pick:
```bash
TTS_DEVICE=cpu python server.py            # force CPU
TTS_DEVICE=mps python server.py            # force MPS (needs lots of RAM)
TTS_MPS_MIN_RAM_GB=24 python server.py     # lower the MPS RAM threshold
```

## Long runs / sleep

Generation is paused if the Mac sleeps (it suspends the worker), so a long job left
unattended can drag across hours of wall-clock. The server holds a wake-lock
(`caffeinate -i`) **only while a job is running**, so it won't idle-sleep
mid-generation and sleeps normally otherwise. Rough timing on CPU: ~4–5× the clip
length (a 15-chunk story ≈ 25–30 min of actual compute).

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
- Audio is generated as WAV and downloads as WAV. **Every generation is also
  auto-saved** to `output/<timestamp>_<voice>.wav` (gitignored) — your artifact archive.
- **Long text** is split into **one sentence per chunk**, generated separately and
  stitched with a clear pause at each sentence (`TTS_SENT_GAP` 0.35s) and paragraph
  (`TTS_PARA_GAP` 0.6s). The UI shows a per-chunk progress bar ("Teil 3 / 11"). A
  punctuation-less paragraph (e.g. a title) gets a period appended so it pauses.

## Using the audio in LingQ (sync caveat)

LingQ auto-aligns imported audio to text and **cannot import external timestamps**, so
its sentence highlight tends to **drift** on any synthetic audio (a known LingQ
limitation, not specific to this tool). The per-sentence pauses above give its aligner
the best chance, but sync isn't guaranteed. **Workaround:** import **shorter lessons**
(per paragraph/scene) so alignment error can't accumulate far before the next anchor.
- **Abbrechen** hard-cancels a running synthesis: it kills the generation worker
  process (instantly freeing the GPU), then respawns it — so the next generation
  waits ~10s for the model to reload. Use it to stop a runaway/too-long generation.
- A refresh during generation **reconnects** to the in-flight job (the job id is
  kept in localStorage); the progress bar resumes where it left off.
- History panel lets you replay recent generations (cleared on refresh).
- The green dot in the header = model loaded and ready (amber/neutral = still loading).
