#!/usr/bin/env python3
"""
German TTS server — Chatterbox Multilingual behind a Flask UI.

Generation runs in a persistent worker subprocess that holds the model. This lets
us HARD-CANCEL a running synthesis by terminating the process (instantly freeing
the GPU), then respawning it. German is native via language_id="de". Runs on
Apple Silicon (MPS) with a CPU fallback.

Run:  python server.py   (inside the .venv — see README)
"""

import io
import multiprocessing as mp
import os
import queue
import threading
import time
import uuid

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder=".")
CORS(app)

VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
LANGUAGE = os.environ.get("TTS_LANGUAGE", "de")
# Chunk long inputs so peak memory stays bounded (generation memory grows with
# input length). 400 tested safe on a 20GB MPS pool; lower if you hit OOM.
MAX_CHARS = int(os.environ.get("TTS_MAX_CHARS", "400"))
# Safety net: kill the worker if a chunk produces no progress for this long — a
# genuine hang. Per-chunk token budgeting (see worker) already bounds normal work,
# so this only needs to catch a truly wedged process; keep it generous.
CHUNK_TIMEOUT = int(os.environ.get("TTS_CHUNK_TIMEOUT", "420"))
# MPS peaks ~24GB generating this model; below this much total RAM it swaps and
# crawls, so we auto-pick CPU instead (set TTS_DEVICE to override either way).
MPS_MIN_RAM_GB = int(os.environ.get("TTS_MPS_MIN_RAM_GB", "32"))
# Silence inserted between stitched chunks (a short beat; within-chunk pauses come
# from the text's own punctuation).
SENT_GAP = float(os.environ.get("TTS_SENT_GAP", "0.18"))
# Every generation is also saved here as <timestamp>_<voice>.wav (gitignored).
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Curated German preset library. `file` is a reference clip in ./voices that
# Chatterbox clones. Add a voice by dropping a clean ~10-15s mono WAV in ./voices
# and adding an entry here. First entry is the default voice (selected on load).
VOICES = [
    {"id": "helmut",      "name": "Helmut", "gender": "Male", "style": "Expressive", "file": "helmut_expressive.wav"},
    {"id": "helmut_calm", "name": "Helmut", "gender": "Male", "style": "Calm",       "file": "helmut_calm.wav"},
]


# =============================================================================
# Worker subprocess: loads the model once, generates chunk-by-chunk, reports
# progress and the finished WAV over a queue. Killed (and respawned) on cancel.
# =============================================================================
def _worker_main(request_q, response_q, language, forced_device):
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import numpy as np
    import soundfile as sf
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    def total_ram_gb():
        try:
            import subprocess
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / (1024 ** 3)
        except Exception:
            return 0  # unknown -> treat as low and prefer CPU (safe)

    if forced_device:
        device = forced_device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and total_ram_gb() >= MPS_MIN_RAM_GB:
        # MPS peaks ~24GB while generating; only worth it with enough unified RAM,
        # else it swaps and crawls. Below the threshold, CPU is faster and safe.
        device = "mps"
    else:
        device = "cpu"

    try:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    except Exception:
        device = "cpu"
        model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")

    # The library hardcodes max_new_tokens=1000 in generate(); a chunk that fails to
    # emit end-of-speech then grinds to that cap (slow, stretched audio, watchdog
    # kills). Patch the T3 inference to honour a per-chunk cap sized to the text.
    _cap = {"v": 1000, "last_tokens": -1}
    _orig_inference = model.t3.inference
    def _capped_inference(*a, **k):
        k["max_new_tokens"] = _cap["v"]
        out = _orig_inference(*a, **k)
        try:
            _cap["last_tokens"] = int(out.shape[-1])
        except Exception:
            _cap["last_tokens"] = -1
        return out
    model.t3.inference = _capped_inference

    response_q.put({"type": "ready", "device": device})

    def free():
        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    while True:
        # Self-terminate if orphaned (parent died) so we never linger holding the
        # model in memory — this is what caused runaway RAM use before.
        if os.getppid() == 1:
            break
        try:
            job = request_q.get(timeout=5)
        except queue.Empty:
            continue
        if job is None:  # shutdown sentinel
            break
        jid = job["job_id"]
        try:
            debug = bool(os.environ.get("TTS_DEBUG"))
            dbg_dir = os.path.join(OUTPUT_DIR, f"debug_{jid[:8]}") if debug else None
            if dbg_dir:
                os.makedirs(dbg_dir, exist_ok=True)
            manifest, cum = [], 0.0

            stitched = []
            chunks = job["chunks"]
            for i, (text, gap) in enumerate(chunks):
                # Cap tokens with headroom over what clean speech needs (~1.5/char),
                # so a chunk that ends naturally fits but a runaway is bounded.
                cap = min(1000, int(len(text) * 1.8) + 120)
                _cap["v"] = cap
                # Some chunks never emit end-of-speech and ramble (hallucinated
                # "spooky" audio) up to the cap. A cap-hit is a reliable signal of
                # that — retry with lower temperature (wanders less) and keep the
                # cleanest take that stops on its own.
                best_audio, best_tokens = None, 10 ** 9
                for temp in (0.8, 0.55, 0.4):
                    with torch.inference_mode():  # no autograd graph -> less memory
                        wav = model.generate(
                            text,
                            language_id=language,
                            audio_prompt_path=job["ref_path"],
                            exaggeration=job["exaggeration"],
                            cfg_weight=job["cfg_weight"],
                            temperature=temp,
                        )
                    a = wav.squeeze(0).detach().to("cpu").numpy()
                    del wav
                    free()
                    if _cap["last_tokens"] < best_tokens:
                        best_audio, best_tokens = a, _cap["last_tokens"]
                    if _cap["last_tokens"] < cap:   # ended naturally -> good take
                        break
                audio = best_audio
                # Declick the boundary with a tiny edge-fade — smooths the seam click
                # WITHOUT trimming speech or the model's natural pauses.
                n = min(int(model.sr * 0.015), audio.shape[0] // 2)
                if n > 0:
                    ramp = np.linspace(0.0, 1.0, n, dtype=audio.dtype)
                    audio[:n] *= ramp
                    audio[-n:] *= ramp[::-1]

                if dbg_dir:
                    dur = len(audio) / model.sr
                    tok, cap = _cap["last_tokens"], _cap["v"]
                    sf.write(os.path.join(dbg_dir, f"chunk_{i:02d}.wav"), audio, model.sr, format="WAV")
                    manifest.append(
                        f"{i:02d}  start={cum:6.1f}s  dur={dur:5.1f}s  chars={len(text):4d}  "
                        f"tokens={tok:4d}/{cap:<4d}{'  <-- CAP HIT' if tok >= cap else ''}\n"
                        f"      {text[:90]}"
                    )
                    cum += dur + (gap if (i < len(chunks) - 1 and gap > 0) else 0)

                stitched.append(audio)
                if i < len(chunks) - 1 and gap > 0:
                    stitched.append(np.zeros(int(model.sr * gap), dtype=audio.dtype))
                response_q.put({"type": "progress", "job_id": jid, "done": i + 1})

            if dbg_dir:
                with open(os.path.join(dbg_dir, "manifest.txt"), "w") as f:
                    f.write("\n".join(manifest) + "\n")

            audio = np.concatenate(stitched) if len(stitched) > 1 else stitched[0]
            buf = io.BytesIO()
            sf.write(buf, audio, model.sr, format="WAV")
            response_q.put({"type": "done", "job_id": jid, "audio": buf.getvalue()})
        except Exception as e:
            response_q.put({"type": "error", "job_id": jid, "error": str(e)})


# =============================================================================
# Main-process worker management + job bookkeeping.
# =============================================================================
_ctx = mp.get_context("spawn")  # required for torch/MPS in a child process
_worker_lock = threading.Lock()
_proc = None
_request_q = None
_worker_gen = 0          # bumped on every (re)spawn; old reader threads exit
_worker_ready = False
_device = None

# job_id -> {status, done, total, audio (bytes), error, voice}
_jobs = {}
_jobs_lock = threading.Lock()

# Hold a wake-lock while a job runs so the Mac doesn't idle-sleep mid-generation
# (sleep suspends the worker — a long job would otherwise drag across hours).
_awake = None


def _hold_awake():
    global _awake
    if _awake is not None and _awake.poll() is None:
        return
    try:
        import subprocess
        _awake = subprocess.Popen(["caffeinate", "-i"])  # prevent idle system sleep
    except Exception:
        _awake = None  # non-macOS or caffeinate missing — best effort


def _save_output(audio_bytes, voice):
    """Persist a finished generation to OUTPUT_DIR as <timestamp>_<voice>.wav."""
    try:
        import datetime
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(os.path.join(OUTPUT_DIR, f"{ts}_{voice}.wav"), "wb") as f:
            f.write(audio_bytes)
    except Exception:
        pass  # saving is best-effort; never fail the request over it


def _release_awake_if_idle():
    """Drop the wake-lock once no job is running."""
    global _awake
    with _jobs_lock:
        busy = any(j["status"] == "running" for j in _jobs.values())
    if not busy and _awake is not None and _awake.poll() is None:
        _awake.terminate()
        _awake = None


def _reader_loop(response_q, gen_id):
    """Drain one worker's response queue into job state until that worker is replaced."""
    global _worker_ready, _device
    while True:
        try:
            msg = response_q.get(timeout=0.5)
        except queue.Empty:
            if _worker_gen != gen_id:   # this worker was replaced; stop reading
                return
            continue
        if msg.get("type") == "ready":
            _device = msg["device"]
            _worker_ready = True
            continue
        jid = msg.get("job_id")
        with _jobs_lock:
            job = _jobs.get(jid)
            if not job or job["status"] != "running":
                continue  # job was cancelled/removed
            t = msg["type"]
            if t == "progress":
                job["done"] = msg["done"]
                job["last_progress"] = time.monotonic()  # reset the watchdog clock
            elif t == "done":
                job["audio"] = msg["audio"]
                job["status"] = "done"
                threading.Thread(target=_save_output, args=(msg["audio"], job["voice"]),
                                 daemon=True).start()
                threading.Thread(target=_release_awake_if_idle, daemon=True).start()
            elif t == "error":
                job["error"] = msg["error"]
                job["status"] = "error"
                threading.Thread(target=_release_awake_if_idle, daemon=True).start()


def _spawn_worker():
    """Start a fresh worker process + reader thread. Caller holds _worker_lock."""
    global _proc, _request_q, _worker_gen, _worker_ready, _device
    _worker_gen += 1
    _worker_ready = False
    _device = None
    _request_q = _ctx.Queue()
    response_q = _ctx.Queue()
    _proc = _ctx.Process(
        target=_worker_main,
        args=(_request_q, response_q, LANGUAGE, os.environ.get("TTS_DEVICE")),
        daemon=True,
    )
    _proc.start()
    threading.Thread(target=_reader_loop, args=(response_q, _worker_gen), daemon=True).start()


def _ensure_worker():
    if _proc is None or not _proc.is_alive():
        _spawn_worker()


def _kill_proc(proc):
    """Terminate a worker, escalating to SIGKILL so it can't linger holding the model."""
    if proc is None or not proc.is_alive():
        return
    proc.terminate()
    proc.join(timeout=5)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)


def _restart_worker():
    """Terminate the running worker (killing any in-flight generation) and respawn."""
    global _proc
    _kill_proc(_proc)
    _spawn_worker()


def _watchdog_loop():
    """Kill the worker if a chunk exceeds CHUNK_TIMEOUT (runaway-generation guard)."""
    while True:
        time.sleep(2)
        now = time.monotonic()
        with _jobs_lock:
            stuck = [jid for jid, j in _jobs.items()
                     if j["status"] == "running" and now - j["last_progress"] > CHUNK_TIMEOUT]
        for jid in stuck:
            with _worker_lock:
                with _jobs_lock:
                    job = _jobs.get(jid)
                    if not job or job["status"] != "running":
                        continue
                    job["status"] = "error"
                    job["error"] = ("Zeitüberschreitung: Ein Abschnitt hat zu lange "
                                    "gedauert und wurde abgebrochen.")
                _restart_worker()


def _split_text(text, max_chars):
    """Split text into (chunk_text, gap_after_seconds) chunks for generation.

    The model pauses naturally at punctuation, so rather than splitting on every
    paragraph (which makes short dialogue lines choppy), we pack sentences greedily
    up to max_chars — but ensure each paragraph ends with terminal punctuation, so a
    heading/title with none isn't run straight into the next sentence. Overlong
    sentences are hard-split on word boundaries; the worker's token cap keeps any
    single chunk from rambling.
    """
    import re
    # Sentence units, with a period added to punctuation-less paragraph ends (titles).
    units = []
    for para in re.split(r"\n\s*\n+", text.strip()):
        para = " ".join(para.split())
        if not para:
            continue
        sents = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", para) if s.strip()]
        if sents and sents[-1][-1] not in ".!?…:\"'»":
            sents[-1] += "."
        units.extend(sents)

    # Hard-split overlong sentences, then pack units greedily up to max_chars.
    atoms = []
    for u in units:
        if len(u) <= max_chars:
            atoms.append(u)
            continue
        buf = ""
        for w in u.split():
            if buf and len(buf) + len(w) + 1 > max_chars:
                atoms.append(buf)
                buf = ""
            buf = f"{buf} {w}".strip()
        if buf:
            atoms.append(buf)

    chunks, cur = [], ""
    for a in atoms:
        if not cur:
            cur = a
        elif len(cur) + len(a) + 1 <= max_chars:
            cur = f"{cur} {a}"
        else:
            chunks.append(cur)
            cur = a
    if cur:
        chunks.append(cur)

    out = [[c, SENT_GAP] for c in chunks] or [[text.strip(), 0.0]]
    out[-1][1] = 0.0  # no trailing gap on the very last chunk
    return out


def _voice_by_id(voice_id):
    for v in VOICES:
        if v["id"] == voice_id:
            return v
    return None


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/voices")
def voices():
    return jsonify([{k: v[k] for k in ("id", "name", "gender", "style")} for v in VOICES])


@app.route("/api/synthesize", methods=["POST"])
def synthesize():
    data = request.json or {}
    text = data.get("text", "").strip()
    voice_id = data.get("voice") or (VOICES[0]["id"] if VOICES else None)
    # Chatterbox knobs. exaggeration ~ expressiveness; cfg_weight ~ pacing/fidelity.
    exaggeration = float(data.get("exaggeration", 0.5))
    cfg_weight = float(data.get("cfg_weight", 0.5))

    if not text:
        return jsonify({"error": "No text provided"}), 400

    voice = _voice_by_id(voice_id)
    if voice is None:
        return jsonify({"error": f"Unknown voice: {voice_id}"}), 400

    ref_path = os.path.join(VOICES_DIR, voice["file"])
    if not os.path.exists(ref_path):
        return jsonify({"error": f"Reference clip missing: {voice['file']}"}), 500

    chunks = _split_text(text, MAX_CHARS)
    job_id = uuid.uuid4().hex
    with _worker_lock:
        with _jobs_lock:
            if any(j["status"] == "running" for j in _jobs.values()):
                return jsonify({"error": "Es läuft bereits eine Synthese."}), 409
            _jobs[job_id] = {"status": "running", "done": 0, "total": len(chunks),
                             "audio": None, "error": None, "voice": voice_id,
                             "last_progress": time.monotonic()}
        _ensure_worker()
        _hold_awake()  # keep the Mac awake for the duration of this job
        _request_q.put({
            "job_id": job_id, "chunks": chunks, "ref_path": ref_path,
            "exaggeration": exaggeration, "cfg_weight": cfg_weight,
        })
    return jsonify({"job_id": job_id, "total": len(chunks)})


@app.route("/api/progress/<job_id>")
def progress(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return jsonify({"error": "Unknown job"}), 404
        return jsonify({k: job[k] for k in ("status", "done", "total", "error")})


@app.route("/api/cancel/<job_id>", methods=["POST"])
def cancel(job_id):
    """Hard-cancel: mark the job cancelled and kill+respawn the worker."""
    with _worker_lock:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job and job["status"] == "running":
                job["status"] = "cancelled"
        _restart_worker()  # terminating the process stops any in-flight generation
    _release_awake_if_idle()
    return jsonify({"ok": True})


@app.route("/api/audio/<job_id>")
def audio(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return jsonify({"error": "Unknown job"}), 404
        if job["status"] != "done":
            return jsonify({"error": "Not ready", "status": job["status"]}), 409
        data = job["audio"]
        voice_id = job["voice"]
        del _jobs[job_id]  # one-shot: free memory once delivered
    return send_file(io.BytesIO(data), mimetype="audio/wav", as_attachment=False,
                     download_name=f"{voice_id}.wav")


@app.route("/api/status")
def status():
    return jsonify({"ok": _worker_ready, "device": _device,
                    "loading": not _worker_ready, "voices": len(VOICES)})


if __name__ == "__main__":
    import atexit
    print("\n🎙  German TTS (Chatterbox Multilingual) at http://localhost:5050")

    def _shutdown():
        _kill_proc(_proc)  # never orphan the worker on exit
        if _awake is not None and _awake.poll() is None:
            _awake.terminate()
    atexit.register(_shutdown)
    threading.Thread(target=_watchdog_loop, daemon=True).start()
    if os.environ.get("TTS_PRELOAD", "1") == "1":
        # Warm the worker (and model) at boot so the first request isn't slow.
        with _worker_lock:
            _ensure_worker()
    try:
        app.run(port=5050, debug=False, threaded=True)
    finally:
        _shutdown()
