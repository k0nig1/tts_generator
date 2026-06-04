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
# Safety net: if a single chunk takes longer than this (seconds), assume the model
# is running away to its token cap, kill the worker and fail the job. Generous
# enough for a slow-but-normal chunk; a runaway hits the 1000-token cap (~10 min).
CHUNK_TIMEOUT = int(os.environ.get("TTS_CHUNK_TIMEOUT", "240"))

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

    if forced_device:
        device = forced_device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    try:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    except Exception:
        device = "cpu"
        model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    response_q.put({"type": "ready", "device": device})

    def free():
        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    while True:
        job = request_q.get()
        if job is None:  # shutdown sentinel
            break
        jid = job["job_id"]
        try:
            pieces = []
            for i, chunk in enumerate(job["chunks"]):
                with torch.inference_mode():  # no autograd graph -> far less memory
                    wav = model.generate(
                        chunk,
                        language_id=language,
                        audio_prompt_path=job["ref_path"],
                        exaggeration=job["exaggeration"],
                        cfg_weight=job["cfg_weight"],
                    )
                pieces.append(wav.squeeze(0).detach().to("cpu").numpy())
                del wav
                free()
                response_q.put({"type": "progress", "job_id": jid, "done": i + 1})

            # Stitch chunks with a short silence so sentences don't run together.
            if len(pieces) > 1:
                gap = np.zeros(int(model.sr * 0.2), dtype=pieces[0].dtype)
                stitched = []
                for i, p in enumerate(pieces):
                    stitched.append(p)
                    if i < len(pieces) - 1:
                        stitched.append(gap)
                audio = np.concatenate(stitched)
            else:
                audio = pieces[0]

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
            elif t == "error":
                job["error"] = msg["error"]
                job["status"] = "error"


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


def _restart_worker():
    """Terminate the running worker (killing any in-flight generation) and respawn."""
    global _proc
    if _proc is not None and _proc.is_alive():
        _proc.terminate()
        _proc.join(timeout=5)
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
    """Split text into chunks on sentence/line boundaries, avoiding tiny fragments.

    Generation memory grows with input length, so long inputs must be chunked to
    stay within the device memory budget. But the model tends to *ramble* (sampling
    toward its token cap) when handed a short fragment like a title or a one-line
    paragraph — so we merge sub-`min_chars` pieces into a neighbour, allowing a
    chunk to grow up to `hard_max` to absorb them. Sentences longer than a whole
    chunk are hard-split on word boundaries.
    """
    import re
    hard_max = int(max_chars * 1.3)          # ceiling when absorbing a short fragment
    min_chars = max(40, max_chars // 6)      # anything shorter is a "tiny" fragment
    parts = re.split(r"(?<=[.!?…])\s+|\n+", text.strip())

    # 1) Break parts that are longer than a whole chunk into word-bounded atoms.
    atoms = []
    for p in (s.strip() for s in parts):
        if not p:
            continue
        if len(p) <= max_chars:
            atoms.append(p)
            continue
        buf = ""
        for w in p.split():
            if buf and len(buf) + len(w) + 1 > max_chars:
                atoms.append(buf)
                buf = ""
            buf = f"{buf} {w}".strip()
        if buf:
            atoms.append(buf)

    # 2) Greedily pack atoms; absorb extra into a still-too-short chunk.
    chunks, cur = [], ""
    for a in atoms:
        if not cur:
            cur = a
        elif (len(cur) + len(a) + 1 <= max_chars
              or (len(cur) < min_chars and len(cur) + len(a) + 1 <= hard_max)):
            cur = f"{cur} {a}"
        else:
            chunks.append(cur)
            cur = a
    if cur:
        chunks.append(cur)

    # 3) Fold any leftover tiny chunk into an adjacent one (e.g. a trailing line).
    out = []
    for c in chunks:
        if out and (len(c) < min_chars or len(out[-1]) < min_chars) \
                and len(out[-1]) + len(c) + 1 <= hard_max:
            out[-1] = f"{out[-1]} {c}"
        else:
            out.append(c)
    return out or [text.strip()]


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
    print("\n🎙  German TTS (Chatterbox Multilingual) at http://localhost:5050")
    threading.Thread(target=_watchdog_loop, daemon=True).start()
    if os.environ.get("TTS_PRELOAD", "1") == "1":
        # Warm the worker (and model) at boot so the first request isn't slow.
        with _worker_lock:
            _ensure_worker()
    app.run(port=5050, debug=False, threaded=True)
