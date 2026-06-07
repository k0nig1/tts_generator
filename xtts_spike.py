"""Spike: try XTTS-v2 on the same German chunks, cloning Helmut. Compare to Chatterbox.

Generates each chunk, saves to xtts_out/, and reports per-chunk timing + a crude
artifact check (sustained low-energy/low-pitch drone regions). No token cap needed —
XTTS is a different architecture; this is purely an exploratory quality/stability test.
"""
import os, time, glob
import numpy as np
import soundfile as sf

# Same reference + text region we've been testing on Chatterbox.
REF = "voices/helmut_calm.wav"
CHUNKS = [
    "Das Signal aus Sektor Neun, Teil zwei: Die Entscheidung.",
    ("Die nächsten Stunden vergingen in einem Zustand angespannter Stille, in dem jedes "
     "Geräusch wie ein Alarmsignal wirkte. Lena und Okafor hatten sich zurückgezogen."),
    ('"Und doch." Lena lehnte sich zurück und rieb sich die Schläfen. "Die Frage ist nicht '
     'mehr, woher das Signal kommt, sondern was Becker geantwortet hat."'),
    ("Okafor sah sie schief an. Die Gegenseite. Hast du ein besseres Wort? Er hatte keines."),
]

os.makedirs("xtts_out", exist_ok=True)
from TTS.api import TTS

t0 = time.time()
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")  # CPU by default here
print(f"model loaded in {time.time()-t0:.0f}s")


def drone_secs(y, sr):
    """Crude: seconds of sustained low-energy + low-pitch (drone-like) audio."""
    import librosa
    hop = int(sr * 0.1)
    rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=hop)[0]
    norm = rms / (rms.max() + 1e-9)
    return float(np.sum((norm < 0.3) & (cent < 1500)) * 0.1)


for i, text in enumerate(CHUNKS):
    t = time.time()
    out = f"xtts_out/chunk_{i:02d}.wav"
    tts.tts_to_file(text=text, file_path=out, speaker_wav=REF, language="de")
    y, sr = sf.read(out)
    if y.ndim > 1:
        y = y.mean(1)
    dur = len(y) / sr
    print(f"chunk {i}: {len(text):3d} chars -> {dur:5.1f}s in {time.time()-t:4.0f}s "
          f"(RTF {(time.time()-t)/dur:.2f})  drone~{drone_secs(np.asarray(y),sr):.1f}s -> {out}")
print("done. listen to xtts_out/chunk_*.wav")
