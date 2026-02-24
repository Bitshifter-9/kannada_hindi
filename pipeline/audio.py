from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln


def merge_and_master(tts_path, original_audio_path, clip_duration, out_dir="outputs"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = str(out / "hindi_audio.wav")

    tts_audio, sr = librosa.load(tts_path, sr=None, mono=True)
    orig_audio, _ = librosa.load(original_audio_path, sr=sr, mono=True)

    target_samples = int(clip_duration * sr)
    timeline = np.zeros(target_samples, dtype=np.float32)

    tts_samples = min(len(tts_audio), target_samples)
    timeline[:tts_samples] += tts_audio[:tts_samples]

    orig_samples = min(len(orig_audio), target_samples)
    bg = orig_audio[:orig_samples] * 0.08
    timeline[:orig_samples] += bg

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(timeline.astype(np.float64))
    normalized = pyln.normalize.loudness(timeline.astype(np.float64), loudness, -14.0)
    normalized = np.clip(normalized, -1.0, 1.0).astype(np.float32)

    sf.write(out_path, normalized, sr)
    print(f"[audio] Done — {clip_duration:.2f}s @ -14 LUFS → {out_path}")
    return out_path
