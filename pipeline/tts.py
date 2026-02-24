import gc
import json
from pathlib import Path

import torch
import librosa
import soundfile as sf
import numpy as np
from TTS.api import TTS


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def _build_full_hindi_text(translated_path):
    with open(translated_path, encoding="utf-8") as f:
        segments = json.load(f)
    return " ".join(s["hindi"] for s in segments), segments


def _time_stretch_to_duration(audio, sr, target_duration):
    current_duration = len(audio) / sr
    if abs(current_duration - target_duration) < 0.05:
        return audio
    rate = current_duration / target_duration
    rate = max(0.7, min(rate, 1.5))
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    target_samples = int(target_duration * sr)
    if len(stretched) >= target_samples:
        return stretched[:target_samples]
    padded = np.zeros(target_samples, dtype=np.float32)
    padded[:len(stretched)] = stretched
    return padded


def synthesise(translated_path, clip_audio_path, clip_duration, out_dir="outputs"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = str(out / "hindi_tts.wav")

    full_text, _ = _build_full_hindi_text(translated_path)

    print(f"[tts] Loading XTTS v2 on {DEVICE} ...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)

    ref_audio, ref_sr = librosa.load(clip_audio_path, sr=22050, mono=True, duration=6.0)
    ref_tmp = str(out / "ref_speaker.wav")
    sf.write(ref_tmp, ref_audio, ref_sr)

    print(f"[tts] Synthesising Hindi text ({len(full_text)} chars) ...")
    tts.tts_to_file(
        text=full_text,
        speaker_wav=ref_tmp,
        language="hi",
        file_path=out_path,
    )

    audio, sr = librosa.load(out_path, sr=None, mono=True)
    audio = _time_stretch_to_duration(audio, sr, clip_duration)
    sf.write(out_path, audio, sr)

    del tts
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    actual_dur = len(audio) / sr
    print(f"[tts] Done — {actual_dur:.2f}s (target {clip_duration}s) → {out_path}")
    return out_path
