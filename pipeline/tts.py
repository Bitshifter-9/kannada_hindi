import json
import asyncio
from pathlib import Path

import edge_tts
import librosa
import soundfile as sf
import numpy as np


VOICE_FEMALE = "hi-IN-SwaraNeural"
VOICE_MALE = "hi-IN-MadhurNeural"


def _time_stretch_to_duration(audio, sr, target_duration):
    current_duration = len(audio) / sr
    if abs(current_duration - target_duration) < 0.05:
        return audio
    rate = current_duration / target_duration
    rate = max(0.75, min(rate, 1.4))
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    target_samples = int(target_duration * sr)
    if len(stretched) >= target_samples:
        return stretched[:target_samples]
    padded = np.zeros(target_samples, dtype=np.float32)
    padded[:len(stretched)] = stretched
    return padded


async def _synthesize(text, output_path, voice):
    communicate = edge_tts.Communicate(text, voice, rate="+0%", pitch="+0Hz")
    await communicate.save(output_path)


def synthesise(translated_path, clip_duration, out_dir="outputs", voice="male"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = str(out / "hindi_tts_raw.wav")
    out_path = str(out / "hindi_tts.wav")

    with open(translated_path, encoding="utf-8") as f:
        segments = json.load(f)
    hindi_text = " ".join(s["hindi"] for s in segments)

    selected_voice = VOICE_MALE if voice == "male" else VOICE_FEMALE
    print(f"[tts] Synthesising with {selected_voice} ...")
    print(f"[tts] Text: {hindi_text[:100]}")

    asyncio.run(_synthesize(hindi_text, raw_path, selected_voice))

    audio, sr = librosa.load(raw_path, sr=None, mono=True)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    audio = _time_stretch_to_duration(audio, sr, clip_duration)
    sf.write(out_path, audio, sr)

    actual_dur = len(audio) / sr
    print(f"[tts] Done — {actual_dur:.2f}s (target {clip_duration}s) → {out_path}")
    return out_path
