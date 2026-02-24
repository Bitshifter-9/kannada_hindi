import json
import gc
from pathlib import Path

import torch
import whisper_timestamped as whisper_ts


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def transcribe(audio_path: str, out_dir: str = "outputs") -> str:

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    segments_path = str(out / "segments.json")

    print(f"[asr] Loading whisper large-v2 on {DEVICE} ...")
    model = whisper_ts.load_model("large-v2", device=DEVICE)

    print(f"[asr] Transcribing {audio_path} ...")
    audio = whisper_ts.load_audio(audio_path)
    result = whisper_ts.transcribe(
        model,
        audio,
        language="kn",         
        detect_disfluencies=False,
        vad=False,              
    )
    segments = []
    for seg in result["segments"]:
        segments.append(
            {
                "text": seg["text"].strip(),
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "words": [
                    {
                        "word": w["text"].strip(),
                        "start": round(w["start"], 3),
                        "end": round(w["end"], 3),
                    }
                    for w in seg.get("words", [])
                ],
            }
        )
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    del model
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    print(f"[asr] Done — {len(segments)} segments → {segments_path}")
    return segments_path
