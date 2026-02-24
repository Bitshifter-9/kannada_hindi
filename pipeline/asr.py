import json
import gc
from pathlib import Path

import torch
import whisper
import numpy as np


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def transcribe(audio_path, out_dir="outputs"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    segments_path = str(out / "segments.json")

    print(f"[asr] Loading whisper medium on {DEVICE} ...")
    model = whisper.load_model("medium", device=DEVICE)

    print(f"[asr] Transcribing {audio_path} ...")
    result = model.transcribe(
        str(audio_path),
        language="kn",
        word_timestamps=False,
        condition_on_previous_text=False,
        no_speech_threshold=0.3,
        logprob_threshold=-1.5,
        compression_ratio_threshold=2.8,
        fp16=False,
    )

    segments = []
    for seg in result["segments"]:
        words = []
        for w in seg.get("words", []):
            words.append({
                "word": w["word"].strip(),
                "start": round(w["start"], 3),
                "end": round(w["end"], 3),
            })
        segments.append({
            "text": seg["text"].strip(),
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "words": words,
        })

    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    del model
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    full_text = " ".join(s["text"] for s in segments)
    print(f"[asr] {len(segments)} segments, last end={segments[-1]['end'] if segments else 0}s")
    print(f"[asr] Full text: {full_text[:120]}")
    return segments_path
