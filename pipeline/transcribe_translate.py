import json
import os
from pathlib import Path

import google.generativeai as genai


def transcribe_and_translate(audio_path, out_dir="outputs"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    segments_path = str(out / "translated_segments.json")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY environment variable")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    print(f"[transcribe] Uploading {audio_path} to Gemini ...")
    audio_file = genai.upload_file(str(audio_path), mime_type="audio/wav")

    prompt = """Listen to this Kannada audio carefully.

1. Transcribe the exact Kannada speech
2. Translate it to natural, conversational Hindi

Return ONLY valid JSON in this exact format:
{"kannada": "...", "hindi": "..."}

No explanation, no markdown, just the JSON."""

    print("[transcribe] Waiting for Gemini response ...")
    response = model.generate_content([prompt, audio_file])
    text = response.text.strip()

    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    result = json.loads(text)
    kannada = result["kannada"]
    hindi = result["hindi"]

    print(f"[transcribe] Kannada: {kannada}")
    print(f"[transcribe] Hindi:   {hindi}")

    segments = [{
        "start": 0.0,
        "end": 15.0,
        "source": kannada,
        "hindi": hindi,
    }]

    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    audio_file.delete()
    print(f"[transcribe] Done → {segments_path}")
    return segments_path
