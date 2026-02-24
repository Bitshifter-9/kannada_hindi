import json
import gc
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_NAME = "Helsinki-NLP/opus-mt-kn-hi"


def _load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    return tokenizer, model


def _translate_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def _ollama_shorten(text, max_chars):
    try:
        import subprocess
        prompt = f"Shorten this Hindi sentence to under {max_chars} characters while keeping the full meaning. Return only the shortened sentence:\n{text}"
        result = subprocess.run(
            ["ollama", "run", "llama3.2:3b", prompt],
            capture_output=True, text=True, timeout=30
        )
        shortened = result.stdout.strip()
        if shortened and len(shortened) < len(text):
            return shortened
    except Exception:
        pass
    return text


def translate(segments_path, out_dir="outputs"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = str(out / "translated_segments.json")

    with open(segments_path, encoding="utf-8") as f:
        segments = json.load(f)

    print(f"[translate] Loading {MODEL_NAME} on {DEVICE} ...")
    tokenizer, model = _load_model()

    translated = []
    for seg in segments:
        src_text = seg["text"]
        hi_text = _translate_text(src_text, tokenizer, model)

        src_len = len(src_text)
        if len(hi_text) > src_len * 1.15:
            hi_text = _ollama_shorten(hi_text, int(src_len * 1.1))

        translated.append({
            "start": seg["start"],
            "end": seg["end"],
            "source": src_text,
            "hindi": hi_text,
        })
        print(f"[translate] {src_text[:40]!r} → {hi_text[:40]!r}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)

    del model, tokenizer
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    print(f"[translate] Done — {len(translated)} segments → {out_path}")
    return out_path
