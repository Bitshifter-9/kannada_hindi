# Kannada → Hindi Video Dubbing Pipeline

A fully automated, open-source pipeline that converts Kannada video to Hindi-dubbed output with natural voice cloning, lip sync, and face restoration — **100% local, zero ongoing cost**.

Built for **Apple Silicon (M1/M2/M3)** using PyTorch MPS. No CUDA required.

---

## Features

- 🎙️ **Voice cloning** — XTTS v2 captures the original speaker's timbre
- 🌐 **High-quality translation** — Helsinki-NLP Kannada→Hindi with LLM post-edit via Ollama
- 🎬 **Lip sync** — Wav2Lip aligns mouth movement to Hindi speech
- ✨ **Face restoration** — Real-ESRGAN sharpens every frame
- ✅ **Auto-validation** — checks duration, SSIM, PESQ; retries if needed
- 🔇 **No silent gaps** — single continuous audio waveform, never segment-stitched

---

## Quick Start

### 1. Clone & setup

```bash
git clone <your-repo-url> kannada-hindi-dubbing
cd kannada-hindi-dubbing

conda create -n dubbing python=3.11 -y
conda activate dubbing

bash setup.sh
```

### 2. Run

```bash
python dub_video.py --input input.mp4 --start 15 --end 30
```

Output: `outputs/final_dubbed.mp4`

---

## Pipeline Stages

```
input.mp4
   │
   ├─ 1. extractor.py   → outputs/clip.mp4 + clip.wav
   ├─ 2. asr.py         → outputs/segments.json  (Whisper large-v2, word timestamps)
   ├─ 3. translate.py   → outputs/translated_segments.json  (Helsinki-NLP + Ollama)
   ├─ 4. tts.py         → outputs/hindi_tts.wav  (XTTS v2 voice clone, time-stretched)
   ├─ 5. audio.py       → outputs/hindi_audio.wav  (loudness-normalised, -14 LUFS)
   ├─ 6. lipsync.py     → outputs/lipsynced.mp4  (Wav2Lip)
   ├─ 7. restore.py     → outputs/restored.mp4   (Real-ESRGAN x2)
   └─ 8. validate.py    → pass/fail → retry up to 3×
              │
              └─ outputs/final_dubbed.mp4  ✅
```

---

## Model Stack

| Stage | Model | Notes |
|---|---|---|
| ASR | `whisper-timestamped large-v2` | MPS accelerated |
| Translation | `Helsinki-NLP/opus-mt-kn-hi` | + Ollama `llama3.2:3b` post-edit |
| TTS | `Coqui XTTS v2` | Voice cloning, MPS |
| Lip sync | `Wav2Lip GAN` | CPU/MPS compatible |
| Restoration | `Real-ESRGAN x2plus` | MPS compatible |

---

## Scaling to 500 Hours of Content

| Layer | Approach |
|---|---|
| **Parallelism** | Spin up N worker processes (1 per GPU/machine) using Python `multiprocessing` |
| **Batching** | WhisperX batch mode; XTTS batched synthesis for ASR/TTS throughput |
| **Cloud** | AWS SageMaker `ml.g5.xlarge` (A10G GPU) for CUDA-accelerated Wav2Lip at scale |
| **Storage** | S3 bucket for input videos + outputs; signed URLs for delivery |
| **Orchestration** | Airflow or Prefect DAG — one task per pipeline stage, retries built-in |
| **Cost estimate** | ~$0.20/hr per SageMaker instance × ~5 min/video = ~$0.017/video dubbed |

---

## Requirements

- macOS 13+ with Apple Silicon (M1/M2/M3)
- `conda` (Python 3.11 recommended)
- `ffmpeg` — `brew install ffmpeg`
- `ollama` (optional, for LLM post-edit) — [ollama.com](https://ollama.com)

---

## License

MIT
