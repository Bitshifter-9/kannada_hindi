# Kannada → Hindi Video Dubbing Pipeline

A high-fidelity open-source dubbing system. Takes a Kannada video and produces a lip-synced Hindi dubbed version using voice cloning.

## Stack
| Stage | Tool |
|---|---|
| Transcription | WhisperX large-v3 |
| Translation | IndicTrans2 / Helsinki-NLP fallback |
| Voice Cloning | Coqui XTTS v2 |
| Lip Sync | MuseTalk / VideoReTalking fallback |
| Face Restoration | CodeFormer |

## Setup

```bash
chmod +x setup.sh
./setup.sh
```

## Usage

```bash
source .venv/bin/activate
python dub_video.py --input video.mp4 --start 15 --end 30 --output final.mp4
```

## Project Structure

```
kannada-hindi-dubbing/
├── dub_video.py       # main entry point
├── pipeline/
│   ├── extractor.py   # clip + audio extraction
│   ├── asr.py         # WhisperX transcription
│   ├── translate.py   # Kannada → Hindi
│   ├── tts.py         # voice cloning + synthesis
│   ├── audio.py       # audio assembly + normalisation
│   ├── lipsync.py     # lip sync
│   ├── restore.py     # face restoration
│   └── validate.py    # quality checks + retry
├── requirements.txt
└── setup.sh
```

## Scaling to 500 Hours of Video
- Batch processing via a job queue (Celery + Redis)
- Parallel GPU workers (one worker per GPU)
- Cloud storage for intermediate artifacts (AWS S3)
- Progress tracking dashboard per video job
