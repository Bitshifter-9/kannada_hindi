#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-command setup for the Kannada→Hindi dubbing pipeline
# Compatible with Mac M3 Pro (Apple Silicon / MPS)
# =============================================================================
set -e

BOLD="\033[1m"
RESET="\033[0m"
GREEN="\033[32m"
YELLOW="\033[33m"

log()  { echo -e "${BOLD}${GREEN}[setup]${RESET} $*"; }
warn() { echo -e "${BOLD}${YELLOW}[warn ]${RESET} $*"; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$ROOT/models"
OUTPUTS_DIR="$ROOT/outputs"

mkdir -p "$MODELS_DIR" "$OUTPUTS_DIR" "$OUTPUTS_DIR/tts_segments"

# ── 1. Python dependencies ────────────────────────────────────────────────────
log "Installing Python dependencies..."
pip install -r "$ROOT/requirements.txt" --quiet

# ── 2. Wav2Lip ────────────────────────────────────────────────────────────────
WAV2LIP_DIR="$MODELS_DIR/Wav2Lip"
if [ ! -d "$WAV2LIP_DIR" ]; then
    log "Cloning Wav2Lip..."
    git clone --depth 1 https://github.com/Rudrabha/Wav2Lip.git "$WAV2LIP_DIR"
fi

WAV2LIP_WEIGHTS="$MODELS_DIR/wav2lip_gan.pth"
if [ ! -f "$WAV2LIP_WEIGHTS" ]; then
    log "Downloading Wav2Lip GAN weights (~272 MB)..."
    # Wav2Lip GAN checkpoint (public Google Drive)
    gdown --id 1P4ifER73FK20NHRxogmSKnDSjQLFKVWX \
          --output "$WAV2LIP_WEIGHTS"
fi

# Face detection model (s3fd) used by Wav2Lip
FACE_DET_DIR="$WAV2LIP_DIR/face_detection/detection/sfd"
mkdir -p "$FACE_DET_DIR"
FACE_DET_WEIGHTS="$FACE_DET_DIR/s3fd.pth"
if [ ! -f "$FACE_DET_WEIGHTS" ]; then
    log "Downloading face detection weights (~86 MB)..."
    gdown --id 1ZjQLit8mvRvkTfZbYK3HMfPQ6SBRxjn6 \
          --output "$FACE_DET_WEIGHTS"
fi

# ── 3. Real-ESRGAN weights ────────────────────────────────────────────────────
REALESRGAN_WEIGHTS="$MODELS_DIR/RealESRGAN_x2plus.pth"
if [ ! -f "$REALESRGAN_WEIGHTS" ]; then
    log "Downloading Real-ESRGAN x2plus weights (~67 MB)..."
    curl -L \
      "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" \
      -o "$REALESRGAN_WEIGHTS" --progress-bar
fi

# ── 4. Ollama + llama3.2:3b ───────────────────────────────────────────────────
if command -v ollama &> /dev/null; then
    log "Pulling Ollama model llama3.2:3b (for translation post-edit)..."
    ollama pull llama3.2:3b
else
    warn "Ollama not found. Install from https://ollama.com — then run: ollama pull llama3.2:3b"
    warn "The pipeline will fall back to Helsinki-NLP translation without LLM post-editing."
fi

# ── 5. ffmpeg check ───────────────────────────────────────────────────────────
if ! command -v ffmpeg &> /dev/null; then
    warn "ffmpeg not found. Install with: brew install ffmpeg"
    exit 1
fi

log "✅  Setup complete! Run the pipeline with:"
echo ""
echo "  python dub_video.py --input input.mp4 --start 15 --end 30"
echo ""
