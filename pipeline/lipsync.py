import subprocess
import sys
from pathlib import Path


WAV2LIP_DIR = Path("models/Wav2Lip")
WAV2LIP_WEIGHTS = Path("models/wav2lip_gan.pth")


def run_lipsync(video_path, audio_path, out_dir="outputs"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = str(out / "lipsynced.mp4")

    if not WAV2LIP_DIR.exists():
        raise FileNotFoundError("Wav2Lip not found. Run setup.sh first.")

    print(f"[lipsync] Running Wav2Lip ...")
    cmd = [
        sys.executable,
        str(WAV2LIP_DIR / "inference.py"),
        "--checkpoint_path", str(WAV2LIP_WEIGHTS),
        "--face", video_path,
        "--audio", audio_path,
        "--outfile", out_path,
        "--pads", "0", "10", "0", "0",
        "--resize_factor", "1",
        "--nosmooth",
    ]

    result = subprocess.run(cmd, cwd=str(WAV2LIP_DIR), capture_output=False)
    if result.returncode != 0:
        raise RuntimeError("Wav2Lip inference failed. Check the output above.")

    print(f"[lipsync] Done → {out_path}")
    return out_path
