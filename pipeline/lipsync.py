import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WAV2LIP_DIR = ROOT / "models" / "Wav2Lip"
WAV2LIP_WEIGHTS = ROOT / "models" / "wav2lip_gan.pth"


def _get_fps(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", video_path],
        capture_output=True, text=True
    )
    num, den = result.stdout.strip().split("/")
    return float(num) / float(den)


def run_lipsync(video_path, audio_path, out_dir="outputs"):
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not WAV2LIP_DIR.exists():
        raise FileNotFoundError("Wav2Lip not found. Run setup.sh first.")

    orig_fps = _get_fps(str(Path(video_path).resolve()))
    target_fps = min(orig_fps, 25.0)

    resampled = str(out / "clip_25fps.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(Path(video_path).resolve()),
         "-vf", f"fps={target_fps}", "-c:v", "libx264", "-crf", "16",
         "-c:a", "aac", resampled],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    wav2lip_out = str(out / "lipsynced_25fps.mp4")
    out_path = str(out / "lipsynced.mp4")

    print(f"[lipsync] Running Wav2Lip on {target_fps:.0f}fps clip ...")
    cmd = [
        sys.executable,
        str(WAV2LIP_DIR / "inference.py"),
        "--checkpoint_path", str(WAV2LIP_WEIGHTS),
        "--face", resampled,
        "--audio", str(Path(audio_path).resolve()),
        "--outfile", wav2lip_out,
        "--pads", "0", "10", "0", "0",
        "--resize_factor", "1",
        "--nosmooth",
    ]

    result = subprocess.run(cmd, cwd=str(WAV2LIP_DIR), capture_output=False)
    if result.returncode != 0:
        raise RuntimeError("Wav2Lip inference failed.")

    subprocess.run(
        ["ffmpeg", "-y", "-i", wav2lip_out,
         "-vf", f"fps={orig_fps}", "-c:v", "libx264", "-crf", "16",
         "-c:a", "aac", out_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    print(f"[lipsync] Done → {out_path}")
    return out_path
