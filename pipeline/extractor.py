import subprocess
from pathlib import Path

def extract_clip(
    source: str,
    start: float,
    end: float,
    out_dir: str = "outputs",
) -> tuple[str, str]:

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    video_out = str(out / "clip.mp4")
    audio_out = str(out / "clip.wav")
    duration = end - start
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", source,
            "-t", str(duration),
            "-c", "copy",
            video_out,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_out,
            "-ar", "16000",
            "-ac", "1",
            "-vn",
            audio_out,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print(f"[extractor] clip  → {video_out}")
    print(f"[extractor] audio → {audio_out}")
    return video_out, audio_out
