import subprocess
from pathlib import Path


def merge_audio_video(video_path, audio_path, out_dir="outputs"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = str(out / "final_dubbed.mp4")

    video_path = str(Path(video_path).resolve())
    audio_path = str(Path(audio_path).resolve())

    print(f"[merge] Overlaying Hindi audio on video ...")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            out_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print(f"[merge] Done → {out_path}")
    return out_path
