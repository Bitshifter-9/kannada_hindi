import subprocess
import shutil
from pathlib import Path

import cv2
import torch
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
WEIGHTS_PATH = "models/RealESRGAN_x2plus.pth"


def _build_upsampler():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    upsampler = RealESRGANer(
        scale=2,
        model_path=WEIGHTS_PATH,
        model=model,
        tile=256,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=DEVICE,
    )
    return upsampler


def restore(video_path, out_dir="outputs"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"
    restored_frames_dir = out / "frames_restored"
    frames_dir.mkdir(exist_ok=True)
    restored_frames_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_paths = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        p = str(frames_dir / f"{idx:05d}.png")
        cv2.imwrite(p, frame)
        frame_paths.append(p)
        idx += 1
    cap.release()

    print(f"[restore] {len(frame_paths)} frames extracted. Loading Real-ESRGAN ...")
    upsampler = _build_upsampler()

    for i, fp in enumerate(frame_paths):
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        restored, _ = upsampler.enhance(img, outscale=1)
        restored_resized = cv2.resize(restored, (width, height), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(restored_frames_dir / f"{i:05d}.png"), restored_resized)
        if i % 30 == 0:
            print(f"[restore] Frame {i}/{len(frame_paths)}")

    out_video = str(out / "restored_noaudio.mp4")
    final_out = str(out / "restored.mp4")

    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(fps),
         "-i", str(restored_frames_dir / "%05d.png"),
         "-c:v", "libx264", "-crf", "16", "-preset", "slow",
         "-pix_fmt", "yuv420p", out_video],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    subprocess.run(
        ["ffmpeg", "-y", "-i", out_video, "-i", video_path,
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
         "-shortest", final_out],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    shutil.rmtree(frames_dir)
    shutil.rmtree(restored_frames_dir)

    print(f"[restore] Done → {final_out}")
    return final_out
