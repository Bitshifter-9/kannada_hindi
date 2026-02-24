import argparse
import subprocess
from pathlib import Path

from pipeline.extractor import extract_clip
from pipeline.asr import transcribe
from pipeline.translate import translate
from pipeline.tts import synthesise
from pipeline.audio import merge_and_master
from pipeline.lipsync import run_lipsync
from pipeline.restore import restore
from pipeline import validate


def final_encode(restored_path, out_dir="outputs"):
    out = Path(out_dir)
    final = str(out / "final_dubbed.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-i", restored_path,
         "-c:v", "libx264", "-crf", "16", "-preset", "slow",
         "-c:a", "aac", "-b:a", "192k", final],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return final


def run(input_path, start, end, out_dir="outputs"):
    clip_duration = end - start

    print("\n── Stage 1: Extract clip ──")
    clip_video, clip_audio = extract_clip(input_path, start, end, out_dir)

    print("\n── Stage 2: Transcribe (Whisper) ──")
    segments_path = transcribe(clip_audio, out_dir)

    print("\n── Stage 3: Translate Kannada → Hindi ──")
    translated_path = translate(segments_path, out_dir)

    print("\n── Stage 4/5: TTS + Audio master ──")
    tts_path = None
    audio_path = None
    passed = False

    for attempt in range(1, 4):
        print(f"\n[dub] TTS attempt {attempt}/3")
        tts_path = synthesise(translated_path, clip_audio, clip_duration, out_dir)
        audio_path = merge_and_master(tts_path, clip_audio, clip_duration, out_dir)

        dur_ok, _ = validate.check_duration(audio_path, clip_duration)
        if dur_ok:
            passed = True
            break
        print(f"[dub] Duration check failed, retrying ...")

    if not passed:
        print("[dub] WARNING: Duration tolerance not met after 3 attempts, continuing anyway.")

    print("\n── Stage 6: Lip sync (Wav2Lip) ──")
    lipsynced_path = run_lipsync(clip_video, audio_path, out_dir)

    print("\n── Stage 7: Face restoration (Real-ESRGAN) ──")
    restored_path = restore(lipsynced_path, out_dir)

    print("\n── Stage 8: Validate ──")
    ok, metrics = validate.run_all(audio_path, clip_video, restored_path, clip_duration)
    print(f"[dub] Metrics: {metrics}")

    print("\n── Final encode ──")
    final = final_encode(restored_path, out_dir)
    print(f"Done → {final}")
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kannada → Hindi video dubbing pipeline")
    parser.add_argument("--input", required=True, help="Path to source video")
    parser.add_argument("--start", type=float, required=True, help="Clip start time in seconds")
    parser.add_argument("--end", type=float, required=True, help="Clip end time in seconds")
    parser.add_argument("--out", default="outputs", help="Output directory (default: outputs)")
    args = parser.parse_args()

    run(args.input, args.start, args.end, args.out)
