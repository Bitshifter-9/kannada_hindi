import argparse
from pipeline.extractor import extract_clip
from pipeline.transcribe_translate import transcribe_and_translate
from pipeline.tts import synthesise
from pipeline.audio import merge_and_master
from pipeline.merge import merge_audio_video


def run(input_path, start, end, out_dir="outputs", voice="male"):
    clip_duration = end - start

    print("\n── Stage 1: Extract clip ──")
    clip_video, clip_audio = extract_clip(input_path, start, end, out_dir)

    print("\n── Stage 2: Transcribe + Translate (Gemini) ──")
    translated_path = transcribe_and_translate(clip_audio, out_dir)

    print("\n── Stage 3: Hindi TTS (Edge TTS) ──")
    tts_path = synthesise(translated_path, clip_duration, out_dir, voice)

    print("\n── Stage 4: Audio master ──")
    audio_path = merge_and_master(tts_path, clip_audio, clip_duration, out_dir)

    print("\n── Stage 5: Merge audio + video ──")
    final = merge_audio_video(clip_video, audio_path, out_dir)

    print(f"\nDone → {final}")
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kannada to Hindi video dubbing")
    parser.add_argument("--input", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--voice", default="male", choices=["male", "female"])
    args = parser.parse_args()

    run(args.input, args.start, args.end, args.out, args.voice)
