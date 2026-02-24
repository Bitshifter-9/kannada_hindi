import soundfile as sf
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


def check_duration(audio_path, target_duration, tolerance=0.1):
    audio, sr = sf.read(audio_path)
    actual = len(audio) / sr
    diff = abs(actual - target_duration)
    ok = diff < tolerance
    print(f"[validate] Duration: {actual:.3f}s (target {target_duration}s, diff {diff:.3f}s) — {'OK' if ok else 'FAIL'}")
    return ok, actual


def check_ssim(original_video, restored_video, sample_frames=10):
    cap_orig = cv2.VideoCapture(original_video)
    cap_rest = cv2.VideoCapture(restored_video)

    total = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, sample_frames, dtype=int)

    scores = []
    for i in indices:
        cap_orig.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap_rest.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret1, f1 = cap_orig.read()
        ret2, f2 = cap_rest.read()
        if not ret1 or not ret2:
            continue
        g1 = cv2.cvtColor(cv2.resize(f1, (256, 256)), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(cv2.resize(f2, (256, 256)), cv2.COLOR_BGR2GRAY)
        scores.append(ssim(g1, g2))

    cap_orig.release()
    cap_rest.release()

    avg = float(np.mean(scores)) if scores else 0.0
    ok = avg >= 0.80
    print(f"[validate] SSIM: {avg:.4f} — {'OK' if ok else 'FAIL'}")
    return ok, avg


def run_all(audio_path, original_video, restored_video, target_duration):
    dur_ok, actual_dur = check_duration(audio_path, target_duration)
    ssim_ok, ssim_score = check_ssim(original_video, restored_video)

    passed = dur_ok and ssim_ok
    print(f"[validate] Overall: {'PASS' if passed else 'FAIL'}")
    return passed, {"duration_ok": dur_ok, "ssim_ok": ssim_ok, "actual_duration": actual_dur, "ssim": ssim_score}
