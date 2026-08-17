"""
The format of video tag file in data/ example.mp4 with example.txt
[00:00:00-00:01:39] game
[00:01:41-00:03:56] live2d
[00:03:57-00:05:54] game
"""

import glob
import os

import cv2
from PIL import Image

DATA_DIR = "data" # video data with tag file, e.g. example.mp4 with example.txt
DATASET_DIR = "dataset" # tag result of video
MAX_FRAMES_PER_VIDEO = 300  # cap per video; the long clip would dominate otherwise


def parse_timestamp(text):
    """'hh:mm:ss' (or mm:ss / ss) -> seconds."""
    seconds = 0
    for part in text.strip().split(":"):
        seconds = seconds * 60 + int(part)
    return seconds


def parse_annotation(txt_path):
    """Return a whole-video label (str) or a list of (start_sec, end_sec, label)
    segments parsed from '[hh:mm:ss-hh:mm:ss] label' lines."""
    with open(txt_path, encoding="utf-8") as f:
        content = f.read().strip()
    if "[" not in content:
        return content
    segments = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith("[") or "]" not in line:
            raise ValueError(
                f"{txt_path}: bad line {line!r} (want '[hh:mm:ss-hh:mm:ss] label')"
            )
        span, _, label = line[1:].partition("]")
        first, _, last = span.partition("-")
        label = label.strip()
        if not last or not label:
            raise ValueError(
                f"{txt_path}: bad line {line!r} (want '[hh:mm:ss-hh:mm:ss] label')"
            )
        segments.append((parse_timestamp(first), parse_timestamp(last), label))
    if not segments:
        raise ValueError(f"{txt_path}: no labels found")
    return segments


def label_at(annotation, sec):
    """Label covering second `sec`, or None if outside all segments."""
    if isinstance(annotation, str):
        return annotation
    for start_sec, end_sec, label in annotation:
        if start_sec <= sec < end_sec:
            return label
    return None


# Collect (video and annotation) pairs
pairs = []
for video_path in sorted(glob.glob(os.path.join(DATA_DIR, "*.mp4"))):
    txt_path = os.path.splitext(video_path)[0] + ".txt"
    if not os.path.exists(txt_path):
        print(f"Skipping {video_path}: missing {txt_path}")
        continue
    pairs.append((video_path, parse_annotation(txt_path)))
if not pairs:
    raise SystemExit(f"No labelled .mp4 files found in {DATA_DIR}/")

# Sample one frame per second (evenly capped) and save under dataset/<tag>/
os.makedirs(DATASET_DIR, exist_ok=True)
for video_path, annotation in pairs:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seconds = int(total_frames // fps)
    step = max(1, seconds // MAX_FRAMES_PER_VIDEO)
    kept = skipped = 0
    for sec in range(0, seconds, step):
        label = label_at(annotation, sec)
        if label is None:  # second not covered by any segment
            skipped += 1
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
        ret, frame = cap.read()
        if not ret:
            continue
        out_dir = os.path.join(DATASET_DIR, label)
        os.makedirs(out_dir, exist_ok=True)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img.save(os.path.join(out_dir, f"{stem}_s{sec:06d}.jpg"), quality=95)
        kept += 1
    cap.release()
    desc = (
        f"[{annotation}]"
        if isinstance(annotation, str)
        else f"({len(annotation)} segments)"
    )
    msg = f"{os.path.basename(video_path)} {desc}: {kept} frames"
    if skipped:
        msg += f", {skipped}s outside segments skipped"
    print(msg)

# print(f"Saved frames to ./{DATASET_DIR}/<tag>/")
