import argparse
import cv2
import json
import subprocess
import statistics
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "face_detection_yunet_2023mar.onnx"
FFMPEG = ROOT / "ffmpeg.exe"
OUTPUT_ROOT = ROOT / "Finished_Shorts"
CAPTIONS = ROOT / "captions_test.ass"

parser = argparse.ArgumentParser(
    description="Render a portrait gaming Short from a stream and its candidates."
)
parser.add_argument("video", help="Source video filename")
parser.add_argument("candidates", help="Matching candidates JSON filename")
parser.add_argument("--count", type=int, default=1,
                    help="Number of highest-scored clips to render")
args = parser.parse_args()

VIDEO = (ROOT / args.video).resolve()
CANDIDATES = (ROOT / args.candidates).resolve()

for path in (VIDEO, CANDIDATES, MODEL, FFMPEG, CAPTIONS):
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")

def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", Path(name).stem).strip("_")

stream_name = safe_name(VIDEO.name)
OUTPUT_DIR = OUTPUT_ROOT / stream_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = total_frames / fps if fps else 0

if fps <= 0 or W <= 0 or H <= 0 or duration <= 0:
    cap.release()
    raise RuntimeError("Could not read valid video properties.")

detector = cv2.FaceDetectorYN.create(
    str(MODEL), "", (960, 540),
    score_threshold=0.75, nms_threshold=0.3, top_k=5000
)

clusters = []
samples = 36

for i in range(1, samples + 1):
    t = duration * i / (samples + 1)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    if not ok:
        continue

    small = cv2.resize(frame, (960, 540))
    detector.setInputSize((960, 540))
    _, faces = detector.detect(small)
    if faces is None:
        continue

    for face in faces:
        x, y, w, h = [float(v) for v in face[:4]]
        cx, cy = x + w / 2, y + h / 2

        if cx > 0.62 * 960 and cy < 0.42 * 540:
            corner = "top-right"
        elif cx < 0.38 * 960 and cy > 0.58 * 540:
            corner = "bottom-left"
        else:
            continue

        box = [x * W / 960, y * H / 540,
               w * W / 960, h * H / 540]

        match = None
        for cluster in clusters:
            mx, my = cluster["center"]
            mw, mh = cluster["size"]
            if (cluster["corner"] == corner
                    and abs(cx - mx * 960 / W) < 0.10 * 960
                    and abs(cy - my * 540 / H) < 0.12 * 540
                    and abs(box[2] - mw) < max(0.08 * W, mw * 0.8)
                    and abs(box[3] - mh) < max(0.10 * H, mh * 0.8)):
                match = cluster
                break

        if match is None:
            match = {
                "corner": corner, "boxes": [], "times": [],
                "center": (box[0] + box[2] / 2, box[1] + box[3] / 2),
                "size": (box[2], box[3])
            }
            clusters.append(match)

        match["boxes"].append(box)
        match["times"].append(t)
        match["center"] = (
            statistics.median(b[0] + b[2] / 2 for b in match["boxes"]),
            statistics.median(b[1] + b[3] / 2 for b in match["boxes"])
        )
        match["size"] = (
            statistics.median(b[2] for b in match["boxes"]),
            statistics.median(b[3] for b in match["boxes"])
        )

cap.release()
clusters.sort(key=lambda c: len(c["boxes"]), reverse=True)

print(f"\nVideo: {VIDEO.name} | {W}x{H} | {duration:.1f}s")
for c in clusters:
    print(f"  {c['corner']}: {len(c['boxes'])} detections")

if not clusters or len(set(clusters[0]["times"])) < 5:
    raise RuntimeError(
        "Could not confidently detect the webcam in the supported corners. "
        "No clips rendered."
    )

corner = clusters[0]["corner"]
print("Estimated webcam corner:", corner)

# Adaptive webcam crop based on the median detected face.
face_boxes = clusters[0]["boxes"]
face_x = statistics.median(b[0] for b in face_boxes)
face_y = statistics.median(b[1] for b in face_boxes)
face_w = statistics.median(b[2] for b in face_boxes)
face_h = statistics.median(b[3] for b in face_boxes)
face_cx = face_x + face_w / 2
face_cy = face_y + face_h / 2

# Match the webcam panel aspect ratio (1080 x 650).
cam_w = min(round(W * 0.45), max(round(W * 0.25), round(face_w * 4.5)))
cam_h = min(H, round(cam_w * 650 / 1080))
cam_w = min(W, round(cam_h * 1080 / 650))

if corner == "top-right":
    cam_x = max(round(W * 0.50), min(W - cam_w, round(face_cx - cam_w * 0.50)))
    cam_y = max(0, min(round(H * 0.50) - cam_h, round(face_cy - cam_h * 0.38)))
else:
    cam_x = max(0, min(round(W * 0.50) - cam_w, round(face_cx - cam_w * 0.50)))
    cam_y = max(round(H * 0.50), min(H - cam_h, round(face_cy - cam_h * 0.38)))

print(f"Adaptive webcam crop: x={cam_x}, y={cam_y}, w={cam_w}, h={cam_h}")

# Center crop is used for gameplay. It may need adjustment for some games.
game_w = min(W, round(H * 0.83))
game_x = max(0, (W - game_w) // 2)

candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
if not candidates:
    raise RuntimeError(f"No candidates in {CANDIDATES.name}")

candidates = sorted(candidates, key=lambda c: float(c.get("score", 0)), reverse=True)
chosen = candidates[:max(1, args.count)]

filters = (
    f"[0:v]crop={cam_w}:{cam_h}:{cam_x}:{cam_y},"
    "scale=1080:650:force_original_aspect_ratio=increase,"
    "crop=1080:650[cam];"
    f"[0:v]crop={game_w}:{H}:{game_x}:0,"
    "scale=1080:1270:force_original_aspect_ratio=increase,"
    "crop=1080:1270[game];"
    "[cam][game]vstack=inputs=2,setsar=1[v]"
)

for i, clip in enumerate(chosen, 1):
    start = max(0.0, float(clip["start"]))
    end = min(duration, float(clip["end"]))
    if end <= start:
        print(f"Skipping invalid candidate {i}: {start}–{end}")
        continue

    output = OUTPUT_DIR / f"{stream_name}_short_{i:02d}.mp4"

    # Create a clip-specific ASS file with timestamps relative to this Short.
    ass_text = CAPTIONS.read_text(encoding="utf-8-sig")
    clip_ass = OUTPUT_DIR / f"{stream_name}_short_{i:02d}_captions.ass"

    def ass_seconds(stamp):
        h, m, sec = stamp.split(":")
        return int(h) * 3600 + int(m) * 60 + float(sec)

    def ass_stamp(seconds):
        seconds = max(0.0, seconds)
        centiseconds = round(seconds * 100)
        h, rem = divmod(centiseconds, 360000)
        m, rem = divmod(rem, 6000)
        sec, cs = divmod(rem, 100)
        return f"{h}:{m:02d}:{sec:02d}.{cs:02d}"

    shifted_lines = []
    for line in ass_text.splitlines():
        if not line.startswith("Dialogue:"):
            shifted_lines.append(line)
            continue
        match = re.match(r"^(Dialogue:\s*[^,]*,)([^,]+),([^,]+)(,.*)$", line)
        if not match:
            shifted_lines.append(line)
            continue
        line_start = ass_seconds(match.group(2))
        line_end = ass_seconds(match.group(3))
        if line_end <= start or line_start >= end:
            continue
        new_start = max(0.0, line_start - start)
        new_end = min(end - start, line_end - start)
        if new_end <= new_start:
            continue
        shifted_lines.append(
            match.group(1) + ass_stamp(new_start) + "," +
            ass_stamp(new_end) + match.group(4)
        )

    clip_ass.write_text("\n".join(shifted_lines) + "\n", encoding="utf-8")

    # Burn captions after webcam and gameplay are combined into portrait format.
    caption_path = clip_ass.relative_to(ROOT).as_posix()
    caption_filters = filters.replace(
        "[cam][game]vstack=inputs=2,setsar=1[v]",
        f"[cam][game]vstack=inputs=2,setsar=1,ass='{caption_path}'[v]"
    )

    cmd = [
        str(FFMPEG), "-y",
        "-ss", str(start), "-i", str(VIDEO),
        "-t", str(end - start),
        "-filter_complex", caption_filters,
        "-map", "[v]", "-map", "0:a:0?",
        "-c:v", "h264_nvenc", "-preset", "p4",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(output)
    ]

    print(f"\nRendering candidate {i}: {start:.1f}–{end:.1f}s")
    subprocess.run(cmd, check=True)
    print("Created:", output)

print("\nFinished. Output folder:", OUTPUT_DIR)
