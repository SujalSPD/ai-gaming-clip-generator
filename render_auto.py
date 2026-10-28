import cv2
import json
import subprocess
import statistics
from pathlib import Path

VIDEO = "Stream.mp4"
MODEL = "face_detection_yunet_2023mar.onnx"
OUTPUT_DIR = Path("Finished_Shorts")

# Approximate webcam overlay dimensions as a fraction of the source.
CAM_W_RATIO = 0.25
CAM_H_RATIO = 0.26

SAMPLES = 36
MIN_DETECTIONS = 5

for filename in [VIDEO, MODEL, "candidates.json", "ffmpeg.exe"]:
    if not Path(filename).is_file():
        raise FileNotFoundError(f"Required file missing: {filename}")

# Read source properties.
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = total_frames / fps if fps else 0

detector = cv2.FaceDetectorYN.create(
    MODEL, "", (960, 540),
    score_threshold=0.75,
    nms_threshold=0.3,
    top_k=5000
)

# Track stable face locations, prioritizing the two supported corners.
clusters = []
frames_with_detection = 0

for i in range(1, SAMPLES + 1):
    t = duration * i / (SAMPLES + 1)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    if not ok:
        continue

    small = cv2.resize(frame, (960, 540))
    detector.setInputSize((960, 540))
    _, faces = detector.detect(small)

    if faces is None:
        continue

    found_this_frame = False
    for face in faces:
        x, y, w, h = [float(v) for v in face[:4]]
        cx, cy = x + w / 2, y + h / 2

        # Exclude most faces far from the supported webcam corners.
        if cx > 0.62 * 960 and cy < 0.42 * 540:
            corner = "top-right"
        elif cx < 0.38 * 960 and cy > 0.58 * 540:
            corner = "bottom-left"
        else:
            continue

        found_this_frame = True
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
                "center": (box[0] + box[2]/2, box[1] + box[3]/2),
                "size": (box[2], box[3])
            }
            clusters.append(match)

        match["boxes"].append(box)
        match["times"].append(round(t, 2))
        boxes = match["boxes"]
        match["center"] = (
            statistics.median(b[0] + b[2]/2 for b in boxes),
            statistics.median(b[1] + b[3]/2 for b in boxes)
        )
        match["size"] = (
            statistics.median(b[2] for b in boxes),
            statistics.median(b[3] for b in boxes)
        )

    if found_this_frame:
        frames_with_detection += 1

cap.release()

clusters.sort(key=lambda c: len(c["boxes"]), reverse=True)

print(f"Video: {W}x{H}, {duration:.1f}s")
print("Stable corner clusters:")
for c in clusters:
    print(f"  {c['corner']}: {len(c['boxes'])} detections")

if not clusters or len(clusters[0]["boxes"]) < MIN_DETECTIONS:
    raise RuntimeError(
        "Facecam position is uncertain. No clip rendered; "
        "we need to inspect detections or adjust the detector."
    )

best = clusters[0]
# Require evidence across multiple sampled frames, not just one frame.
unique_times = len(set(best["times"]))
if unique_times < MIN_DETECTIONS:
    raise RuntimeError("Not enough separate frames confirm the facecam position.")

corner = best["corner"]
print(f"Estimated facecam corner: {corner}")

# Approximate webcam box. This is a configurable estimate, not
# guaranteed to match the overlay's exact boundaries.
cam_w = round(W * CAM_W_RATIO)
cam_h = round(H * CAM_H_RATIO)
cam_x = W - cam_w if corner == "top-right" else 0
cam_y = 0 if corner == "top-right" else H - cam_h

# Use the strongest speech-scored candidate.
candidates = json.loads(
    Path("candidates.json").read_text(encoding="utf-8")
)
if not candidates:
    raise RuntimeError("No candidates found in candidates.json")

clip = max(candidates, key=lambda c: c["score"])
start, end = float(clip["start"]), float(clip["end"])

# Central gameplay crop avoids both supported webcam corners.
game_w = min(W, round(H * 0.83))
game_x = max(0, (W - game_w) // 2)

# Portrait layout: 650px webcam + 1270px gameplay = 1920px.
output = OUTPUT_DIR / "auto_short_01.mp4"
OUTPUT_DIR.mkdir(exist_ok=True)

filters = (
    f"[0:v]crop={cam_w}:{cam_h}:{cam_x}:{cam_y},"
    "scale=1080:650:force_original_aspect_ratio=increase,"
    "crop=1080:650[cam];"
    f"[0:v]crop={game_w}:{H}:{game_x}:0,"
    "scale=1080:1270:force_original_aspect_ratio=increase,"
    "crop=1080:1270[game];"
    "[cam][game]vstack=inputs=2,setsar=1[v]"
)

cmd = [
    str(Path("ffmpeg.exe").resolve()), "-y",
    "-ss", str(start), "-i", VIDEO,
    "-t", str(end - start),
    "-filter_complex", filters,
    "-map", "[v]", "-map", "0:a:0?",
    "-c:v", "h264_nvenc", "-preset", "p4",
    "-c:a", "aac", "-b:a", "192k",
    "-movflags", "+faststart",
    str(output)
]

print(f"Rendering {start:.1f}s–{end:.1f}s...")
subprocess.run(cmd, check=True)
print(f"Created: {output.resolve()}")
print("Please check the facecam crop and gameplay framing.")
