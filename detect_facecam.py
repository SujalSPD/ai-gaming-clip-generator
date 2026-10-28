import cv2
import json
import statistics
from pathlib import Path

VIDEO = "Stream.mp4"
MODEL = "face_detection_yunet_2023mar.onnx"
OUTPUT_JSON = "facecam_analysis.json"
OUTPUT_IMAGE = "facecam_contact_sheet.jpg"

if not Path(VIDEO).is_file():
    raise FileNotFoundError(f"Video not found: {VIDEO}")
if not Path(MODEL).is_file():
    raise FileNotFoundError(f"Model not found: {MODEL}")

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Could not open {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = total / fps if fps else 0

detector = cv2.FaceDetectorYN.create(
    MODEL, "", (960, 540),
    score_threshold=0.75,
    nms_threshold=0.3,
    top_k=5000
)

# Sample 36 evenly spaced frames.
observations = []
preview_frames = []
sample_times = [duration * i / 37 for i in range(1, 37)]

for index, timestamp in enumerate(sample_times):
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ok, frame = cap.read()
    if not ok:
        continue

    small = cv2.resize(frame, (960, 540))
    _, faces = detector.detect(small)
    frame_detections = []

    if faces is not None:
        for face in faces:
            x, y, w, h = [float(v) for v in face[:4]]
            score = float(face[-1])

            # Convert detection coordinates to original video pixels.
            box = [x * W / 960, y * H / 540,
                   w * W / 960, h * H / 540]

            obs = {
                "time": round(timestamp, 2),
                "box": [round(v, 1) for v in box],
                "confidence": round(score, 3)
            }
            observations.append(obs)
            frame_detections.append(obs)

    # Save 12 annotated thumbnails for visual inspection.
    if index % 3 == 0:
        preview = cv2.resize(frame, (480, 270))
        for det in frame_detections:
            x, y, w, h = det["box"]
            x, y, w, h = [round(v / 2) for v in (x, y, w, h)]
            cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(preview, f"{timestamp:.1f}s", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        preview_frames.append(preview)

cap.release()

# Cluster face detections by stable position and approximate size.
clusters = []
for obs in observations:
    x, y, w, h = obs["box"]
    cx, cy = x + w / 2, y + h / 2
    match = None

    for cluster in clusters:
        mx, my = cluster["center"]
        mw, mh = cluster["size"]
        if (abs(cx - mx) < W * 0.10
                and abs(cy - my) < H * 0.12
                and abs(w - mw) < max(W * 0.08, mw * 0.8)
                and abs(h - mh) < max(H * 0.10, mh * 0.8)):
            match = cluster
            break

    if match is None:
        match = {"items": [], "center": (cx, cy), "size": (w, h)}
        clusters.append(match)

    match["items"].append(obs)
    boxes = [item["box"] for item in match["items"]]
    match["center"] = (
        statistics.median(b[0] + b[2]/2 for b in boxes),
        statistics.median(b[1] + b[3]/2 for b in boxes)
    )
    match["size"] = (
        statistics.median(b[2] for b in boxes),
        statistics.median(b[3] for b in boxes)
    )

clusters.sort(key=lambda c: len(c["items"]), reverse=True)
results = []

for cluster in clusters:
    x, y = cluster["center"]
    w, h = cluster["size"]
    quadrant = ("top-" if y < H/2 else "bottom-") + (
        "left" if x < W/2 else "right"
    )
    results.append({
        "detections": len(cluster["items"]),
        "quadrant": quadrant,
        "median_face_box_xywh": [
            round(x-w/2, 1), round(y-h/2, 1),
            round(w, 1), round(h, 1)
        ],
        "sample_times_seconds": [
            item["time"] for item in cluster["items"]
        ]
    })

report = {
    "video": VIDEO,
    "resolution": [W, H],
    "duration_seconds": round(duration, 2),
    "frames_sampled": len(sample_times),
    "total_face_detections": len(observations),
    "position_clusters": results
}

Path(OUTPUT_JSON).write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)

if preview_frames:
    sheet = cv2.vconcat([
        cv2.hconcat(preview_frames[i:i+4])
        for i in range(0, len(preview_frames), 4)
    ])
    cv2.imwrite(OUTPUT_IMAGE, sheet)

print(json.dumps(report, indent=2))
print(f"\nSaved analysis: {Path(OUTPUT_JSON).resolve()}")
print(f"Saved visual preview: {Path(OUTPUT_IMAGE).resolve()}")
