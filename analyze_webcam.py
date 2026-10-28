import cv2
import statistics

video = "Stream_Left.mp4"
model = "face_detection_yunet_2023mar.onnx"

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {video}")

fps = cap.get(cv2.CAP_PROP_FPS)
total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = total / fps

detector = cv2.FaceDetectorYN.create(
    model, "", (960, 540),
    score_threshold=0.75, nms_threshold=0.3, top_k=5000
)

results = []
for t in range(1, 61):
    sec = duration * t / 61
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    ok, frame = cap.read()
    if not ok:
        continue
    small = cv2.resize(frame, (960, 540))
    detector.setInputSize((960, 540))
    _, faces = detector.detect(small)
    if faces is None:
        continue
    for f in faces:
        x, y, w, h = map(float, f[:4])
        cx, cy = x + w / 2, y + h / 2
        if cx < 0.38 * 960 and cy > 0.58 * 540:
            results.append({
                "t": round(sec, 2),
                "x": round(x * W / 960),
                "y": round(y * H / 540),
                "w": round(w * W / 960),
                "h": round(h * H / 540),
                "confidence": round(float(f[-1]), 3)
            })

cap.release()
print(f"Video: {W}x{H}, {duration:.1f}s")
print(f"Bottom-left face detections: {len(results)}")
if results:
    for key in ("x", "y", "w", "h"):
        vals = [r[key] for r in results]
        print(f"{key}: min={min(vals)}, median={statistics.median(vals)}, max={max(vals)}")
    print("Sample detections:")
    for r in results[:10]:
        print(r)
else:
    print("No bottom-left face detections found.")
