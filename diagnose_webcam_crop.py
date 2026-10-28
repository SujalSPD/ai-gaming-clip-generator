import cv2
import json

video = "Stream_Left.mp4"
model = "face_detection_yunet_2023mar.onnx"
output = "webcam_crop_diagnostic.jpg"

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {video}")

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Inspect a representative frame near the start.
cap.set(cv2.CAP_PROP_POS_MSEC, 5000)
ok, frame = cap.read()
cap.release()
if not ok:
    raise RuntimeError("Could not read diagnostic frame.")

detector = cv2.FaceDetectorYN.create(
    model, "", (960, 540),
    score_threshold=0.75, nms_threshold=0.3, top_k=5000
)
small = cv2.resize(frame, (960, 540))
_, faces = detector.detect(small)

if faces is None:
    raise RuntimeError("No face detected at 5 seconds.")

# Choose the bottom-left face.
face = min(
    (f for f in faces if f[0] + f[2] / 2 < 0.38 * 960
     and f[1] + f[3] / 2 > 0.58 * 540),
    key=lambda f: f[-1],
    default=None
)
if face is None:
    raise RuntimeError("No bottom-left face found at 5 seconds.")

x, y, w, h = [int(round(v * 2)) for v in face[:4]]
face_cx, face_cy = x + w // 2, y + h // 2

# Diagnostic crop proposal, with space around the detected face.
crop_w = 760
crop_h = round(crop_w * 650 / 1080)
crop_x = max(0, min(W - crop_w, face_cx - crop_w // 2))
crop_y = max(0, min(H - crop_h, face_cy - round(crop_h * 0.38)))

preview = frame.copy()
cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 4)
cv2.rectangle(
    preview, (crop_x, crop_y),
    (crop_x + crop_w, crop_y + crop_h),
    (0, 0, 255), 4
)
cv2.putText(
    preview, "GREEN = detected face | RED = proposed crop",
    (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
)
cv2.imwrite(output, cv2.resize(preview, (960, 540)))
print(f"Saved diagnostic image: {output}")
print(f"Face box: x={x}, y={y}, w={w}, h={h}")
print(f"Proposed crop: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")
