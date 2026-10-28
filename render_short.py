import json
import subprocess
from pathlib import Path

ffmpeg = str(Path("ffmpeg.exe").resolve())
output_dir = Path("Finished_Shorts")
output_dir.mkdir(exist_ok=True)

candidates = json.loads(
    Path("candidates.json").read_text(encoding="utf-8")
)
clip = max(candidates, key=lambda c: c["score"])

# Assumption for this first test: facecam is inside the
# top-right 640x360 area of the 1920x1080 source.
start, end = clip["start"], clip["end"]
output = output_dir / "short_01_stacked.mp4"

filters = (
    "[0:v]crop=640:360:1280:0,"
    "scale=1080:700:force_original_aspect_ratio=increase,"
    "crop=1080:700[cam];"
    "[0:v]crop=956:1080:482:0,"
    "scale=1080:1220[game];"
    "[cam][game]vstack=inputs=2[v]"
)

cmd = [
    ffmpeg, "-y",
    "-ss", str(start), "-i", "Stream.mp4",
    "-t", str(end - start),
    "-filter_complex", filters,
    "-map", "[v]", "-map", "0:a:0",
    "-c:v", "h264_nvenc", "-preset", "p4",
    "-c:a", "aac", "-b:a", "192k",
    "-movflags", "+faststart",
    str(output)
]

subprocess.run(cmd, check=True)
print("Created:", output.resolve())
