import imageio_ffmpeg
import subprocess

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

subprocess.run([
    ffmpeg, "-y",
    "-ss", "00:00:20",
    "-i", "Stream.mp4",
    "-t", "20",
    "-filter_complex",
    "[0:v]split=2[bg][fg];"
    "[bg]scale=608:1080:force_original_aspect_ratio=increase,"
    "crop=608:1080,boxblur=20:10[background];"
    "[fg]scale=608:342[gameplay];"
    "[background][gameplay]overlay=0:100,"
    "setsar=1[out]",
    "-map", "[out]",
    "-map", "0:a?",
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-crf", "23",
    "-c:a", "aac",
    "test_vertical.mp4"
], check=True)

print("Created test_vertical.mp4")
