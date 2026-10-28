from pathlib import Path
import argparse
import json
import re
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "Input_Videos"
WORK_DIR = ROOT / "pipeline_work"
OUTPUT_DIR = ROOT / "Finished_Shorts"
RENDERER = ROOT / "render_batch_caption_test.py"
STYLE_FILE = ROOT / "captions_test.ass"
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}

REACTIONS = {
    "oh shit": 5, "what is happening": 4, "what happened": 4,
    "what is this": 3, "who is there": 2, "very dangerous": 3,
    "crazy": 2, "let's go": 3, "lets go": 3, "no way": 4,
    "oh my god": 5, "help me": 4, "run": 2, "what the hell": 4,
    "are you kidding": 4, "holy shit": 5, "bro": 1
}


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", Path(name).stem).strip("_")


def ass_time(seconds):
    cs = max(0, round(float(seconds) * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def safe_text(value):
    return str(value).strip().replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def transcribe_translate(video, model):
    """Fast full-video pass: translate to English without word alignment."""
    print(f"Transcribing and translating {video.name} to English (fast pass)...")
    return model.transcribe(
        str(video),
        task="translate",
        word_timestamps=False,
        verbose=False
    )


def align_selected_clips(video, candidates, model, work_dir):
    """Run expensive word-level alignment only on selected short clips."""
    aligned_segments = []
    for index, candidate in enumerate(candidates, 1):
        start = max(0.0, float(candidate["start"]))
        end = max(start, float(candidate["end"]))
        if end <= start:
            continue

        audio_path = work_dir / f"{safe_name(video.name)}_clip_{index:02d}.wav"
        print(f"Word timing {index}/{len(candidates)}: {start:.1f}s–{end:.1f}s")
        ffmpeg_bin = str(ROOT / "ffmpeg.exe") if (ROOT / "ffmpeg.exe").is_file() else "ffmpeg"
        cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(start), "-i", str(video), "-t", str(end - start),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(audio_path)
        ]
        try:
            subprocess.run(cmd, check=True, cwd=ROOT)
            clip_result = model.transcribe(
                str(audio_path),
                task="translate",
                word_timestamps=True,
                verbose=False
            )
            for segment in clip_result.get("segments", []):
                seg = dict(segment)
                seg["start"] = float(seg.get("start", 0)) + start
                seg["end"] = float(seg.get("end", 0)) + start
                shifted_words = []
                for word in seg.get("words", []):
                    w = dict(word)
                    w["start"] = float(w.get("start", 0)) + start
                    w["end"] = float(w.get("end", 0)) + start
                    shifted_words.append(w)
                seg["words"] = shifted_words
                aligned_segments.append(seg)
        finally:
            if audio_path.exists():
                audio_path.unlink()

    return {"segments": sorted(aligned_segments, key=lambda s: s.get("start", 0))}


def make_candidates(result, duration, count=5):
    segments = result.get("segments", [])
    ranked = []
    window, step = 20.0, 5.0

    for start in (i * step for i in range(max(1, int(duration / step)))):
        if start >= duration:
            continue
        end = min(start + window, duration)
        relevant = [
            s for s in segments
            if float(s.get("start", 0)) < end
            and float(s.get("end", 0)) > start
        ]
        if not relevant:
            continue

        text = " ".join(s.get("text", "") for s in relevant).lower()
        score = sum(points for phrase, points in REACTIONS.items() if phrase in text)
        score += min(text.count("?") * 2 + text.count("!"), 4)

        word_count = sum(len(str(s.get("text", "")).split()) for s in relevant)
        score += min(word_count / 12.0, 2.0)

        ranked.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "score": round(score, 2),
            "transcript": " ".join(s.get("text", "").strip() for s in relevant)
        })

    ranked.sort(key=lambda c: c["score"], reverse=True)
    selected = []

    for item in ranked:
        if any(
            item["start"] < old["end"] + 2
            and item["end"] > old["start"] - 2
            for old in selected
        ):
            continue
        selected.append(item)
        if len(selected) >= count:
            break

    return selected


def make_captions(result, output):
    original = STYLE_FILE.read_text(encoding="utf-8-sig")
    if "[Events]" not in original:
        raise RuntimeError("captions_test.ass has no [Events] section.")

    header = original.split("[Events]", 1)[0]
    events_header = (
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\n"
    )
    events = []

    for segment in result.get("segments", []):
        words = [
            w for w in segment.get("words", [])
            if float(w.get("end", 0)) > float(w.get("start", 0))
            and str(w.get("word", "")).strip()
        ]

        for offset in range(0, len(words), 5):
            group = words[offset:offset + 5]
            if not group:
                continue

            group_start = float(group[0]["start"])
            group_end = float(group[-1]["end"])

            for current in group:
                start = max(group_start, float(current["start"]))
                end = min(group_end, float(current["end"]))
                if end <= start:
                    continue

                rendered = []
                for word in group:
                    text = safe_text(word.get("word", ""))
                    if word is current:
                        rendered.append(r"{\c&H0000FFFF&}" + text + r"{\c&H00FFFFFF&}")
                    else:
                        rendered.append(text)

                events.append(
                    f"Dialogue: 0,{ass_time(start)},{ass_time(end)},"
                    f"Default,,0,0,0,,{' '.join(rendered)}"
                )

    output.write_text(
        header + events_header + "\n".join(events) + "\n",
        encoding="utf-8-sig"
    )
    print(f"Created captions: {len(events)} timed word events")


def process_video(video, count):
    video = video.resolve()
    print("\n" + "=" * 65)
    print("PROCESSING:", video.name)
    print("=" * 65)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stem = safe_name(video.name)
    transcript_path = WORK_DIR / f"{stem}_english_transcript.json"
    candidates_path = WORK_DIR / f"{stem}_candidates.json"
    captions_path = WORK_DIR / f"{stem}_captions.ass"
    renderer_path = ROOT / f"_renderer_{stem}.py"

    if not video.is_file():
        raise FileNotFoundError(f"Video does not exist: {video}")
    if not video.is_relative_to(ROOT):
        raise ValueError(
            f"Video must be inside the project folder so the renderer can find it: {video}"
        )

    source = RENDERER.read_text(encoding="utf-8-sig")
    original_source = source
    source = source.replace(
        'CAPTIONS = ROOT / "captions_test.ass"',
        f'CAPTIONS = ROOT / "pipeline_work" / "{stem}_captions.ass"'
    )
    if source == original_source:
        raise RuntimeError(
            'Could not find the exact line CAPTIONS = ROOT / "captions_test.ass" '
            "inside render_batch_caption_test.py."
        )

    renderer_path.write_text(source, encoding="utf-8")

    try:
        import cv2
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        if fps <= 0 or frames <= 0:
            raise RuntimeError(f"Cannot read video duration: {video}")

        duration = frames / fps
        print(f"Video duration: {duration / 60:.1f} minutes")

        import whisper
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper small model on {device}...")
        model = whisper.load_model("small", device=device)

        # Important speed fix: no word-level alignment over the full recording.
        result = transcribe_translate(video, model)
        transcript_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        candidates = make_candidates(result, duration, count)
        if not candidates:
            raise RuntimeError(
                "No candidate highlights were generated. The video may have no "
                "recognized speech, or transcription may have failed."
            )

        candidates_path.write_text(
            json.dumps(candidates, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"Found {len(candidates)} candidate highlights.")

        # Word-level alignment is only run on the chosen ~20-second clips.
        aligned_result = align_selected_clips(video, candidates, model, WORK_DIR)
        if not any(seg.get("words") for seg in aligned_result.get("segments", [])):
            raise RuntimeError(
                "Whisper did not return word timings for the selected clips; "
                "captions were not generated."
            )
        make_captions(aligned_result, captions_path)

        # Renderer expects paths relative to project root.
        video_arg = video.relative_to(ROOT).as_posix()
        candidates_arg = candidates_path.relative_to(ROOT).as_posix()

        subprocess.run(
            [
                sys.executable, str(renderer_path),
                video_arg, candidates_arg, "--count", str(count)
            ],
            cwd=ROOT,
            check=True
        )

        print(f"SUCCESS: {video.name}")
        print(f"Shorts folder: {OUTPUT_DIR / safe_name(video.name)}")

    finally:
        if renderer_path.exists():
            renderer_path.unlink()


def resolve_single_video(value):
    candidate = Path(value)

    if candidate.is_absolute():
        options = [candidate.resolve()]
    else:
        # If a path includes a folder, honor that path first; otherwise also
        # search Input_Videos by the supplied filename.
        options = [(ROOT / candidate).resolve()]
        if len(candidate.parts) == 1:
            options.append((INPUT_DIR / candidate.name).resolve())

    for option in options:
        if option.is_file() and option.suffix.lower() in VIDEO_EXTS:
            return option

    checked = "\n".join(f"  {p}" for p in options)
    raise FileNotFoundError(
        f"Video not found or unsupported: {value}\nChecked:\n{checked}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Automatically create captioned vertical gaming Shorts."
    )
    parser.add_argument(
        "--video",
        help="Video filename or path; a filename is searched in Input_Videos."
    )
    parser.add_argument(
        "--count", type=int, default=5,
        help="Maximum number of Shorts to request per video (default: 5)."
    )
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1.")

    if not RENDERER.is_file():
        raise FileNotFoundError(f"Renderer not found: {RENDERER}")
    if not STYLE_FILE.is_file():
        raise FileNotFoundError(f"Caption style file not found: {STYLE_FILE}")

    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.video:
        videos = [resolve_single_video(args.video)]
    else:
        videos = sorted(
            p.resolve() for p in INPUT_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        )

    if not videos:
        print(f"No videos found. Add MP4/MKV/MOV files to: {INPUT_DIR}")
        return

    failures = []
    for video in videos:
        try:
            process_video(video, args.count)
        except Exception as exc:
            failures.append((video.name, str(exc)))
            print(f"\nFAILED: {video.name}: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 65)
    print(f"Completed: {len(videos) - len(failures)}/{len(videos)} videos")

    if failures:
        print("Failures:")
        for name, reason in failures:
            print(f" - {name}: {reason}")

    print("Finished Shorts location:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
