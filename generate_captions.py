import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "word_transcript.json").read_text(encoding="utf-8"))
OUTPUT = ROOT / "captions_test.ass"

# Correct only the known transcription error.
def clean_word(value):
    value = str(value).strip()
    if value.lower() == "copping":
        return "copying"
    return value

def ass_time(seconds):
    total = max(0, round(float(seconds) * 100))
    h, rem = divmod(total, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02}:{s:02}.{cs:02}"

def safe_text(value):
    value = clean_word(value)
    return value.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")

header = """[Script Info]
Title: Readable Gaming Captions
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Bebas Neue,68,&H00FFFFFF,&H0000FFFF,&H00000000,&H90000000,-1,0,0,0,100,100,1,0,1,7,3,2,80,80,330,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

events = []
for segment in data.get("segments", []):
    words = [
        w for w in segment.get("words", [])
        if float(w.get("end", 0)) > float(w.get("start", 0))
        and str(w.get("word", "")).strip()
    ]

    # Build fixed, readable groups of up to 5 words.
    groups = [words[i:i+5] for i in range(0, len(words), 5)]
    for group in groups:
        group_start = float(group[0]["start"])
        group_end = float(group[-1]["end"])
        if group_end <= group_start:
            continue

        for index, current in enumerate(group):
            start = max(group_start, float(current["start"]))
            end = min(group_end, float(current["end"]))
            if end <= start:
                continue

            rendered = []
            for item in group:
                shown = safe_text(item.get("word", ""))
                # ASS color override: yellow current word, white remaining words.
                if item is current:
                    rendered.append(r"{\c&H0000FFFF&}" + shown + r"{\c&H00FFFFFF&}")
                else:
                    rendered.append(shown)

            text = " ".join(rendered)
            events.append(
                f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{text}"
            )

OUTPUT.write_text(header + "\n".join(events) + "\n", encoding="utf-8-sig")
print(f"Generated {OUTPUT.name}: {len(events)} caption events.")
print("Style: bold 88px text, thick black outline, yellow spoken-word highlight.")


