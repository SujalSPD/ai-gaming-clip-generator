import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "word_transcript.json"
OUTPUT = ROOT / "captions_test.ass"

data = json.loads(SOURCE.read_text(encoding="utf-8"))

# Corrections are applied to displayed captions only; audio is unchanged.
corrections = {
    "copping": "copying",
    "cop": "copying",
}

def ass_time(seconds):
    total = max(0, round(float(seconds) * 100))
    hours, rem = divmod(total, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, centis = divmod(rem, 100)
    return f"{hours}:{minutes:02}:{secs:02}.{centis:02}"

def clean_word(word):
    word = word.strip()
    return corrections.get(word.lower(), word)

header = """[Script Info]
Title: Gaming Captions Test
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,68,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,65,65,300,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

events = []
for segment in data.get("segments", []):
    words = segment.get("words", [])
    for index, word in enumerate(words):
        start = max(0.0, float(word.get("start", 0)))
        end = float(word.get("end", start))
        if end <= start:
            continue

        # Display short phrases together; highlight the currently spoken word.
        group_start = max(0, index - 2)
        group_end = min(len(words), index + 3)
        phrase_words = words[group_start:group_end]
        phrase = []
        for item in phrase_words:
            shown = clean_word(item.get("word", ""))
            if not shown:
                continue
            if item is word:
                phrase.append(r"{\c&H0000FFFF&}" + shown + r"{\c&H00FFFFFF&}")
            else:
                phrase.append(shown)

        text = " ".join(phrase).strip()
        text = text.replace("{", r"\{").replace("}", r"\}")
        events.append(
            f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{text}"
        )

OUTPUT.write_text(header + "\n".join(events) + "\n", encoding="utf-8-sig")
print(f"Created {OUTPUT.name} with {len(events)} timed caption events.")
print("Correction preview: copying my accent")
