import json
import re
from pathlib import Path

source = Path("transcript_left_en.json")
destination = Path("candidates_left.json")

data = json.loads(source.read_text(encoding="utf-8"))
segments = data["segments"] if isinstance(data, dict) else data
duration = max((s["end"] for s in segments), default=0)

reaction_phrases = {
    "oh shit": 5,
    "what is happening": 4,
    "what happened": 4,
    "what is this": 3,
    "who is there": 2,
    "very dangerous": 3,
    "crazy": 2,
    "let's go": 3,
    "lets go": 3,
    "no way": 4,
    "oh my god": 5,
    "help me": 4,
    "run": 2,
}

window = 20
step = 5
ranked = []

for start in range(0, max(1, int(duration)), step):
    end = min(start + window, duration)
    relevant = [
        s for s in segments
        if s["start"] < end and s["end"] > start
    ]
    if not relevant:
        continue

    text = " ".join(s.get("text", "") for s in relevant).lower()
    score = sum(
        points for phrase, points in reaction_phrases.items()
        if phrase in text
    )
    score += min(text.count("?") * 2 + text.count("!"), 4)

    if score > 0:
        ranked.append({
            "start": round(max(0, start), 2),
            "end": round(end, 2),
            "score": score,
            "transcript": " ".join(
                s.get("text", "").strip() for s in relevant
            ),
        })

ranked.sort(key=lambda c: c["score"], reverse=True)
selected = []

for candidate in ranked:
    if any(
        candidate["start"] < item["end"] + 2
        and candidate["end"] > item["start"] - 2
        for item in selected
    ):
        continue
    selected.append(candidate)
    if len(selected) >= 5:
        break

selected.sort(key=lambda c: c["start"])
destination.write_text(
    json.dumps(selected, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(f"Analyzed {duration:.1f} seconds of transcript.")
print(f"Selected {len(selected)} candidates:")
for i, c in enumerate(selected, 1):
    print(f'\n{i}. {c["start"]:.1f}s–{c["end"]:.1f}s | score {c["score"]}')
    print(c["transcript"][:250])
print(f"\nSaved to {destination}")
