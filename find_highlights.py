import json
from pathlib import Path

segments = json.loads(
    Path("transcript.json").read_text(encoding="utf-8")
)
duration = max((s["end"] for s in segments), default=0)

cues = {
    "what the fuck": 6,
    "what happened": 5,
    "funny": 3,
    "earthworms": 3,
    "under arrest": 3,
    "meow": 4,
    "accent": 2,
    "repeat after me": 3,
    "you are copping": 4,
    "copping my accent": 4,
}

# Evaluate overlapping 28-second windows.
window = 28
step = 4
ranked = []

for start in range(0, max(1, int(duration - window) + 1), step):
    end = min(start + window, duration)
    relevant = [
        s for s in segments
        if s["start"] < end and s["end"] > start
    ]

    text = " ".join(s.get("text", "") for s in relevant).lower()
    score = sum(points for phrase, points in cues.items()
                if phrase in text)

    # Small bonus for lively dialogue, without counting every "bro".
    score += min(text.count("?") + text.count("!"), 3)

    if score > 0:
        ranked.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "score": score,
            "transcript": " ".join(
                s.get("text", "").strip() for s in relevant
            ),
        })

# Pick the strongest candidates without overlapping clips.
ranked.sort(key=lambda c: c["score"], reverse=True)
selected = []

for candidate in ranked:
    overlaps = any(
        candidate["start"] < item["end"] + 3
        and candidate["end"] > item["start"] - 3
        for item in selected
    )
    if not overlaps:
        selected.append(candidate)
    if len(selected) >= 5:
        break

selected.sort(key=lambda c: c["start"])

Path("candidates.json").write_text(
    json.dumps(selected, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(f"\nSelected {len(selected)} candidate clips:\n")
for i, c in enumerate(selected, 1):
    print(f"{i}. {c['start']:.1f}s–{c['end']:.1f}s"
          f" | score {c['score']}")
    print(f"   {c['transcript'][:220]}\n")

print("Saved to candidates.json")
