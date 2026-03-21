"""
Clean Open-MELON captions by removing non-visual information.
Uses gemma2:9b via Ollama to rewrite captions keeping only visual descriptions.
Saves incrementally to JSONL with resume support.
"""

import json, time, os, requests
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────
MODEL = "gemma2:9b"
OUTPUT_FILE = Path(__file__).parent.parent / "captions_cleaned.jsonl"
INDEX_FILE = Path(__file__).parent / "melanoma_nevus_indices.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
SPLIT = "train"

SYSTEM_PROMPT = (
    "You are a histopathology caption editor. Your job is to rewrite captions "
    "so they contain ONLY visual descriptions and diagnosis names. "
    "Keep: tissue architecture, cell morphology, colors, patterns, structures, "
    "lesion shape, cell types visible, AND diagnosis/tumor names (e.g. melanoma, "
    "carcinoma, nevus, fibrosarcoma). "
    "Remove ALL of the following: staining method (H&E, IHC, etc.), "
    "magnification (×40, 100x, etc.), patient demographics (age, sex), "
    "clinical history, figure/panel references, "
    "study methodology, receptor status, and staging information. "
    "If after removing non-visual info nothing remains, output 'NO_VISUAL_CONTENT'. "
    "Output ONLY the rewritten caption, nothing else."
)
# ────────────────────────────────────────────────────────────────────


def load_completed(path: Path) -> dict[int, dict]:
    """Load completed entries from JSONL, return {index: record}."""
    done = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done[rec["index"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def append_result(path: Path, result: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(result) + "\n")


def ask_gemma(caption: str) -> str:
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": f"Rewrite this caption keeping only visual descriptions:\n\n{caption}",
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 300},
    })
    resp.raise_for_status()
    return resp.json()["response"].strip()


def main():
    from datasets import load_dataset

    print(f"Loading Open-MELON {SPLIT} split...", flush=True)
    ds = load_dataset("MartiHan/Open-MELON-VL-2.5K", split=SPLIT)

    # Load target indices (melanoma + nevus only)
    with open(INDEX_FILE) as f:
        idx_data = json.load(f)
    target_indices = sorted(idx_data["melanoma"] + idx_data["nevus"])
    n = len(target_indices)

    # Build label lookup
    mel_set = set(idx_data["melanoma"])
    label_of = {i: "melanoma" if i in mel_set else "nevus" for i in target_indices}

    done = load_completed(OUTPUT_FILE)
    remaining = len([i for i in target_indices if i not in done])
    print(f"Target: {n} | Done: {len([i for i in target_indices if i in done])} | Remaining: {remaining}", flush=True)

    if remaining == 0:
        print("All captions already cleaned!")
        return

    total_time = 0.0
    processed = 0

    for i in target_indices:
        if i in done:
            continue

        caption = ds[i]["caption"]
        t0 = time.time()
        cleaned = ask_gemma(caption)
        elapsed = time.time() - t0

        result = {
            "index": i,
            "label": label_of[i],
            "original": caption,
            "cleaned": cleaned,
            "time_s": round(elapsed, 2),
        }

        append_result(OUTPUT_FILE, result)
        done[i] = result
        processed += 1
        total_time += elapsed

        avg = total_time / processed
        eta_min = remaining * avg / 60
        remaining -= 1
        print(
            f"[{n - remaining}/{n}] idx={i} | {elapsed:.1f}s | "
            f"avg={avg:.1f}s | ETA={eta_min:.1f}min",
            flush=True,
        )

    print(f"\nDone! Cleaned {processed} captions → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
