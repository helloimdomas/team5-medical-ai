"""
Clean Open-MELON captions by removing non-visual information.
Uses gemma-3-27b-it via Gemini API to rewrite captions keeping only visual descriptions.
Saves incrementally to JSONL with resume support.

Setup:
  export GEMINI_API_KEY="your-api-key"
  uv pip install google-genai datasets
  python clean_captions.py
"""

import json, time, os, argparse
from pathlib import Path

# CONFIG 
MODEL = "gemma-3-27b-it"
SCRIPT_DIR = Path(__file__).parent
INDEX_FILE = SCRIPT_DIR / "indices" / "melanoma_nevus_indices.json"
API_KEY = os.environ.get("GEMINI_API_KEY", "")

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


def ask_gemma(caption: str, client, max_retries: int = 5) -> str:
    """Call Gemma via Gemini API with exponential backoff."""
    prompt = f"Rewrite this caption keeping only visual descriptions:\n\n{caption}"
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[{"role": "user", "parts": [{"text": SYSTEM_PROMPT + "\n\n" + prompt}]}],
                config={"temperature": 0, "max_output_tokens": 300},
            )
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = (2 ** attempt) + 1  # 2, 3, 5, 9, 17 seconds
                print(f"  Rate limited. Waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise
    raise Exception(f"Max retries ({max_retries}) exceeded")


def main():
    from datasets import load_dataset, concatenate_datasets
    from google import genai
    
    parser = argparse.ArgumentParser(description="Clean Open-MELON captions")
    parser.add_argument("--api-key", help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("-n", type=int, default=-1, help="Process only first N images (-1 for all)")
    args = parser.parse_args()
    
    api_key = args.api_key or API_KEY
    if not api_key:
        print("Error: Set GEMINI_API_KEY environment variable or use --api-key")
        return
    
    client = genai.Client(api_key=api_key)
    
    # Output path
    output_file = SCRIPT_DIR / "captions" / "captions_cleaned.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("Loading Open-MELON dataset (all splits)...", flush=True)
    ds_raw = load_dataset("MartiHan/Open-MELON-VL-2.5K")
    ds = concatenate_datasets([ds_raw["train"], ds_raw["validation"], ds_raw["test"]])
    print(f"Total images: {len(ds)}")

    # Load target indices (melanoma + nevus only)
    with open(INDEX_FILE) as f:
        idx_data = json.load(f)
    target_indices = sorted(idx_data["melanoma"] + idx_data["nevus"])
    
    if args.n > 0:
        target_indices = target_indices[:args.n]
    n = len(target_indices)

    # Build label lookup
    mel_set = set(idx_data["melanoma"])
    label_of = {i: "melanoma" if i in mel_set else "nevus" for i in target_indices}

    done = load_completed(output_file)
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
        
        try:
            cleaned = ask_gemma(caption, client)
        except Exception as e:
            print(f"Error on idx={i}: {e}")
            time.sleep(2)
            continue
            
        elapsed = time.time() - t0

        result = {
            "index": i,
            "label": label_of[i],
            "original": caption,
            "cleaned": cleaned,
            "time_s": round(elapsed, 2),
        }

        append_result(output_file, result)
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

    print(f"\nDone! Cleaned {processed} captions → {output_file}")


if __name__ == "__main__":
    main()
