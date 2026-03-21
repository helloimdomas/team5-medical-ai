"""
MedGemma caption-only evaluation on Open-MELON — LOCAL version (Ollama).
Generates captions that include visual descriptions + diagnosis names.
No binary feature tasks. Saves incrementally to JSONL with resume support.

Setup:
  1. brew install ollama && brew services start ollama
  2. ollama pull dcarrascosa/medgemma-1.5-4b-it:Q4_K_M
  3. uv pip install ollama datasets Pillow
  4. python medgemma_captions_local.py
"""

import os, time, json, base64, io

N_IMAGES = -1  # -1 = all images in index file
MODEL = "dcarrascosa/medgemma-1.5-4b-it:Q4_K_M"
INDEX_FILE = os.path.join(os.path.dirname(__file__), "melanoma_nevus_indices.json")
CACHE_DIR = os.path.expanduser("~/.cache/huggingface")

# === PROMPT CONFIGURATIONS ===
# Each config: (output_file, prompt)
PROMPT_CONFIGS = [
    (
        "medgemma_captions_v3.jsonl",
        "You are a pathologist writing a brief report. Describe the key visual findings in 2-3 sentences. Do not list differentials or next steps. Provide your best guess for the diagnosis and your certainty level (either high or low)"
    ),
    (
        "medgemma_captions_v4.jsonl",
        "You are a dermatopathologist examining a skin biopsy. This lesion is either MELANOMA or a BENIGN NEVUS. Describe the key histological features in 2-3 sentences, then state your diagnosis (melanoma or benign nevus) and certainty level (high/low)."
    ),
]


def image_to_base64(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_completed(output_file):
    done = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["index"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def append_result(output_file, result):
    with open(output_file, "a") as f:
        f.write(json.dumps(result) + "\n")


def ask_ollama(model, img_b64, prompt, max_tokens):
    import ollama
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
        options={"num_predict": max_tokens, "temperature": 0},
    )
    return response.message.content.strip()


def main():
    import ollama
    from datasets import load_dataset

    try:
        models = ollama.list()
        model_names = [m.model for m in models.models]
        if not any(MODEL in name for name in model_names):
            print(f"Model '{MODEL}' not found. Run: ollama pull {MODEL}")
            return
    except Exception as e:
        print(f"Ollama not running? Start with: ollama serve\nError: {e}")
        return

    print("Loading Open-MELON dataset...", flush=True)
    ds = load_dataset("MartiHan/Open-MELON-VL-2.5K", split="train", cache_dir=CACHE_DIR)

    # Load target indices (melanoma + nevus only)
    with open(INDEX_FILE) as f:
        idx_data = json.load(f)
    target_indices = sorted(idx_data["melanoma"] + idx_data["nevus"])
    if N_IMAGES > 0:
        target_indices = target_indices[:N_IMAGES]
    n = len(target_indices)

    # Build label lookup
    mel_set = set(idx_data["melanoma"])
    label_of = {i: "melanoma" if i in mel_set else "nevus" for i in target_indices}

    done = load_completed(OUTPUT_FILE)
    print(f"Dataset: {len(ds)} | Target: {n} | Done: {len(done)} | Remaining: {n - len(done)}", flush=True)

    total_time = 0.0
    processed = 0

    for i in target_indices:
        if i in done:
            continue

        sample = ds[i]
        img_b64 = image_to_base64(sample["image"])

        t0 = time.time()
        caption = ask_ollama(MODEL, img_b64, CAPTION_PROMPT, max_tokens=200)
        elapsed = time.time() - t0

        result = {
            "index": i,
            "label": label_of[i],
            "pmc_id": sample["pmc_id"],
            "caption_gt": sample["caption"],
            "caption_gen": caption,
            "time_s": round(elapsed, 2),
        }

        append_result(OUTPUT_FILE, result)
        done.add(i)
        processed += 1
        total_time += elapsed

        avg = total_time / processed
        remaining = (n - len(done)) * avg / 60
        print(
            f"[{len(done)}/{n}] idx={i} | {elapsed:.1f}s | "
            f"avg={avg:.1f}s/img | ETA={remaining:.1f}min",
            flush=True,
        )

    print(f"\nDone! {processed} captions → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
