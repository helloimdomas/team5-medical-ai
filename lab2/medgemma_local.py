"""
MedGemma structured evaluation on Open-MELON — LOCAL version (Ollama).
Processes one image at a time (all features + caption), saves incrementally.
Resumes from where it left off if interrupted.

Setup:
  1. brew install ollama && brew services start ollama
  2. ollama pull dcarrascosa/medgemma-1.5-4b-it:Q4_K_M
  3. uv pip install ollama datasets Pillow
  4. python medgemma_local.py
"""

import os, time, json, base64, io

N_IMAGES = -1  # -1 = all images
MODEL = "dcarrascosa/medgemma-1.5-4b-it:Q4_K_M"
OUTPUT_FILE = "medgemma_eval_local.jsonl"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface")

TASKS = [
    {"name": "cyto_pigment",            "prompt": "Does this histopathology image show pigment (melanin)? Answer ONLY 'yes' or 'no'.", "gt_key": "cyto_pigment",            "gt_map": {True: "yes", False: "no"}, "type": "binary"},
    {"name": "ctx_inflammation",        "prompt": "Does this histopathology image show inflammation (inflammatory infiltrate)? Answer ONLY 'yes' or 'no'.", "gt_key": "ctx_inflammation",        "gt_map": {True: "yes", False: "no"}, "type": "binary"},
    {"name": "ctx_necrosis",            "prompt": "Does this histopathology image show necrosis? Answer ONLY 'yes' or 'no'.", "gt_key": "ctx_necrosis",            "gt_map": {True: "yes", False: "no"}, "type": "binary"},
    {"name": "ctx_fibrosis",            "prompt": "Does this histopathology image show fibrosis? Answer ONLY 'yes' or 'no'.", "gt_key": "ctx_fibrosis",            "gt_map": {True: "yes", False: "no"}, "type": "binary"},
    {"name": "cyto_prominent_nucleoli", "prompt": "Does this histopathology image show prominent nucleoli? Answer ONLY 'yes' or 'no'.", "gt_key": "cyto_prominent_nucleoli", "gt_map": {True: "yes", False: "no"}, "type": "binary"},
    {"name": "arch_growth_pattern",     "prompt": "Is the growth pattern in this histopathology image infiltrative or circumscribed? Answer ONLY 'infiltrative' or 'circumscribed'.", "gt_key": "arch_growth_pattern", "gt_map": {"infiltrative": "infiltrative", "circumscribed": "circumscribed", "unknown": None}, "type": "binary"},
    {"name": "caption",                 "prompt": "Describe the visual findings in this histopathology image in 2-3 sentences. Focus only on what you can see: tissue architecture, cell morphology, pigment, inflammation, and notable features. Do not mention staining method, magnification, or patient history.", "gt_key": "caption", "gt_map": None, "type": "caption"},
]

def image_to_base64(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def parse_answer(text, valid_answers):
    text_lower = text.strip().lower()
    for ans in valid_answers:
        if ans in text_lower:
            return ans
    return text_lower[:30]

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
            print(f"Available: {model_names}")
            return
    except Exception as e:
        print(f"Ollama not running? Start with: ollama serve\nError: {e}")
        return

    print("Loading Open-MELON dataset...", flush=True)
    ds = load_dataset("MartiHan/Open-MELON-VL-2.5K", split="train", cache_dir=CACHE_DIR)
    n = min(N_IMAGES, len(ds)) if N_IMAGES > 0 else len(ds)

    done = load_completed(OUTPUT_FILE)
    print(f"Dataset: {len(ds)} | Target: {n} | Already done: {len(done)} | Remaining: {n - len(done)}", flush=True)

    total_time = 0
    processed = 0

    for i in range(n):
        if i in done:
            continue

        sample = ds[i]
        img_b64 = image_to_base64(sample["image"])

        result = {
            "index": i,
            "pmc_id": sample["pmc_id"],
            "features": {},
            "caption_gt": sample["caption"],
            "caption_gen": None,
            "total_time_s": 0,
        }

        img_start = time.time()

        for task in TASKS:
            max_tokens = 150 if task["type"] == "caption" else 20
            t0 = time.time()
            raw = ask_ollama(MODEL, img_b64, task["prompt"], max_tokens)
            elapsed = time.time() - t0

            if task["type"] == "binary":
                gt_raw = sample[task["gt_key"]]
                gt_label = task["gt_map"].get(gt_raw)
                valid = [v for v in task["gt_map"].values() if v is not None]
                pred = parse_answer(raw, valid)

                result["features"][task["name"]] = {
                    "gt": gt_label,
                    "pred": pred,
                    "correct": pred == gt_label if gt_label is not None else None,
                    "raw": raw,
                    "time_s": round(elapsed, 2),
                }
            else:
                result["caption_gen"] = raw
                result["features"]["caption"] = {"time_s": round(elapsed, 2)}

        img_elapsed = time.time() - img_start
        result["total_time_s"] = round(img_elapsed, 2)

        append_result(OUTPUT_FILE, result)
        done.add(i)
        processed += 1
        total_time += img_elapsed

        avg = total_time / processed
        remaining = (n - len(done)) * avg / 60
        print(f"[{len(done)}/{n}] img={i} | {img_elapsed:.1f}s | avg: {avg:.1f}s/img | ETA: {remaining:.1f}min", flush=True)

    print_summary(OUTPUT_FILE)

def print_summary(output_file):
    records = []
    with open(output_file) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"\n{'='*60}")
    print(f"SUMMARY — {len(records)} images")
    print(f"{'='*60}")

    task_names = [t["name"] for t in TASKS if t["type"] == "binary"]
    for tname in task_names:
        correct, total = 0, 0
        for rec in records:
            feat = rec["features"].get(tname, {})
            if feat.get("gt") is not None:
                total += 1
                if feat.get("correct"):
                    correct += 1
        acc = correct / total if total > 0 else 0
        print(f"  {tname:30s} acc={acc:.1%}  ({correct}/{total})")

    caps = sum(1 for r in records if r.get("caption_gen"))
    print(f"  {'caption':30s} {caps} generated")
    print(f"\nResults in: {output_file}")

if __name__ == "__main__":
    main()
