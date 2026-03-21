"""
MedGemma structured evaluation on Open-MELON — COLAB version (HF Transformers).
Processes one image at a time (all features + caption), saves incrementally.
Resumes from where it left off if interrupted.

Setup in Colab:
  1. Accept license: https://huggingface.co/google/medgemma-4b-it
  2. !pip install transformers accelerate datasets bitsandbytes -q
  3. Set HF_TOKEN below or as env var
  4. Runtime > Change runtime type > T4 GPU
  5. Run this script
"""

import os, time, json, torch

# ── CONFIG ──────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_TOKEN_HERE")
MODEL_ID = "google/medgemma-4b-it"
N_IMAGES = -1  # -1 = all images
OUTPUT_FILE = "medgemma_eval_colab.jsonl"
DRIVE_OUTPUT = "/content/drive/MyDrive/medgemma_eval_colab.jsonl"
USE_4BIT = True  # 4-bit quantization to fit on T4 (15GB)
# ────────────────────────────────────────────────────────────────────

os.environ["HF_TOKEN"] = HF_TOKEN

TASKS = [
    {"name": "cyto_pigment",            "prompt": "Does this histopathology image show pigment (melanin)? Answer ONLY 'yes' or 'no'.", "gt_key": "cyto_pigment",            "gt_map": {True: "yes", False: "no"}, "type": "binary"},
    {"name": "ctx_inflammation",        "prompt": "Does this histopathology image show inflammation (inflammatory infiltrate)? Answer ONLY 'yes' or 'no'.", "gt_key": "ctx_inflammation",        "gt_map": {True: "yes", False: "no"}, "type": "binary"},
    {"name": "ctx_necrosis",            "prompt": "Does this histopathology image show necrosis? Answer ONLY 'yes' or 'no'.", "gt_key": "ctx_necrosis",            "gt_map": {True: "yes", False: "no"}, "type": "binary"},
    {"name": "ctx_fibrosis",            "prompt": "Does this histopathology image show fibrosis? Answer ONLY 'yes' or 'no'.", "gt_key": "ctx_fibrosis",            "gt_map": {True: "yes", False: "no"}, "type": "binary"},
    {"name": "cyto_prominent_nucleoli", "prompt": "Does this histopathology image show prominent nucleoli? Answer ONLY 'yes' or 'no'.", "gt_key": "cyto_prominent_nucleoli", "gt_map": {True: "yes", False: "no"}, "type": "binary"},
    {"name": "arch_growth_pattern",     "prompt": "Is the growth pattern in this histopathology image infiltrative or circumscribed? Answer ONLY 'infiltrative' or 'circumscribed'.", "gt_key": "arch_growth_pattern", "gt_map": {"infiltrative": "infiltrative", "circumscribed": "circumscribed", "unknown": None}, "type": "binary"},
    {"name": "caption",                 "prompt": "Describe the visual findings in this histopathology image in 2-3 sentences. Focus only on what you can see: tissue architecture, cell morphology, pigment, inflammation, and notable features. Do not mention staining method, magnification, or patient history.", "gt_key": "caption", "gt_map": None, "type": "caption"},
]

def parse_answer(text, valid_answers):
    text_lower = text.strip().lower()
    for ans in valid_answers:
        if ans in text_lower:
            return ans
    return text_lower[:30]

def load_completed(output_file):
    done = set()
    for path in [output_file, DRIVE_OUTPUT]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        done.add(json.loads(line)["index"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return done

def append_result(result):
    line = json.dumps(result) + "\n"
    with open(OUTPUT_FILE, "a") as f:
        f.write(line)
    try:
        with open(DRIVE_OUTPUT, "a") as f:
            f.write(line)
    except Exception:
        pass

def ask_model(model, processor, image, prompt, max_tokens=20):
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        generation = generation[0][input_len:]

    return processor.decode(generation, skip_special_tokens=True).strip()

def main():
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from datasets import load_dataset

    # Mount Drive for persistent storage
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
        os.makedirs("/content/drive/MyDrive/hf_cache", exist_ok=True)
        print("Drive mounted, using Drive cache", flush=True)
    except ImportError:
        print("Not on Colab, using default cache", flush=True)

    print(f"Loading model: {MODEL_ID} ({'4-bit' if USE_4BIT else 'bf16'})", flush=True)
    t0 = time.time()

    load_kwargs = {
        "device_map": "auto",
    }
    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        load_kwargs["dtype"] = torch.bfloat16

    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **load_kwargs)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

    print("Loading Open-MELON dataset...", flush=True)
    ds = load_dataset("MartiHan/Open-MELON-VL-2.5K", split="train")
    n = min(N_IMAGES, len(ds)) if N_IMAGES > 0 else len(ds)

    done = load_completed(OUTPUT_FILE)
    print(f"Dataset: {len(ds)} | Target: {n} | Already done: {len(done)} | Remaining: {n - len(done)}", flush=True)

    total_time = 0
    processed = 0

    for i in range(n):
        if i in done:
            continue

        sample = ds[i]

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
            raw = ask_model(model, processor, sample["image"], task["prompt"], max_tokens)
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

        append_result(result)
        done.add(i)
        processed += 1
        total_time += img_elapsed

        avg = total_time / processed
        remaining = (n - len(done)) * avg / 60
        print(f"[{len(done)}/{n}] img={i} | {img_elapsed:.1f}s | avg: {avg:.1f}s/img | ETA: {remaining:.1f}min", flush=True)

    print_summary()

def print_summary():
    # Try Drive copy first, fallback to local
    path = DRIVE_OUTPUT if os.path.exists(DRIVE_OUTPUT) else OUTPUT_FILE
    records = []
    with open(path) as f:
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
    print(f"\nResults saved to: {OUTPUT_FILE}")
    if os.path.exists(DRIVE_OUTPUT):
        print(f"Drive backup: {DRIVE_OUTPUT}")

if __name__ == "__main__":
    main()
