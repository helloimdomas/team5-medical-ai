"""
MedGemma caption generation on Open-MELON — LOCAL version (Ollama).
Generates captions that include visual descriptions + diagnosis names.
Uses configs/prompts.yaml for prompt definitions.

Setup:
  1. brew install ollama && brew services start ollama
  2. ollama pull dcarrascosa/medgemma-1.5-4b-it:Q4_K_M
  3. uv pip install ollama datasets Pillow pyyaml
  4. python generate_captions.py --prompt-id dermatopathologist

Usage:
  python generate_captions.py --prompt-id PROMPT_ID [--limit N]
  
  Available prompt IDs are defined in configs/prompts.yaml
"""

import os, time, json, base64, io, argparse
from pathlib import Path

# === CONFIG ===
MODEL = "dcarrascosa/medgemma-1.5-4b-it:Q4_K_M"
SCRIPT_DIR = Path(__file__).parent
CONFIG_FILE = SCRIPT_DIR / "configs" / "prompts.yaml"
INDEX_FILE = SCRIPT_DIR / "indices" / "melanoma_nevus_indices.json"
CAPTIONS_DIR = SCRIPT_DIR / "captions"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface")


def load_prompts():
    """Load prompt configurations from YAML."""
    import yaml
    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)
    return config["prompts"]


def image_to_base64(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_completed(output_file):
    done = set()
    if output_file.exists():
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
    parser = argparse.ArgumentParser(description="Generate captions using MedGemma")
    parser.add_argument("--prompt-id", default=None,
                        help="Prompt ID from configs/prompts.yaml")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit number of images (-1 = all)")
    parser.add_argument("--list-prompts", action="store_true",
                        help="List available prompt IDs and exit")
    args = parser.parse_args()
    
    # Load prompts
    prompts = load_prompts()
    
    if args.list_prompts:
        print("Available prompt IDs:")
        for pid, pconfig in prompts.items():
            print(f"  {pid}: {pconfig['name']}")
        return
    
    if not args.prompt_id:
        print("Error: --prompt-id is required")
        print(f"Available: {', '.join(prompts.keys())}")
        return
    
    if args.prompt_id not in prompts:
        print(f"Error: Unknown prompt ID '{args.prompt_id}'")
        print(f"Available: {', '.join(prompts.keys())}")
        return
    
    prompt_config = prompts[args.prompt_id]
    prompt = prompt_config["prompt"].strip()
    
    import ollama
    from datasets import load_dataset, concatenate_datasets

    try:
        models = ollama.list()
        model_names = [m.model for m in models.models]
        if not any(MODEL in name for name in model_names):
            print(f"Model '{MODEL}' not found. Run: ollama pull {MODEL}")
            return
    except Exception as e:
        print(f"Ollama not running? Start with: ollama serve\nError: {e}")
        return

    print(f"Prompt ID: {args.prompt_id}")
    print(f"Prompt: {prompt[:80]}...")
    print("Loading Open-MELON dataset (all splits)...", flush=True)
    
    # Load all splits and concatenate (indices are global across all splits)
    ds_dict = load_dataset("MartiHan/Open-MELON-VL-2.5K", cache_dir=CACHE_DIR)
    ds = concatenate_datasets([ds_dict["train"], ds_dict["validation"], ds_dict["test"]])
    print(f"Total images: {len(ds)}")

    # Load target indices
    with open(INDEX_FILE) as f:
        idx_data = json.load(f)
    target_indices = sorted(idx_data["melanoma"] + idx_data["nevus"])
    if args.limit > 0:
        target_indices = target_indices[:args.limit]
    n = len(target_indices)

    # Build label lookup
    mel_set = set(idx_data["melanoma"])
    label_of = {i: "melanoma" if i in mel_set else "nevus" for i in target_indices}

    # Setup output path
    output_dir = CAPTIONS_DIR / args.prompt_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "full.jsonl"

    done = load_completed(output_file)
    print(f"Dataset: {len(ds)} | Target: {n} | Done: {len(done)} | Remaining: {n - len(done)}", flush=True)

    if len(done) >= n:
        print("Already complete!")
        return

    total_time = 0.0
    processed = 0

    for i in target_indices:
        if i in done:
            continue

        sample = ds[i]
        img_b64 = image_to_base64(sample["image"])

        t0 = time.time()
        caption = ask_ollama(MODEL, img_b64, prompt, max_tokens=200)
        elapsed = time.time() - t0

        result = {
            "index": i,
            "label": label_of[i],
            "pmc_id": sample["pmc_id"],
            "caption_gt": sample["caption"],
            "caption_gen": caption,
            "time_s": round(elapsed, 2),
        }

        append_result(output_file, result)
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

    print(f"\nDone! {processed} captions → {output_file}")


if __name__ == "__main__":
    main()
