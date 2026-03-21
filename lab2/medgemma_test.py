"""
MedGemma 4B test script — run on Google Colab (free T4 GPU).
Generates descriptions for 50 Open-MELON images and saves results.

Usage (in Colab):
  1. Upload this file or paste into a cell
  2. Set your HF token below (or use huggingface-cli login)
  3. Run: !pip install transformers accelerate datasets -q
  4. Run the script
"""

import os, time, json

# ── CONFIG ──────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Set HF_TOKEN environment variable
N_IMAGES = 50
MODEL_ID = "google/medgemma-4b-it"
OUTPUT_FILE = "medgemma_results_50.json"
# ────────────────────────────────────────────────────────────────────

os.environ["HF_TOKEN"] = HF_TOKEN

from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

def main():
    print("Loading Open-MELON dataset...")
    ds = load_dataset("MartiHan/Open-MELON-VL-2.5K", split="train")
    print(f"Dataset loaded: {len(ds)} samples. Using first {N_IMAGES}.")

    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"Model loaded in {time.time()-t0:.1f}s")

    prompt = (
        "You are an expert pathologist. Describe this histopathology image in detail. "
        "Include: tissue type, cell morphology, growth pattern, presence of pigment, "
        "inflammation, necrosis, and any other notable features. "
        "End with your most likely diagnosis."
    )

    results = []
    total_time = 0

    for i in range(min(N_IMAGES, len(ds))):
        sample = ds[i]
        image = sample["image"]
        caption = sample["caption"]

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

        t0 = time.time()
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=300, do_sample=False)
            generation = generation[0][input_len:]
        elapsed = time.time() - t0
        total_time += elapsed

        decoded = processor.decode(generation, skip_special_tokens=True)

        results.append({
            "index": i,
            "pmc_id": sample["pmc_id"],
            "ground_truth_caption": caption,
            "medgemma_output": decoded,
            "inference_time_s": round(elapsed, 2),
        })

        print(f"[{i+1}/{N_IMAGES}] {elapsed:.1f}s | {decoded[:100]}...")

    # Save results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    avg_time = total_time / len(results) if results else 0
    print(f"\nDone! {len(results)} images processed.")
    print(f"Average time per image: {avg_time:.1f}s")
    print(f"Total time: {total_time:.1f}s")
    print(f"Results saved to {OUTPUT_FILE}")
    print(f"\nEstimated time for full dataset (2126 images): {avg_time * 2126 / 3600:.1f} hours")

if __name__ == "__main__":
    main()
