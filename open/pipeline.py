#!/usr/bin/env python3
"""
Parallel caption generation and evaluation pipeline.

Runs 3 threads concurrently:
  1. Generator: MedGemma caption generation (Ollama)
  2. Classifier: Keyword-based melanoma/nevus classification (instant)
  3. Evaluator: RAGAS faithfulness/relevance scoring (Gemini API)

After completing one prompt, automatically moves to the next.

Usage:
  export GEMINI_API_KEY="..."
  python pipeline.py                     # Run all prompts
  python pipeline.py --prompt-id baseline # Run specific prompt
  python pipeline.py -n 50               # Test with first 50 images
"""

import os
import sys
import json
import time
import re
import io
import base64
import argparse
import threading
from queue import Queue, Empty
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import yaml

# === CONFIG ===
SCRIPT_DIR = Path(__file__).parent
CONFIG_FILE = SCRIPT_DIR / "configs" / "prompts.yaml"
INDEX_FILE = SCRIPT_DIR / "indices" / "melanoma_nevus_indices.json"
CAPTIONS_FILE = SCRIPT_DIR / "captions" / "captions_cleaned.jsonl"
OLLAMA_MODEL = "dcarrascosa/medgemma-1.5-4b-it:Q4_K_M"
GEMINI_MODEL = "gemma-3-27b-it"
API_KEY = os.environ.get("GEMINI_API_KEY", "")


@dataclass
class Caption:
    """Container for caption data flowing through pipeline."""
    index: int
    label: str  # ground truth
    pmc_id: str
    caption_gt: str
    caption_gen: str = ""
    pred_label: str = ""
    correct: bool = False
    faithfulness: float = 0.0
    relevance: float = 0.0
    faith_reason: str = ""
    rel_reason: str = ""


def load_config():
    """Load prompt configs from YAML."""
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)["prompts"]


def load_indices():
    """Load melanoma/nevus indices."""
    with open(INDEX_FILE) as f:
        data = json.load(f)
    melanoma_set = set(data["melanoma"])
    all_indices = sorted(data["melanoma"] + data["nevus"])
    return all_indices, melanoma_set


def load_cleaned_captions():
    """Load cleaned captions for RAGAS evaluation from captions_cleaned.jsonl."""
    captions = {}
    with open(CAPTIONS_FILE) as f:
        for line in f:
            entry = json.loads(line)
            captions[entry["index"]] = {
                "label": entry["label"],
                "cleaned": entry["cleaned"],
                "original": entry.get("original", ""),
            }
    return captions


def image_to_base64(pil_image):
    """Convert PIL image to base64."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def strip_confidence(caption: str) -> str:
    """Strip confidence level from caption format: 'Description. Diagnosis. Confidence.'"""
    # Remove common confidence patterns like "High.", "Low.", "Medium.", "High confidence.", etc.
    import re
    # Remove trailing confidence indicators
    caption = re.sub(r'\s*(High|Medium|Low|Moderate)\.?\s*$', '', caption, flags=re.IGNORECASE)
    caption = re.sub(r'\s*(High|Medium|Low|Moderate)\s+confidence\.?\s*$', '', caption, flags=re.IGNORECASE)
    return caption.strip()


# === THREAD 1: GENERATOR ===
def generator_thread(
    prompt: str,
    target_indices: list,
    melanoma_set: set,
    dataset,
    cleaned_captions: dict,
    output_queue: Queue,
    caption_file: Path,
    done_indices: set,
    stop_event: threading.Event,
):
    """Generate captions using MedGemma via Ollama."""
    import ollama

    for idx in target_indices:
        if stop_event.is_set():
            break
        if idx in done_indices:
            continue

        sample = dataset[idx]
        img_b64 = image_to_base64(sample["image"])
        
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
                options={"num_predict": 200, "temperature": 0},
            )
            caption_gen = response.message.content.strip()
        except Exception as e:
            print(f"  [GEN] Error idx={idx}: {e}")
            continue

        label = "melanoma" if idx in melanoma_set else "nevus"
        
        # Use cleaned caption from captions_cleaned.jsonl for RAGAS evaluation
        caption_gt = cleaned_captions.get(idx, {}).get("cleaned", sample.get("caption", ""))
        
        cap = Caption(
            index=idx,
            label=label,
            pmc_id=sample["pmc_id"],
            caption_gt=caption_gt,
            caption_gen=caption_gen,
        )

        # Save to file immediately
        with open(caption_file, "a") as f:
            f.write(json.dumps({
                "index": cap.index,
                "label": cap.label,
                "pmc_id": cap.pmc_id,
                "caption_gt": cap.caption_gt,
                "caption_gen": cap.caption_gen,
            }) + "\n")

        output_queue.put(cap)
        done_indices.add(idx)

    output_queue.put(None)  # Signal done


# === THREAD 2: CLASSIFIER ===
def classifier_thread(
    input_queue: Queue,
    output_queue: Queue,
    results: dict,
    stop_event: threading.Event,
):
    """Classify captions using keyword matching."""
    melanoma_kw = ["melanoma"]
    nevus_kw = ["nevus", "nevi", "naevus", "naevi"]

    while not stop_event.is_set():
        try:
            cap = input_queue.get(timeout=1)
        except Empty:
            continue

        if cap is None:
            output_queue.put(None)
            break

        text = cap.caption_gen.lower()
        has_mel = any(kw in text for kw in melanoma_kw)
        has_nev = any(kw in text for kw in nevus_kw)

        if has_mel and not has_nev:
            cap.pred_label = "melanoma"
        elif has_nev and not has_mel:
            cap.pred_label = "nevus"
        elif has_mel and has_nev:
            cap.pred_label = "melanoma"  # melanoma takes priority
        else:
            cap.pred_label = "unknown"

        cap.correct = cap.pred_label == cap.label

        # Update running stats
        results["total"] += 1
        if cap.correct:
            results["correct"] += 1
        if cap.label == "melanoma":
            results["melanoma_total"] += 1
            if cap.correct:
                results["melanoma_correct"] += 1
        else:
            results["nevus_total"] += 1
            if cap.correct:
                results["nevus_correct"] += 1

        output_queue.put(cap)


# === THREAD 3: RAGAS EVALUATOR ===
def evaluator_thread(
    input_queue: Queue,
    results: list,
    eval_file: Path,
    stop_event: threading.Event,
):
    """Evaluate captions using RAGAS metrics via Gemini API."""
    from google import genai

    if not API_KEY:
        print("  [EVAL] No GEMINI_API_KEY - skipping RAGAS evaluation")
        while not stop_event.is_set():
            try:
                cap = input_queue.get(timeout=1)
                if cap is None:
                    break
            except Empty:
                continue
        return

    client = genai.Client(api_key=API_KEY)

    def call_gemma(prompt: str, max_retries: int = 5) -> dict:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[{"role": "user", "parts": [{"text": prompt}]}],
                    config={"temperature": 0, "max_output_tokens": 500},
                )
                text = response.text.strip()
                # Parse score and reason
                score_match = re.search(r'"?score"?\s*[:=]\s*([0-9.]+)', text)
                reason_match = re.search(r'"?reason"?\s*[:=]\s*"?([^"]+)"?', text, re.IGNORECASE)
                return {
                    "score": float(score_match.group(1)) if score_match else 0.0,
                    "reason": reason_match.group(1).strip() if reason_match else text[:200],
                }
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = (2 ** attempt) + 1
                    time.sleep(wait)
                else:
                    return {"score": 0.0, "reason": f"Error: {e}"}
        return {"score": 0.0, "reason": "Max retries exceeded"}

    faith_prompt_template = """Evaluate faithfulness: are claims in the generated caption supported by the reference?

Reference: {reference}
Generated: {generated}

Score 0.0-1.0 where 1.0 = all claims supported.
Reply ONLY: {{"score": X.XX, "reason": "brief explanation"}}"""

    rel_prompt_template = """Evaluate relevance: does the generated caption address the same medical findings as the reference?

Reference: {reference}
Generated: {generated}

Score 0.0-1.0 where 1.0 = fully relevant.
Reply ONLY: {{"score": X.XX, "reason": "brief explanation"}}"""

    while not stop_event.is_set():
        try:
            cap = input_queue.get(timeout=1)
        except Empty:
            continue

        if cap is None:
            break

        # Strip confidence level from generated caption for RAGAS evaluation
        caption_for_eval = strip_confidence(cap.caption_gen)

        # Faithfulness
        faith_prompt = faith_prompt_template.format(
            reference=cap.caption_gt[:500],
            generated=caption_for_eval[:500],
        )
        faith_result = call_gemma(faith_prompt)
        cap.faithfulness = faith_result["score"]
        cap.faith_reason = faith_result["reason"]

        # Relevance
        rel_prompt = rel_prompt_template.format(
            reference=cap.caption_gt[:500],
            generated=caption_for_eval[:500],
        )
        rel_result = call_gemma(rel_prompt)
        cap.relevance = rel_result["score"]
        cap.rel_reason = rel_result["reason"]

        # Save result
        result_entry = {
            "index": cap.index,
            "label": cap.label,
            "pred_label": cap.pred_label,
            "correct": cap.correct,
            "faithfulness": cap.faithfulness,
            "relevance": cap.relevance,
            "faith_reason": cap.faith_reason,
            "rel_reason": cap.rel_reason,
        }
        results.append(result_entry)

        # Append to file
        with open(eval_file, "a") as f:
            f.write(json.dumps(result_entry) + "\n")


def load_done_indices(caption_file: Path) -> set:
    """Load indices already processed from JSONL."""
    done = set()
    if caption_file.exists():
        with open(caption_file) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["index"])
                except:
                    pass
    return done


def print_progress(results: dict, total: int, prompt_id: str):
    """Print live progress stats."""
    if results["total"] == 0:
        return
    acc = results["correct"] / results["total"] * 100
    sens = results["melanoma_correct"] / max(1, results["melanoma_total"]) * 100
    spec = results["nevus_correct"] / max(1, results["nevus_total"]) * 100
    print(
        f"\r[{prompt_id}] {results['total']}/{total} | "
        f"Acc: {acc:.1f}% | Sens: {sens:.1f}% | Spec: {spec:.1f}%",
        end="", flush=True
    )


def run_prompt(prompt_id: str, prompt_text: str, target_indices: list, melanoma_set: set, dataset, cleaned_captions: dict):
    """Run full pipeline for one prompt."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {prompt_id}")
    print(f"Prompt: {prompt_text[:80]}...")
    print(f"{'='*60}")

    # Setup paths
    caption_dir = SCRIPT_DIR / "captions" / prompt_id
    result_dir = SCRIPT_DIR / "results" / prompt_id
    caption_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    caption_file = caption_dir / "captions.jsonl"
    eval_file = result_dir / "eval_live.jsonl"

    # Load already done
    done_indices = load_done_indices(caption_file)
    remaining = len([i for i in target_indices if i not in done_indices])
    print(f"Total: {len(target_indices)} | Done: {len(done_indices)} | Remaining: {remaining}")

    if remaining == 0:
        print("Already complete!")
        return

    # Queues
    gen_to_class = Queue(maxsize=10)
    class_to_eval = Queue(maxsize=10)

    # Shared state
    class_results = {
        "total": 0, "correct": 0,
        "melanoma_total": 0, "melanoma_correct": 0,
        "nevus_total": 0, "nevus_correct": 0,
    }
    eval_results = []
    stop_event = threading.Event()

    # Start threads
    gen_thread = threading.Thread(
        target=generator_thread,
        args=(prompt_text, target_indices, melanoma_set, dataset, cleaned_captions, gen_to_class, caption_file, done_indices, stop_event),
    )
    class_thread = threading.Thread(
        target=classifier_thread,
        args=(gen_to_class, class_to_eval, class_results, stop_event),
    )
    eval_thread = threading.Thread(
        target=evaluator_thread,
        args=(class_to_eval, eval_results, eval_file, stop_event),
    )

    gen_thread.start()
    class_thread.start()
    eval_thread.start()

    # Progress monitoring
    try:
        while gen_thread.is_alive() or class_thread.is_alive():
            print_progress(class_results, len(target_indices), prompt_id)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()

    gen_thread.join()
    class_thread.join()
    eval_thread.join()

    # Final summary
    print()
    if class_results["total"] > 0:
        acc = class_results["correct"] / class_results["total"] * 100
        sens = class_results["melanoma_correct"] / max(1, class_results["melanoma_total"]) * 100
        spec = class_results["nevus_correct"] / max(1, class_results["nevus_total"]) * 100
        
        summary = {
            "prompt_id": prompt_id,
            "timestamp": datetime.now().isoformat(),
            "total": class_results["total"],
            "accuracy": round(acc, 2),
            "sensitivity": round(sens, 2),
            "specificity": round(spec, 2),
        }
        
        # Add RAGAS averages if available
        if eval_results:
            summary["avg_faithfulness"] = round(sum(r["faithfulness"] for r in eval_results) / len(eval_results), 3)
            summary["avg_relevance"] = round(sum(r["relevance"] for r in eval_results) / len(eval_results), 3)
        
        # Save summary
        with open(result_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ {prompt_id} complete: Acc={acc:.1f}%, Sens={sens:.1f}%, Spec={spec:.1f}%")
        print(f"  Saved: {caption_file}")
        print(f"  Saved: {result_dir / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="Parallel caption generation and evaluation pipeline")
    parser.add_argument("--prompt-id", help="Run specific prompt only")
    parser.add_argument("-n", type=int, default=-1, help="Process only first N images")
    args = parser.parse_args()

    # Load config and indices
    prompts = load_config()
    all_indices, melanoma_set = load_indices()
    
    # Load cleaned captions for RAGAS evaluation
    print("Loading cleaned captions from captions_cleaned.jsonl...", flush=True)
    cleaned_captions = load_cleaned_captions()
    print(f"Loaded {len(cleaned_captions)} cleaned captions")
    
    if args.n > 0:
        all_indices = all_indices[:args.n]
    
    print(f"Total images to process: {len(all_indices)}")
    print(f"Available prompts: {list(prompts.keys())}")

    # Load dataset
    print("Loading dataset...", flush=True)
    from datasets import load_dataset, concatenate_datasets
    ds_raw = load_dataset("MartiHan/Open-MELON-VL-2.5K")
    dataset = concatenate_datasets([ds_raw["train"], ds_raw["validation"], ds_raw["test"]])
    print(f"Dataset loaded: {len(dataset)} images")

    # Run prompts
    prompt_ids = [args.prompt_id] if args.prompt_id else list(prompts.keys())
    
    for pid in prompt_ids:
        if pid not in prompts:
            print(f"Unknown prompt: {pid}")
            continue
        run_prompt(pid, prompts[pid]["prompt"], all_indices, melanoma_set, dataset, cleaned_captions)

    print("\n" + "="*60)
    print("ALL PROMPTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
