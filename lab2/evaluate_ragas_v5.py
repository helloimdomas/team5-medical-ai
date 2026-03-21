"""
RAGAS-style evaluation for MedGemma captions using Gemma API.

Evaluates:
- Faithfulness: Are claims in generated caption supported by GT caption?
- Relevancy: Is the response relevant to the prompt?

Usage:
    python lab2/evaluate_ragas.py --captions medgemma_captions_v4.jsonl
"""

import json
import os
import argparse
import time
from pathlib import Path

# You can also set GEMINI_API_KEY environment variable
API_KEY = os.environ.get("GEMINI_API_KEY", "")

FAITHFULNESS_PROMPT = """You are evaluating the faithfulness of a generated medical caption against a reference caption.

Reference Caption (ground truth):
{context}

Generated Caption:
{answer}

Task: Determine what fraction of claims in the generated caption are supported by the reference caption.
- A claim is "supported" if it describes the same finding, diagnosis, or characteristic mentioned in the reference.
- Claims about completely different tissues/diagnoses are NOT supported.
- General descriptive terms (e.g., "cells", "tissue") that match are supported.

Score from 0.0 to 1.0 where:
- 1.0 = All claims in generated caption are supported by reference
- 0.5 = About half the claims are supported
- 0.0 = No claims are supported (completely different diagnosis/tissue)

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief explanation>"}}"""

RELEVANCY_PROMPT = """You are evaluating whether a medical caption is relevant to the prompt that generated it.

Prompt:
{question}

Generated Caption:
{answer}

Task: Determine if the generated caption addresses what the prompt asked for.
- Does it describe visual findings as requested?
- Does it provide a diagnosis if asked?
- Does it include certainty level if asked?

Score from 0.0 to 1.0 where:
- 1.0 = Fully addresses all aspects of the prompt
- 0.5 = Partially addresses the prompt
- 0.0 = Completely irrelevant response

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief explanation>"}}"""


def evaluate_with_gemma(prompt: str, api_key: str, model: str = "gemma-3-27b-it") -> dict:
    """Call Gemma API for evaluation."""
    from google import genai
    
    client = genai.Client(api_key=api_key)
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        text = response.text.strip()
        
        # Parse JSON response
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        return json.loads(text.strip())
    except json.JSONDecodeError:
        # Try to extract score from text
        import re
        match = re.search(r'"score":\s*([\d.]+)', text)
        if match:
            return {"score": float(match.group(1)), "reason": text}
        return {"score": 0.0, "reason": f"Failed to parse: {text[:200]}"}
    except Exception as e:
        return {"score": 0.0, "reason": f"API error: {str(e)}"}


def load_captions(filepath: str) -> list[dict]:
    """Load JSONL caption file."""
    captions = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                captions.append(json.loads(line))
    return captions


def load_completed(output_file: str) -> dict:
    """Load already-completed evaluations for resume support."""
    done = {}
    if os.path.exists(output_file):
        try:
            with open(output_file) as f:
                data = json.load(f)
                for r in data.get("results", []):
                    done[r["index"]] = r
        except (json.JSONDecodeError, KeyError):
            pass
    return done


def save_progress(output_file: str, results: list, faithfulness_scores: list, relevancy_scores: list):
    """Save current progress to file."""
    avg_f = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
    avg_r = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0
    
    with open(output_file, "w") as f:
        json.dump({
            "summary": {
                "samples": len(results),
                "avg_faithfulness": avg_f,
                "avg_relevancy": avg_r,
            },
            "results": results,
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="RAGAS-style evaluation for captions")
    parser.add_argument("--captions", default="medgemma_captions_v5.jsonl",
                        help="Path to generated captions JSONL")
    parser.add_argument("--prompt", default="You are a pathologist examining a biopsy. This lesion is either MELANOMA or BENIGN (e.g., nevus). Describe the key histological features in 2-3 sentences, then state your diagnosis (melanoma or benign) and certainty level (high/low).",
                        help="The prompt used to generate captions")
    parser.add_argument("--sample", type=int, default=0,
                        help="Number of samples to evaluate (0 = all)")
    parser.add_argument("--api-key", default="",
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--model", default="gemma-3-27b-it",
                        help="Gemma model to use")
    args = parser.parse_args()
    
    api_key = args.api_key or API_KEY
    if not api_key:
        print("Error: Set GEMINI_API_KEY environment variable or use --api-key")
        return
    
    # Resolve paths
    script_dir = Path(__file__).parent.parent
    captions_path = script_dir / args.captions if not Path(args.captions).is_absolute() else Path(args.captions)
    output_path = captions_path.parent / f"{captions_path.stem}_ragas.json"
    
    print(f"Loading captions from: {captions_path}")
    print(f"Output will be saved to: {output_path}")
    
    captions = load_captions(captions_path)
    
    # Sample if requested
    if args.sample > 0 and args.sample < len(captions):
        import random
        random.seed(42)
        captions = random.sample(captions, args.sample)
    
    # Load already completed evaluations
    done = load_completed(output_path)
    print(f"Total captions: {len(captions)} | Already done: {len(done)} | Remaining: {len(captions) - len(done)}")
    
    # Restore previous results
    results = list(done.values())
    faithfulness_scores = [r["faithfulness"].get("score", 0) for r in results]
    relevancy_scores = [r["relevancy"].get("score", 0) for r in results]
    
    # Track timing
    total_time = 0.0
    processed = 0
    
    try:
        for i, cap in enumerate(captions):
            if cap["index"] in done:
                continue
            
            t0 = time.time()
            
            # Faithfulness evaluation
            faith_prompt = FAITHFULNESS_PROMPT.format(
                context=cap["caption_gt"],
                answer=cap["caption_gen"]
            )
            faith_result = evaluate_with_gemma(faith_prompt, api_key, args.model)
            
            # Relevancy evaluation  
            rel_prompt = RELEVANCY_PROMPT.format(
                question=args.prompt,
                answer=cap["caption_gen"]
            )
            rel_result = evaluate_with_gemma(rel_prompt, api_key, args.model)
            
            elapsed = time.time() - t0
            total_time += elapsed
            processed += 1
            
            result = {
                "index": cap["index"],
                "label": cap["label"],
                "faithfulness": faith_result,
                "relevancy": rel_result,
                "caption_gen": cap["caption_gen"][:200],
                "caption_gt": cap["caption_gt"][:200],
            }
            
            results.append(result)
            done[cap["index"]] = result
            
            faithfulness_scores.append(faith_result.get("score", 0))
            relevancy_scores.append(rel_result.get("score", 0))
            
            # Calculate stats
            avg_f = sum(faithfulness_scores) / len(faithfulness_scores)
            avg_r = sum(relevancy_scores) / len(relevancy_scores)
            avg_time = total_time / processed
            remaining = len(captions) - len(done)
            eta_min = remaining * avg_time / 60
            
            print(
                f"[{len(done)}/{len(captions)}] idx={cap['index']} | "
                f"F={faith_result.get('score', 0):.2f} R={rel_result.get('score', 0):.2f} | "
                f"Avg F={avg_f:.3f} R={avg_r:.3f} | "
                f"{elapsed:.1f}s (avg {avg_time:.1f}s) | ETA={eta_min:.1f}min"
            )
            
            # Save progress every 10 samples
            if processed % 10 == 0:
                save_progress(output_path, results, faithfulness_scores, relevancy_scores)
            
            # Rate limiting
            time.sleep(0.3)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving progress...")
        save_progress(output_path, results, faithfulness_scores, relevancy_scores)
        print(f"Progress saved to: {output_path}")
        print("Run again to resume from where you left off.")
        return
    
    # Final summary
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
    avg_relevancy = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0
    
    print("\n" + "="*60)
    print("RAGAS-STYLE EVALUATION RESULTS")
    print("="*60)
    print(f"Samples evaluated: {len(results)}")
    print(f"Average Faithfulness: {avg_faithfulness:.4f}")
    print(f"Average Relevancy:    {avg_relevancy:.4f}")
    
    # Show some examples
    sorted_by_faith = sorted(results, key=lambda x: x["faithfulness"].get("score", 0))
    
    print("\n--- LOWEST FAITHFULNESS ---")
    for r in sorted_by_faith[:3]:
        print(f"idx={r['index']} | score={r['faithfulness'].get('score', 0):.2f}")
        print(f"  Reason: {r['faithfulness'].get('reason', '')[:100]}")
        print(f"  GT: {r['caption_gt'][:100]}...")
        print(f"  Gen: {r['caption_gen'][:100]}...")
        print()
    
    print("--- HIGHEST FAITHFULNESS ---")
    for r in sorted_by_faith[-3:]:
        print(f"idx={r['index']} | score={r['faithfulness'].get('score', 0):.2f}")
        print(f"  Reason: {r['faithfulness'].get('reason', '')[:100]}")
        print()
    
    # Final save
    save_progress(output_path, results, faithfulness_scores, relevancy_scores)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
