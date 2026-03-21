"""
Evaluate MedGemma captions for diagnosis accuracy + BERTScore.

Usage:
    python lab2/evaluate_captions.py [--captions FILE] [--indices FILE] [--top N]

Example:
    python lab2/evaluate_captions.py --captions medgemma_captions_v2.jsonl
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

# Keyword patterns for classification
MELANOMA_PATTERNS = [
    r'\bmelanoma\b',
    r'\bmalignant\s+melanocyt',
    r'\bmetastatic\s+melanoma\b',
]

BENIGN_PATTERNS = [
    r'\bnevus\b',
    r'\bnevi\b',
    r'\bbenign\b',
    r'\bspitz\b',
    r'\bblue\s+nevus\b',
    r'\bdysplastic\b',
    r'\bcompound\s+nevus\b',
    r'\bjunctional\s+nevus\b',
    r'\bintradermal\b',
    r'\bcongenital\s+nevus\b',
    r'\breed\s+nevus\b',
    r'\bnon-?malignant\b',
]


def classify_caption(text: str) -> str | None:
    """
    Classify caption as 'melanoma', 'benign', or None (indeterminate).
    Returns the predicted class based on keyword matching.
    """
    text_lower = text.lower()
    
    has_melanoma = any(re.search(p, text_lower) for p in MELANOMA_PATTERNS)
    has_benign = any(re.search(p, text_lower) for p in BENIGN_PATTERNS)
    
    if has_melanoma and not has_benign:
        return "melanoma"
    elif has_benign and not has_melanoma:
        return "benign"
    elif has_melanoma and has_benign:
        # Both present - check which appears first or dominates
        # Default to melanoma if explicitly mentioned
        return "melanoma" if "melanoma" in text_lower else "benign"
    else:
        return None  # Indeterminate


def load_captions(filepath: str) -> list[dict]:
    """Load JSONL caption file."""
    captions = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                captions.append(json.loads(line))
    return captions


def load_indices(filepath: str) -> dict:
    """Load ground truth indices."""
    with open(filepath) as f:
        return json.load(f)


def compute_bertscore(generated: list[str], references: list[str], batch_size: int = 32):
    """Compute BERTScore for generated vs reference captions."""
    try:
        from bert_score import score
    except ImportError:
        print("Installing bert-score...")
        import subprocess
        subprocess.run(["pip", "install", "bert-score", "-q"], check=True)
        from bert_score import score
    
    print("Computing BERTScore (this may take a minute)...")
    P, R, F1 = score(generated, references, lang="en", verbose=False, batch_size=batch_size)
    return P.tolist(), R.tolist(), F1.tolist()


def main():
    parser = argparse.ArgumentParser(description="Evaluate MedGemma captions")
    parser.add_argument("--captions", default="medgemma_captions_v2.jsonl",
                        help="Path to generated captions JSONL")
    parser.add_argument("--indices", default="lab2/melanoma_nevus_indices.json",
                        help="Path to ground truth indices JSON")
    parser.add_argument("--top", type=int, default=5,
                        help="Number of best/worst examples to show")
    parser.add_argument("--no-bert", action="store_true",
                        help="Skip BERTScore computation")
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent.parent
    captions_path = script_dir / args.captions if not Path(args.captions).is_absolute() else Path(args.captions)
    indices_path = script_dir / args.indices if not Path(args.indices).is_absolute() else Path(args.indices)
    
    print(f"Loading captions from: {captions_path}")
    print(f"Loading indices from: {indices_path}")
    
    captions = load_captions(captions_path)
    indices = load_indices(indices_path)
    
    # Build ground truth lookup
    melanoma_set = set(indices["melanoma"])
    nevus_set = set(indices["nevus"])
    
    def get_gt_label(idx):
        if idx in melanoma_set:
            return "melanoma"
        elif idx in nevus_set:
            return "benign"
        return "unknown"
    
    # Classify and evaluate
    results = []
    confusion = defaultdict(int)  # (gt, pred) -> count
    
    for item in captions:
        idx = item["index"]
        gt_label = get_gt_label(idx)
        pred_label = classify_caption(item["caption_gen"])
        
        # Map gt melanoma/nevus to melanoma/benign for comparison
        gt_binary = "melanoma" if gt_label == "melanoma" else "benign"
        
        correct = (pred_label == gt_binary) if pred_label else False
        
        results.append({
            "index": idx,
            "gt_label": gt_binary,
            "pred_label": pred_label,
            "correct": correct,
            "caption_gen": item["caption_gen"],
            "caption_gt": item["caption_gt"],
        })
        
        confusion[(gt_binary, pred_label or "indeterminate")] += 1
    
    # Summary statistics
    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    indeterminate = sum(1 for r in results if r["pred_label"] is None)
    determinate = total - indeterminate
    
    print("\n" + "="*60)
    print("DIAGNOSIS CLASSIFICATION RESULTS")
    print("="*60)
    print(f"Total captions: {total}")
    print(f"Determinate (has keyword): {determinate} ({100*determinate/total:.1f}%)")
    print(f"Indeterminate (no keyword): {indeterminate} ({100*indeterminate/total:.1f}%)")
    print(f"\nAmong determinate predictions:")
    determinate_correct = sum(1 for r in results if r["pred_label"] and r["correct"])
    if determinate > 0:
        print(f"  Correct: {determinate_correct}/{determinate} ({100*determinate_correct/determinate:.1f}%)")
    
    print(f"\nOverall (counting indeterminate as wrong):")
    print(f"  Correct: {correct_count}/{total} ({100*correct_count/total:.1f}%)")
    
    # Confusion matrix
    print("\nConfusion Matrix (rows=GT, cols=Pred):")
    print(f"{'':15} {'melanoma':>12} {'benign':>12} {'indeterminate':>15}")
    for gt in ["melanoma", "benign"]:
        row = f"{gt:15}"
        for pred in ["melanoma", "benign", "indeterminate"]:
            row += f" {confusion[(gt, pred)]:>12}"
        print(row)
    
    # BERTScore
    if not args.no_bert:
        generated = [r["caption_gen"] for r in results]
        references = [r["caption_gt"] for r in results]
        
        P, R, F1 = compute_bertscore(generated, references)
        
        # Add scores to results
        for i, r in enumerate(results):
            r["bert_p"] = P[i]
            r["bert_r"] = R[i]
            r["bert_f1"] = F1[i]
        
        avg_p = sum(P) / len(P)
        avg_r = sum(R) / len(R)
        avg_f1 = sum(F1) / len(F1)
        
        print("\n" + "="*60)
        print("BERTSCORE RESULTS")
        print("="*60)
        print(f"Average Precision: {avg_p:.4f}")
        print(f"Average Recall:    {avg_r:.4f}")
        print(f"Average F1:        {avg_f1:.4f}")
        
        # Best and worst captions by F1
        sorted_by_f1 = sorted(results, key=lambda x: x.get("bert_f1", 0), reverse=True)
        
        print(f"\n{'='*60}")
        print(f"TOP {args.top} BEST CAPTIONS (by BERTScore F1)")
        print("="*60)
        for i, r in enumerate(sorted_by_f1[:args.top], 1):
            print(f"\n--- #{i} (idx={r['index']}, F1={r['bert_f1']:.4f}) ---")
            print(f"GT label: {r['gt_label']} | Pred: {r['pred_label'] or 'indeterminate'} | Correct: {r['correct']}")
            print(f"GT caption: {r['caption_gt'][:200]}...")
            print(f"Gen caption: {r['caption_gen'][:200]}...")
        
        print(f"\n{'='*60}")
        print(f"TOP {args.top} WORST CAPTIONS (by BERTScore F1)")
        print("="*60)
        for i, r in enumerate(sorted_by_f1[-args.top:], 1):
            print(f"\n--- #{i} (idx={r['index']}, F1={r['bert_f1']:.4f}) ---")
            print(f"GT label: {r['gt_label']} | Pred: {r['pred_label'] or 'indeterminate'} | Correct: {r['correct']}")
            print(f"GT caption: {r['caption_gt'][:200]}...")
            print(f"Gen caption: {r['caption_gen'][:200]}...")
    
    # Save detailed results
    output_path = captions_path.parent / f"{captions_path.stem}_eval.json"
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "total": total,
                "determinate": determinate,
                "indeterminate": indeterminate,
                "correct": correct_count,
                "accuracy_overall": correct_count / total if total else 0,
                "accuracy_determinate": determinate_correct / determinate if determinate else 0,
                "bert_avg_p": avg_p if not args.no_bert else None,
                "bert_avg_r": avg_r if not args.no_bert else None,
                "bert_avg_f1": avg_f1 if not args.no_bert else None,
            },
            "confusion": {f"{k[0]}_{k[1]}": v for k, v in confusion.items()},
            "results": results,
        }, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
