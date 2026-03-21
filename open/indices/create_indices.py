#!/usr/bin/env python3
"""
create_indices.py - Single Source of Truth for Melanoma/Nevus Index Creation

This script creates the ground truth indices for melanoma vs nevus classification
from the Open-MELON dataset. It uses keyword matching on captions since Open-MELON
does not have explicit diagnosis labels.

CLASSIFICATION RULES:
1. Search caption for keywords (case-insensitive)
2. "melanoma" keyword ONLY → melanoma class (includes uveal, oral, metastatic)
3. "nevus" or "nevi" keyword ONLY → nevus class (includes all nevus subtypes)
4. If BOTH keywords present → EXCLUDED (ambiguous cases like nevus-associated 
   melanoma, melanoma arising from nevus, or differential diagnosis discussions)
5. If NEITHER keyword present → EXCLUDED

RATIONALE FOR EXCLUDING "BOTH":
- ~110 images mention both "melanoma" and "nevus/nevi"
- These are often: nevus-associated melanoma, melanoma arising from a nevus,
  or images showing both lesions side by side
- Including them would contaminate the binary classification benchmark

OUTPUT: melanoma_nevus_indices.json
{
    "melanoma": [list of global indices],
    "nevus": [list of global indices],
    "excluded_both": [list of indices with both keywords],
    "metadata": {...}
}

Usage:
    python create_indices.py [--output OUTPUT_FILE]
"""

import json
import argparse
from datetime import datetime
from pathlib import Path


def create_indices(output_path: Path, dry_run: bool = False) -> dict:
    """Create melanoma/nevus indices from Open-MELON dataset."""
    from datasets import load_dataset
    
    print("Loading Open-MELON dataset (all splits)...")
    ds = load_dataset(
        "MartiHan/Open-MELON-VL-2.5K",
        cache_dir="~/.cache/huggingface"
    )
    
    melanoma_indices = []
    nevus_indices = []
    excluded_both = []  # Cases with both keywords (ambiguous)
    
    # Classification rules
    melanoma_keywords = ["melanoma"]
    nevus_keywords = ["nevus", "nevi"]
    
    # Track statistics
    stats = {
        "train": {"melanoma": 0, "nevus": 0, "both": 0, "neither": 0},
        "validation": {"melanoma": 0, "nevus": 0, "both": 0, "neither": 0},
        "test": {"melanoma": 0, "nevus": 0, "both": 0, "neither": 0},
    }
    
    # Process each split with global index offset
    offset = 0
    split_info = {}
    
    for split_name in ["train", "validation", "test"]:
        split = ds[split_name]
        split_info[split_name] = {
            "start_idx": offset,
            "end_idx": offset + len(split) - 1,
            "size": len(split),
        }
        
        for i, sample in enumerate(split):
            caption_lower = sample["caption"].lower()
            global_idx = offset + i
            
            # Check for keywords
            has_melanoma = any(kw in caption_lower for kw in melanoma_keywords)
            has_nevus = any(kw in caption_lower for kw in nevus_keywords)
            
            # Classify according to rules
            if has_melanoma and has_nevus:
                # Both present → EXCLUDE (ambiguous)
                excluded_both.append(global_idx)
                stats[split_name]["both"] += 1
            elif has_melanoma:
                melanoma_indices.append(global_idx)
                stats[split_name]["melanoma"] += 1
            elif has_nevus:
                nevus_indices.append(global_idx)
                stats[split_name]["nevus"] += 1
            else:
                stats[split_name]["neither"] += 1
        
        offset += len(split)
    
    # Build output structure
    result = {
        "melanoma": sorted(melanoma_indices),
        "nevus": sorted(nevus_indices),
        "excluded_both": sorted(excluded_both),
        "metadata": {
            "created": datetime.now().isoformat(),
            "script": "create_indices.py",
            "dataset": "MartiHan/Open-MELON-VL-2.5K",
            "total_dataset_size": offset,
            "splits": split_info,
            "classification_rules": {
                "melanoma_keywords": melanoma_keywords,
                "nevus_keywords": nevus_keywords,
                "both_keywords_policy": "EXCLUDED (ambiguous cases)",
            },
            "counts": {
                "melanoma": len(melanoma_indices),
                "nevus": len(nevus_indices),
                "excluded_both": len(excluded_both),
                "total_classified": len(melanoma_indices) + len(nevus_indices),
            },
            "per_split_stats": stats,
        },
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("INDEX CREATION SUMMARY")
    print("=" * 60)
    print(f"Total dataset: {offset} images")
    print(f"Melanoma (only): {len(melanoma_indices)}")
    print(f"Nevus (only): {len(nevus_indices)}")
    print(f"Excluded (both keywords): {len(excluded_both)}")
    print(f"Excluded (no keywords): {offset - len(melanoma_indices) - len(nevus_indices) - len(excluded_both)}")
    print(f"Total classified: {len(melanoma_indices) + len(nevus_indices)}")
    
    print("\nPer-split breakdown:")
    for split_name, s in stats.items():
        info = split_info[split_name]
        print(f"  {split_name}: {info['size']} images "
              f"(mel={s['melanoma']}, nev={s['nevus']}, both={s['both']}, excl={s['neither']})")
    
    if not dry_run:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: {output_path}")
    else:
        print("\n[DRY RUN - not saved]")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Create melanoma/nevus indices from Open-MELON dataset"
    )
    parser.add_argument(
        "--output",
        default="melanoma_nevus_indices.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show statistics without saving"
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    output_path = script_dir / args.output
    
    create_indices(output_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
