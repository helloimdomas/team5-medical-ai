#!/usr/bin/env python3
"""
balanced_accuracy.py - Evaluate classifiers with balanced class distribution.

The original dataset has 618 melanoma vs 302 nevus (2:1 ratio), which can bias
accuracy metrics. This script randomly samples equal numbers from each class
to compute a fair accuracy comparison.

Usage:
    python balanced_accuracy.py
"""

import json
import random
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

SCRIPT_DIR = Path(__file__).parent
RANDOM_SEED = 42
N_TRIALS = 10  # Number of random sampling trials to average

def load_data():
    """Load embeddings and captions."""
    # Load BiomedCLIP embeddings
    embeddings = np.load(SCRIPT_DIR / "embeddings" / "biomedclip_embeddings.npz")
    X, y, indices = embeddings['X'], embeddings['y'], embeddings['indices']
    
    # Load MedGemma predictions
    medgemma_preds = {}
    caption_file = SCRIPT_DIR / "captions" / "binary_choice" / "captions.jsonl"
    with open(caption_file) as f:
        for line in f:
            entry = json.loads(line)
            text = entry["caption_gen"].lower()
            
            # Apply same keyword classification logic
            has_mel = "melanoma" in text
            has_nev = "nevus" in text or "nevi" in text or "benign" in text
            
            if has_mel and not has_nev:
                pred = 1  # melanoma
            elif has_nev and not has_mel:
                pred = 0  # nevus
            elif has_mel and has_nev:
                pred = 1  # melanoma priority
            else:
                pred = 1  # default to melanoma (unknown)
            
            medgemma_preds[entry["index"]] = pred
    
    return X, y, indices, medgemma_preds


def balanced_sample(X, y, indices, rng):
    """Sample equal numbers from each class."""
    melanoma_mask = y == 1
    nevus_mask = y == 0
    
    melanoma_indices = np.where(melanoma_mask)[0]
    nevus_indices = np.where(nevus_mask)[0]
    
    # Sample min(len(melanoma), len(nevus)) from each class
    n_samples = min(len(melanoma_indices), len(nevus_indices))
    
    mel_sample = rng.choice(melanoma_indices, size=n_samples, replace=False)
    nev_sample = rng.choice(nevus_indices, size=n_samples, replace=False)
    
    selected = np.concatenate([mel_sample, nev_sample])
    rng.shuffle(selected)
    
    return X[selected], y[selected], indices[selected]


def evaluate_balanced(n_trials=N_TRIALS):
    """Evaluate with balanced sampling over multiple trials."""
    print("Loading data...")
    X, y, indices, medgemma_preds = load_data()
    
    print(f"\nOriginal distribution: {sum(y == 1)} melanoma, {sum(y == 0)} nevus")
    
    # Store results across trials
    results = {
        "MedGemma (binary_choice)": [],
        "BiomedCLIP + LogReg": [],
        "BiomedCLIP + SVM": [],
        "BiomedCLIP + RandomForest": [],
    }
    
    for trial in range(n_trials):
        rng = np.random.default_rng(RANDOM_SEED + trial)
        
        # Balanced sample
        X_bal, y_bal, idx_bal = balanced_sample(X, y, indices, rng)
        
        # Split 80/20
        n = len(y_bal)
        perm = rng.permutation(n)
        n_train = int(0.8 * n)
        
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]
        
        X_train, X_test = X_bal[train_idx], X_bal[test_idx]
        y_train, y_test = y_bal[train_idx], y_bal[test_idx]
        idx_test = idx_bal[test_idx]
        
        # MedGemma accuracy on test set
        mg_preds = [medgemma_preds.get(int(idx), 1) for idx in idx_test]
        mg_acc = accuracy_score(y_test, mg_preds)
        results["MedGemma (binary_choice)"].append(mg_acc)
        
        # BiomedCLIP classifiers
        for name, clf in [
            ("BiomedCLIP + LogReg", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ("BiomedCLIP + SVM", SVC(kernel="rbf", class_weight="balanced")),
            ("BiomedCLIP + RandomForest", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)),
        ]:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name].append(acc)
    
    # Print results
    print("\n" + "=" * 60)
    print(f"BALANCED ACCURACY (averaged over {n_trials} trials)")
    print(f"Each trial uses {sum(y == 0)} samples per class (balanced)")
    print("=" * 60)
    
    summary = {}
    for name, accs in results.items():
        mean_acc = np.mean(accs) * 100
        std_acc = np.std(accs) * 100
        print(f"{name:30s}: {mean_acc:.1f}% (+/- {std_acc:.1f}%)")
        summary[name] = {"mean": mean_acc, "std": std_acc}
    
    # Compare to original imbalanced
    print("\n" + "=" * 60)
    print("COMPARISON: Original (imbalanced) vs Balanced")
    print("=" * 60)
    
    # Original MedGemma
    orig_mg_preds = [medgemma_preds.get(int(idx), 1) for idx in indices]
    orig_mg_acc = accuracy_score(y, orig_mg_preds) * 100
    print(f"MedGemma original:     {orig_mg_acc:.1f}%")
    print(f"MedGemma balanced:     {summary['MedGemma (binary_choice)']['mean']:.1f}%")
    print(f"  Difference: {summary['MedGemma (binary_choice)']['mean'] - orig_mg_acc:+.1f}%")
    
    print()
    
    # BiomedCLIP RF original (from saved results)
    try:
        with open(SCRIPT_DIR / "results" / "biomedclip_results.json") as f:
            orig_results = json.load(f)
        orig_rf_acc = orig_results["supervised"]["RandomForest"]["test_accuracy"] * 100
        print(f"BiomedCLIP RF original:  {orig_rf_acc:.1f}%")
        print(f"BiomedCLIP RF balanced:  {summary['BiomedCLIP + RandomForest']['mean']:.1f}%")
        print(f"  Difference: {summary['BiomedCLIP + RandomForest']['mean'] - orig_rf_acc:+.1f}%")
    except:
        pass
    
    # Save results
    output = {
        "n_trials": n_trials,
        "samples_per_class": int(sum(y == 0)),
        "balanced_results": summary,
    }
    
    with open(SCRIPT_DIR / "results" / "balanced_accuracy.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {SCRIPT_DIR / 'results' / 'balanced_accuracy.json'}")
    
    return summary


if __name__ == "__main__":
    evaluate_balanced()
