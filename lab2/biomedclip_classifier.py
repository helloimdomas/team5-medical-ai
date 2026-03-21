"""
BiomedCLIP-based melanoma vs nevus classifier.

Extracts image embeddings using BiomedCLIP and trains a classifier.
Compares accuracy against MedGemma's captioning approach.

Setup:
    pip install open_clip_torch scikit-learn datasets pillow

Usage:
    python lab2/biomedclip_classifier.py
"""

import json
import os
import numpy as np
from pathlib import Path

INDEX_FILE = os.path.join(os.path.dirname(__file__), "melanoma_nevus_indices.json")
CACHE_DIR = os.path.expanduser("~/.cache/huggingface")


def load_indices():
    """Load ground truth indices."""
    with open(INDEX_FILE) as f:
        return json.load(f)


def extract_embeddings(model, preprocess, device, dataset, indices, batch_size=32):
    """Extract BiomedCLIP embeddings for all images."""
    import torch
    from PIL import Image
    
    embeddings = []
    labels = []
    idx_list = []
    
    # Load index data for labels
    idx_data = load_indices()
    melanoma_set = set(idx_data["melanoma"])
    
    print(f"Extracting embeddings for {len(indices)} images...")
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        img = sample["image"]
        
        # Preprocess and get embedding
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = model.encode_image(img_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
        
        embeddings.append(embedding.cpu().numpy().flatten())
        labels.append(1 if idx in melanoma_set else 0)  # 1 = melanoma, 0 = benign
        idx_list.append(idx)
        
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(indices)}] extracted")
    
    return np.array(embeddings), np.array(labels), idx_list


def train_and_evaluate(X, y, indices, test_size=0.2, random_state=42):
    """Train classifier and evaluate with cross-validation."""
    from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    
    print("\n" + "="*60)
    print("TRAINING CLASSIFIERS")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {len(X_train)} (melanoma: {sum(y_train)}, benign: {len(y_train)-sum(y_train)})")
    print(f"Test set: {len(X_test)} (melanoma: {sum(y_test)}, benign: {len(y_test)-sum(y_test)})")
    
    # Define classifiers
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "SVM (RBF)": SVC(kernel="rbf", class_weight="balanced", probability=True),
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        print(f"\n--- {name} ---")
        
        # Cross-validation on training set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
        print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Train on full training set
        clf.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"Sensitivity (melanoma): {sensitivity:.4f}")
        print(f"Specificity (benign): {specificity:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        results[name] = {
            "cv_accuracy": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "test_accuracy": test_acc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "confusion_matrix": cm.tolist(),
        }
    
    return results, classifiers, (X_train, X_test, y_train, y_test, idx_train, idx_test)


def main():
    import torch
    
    print("Loading BiomedCLIP...")
    
    try:
        import open_clip
    except ImportError:
        print("Installing open_clip_torch...")
        import subprocess
        subprocess.run(["pip", "install", "open_clip_torch", "-q"], check=True)
        import open_clip
    
    # Load BiomedCLIP
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    
    # Load dataset
    print("Loading Open-MELON dataset...")
    from datasets import load_dataset
    ds = load_dataset("MartiHan/Open-MELON-VL-2.5K", split="train", cache_dir=CACHE_DIR)
    
    # Load indices
    idx_data = load_indices()
    target_indices = sorted(idx_data["melanoma"] + idx_data["nevus"])
    print(f"Total images: {len(target_indices)} (melanoma: {len(idx_data['melanoma'])}, nevus: {len(idx_data['nevus'])})")
    
    # Extract embeddings
    X, y, indices = extract_embeddings(model, preprocess_val, device, ds, target_indices)
    print(f"Embeddings shape: {X.shape}")
    
    # Train and evaluate
    results, classifiers, splits = train_and_evaluate(X, y, indices)
    
    # Compare with MedGemma
    print("\n" + "="*60)
    print("COMPARISON WITH MEDGEMMA")
    print("="*60)
    print(f"MedGemma V4 (constrained prompt):")
    print(f"  Accuracy: 70.3%")
    print(f"  Sensitivity: 96.6%")
    print(f"  Specificity: 13.9%")
    
    best_clf = max(results.items(), key=lambda x: x[1]["test_accuracy"])
    print(f"\nBest BiomedCLIP classifier ({best_clf[0]}):")
    print(f"  Accuracy: {best_clf[1]['test_accuracy']*100:.1f}%")
    print(f"  Sensitivity: {best_clf[1]['sensitivity']*100:.1f}%")
    print(f"  Specificity: {best_clf[1]['specificity']*100:.1f}%")
    
    if best_clf[1]["test_accuracy"] > 0.703:
        print(f"\n✅ BiomedCLIP BEATS MedGemma by {(best_clf[1]['test_accuracy']-0.703)*100:.1f}%!")
    else:
        print(f"\n⚠️ MedGemma still ahead by {(0.703-best_clf[1]['test_accuracy'])*100:.1f}%")
    
    # Save results
    output_path = Path(__file__).parent.parent / "biomedclip_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            "embedding_dim": X.shape[1],
            "n_samples": len(target_indices),
            "results": results,
            "comparison": {
                "medgemma_v4_accuracy": 0.703,
                "medgemma_v4_sensitivity": 0.966,
                "medgemma_v4_specificity": 0.139,
            }
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Save embeddings for future use
    embeddings_path = Path(__file__).parent.parent / "biomedclip_embeddings.npz"
    np.savez(embeddings_path, X=X, y=y, indices=np.array(indices))
    print(f"Embeddings saved to: {embeddings_path}")


if __name__ == "__main__":
    main()
