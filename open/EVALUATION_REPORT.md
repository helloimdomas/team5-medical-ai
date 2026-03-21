# MedGemma Caption Evaluation Report

## Overview

This report summarizes the evaluation of MedGemma (1.5-4B-IT, Q4_K_M quantized) on the Open-MELON dermatopathology dataset for melanoma vs. benign nevus classification.

**Dataset:** 680 images from Open-MELON (464 melanoma, 216 benign nevus)

**Model:** `dcarrascosa/medgemma-1.5-4b-it:Q4_K_M` via Ollama

---

## Chronological Summary of All Experiments

### Phase 1: Initial Caption Generation

**Files created:**
- `lab2/medgemma_captions_local.py` - Main script for generating captions via Ollama
- `lab2/melanoma_nevus_indices.json` - Ground truth labels (680 images)
- `medgemma_captions_v2.jsonl` - First batch of captions

**Prompt V2 (original):**
```
"You are a pathologist writing a brief report. Describe the key visual findings 
and diagnosis in 2-3 sentences. Do not list differentials or next steps."
```

**Finding:** Model rarely mentioned diagnosis keywords (only 15.3% of captions)

---

### Phase 2: Evaluation Script & BERTScore

**Files created:**
- `lab2/evaluate_captions.py` - Reusable evaluation script (accuracy + BERTScore)
- `medgemma_captions_v2_eval.json` - Detailed evaluation results

**V2 Results:**
| Metric | Value |
|--------|-------|
| Diagnosis coverage | 15.3% |
| Accuracy (determinate) | 53.8% |
| BERTScore F1 | 0.839 |

---

### Phase 3: Prompt Engineering (V3 & V4)

**V3 Prompt (+ certainty):**
```
"...Provide your best guess for the diagnosis and your certainty level (high/low)"
```

**V4 Prompt (constrained to melanoma/nevus):**
```
"You are a dermatopathologist examining a skin biopsy. This lesion is either 
MELANOMA or a BENIGN NEVUS. Describe features... state diagnosis and certainty."
```

**Files created:**
- `medgemma_captions_v3.jsonl`, `medgemma_captions_v3_eval.json`
- `medgemma_captions_v4.jsonl`, `medgemma_captions_v4_eval.json`

**Results Comparison:**

| Version | Prompt Type | Dx Coverage | Accuracy |
|---------|-------------|-------------|----------|
| V2 | Original | 15.3% | 53.8% |
| V3 | +Certainty | 21.5% | 58.2% |
| **V4** | **Constrained** | **99.9%** | **70.3%** |

**Key Finding:** Forcing the model to choose "melanoma or nevus" dramatically improved diagnosis coverage and accuracy.

---

### Phase 4: RAGAS Semantic Evaluation

**Files created:**
- `lab2/evaluate_ragas.py` - RAGAS evaluation via Gemini API
- `lab2/evaluate_ragas_local.py` - RAGAS evaluation via local Ollama
- `medgemma_captions_v4_ragas.json` - Faithfulness/relevancy scores

**V4 RAGAS Results:**
| Metric | Score |
|--------|-------|
| Faithfulness | 0.475 |
| Relevancy | 0.936 |

**Finding:** Correct diagnoses had 2x higher faithfulness (0.56 vs 0.26).

---

### Phase 5: Certainty Calibration Analysis

**Finding:** MedGemma's self-reported certainty correlates with accuracy:

| Certainty | Accuracy | Count |
|-----------|----------|-------|
| High | 74.1% | 410 |
| Low | 65.0% | 254 |

**"Benign + High Certainty"** predictions were 88.9% accurate (but rare).

---

### Phase 6: BiomedCLIP Classifier

**Files created:**
- `lab2/biomedclip_classifier.py` - Embedding extraction + classifier training
- `biomedclip_embeddings.npz` - Cached embeddings (680 images)
- `biomedclip_results.json` - Classification results

**BiomedCLIP Results (RandomForest):**
| Metric | Value |
|--------|-------|
| **Accuracy** | **79.4%** |
| Sensitivity | 95.7% |
| Specificity | 44.2% |

**✅ BiomedCLIP beat MedGemma by 9.1%** with much better specificity (44% vs 14%).

---

### Phase 7: Hard Case Analysis

**Finding:** 119 cases (17.5%) were wrong for BOTH models.

**Almost all hard cases were benign lesions misclassified as melanoma:**
- 118/119 hard cases were benign (not melanoma)
- Hardest subtypes: Spitz nevus (22), blue nevus (15), unclassified (51)

**Both models have strong melanoma bias.**

---

### Phase 8: Dataset Quality Analysis

**Files created:**
- `lab2/melanoma_nevus_indices_skin_only.json` - Filtered to 557 cutaneous cases

**Found 123 non-cutaneous cases:**
| Category | Count |
|----------|-------|
| Cutaneous skin | 557 |
| Oral/mucosal | 54 |
| Uveal (eye) | 36 |
| Metastatic | 18 |
| Non-melanocytic (mislabeled) | 17 |

**Skin-only evaluation showed LOWER accuracy** (both models ~3-9% worse), meaning the non-cutaneous cases were actually easier to classify.

---

### Phase 9: Generic Prompt (V5)

**V5 Prompt (removed "dermatopathologist/skin"):**
```
"You are a pathologist examining a biopsy. This lesion is either MELANOMA or 
BENIGN (e.g., nevus). Describe features... state diagnosis and certainty."
```

**Files created:**
- `medgemma_captions_v5.jsonl`, `medgemma_captions_v5_eval.json`
- `lab2/evaluate_ragas_v5.py`
- `medgemma_captions_v5_ragas.json`

**V5 Results:**
| Metric | V4 | V5 | Winner |
|--------|-----|-----|--------|
| Accuracy | **70.3%** | 68.7% | V4 |
| Sensitivity | 96.6% | 99.1% | V5 |
| Specificity | **13.9%** | 3.2% | V4 |
| Faithfulness | 0.475 | **0.528** | V5 |
| Relevancy | 0.936 | **0.955** | V5 |

**Key Finding:** Removing "dermatopathologist" made the model MORE melanoma-biased. V4 remains best for classification.

---

## Final Summary Table

| Model/Prompt | Accuracy | Sens | Spec | Faithfulness |
|--------------|----------|------|------|--------------|
| MedGemma V2 | 8.2% | - | - | - |
| MedGemma V3 | 12.5% | - | - | - |
| **MedGemma V4** | **70.3%** | 96.6% | 13.9% | 0.475 |
| MedGemma V5 | 68.7% | 99.1% | 3.2% | 0.528 |
| **BiomedCLIP** | **79.4%** | 95.7% | **44.2%** | - |

---

## Key Conclusions

1. **BiomedCLIP is best for classification** (79.4% accuracy, 44% specificity)
2. **MedGemma V4 is best for captioning** (70.3% accuracy with interpretable descriptions)
3. **Both models have strong melanoma bias** (poor specificity)
4. **Certainty is calibrated** in V4 - high certainty = higher accuracy
5. **Domain constraints help** - "skin biopsy" prompt performed better than generic
6. **Dataset has quality issues** - ~17 mislabeled cases, 123 non-cutaneous

---

## Files Reference

| File | Purpose |
|------|---------|
| `lab2/medgemma_captions_local.py` | Caption generation script |
| `lab2/evaluate_captions.py` | Accuracy + BERTScore evaluation |
| `lab2/evaluate_ragas.py` | RAGAS faithfulness evaluation |
| `lab2/biomedclip_classifier.py` | BiomedCLIP embedding classifier |
| `lab2/melanoma_nevus_indices.json` | Full dataset labels (680) |
| `lab2/melanoma_nevus_indices_skin_only.json` | Filtered labels (557) |
| `medgemma_captions_v[2-5].jsonl` | Generated captions |
| `medgemma_captions_v[2-5]_eval.json` | Evaluation results |
| `medgemma_captions_v[4-5]_ragas.json` | RAGAS scores |
| `biomedclip_embeddings.npz` | Cached image embeddings |
| `biomedclip_results.json` | Classifier results |
