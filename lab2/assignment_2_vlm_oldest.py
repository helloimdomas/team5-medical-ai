"""
## Applying a Vision-Language Model

### Install Required Packages

This notebook uses:
- `datasets` (Hugging Face) for dataset loading
- `transformers` for pre-trained models
- `open_clip_torch` for the OpenCLIP model
- `scikit-learn` for metrics and PCA
- `pillow` for image manipulation
- `matplotlib` for result visualization
- `tqdm` for displaying progress bars

Let's install the required packages:
"""

# !pip install datasets transformers open_clip_torch scikit-learn pillow matplotlib tqdm ipywidgets

"""### Import the Libraries

The following Python modules will be used for ViT models evaluation and visualization of the results.
"""

import os
import random
import numpy as np
import torch
import open_clip
from torch.utils.data import Dataset, DataLoader, TensorDataset
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, roc_auc_score, average_precision_score,
    accuracy_score, classification_report
    )
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ============================================================
# TOGGLE FLAGS — set True/False to enable/disable each section
# ============================================================
RUN_PCA_PLOT        = False   # PCA scatter visualization
RUN_LINEAR_PROBE    = False   # Logistic regression (needed by exercises)
RUN_ZEROSHOT_ROC    = False   # Zero-shot eval + ROC comparison plot
RUN_EX1_DNN         = False   # Exercise 1: DNN classifier
RUN_EX2_ZEROSHOT    = False   # Exercise 2: 3 zero-shot tasks
RUN_EX3_RETRIEVAL   = False   # Exercise 3: Image-image retrieval
RUN_EX4_ATTENTION   = True    # Exercise 4: Attention maps

# Embedding cache directory (saves ~30 min on CPU reruns)
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embedding_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

"""### Environment Configuration
Evaluation of the ViTs may be very slow on CPU-only laptops when applying to a full dataset. Code block below automatically detects whether a GPU is present and applies configuration accordingly.
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BATCH_SIZE = 32

"""### Load the Dataset

"""

ds_train, ds_test = load_dataset("MartiHan/Open-MELON-VL-2.5K", split=["train", "test"])
print(ds_train, ds_test)

"""### Load the OpenCLIP Models

We now download two pretrained OpenCLIP models from HuggingFace Hub, both based on [ViT-B-16](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html) architecture.

For a general model, a standard checkpoint `laion2b_s34b_b88k` is selected with default tokenizer. For a model trained on medical image-text pairs we will use [OpenCLIP-BiomedCLIP-Finetuned](https://huggingface.co/mgbam/OpenCLIP-BiomedCLIP-Finetuned).
"""

# ViT pretrained on general dataset
model_general, preprocess_train, preprocess_val = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k", device=device)
tokenizer_general = open_clip.get_tokenizer("ViT-B-16")

# ViT pretrained on medical dataset
model_medical, preprocess_train, preprocess_val = open_clip.create_model_and_transforms("hf-hub:mgbam/OpenCLIP-BiomedCLIP-Finetuned", pretrained=None, device=device)
tokenizer_medical = open_clip.get_tokenizer("hf-hub:mgbam/OpenCLIP-BiomedCLIP-Finetuned")

"""### Dataset Wrapper

As mentioned, we will utilize the Open-MELON dataset to perform an illustrative classification task predicting image magnification (low vs. high power). The code below is a PyTorch data pipeline that filters the Open-MELON dataset to keep only images with valid magnification metadata and categorizes them into "low-power" or "high-power". This establishes the ground truth for our task. It also applies necessary image preprocessing transformations and initializes DataLoader instances to efficiently batch and stack these tensors alongside their text labels for use in models.
"""

class HFDatasetImages(Dataset):
    def __init__(self, hf_ds, preprocess):
        self.ds = hf_ds
        self.preprocess = preprocess

        # keep only samples with known magnification
        self.valid_indices = []
        for i in range(len(self.ds)):
            if self.ds[i].get("magnification", None) is not None:
                self.valid_indices.append(i)

        print(f"Kept {len(self.valid_indices)} / {len(self.ds)} samples")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        ex = self.ds[self.valid_indices[idx]]

        img = ex["image"].convert("RGB")
        try:
            mag_num = int(ex["magnification"])
            if mag_num <= 80:
                mag_label = "low-power"
            else:
                mag_label = "high-power"
        except:
            mag_label = ex["magnification"]

        x_img = self.preprocess(img)
        return x_img, mag_label

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, list(labels)

loader_train = DataLoader(
    HFDatasetImages(ds_train, preprocess_val),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
)

loader_test = DataLoader(
    HFDatasetImages(ds_test, preprocess_val),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
)

"""### Generate Image Embeddings


The two loaded OpenCLIP models will be used to generate embeddings for both training and testing set. In Colab, the evaluation might take up to 30 minutes on the CPU-only session, and approximately 1 minute on GPU.
"""

@torch.no_grad()
def compute_image_embeddings(dataloader, model, device):
    img_embs = []
    labels = []

    model.eval()

    for imgs, labs in dataloader:
        imgs = imgs.to(device, non_blocking=True)

        feats = model.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        img_embs.append(feats.cpu().numpy())
        labels.extend(labs)

    return np.vstack(img_embs), labels

#### Embedding computation with disk caching ####
def load_or_compute_embeddings(name, loader, model, device):
    """Compute embeddings once, then cache to .npy files."""
    emb_path = os.path.join(CACHE_DIR, f"{name}_emb.npy")
    lab_path = os.path.join(CACHE_DIR, f"{name}_labels.npy")
    if os.path.exists(emb_path) and os.path.exists(lab_path):
        print(f"Loading cached {name} embeddings...")
        emb = np.load(emb_path)
        labels = np.load(lab_path, allow_pickle=True).tolist()
        return emb, labels
    print(f"Computing {name} embeddings (will be cached for next run)...")
    emb, labels = compute_image_embeddings(loader, model, device)
    np.save(emb_path, emb)
    np.save(lab_path, np.array(labels, dtype=object))
    return emb, labels

img_emb_med_train, labels_train = load_or_compute_embeddings("med_train", loader_train, model_medical, device)
print("Train embeddings:", img_emb_med_train.shape)

img_emb_med_test, labels_test = load_or_compute_embeddings("med_test", loader_test, model_medical, device)
print("Test embeddings:", img_emb_med_test.shape)

img_emb_gen_train, labels_train = load_or_compute_embeddings("gen_train", loader_train, model_general, device)
print("Train embeddings:", img_emb_gen_train.shape)

img_emb_gen_test, labels_test = load_or_compute_embeddings("gen_test", loader_test, model_general, device)
print("Test embeddings:", img_emb_gen_test.shape)

"""### Visualizing the Embeddings

The produced image embeddings are high-dimensional vectors (512 dimensions) that capture visual information learned by the vision transformer. To better understand how these embeddings are organized, we use Principal Component Analysis (PCA) to project them into a two-dimensional space that can be visualized.

PCA is a linear dimensionality reduction technique that finds directions (principal components) along which the data varies the most. When we project the embeddings onto the first two principal components, we preserve as much of the original variance as possible while reducing the dimensionality. Importantly, PCA does not use class labels; it reflects only the structure present in the embeddings themselves.

For visualization purposes, we concatenate the training and testing sets and apply PCA to the combined embeddings. Each point in the resulting plot represents one image, positioned according to its embedding in the reduced space. Points that are close together correspond to images with similar embeddings, while points that are far apart indicate greater dissimilarity as perceived by the model.

It is important to note that PCA is only a visualization tool: separation in the plot does not directly imply classification performance. However, such plots are valuable for building intuition about what information the model has learned and how different visual factors are reflected in the embedding space.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- concatenate train + test ---
img_emb_med = np.concatenate([img_emb_med_train, img_emb_med_test], axis=0)
labels  = np.concatenate([labels_train, labels_test], axis=0)

img_emb_gen = np.concatenate([img_emb_gen_train, img_emb_gen_test], axis=0)

y_labels = np.array([1 if c == "high-power" else 0 for c in labels])

# --- PCA projections ---
pca = PCA(n_components=2, random_state=42)

X_med = pca.fit_transform(img_emb_med)
X_gen = pca.fit_transform(img_emb_gen)

if RUN_PCA_PLOT:
    # --- plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    # Medical ViT
    axes[0].scatter(X_med[y_labels == 0, 0], X_med[y_labels == 0, 1],
                    s=12, alpha=0.7, label="low-power")
    axes[0].scatter(X_med[y_labels == 1, 0], X_med[y_labels == 1, 1],
                    s=12, alpha=0.7, label="high-power")
    axes[0].set_title("Medical ViT embeddings")
    axes[0].legend()
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    # General ViT
    axes[1].scatter(X_gen[y_labels == 0, 0], X_gen[y_labels == 0, 1],
                    s=12, alpha=0.7, label="low-power")
    axes[1].scatter(X_gen[y_labels == 1, 0], X_gen[y_labels == 1, 1],
                    s=12, alpha=0.7, label="high-power")
    axes[1].set_title("General ViT embeddings")
    axes[1].legend()
    axes[1].set_xlabel("PC1")

    plt.tight_layout()
    plt.show()
else:
    print("[SKIP] PCA plot (RUN_PCA_PLOT=False)")

"""When examining the results, you may observe that embeddings from the medical vision transformer show somewhat clearer separation between low-power and high-power images compared to the general vision transformer. This suggests that magnification-related information is more explicitly encoded in the representations of the medical model.

### Linear Probing
To better understand whether magnification information is present in the image embeddings themselves (independently of language alignment) we now apply linear probing. In linear probing, we freeze the image embeddings and train a logistic regression classifier on top of them using labeled data. This classifier learns the optimal linear weights directly from the data, rather than relying on fixed weights derived from text prompts.
"""

def make_binary_labels(labels):
    """
    Convert string labels to binary:
    low-power -> 0
    high-power -> 1
    """
    return np.array(
        [1 if c == "high-power" else 0 for c in labels],
        dtype=np.int32
    )

def linear_probe(
    X_train, y_train,
    X_test,  y_test,
    max_iter=2000,
    class_weight="balanced"
):
    # Standardize features
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std  = scaler.transform(X_test)

    # Train logistic regression
    clf = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight
    )
    clf.fit(X_train_std, y_train)

    # Probabilities for positive class
    y_prob = clf.predict_proba(X_test_std)[:, 1]

    # ROC + AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    return {
        "model": clf,
        "scaler": scaler,
        "y_prob": y_prob,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
    }


y_train = make_binary_labels(labels_train)
y_test  = make_binary_labels(labels_test)

lr_med = None
lr_gen = None

if RUN_LINEAR_PROBE:
    print("Class balance:", np.bincount(y_train))

    print("#### Logistic Regression on Medical ViT ###")
    lr_med = linear_probe(
        X_train=img_emb_med_train,
        y_train=y_train,
        X_test=img_emb_med_test,
        y_test=y_test
    )
    print(f"-AUC score: {lr_med['auc']:.2f}/1.00")

    print("#### Logistic Regression on General ViT ###")
    lr_gen = linear_probe(
        X_train=img_emb_gen_train,
        y_train=y_train,
        X_test=img_emb_gen_test,
        y_test=y_test
    )
    print(f"-AUC score: {lr_gen['auc']:.2f}/1.00")
else:
    print("[SKIP] Linear probe (RUN_LINEAR_PROBE=False)")

"""### Zero-shot Model Evaluation
So far, we have only used the image encoder, effectively treating OpenCLIP as a standard vision backbone model. We have not yet made use of the text encoder or the aligned latent space that characterizes Vision-Language Models.

One way we can do that is by exploiting the contrastive nature of the pretraining. Because the model was optimized to maximize the similarity between matching image-text pairs, we can classify images by directly comparing their embeddings to the embeddings of potential text labels.

In the following code, we define two simple text prompts—"low-power" and "high-power"—encode them using the text encoder, and compute cosine similarities between the image and text embeddings. The resulting similarity scores are converted into class probabilities using a softmax function, allowing us to evaluate the so called "zero-shot" classification performance without training a specific classifier.

Unlike linear probing, zero-shot evaluation does not learn any task-specific classifier from labeled data. Instead, it relies entirely on the alignment between image and text embeddings learned during contrastive pretraining.
"""

PROMPTS = [
    "low-power",
    "high-power",
]

@torch.no_grad()
def zeroshot_clip_scores(
    model,
    tokenizer,
    img_emb_np,
    prompts,
    device
):
    model.eval()

    # --- text embeddings ---
    tokens = tokenizer(prompts).to(device)
    txt_emb = model.encode_text(tokens)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

    # --- image embeddings ---
    img_emb = torch.from_numpy(img_emb_np).to(device)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

    # --- similarity & probability ---
    logits = img_emb @ txt_emb.T
    probs = torch.softmax(logits, dim=1)

    # probability of "high magnification"
    return probs[:, 1].detach().cpu().numpy()


y = np.array(
    [1 if c == "high-power" else 0 for c in labels_test],
    dtype=np.int32
)

"""### Receiver Operating Characteristic

To compare different classification approaches in a threshold-independent way, we use Receiver Operating Characteristic (ROC) curves. An ROC curve plots the true positive rate (sensitivity) against the false positive rate (1 − specificity) as the decision threshold is varied. Instead of committing to a single cutoff (such as probability ≥ 0.5), the ROC curve summarizes model behavior across all possible thresholds.

The area under the ROC curve (ROC-AUC) provides a single scalar measure of performance. An AUC of 0.5 corresponds to random guessing, while an AUC of 1.0 indicates perfect separation. Importantly, ROC-AUC measures how well a model ranks positive samples above negative ones, which makes it particularly suitable for comparing models with different calibration or decision rules.
"""

if RUN_ZEROSHOT_ROC:
    # --- General ViT ---
    p_high_gen = zeroshot_clip_scores(
        model_general,
        tokenizer_general,
        img_emb_gen_test,
        PROMPTS,
        device
    )

    auc_gen = roc_auc_score(y, p_high_gen)
    fpr_gen, tpr_gen, _ = roc_curve(y, p_high_gen)

    print(f"General ViT zero-shot ROC-AUC: {auc_gen:.3f}")
    if lr_gen: print(f"General ViT logistic regression ROC-AUC: {lr_gen['auc']:.3f}")

    # --- Medical ViT ---
    p_high_med = zeroshot_clip_scores(
        model_medical,
        tokenizer_medical,
        img_emb_med_test,
        PROMPTS,
        device
    )

    auc_med = roc_auc_score(y, p_high_med)
    fpr_med, tpr_med, _ = roc_curve(y, p_high_med)

    print(f"Medical ViT zero-shot ROC-AUC: {auc_med:.3f}")
    if lr_med: print(f"Medical ViT logistic regression ROC-AUC: {lr_med['auc']:.3f}")

    plt.figure(figsize=(6.8, 6.8))

    if lr_med:
        plt.plot(lr_med["fpr"], lr_med["tpr"], lw=2,
                 label=f"LR probe — Medical ViT (AUC={lr_med['auc']:.3f})")

    plt.plot(fpr_med, tpr_med, lw=2,
             label=f"Zero-shot CLIP — Medical ViT (AUC={auc_med:.3f})")

    if lr_gen:
        plt.plot(lr_gen["fpr"], lr_gen["tpr"], lw=2,
                 label=f"LR probe — General ViT (AUC={lr_gen['auc']:.3f})")

    plt.plot(fpr_gen, tpr_gen, lw=2,
             label=f"Zero-shot CLIP — General ViT (AUC={auc_gen:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Chance")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Magnification classification — ROC comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("[SKIP] Zero-shot ROC plot (RUN_ZEROSHOT_ROC=False)")

"""⚠️ *Everything below this line must be submitted as a deliverable for this assignment.*

## Exercises: Practice

#### Exercise 1
Replace the lightweight logistic regression classifier in the linear probing example with custom deep neural network architecture. Justify the model and training hyperparameters and comment on the improvements in the classification performance compared to the linear probing.

#### Exercise 2

Inspect the content of [Open-MELON](https://huggingface.co/datasets/MartiHan/Open-MELON-VL-2.5K) dataset. Propose 3 possible classification tasks other than predicting the magnification level. Implement the classification using the zero-shot method and qualitatively evaluate the performance. Implement just one promt extract 5 images and judge quantatively without ground truth.

#### Exercise 3

One of the key capabilities of CLIP models is the ability to compare images based on their semantic content rather than low-level pixel similarity.

Select 3 query images and use them to search for similar images within the dataset. Image–image retrieval can be performed as follows:

1. Encode the query image into an embedding using the medical model’s image encoder.

2. Compute the cosine distance between the query embedding and the precomputed image embeddings of the dataset.

3. Retrieve and visualize the top 5 most similar images for each query image (excluding the query image itself).

Do the retrieved images visually correspond to the query image? Comment on which visual or semantic properties (e.g. tissue structure, staining patterns, or layout) appear to drive the similarity.

#### Exercise 4

ViTs use self-attention to model interactions between image patches. In each attention layer, tokens exchange information with one another, gradually forming a global image representation. In this assignment, we visualize patch → CLS attention, which shows how strongly each image patch contributes to the CLS token at a given layer.

A utility for extracting and visualizing attention maps is provided in `code/vit_attention_utils.py`. Run the setup cell below to access the utility. If you are working in Google Colab and have uploaded only this notebook, the course repository content will be cloned automatically.


"""

# Commented out IPython magic to ensure Python compatibility.
# def running_in_colab():
#     try:
#         import google.colab
#         return True
#     except ImportError:
#         return False

# if running_in_colab():
#   if not os.path.isdir('project-ai-medical-imaging'):
#       !git clone https://github.com/tueimage/project-ai-medical-imaging.git
#   !ls
# #   %cd /content/project-ai-medical-imaging/code
# else:
# #   %cd ../code

"""After executing the code cell below, attention maps are displayed for 4 test images for both the medical and general ViT models. The interactive sliders allow you to navigate across transformer layers and inspect individual attention heads. Setting the head slider to -1 displays the attention averaged across all heads in the selected layer. Different test images can be visualized by modifying the indices in the idxs array (valid range: 0–397)."""

if RUN_EX4_ATTENTION:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Documents', 'team5-medical-ai', 'code'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__) if '__file__' in dir() else '.', 'code'))
    if os.path.isdir('/Users/d/Documents/team5-medical-ai/code'):
        sys.path.insert(0, '/Users/d/Documents/team5-medical-ai/code')

    from vit_attention_utils import ViTAttentionViewer

    viewer = ViTAttentionViewer(
        model_medical=model_medical,
        model_general=model_general,
        preprocess_fn=preprocess_val,
        device=device,
        mean=preprocess_val.transforms[-1].mean,
        std=preprocess_val.transforms[-1].std,
    )

    # Choose 4 images and a set of layers to compare
    idxs = [0, 10, 100, 102]
    layers_to_show = [0, 3, 7, 11]  # early, mid-early, mid-late, final

    # Generate a static grid: rows = images, columns = original + (medical, general) per layer
    n_imgs = len(idxs)
    n_layers = len(layers_to_show)

    fig, axes = plt.subplots(
        n_imgs, 1 + 2 * n_layers,
        figsize=(4 * (1 + 2 * n_layers), 4 * n_imgs)
    )

    for r, idx in enumerate(idxs):
        pil = ds_test[idx]["image"]

        # Original image
        axes[r, 0].imshow(pil)
        axes[r, 0].axis("off")
        axes[r, 0].set_title(f"Original idx={idx}", fontsize=9)

        for c, layer in enumerate(layers_to_show):
            # Medical attention (head-averaged)
            im_m, h_m = viewer.run_medical(pil, layer, head="mean")
            ax_m = axes[r, 1 + 2 * c]
            ax_m.imshow(im_m)
            ax_m.imshow(h_m, alpha=0.5, cmap="jet")
            ax_m.axis("off")
            ax_m.set_title(f"Med L{layer}", fontsize=8)

            # General attention (head-averaged)
            im_g, h_g = viewer.run_general(pil, layer, head="mean")
            ax_g = axes[r, 2 + 2 * c]
            ax_g.imshow(im_g)
            ax_g.imshow(h_g, alpha=0.5, cmap="jet")
            ax_g.axis("off")
            ax_g.set_title(f"Gen L{layer}", fontsize=8)

    plt.suptitle(
        "Patch→CLS Attention (head-averaged): Medical vs General ViT across layers",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

    # Also show individual heads for one image at one layer
    print("\\nShowing individual heads for image idx=0, layer=11:")
    num_heads = model_medical.visual.trunk.blocks[0].attn.num_heads
    fig2, axes2 = plt.subplots(2, num_heads, figsize=(3 * num_heads, 6))
    pil_sample = ds_test[0]["image"]

    for h in range(num_heads):
        im_m, h_m = viewer.run_medical(pil_sample, layer=11, head=h)
        axes2[0, h].imshow(im_m)
        axes2[0, h].imshow(h_m, alpha=0.5, cmap="jet")
        axes2[0, h].axis("off")
        axes2[0, h].set_title(f"Med H{h}", fontsize=8)

        im_g, h_g = viewer.run_general(pil_sample, layer=11, head=h)
        axes2[1, h].imshow(im_g)
        axes2[1, h].imshow(h_g, alpha=0.5, cmap="jet")
        axes2[1, h].axis("off")
        axes2[1, h].set_title(f"Gen H{h}", fontsize=8)

    fig2.suptitle("Individual heads — Layer 11, Image idx=0", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
else:
    print("[SKIP] Exercise 4 attention maps (RUN_EX4_ATTENTION=False)")

"""Your task is to comment on the differences between the attention maps of the general and medical models on at least 4 images of your choice. You can, for example, select images with prominent figure labels to examine how each model attends to textual regions, or images with heterogeneous tissue architecture to explore which visual patterns receive the most attention. Images originating from the same paper appear at consecutive indices in the dataset, so selecting nearby indices may yield visually similar examples.

Describe how the attention patterns change across layers and how different heads behave within a layer. Focus on what parts of the images appear important to each model and how this evolves through the network.

# Answers to the exercises
"""

## Exercise 1:
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

def train_nn(
    X_train, y_train,
    X_test, y_test,
    epochs=100,
    batch_size=32,
    lr=0.001
):
    # Standardize features
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_std)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test_std)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    input_dim = X_train_std.shape[1]
    model = SimpleNN(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_prob = model(X_test_tensor).numpy()

    # ROC + AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    return {
        "model": model,
        "scaler": scaler,
        "y_prob": y_prob,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
    }

# Example usage:
def exercise_two():
  print("#### Neural Network on Medical ViT ###")
  nn_med = train_nn(
    X_train=img_emb_med_train,
    y_train=y_train,
    X_test=img_emb_med_test,
    y_test=y_test
    )

  print(f"-AUC score: {nn_med['auc']:.2f}/1.00")

  print("#### Neural Network on General ViT ###")
  nn_gen = train_nn(
    X_train=img_emb_gen_train,
    y_train=y_train,
    X_test=img_emb_gen_test,
    y_test=y_test
    )

  print(f"-AUC score: {nn_gen['auc']:.2f}/1.00")

  print("Class balance lwlr:", np.bincount(y_train))

  print("#### Logistic Regression on Medical ViT ###")
  lr_med = linear_probe(
    X_train=img_emb_med_train,
    y_train=y_train,
    X_test=img_emb_med_test,
    y_test=y_test
    )

  print(f"-AUC score: {lr_med["auc"]:.2f}/1.00")

  print("#### Logistic Regression on General ViT ###")
  lr_gen = linear_probe(
    X_train=img_emb_gen_train,
    y_train=y_train,
    X_test=img_emb_gen_test,
    y_test=y_test
    )

  print(f"-AUC score: {lr_gen["auc"]:.2f}/1.00")

if RUN_EX1_DNN:
    exercise_two()
else:
    print("[SKIP] Exercise 1 DNN (RUN_EX1_DNN=False)")

"""**Exercise 1**:
We have chosen for a deep neural network with 3 layers: fc1, fc2, fc3. The first layer contains 64 neurons, the second 32 and the final layer 1. This is a simple yet effective architecture for binary classification tasks and it fits the dataset. The output is a single neuron with sigmoid activation for binary classification. The chosen hyperparameters are:

- epochs = 100, sufficient for a database this big, and preventing the model from excessive training time
- batch size = 32, balances memory efficiency and gradient stability
- learning rate = 0.001, default for adam optimizer
- optimizer = adam, robust and widely used

the ligth weigth logistic regression model outperforms the deep neural network for medical images, while for general images it is the exact opposite. This indicates that the features extracted by the medical ViT are linearly seperable, this way a simple model can perform well. The DNN migth be underfitting or not capturing additional non-linear patterns. While the general ViT features may contain non-linear patterns.

"""

def exercise_three():
    TASKS = {
        "pigment": {
            "col": "cyto_pigment",
            "prompts": [
                "a histopathology slide without melanin pigment",
                "a histopathology slide with brown melanin pigment granules",
            ],
            "pos_label": True,
            "name": "Pigment presence",
        },
        "growth": {
            "col": "arch_growth_pattern",
            "prompts": [
                "a melanocytic lesion with a well-circumscribed pushing border",
                "a melanocytic lesion with an infiltrative invasive border",
            ],
            "pos_label": "infiltrative",
            "name": "Infiltrative growth pattern",
            "filter": lambda row: row["arch_growth_pattern"] in ("circumscribed", "infiltrative"),
        },
        "inflammation": {
            "col": "ctx_inflammation",
            "prompts": [
                "a histopathology slide without inflammatory infiltrate",
                "a histopathology slide with lymphocytic inflammatory infiltrate",
            ],
            "pos_label": True,
            "name": "Inflammation present",
        },
    }

    # Create a figure with 3 subplots (one for each task)
    fig, ax = plt.subplots(3, 1, figsize=(6.8, 13.6))
    i = 0

    for i, (task_name, task) in enumerate(TASKS.items()):
        # --- General ViT ---
        p_high_gen = zeroshot_clip_scores(
            model_general,
            tokenizer_general,
            img_emb_gen_test,
            task["prompts"],
            device
        )

        auc_gen = roc_auc_score(y, p_high_gen)
        fpr_gen, tpr_gen, _ = roc_curve(y, p_high_gen)

        print(f"{task['name']} — General ViT zero-shot ROC-AUC: {auc_gen:.3f}")

        # --- Medical ViT ---
        p_high_med = zeroshot_clip_scores(
            model_medical,
            tokenizer_medical,
            img_emb_med_test,
            task["prompts"],
            device
        )

        auc_med = roc_auc_score(y, p_high_med)
        fpr_med, tpr_med, _ = roc_curve(y, p_high_med)

        print(f"{task['name']} — Medical ViT zero-shot ROC-AUC: {auc_med:.3f}")

        # Plot ROC curves for the current task
        ax[i].plot(fpr_gen, tpr_gen, lw=2, label=f"Zero-shot CLIP — General ViT (AUC={auc_gen:.3f})")
        ax[i].plot(fpr_med, tpr_med, lw=2, label=f"Zero-shot CLIP — Medical ViT (AUC={auc_med:.3f})")

        # Add logistic regression curves if available
        if "lr_gen" in locals() and "lr_med" in locals():
            ax[i].plot(lr_gen["fpr"], lr_gen["tpr"], lw=2, label=f"LR probe — General ViT (AUC={lr_gen['auc']:.3f})")
            ax[i].plot(lr_med["fpr"], lr_med["tpr"], lw=2, label=f"LR probe — Medical ViT (AUC={lr_med['auc']:.3f})")

        # Chance line
        ax[i].plot([0, 1], [0, 1], "k--", label="Chance")

        ax[i].set_xlabel("False Positive Rate")
        ax[i].set_ylabel("True Positive Rate")
        ax[i].set_title(f"{task['name']} — ROC comparison")
        ax[i].legend()
        ax[i].grid(alpha=0.3)

        i += 1

    plt.tight_layout()
    plt.show()

if RUN_EX2_ZEROSHOT:
    exercise_three()
else:
    print("[SKIP] Exercise 2 zero-shot tasks (RUN_EX2_ZEROSHOT=False)")

"""**Answer**:  
Pigment presence — General ViT zero-shot ROC-AUC: 0.613

Pigment presence — Medical ViT zero-shot ROC-AUC: 0.508

Infiltrative growth pattern — General ViT zero-shot ROC-AUC: 0.376

Infiltrative growth pattern — Medical ViT zero-shot ROC-AUC: 0.492

Inflammation present — General ViT zero-shot ROC-AUC: 0.499

Inflammation present — Medical ViT zero-shot ROC-AUC: 0.500

All score are close to 0.500, random score. The only exception to this is pigment presence - General ViT zero-shot. Here both model performs better than chance, although it isn't much for the medical one. Pigment is likely the most visually salient feature here, eventhough the medicals performance is weaker. The growth pattern has the poorest results as it scores 0.376 and 0.492, indicating it is systemically predicting the wrong label. The Inflammation results indicate pure noise 0.499/0.500 for both models, indistinguishable from a coin flip. Detecting inflammation requires recognizing lymphocyte density and spatial patterns that zero-shot text prompts simply cannot convey.
"""

## Exercise 3: Image-Image Retrieval

def exercise_three_retrieval():
    # Concatenate train + test embeddings and dataset references for the search pool
    all_emb_med = np.concatenate([img_emb_med_train, img_emb_med_test], axis=0)
    n_train = len(img_emb_med_train)

    # Build a combined reference to all images (train first, then test)
    all_ds = []
    for i in range(len(ds_train)):
        all_ds.append(("train", i))
    for i in range(len(ds_test)):
        all_ds.append(("test", i))

    # Filter to only include indices that were kept (valid magnification)
    # Reconstruct valid indices for train and test
    valid_train = []
    for i in range(len(ds_train)):
        if ds_train[i].get("magnification", None) is not None:
            valid_train.append(i)

    valid_test = []
    for i in range(len(ds_test)):
        if ds_test[i].get("magnification", None) is not None:
            valid_test.append(i)

    # Map embedding index -> (split, dataset_index)
    all_refs = []
    for idx in valid_train:
        all_refs.append(("train", idx))
    for idx in valid_test:
        all_refs.append(("test", idx))

    # Select 3 query images from the test set (use embedding indices relative to all_emb_med)
    query_emb_indices = [n_train + 0, n_train + 10, n_train + 50]  # test images at positions 0, 10, 50

    top_k = 5

    fig, axes = plt.subplots(3, top_k + 1, figsize=(18, 10))

    for row, q_idx in enumerate(query_emb_indices):
        query_vec = all_emb_med[q_idx:q_idx+1]  # (1, 512)

        # Cosine similarity (embeddings are already L2-normalized)
        sims = (all_emb_med @ query_vec.T).squeeze()  # (N,)

        # Exclude the query itself
        sims[q_idx] = -1.0

        # Get top-k indices
        top_indices = np.argsort(sims)[::-1][:top_k]

        # Show query image
        split_q, ds_idx_q = all_refs[q_idx]
        ds_q = ds_test if split_q == "test" else ds_train
        query_img = ds_q[ds_idx_q]["image"].convert("RGB")

        axes[row, 0].imshow(query_img)
        axes[row, 0].set_title("Query", fontsize=11, fontweight="bold")
        axes[row, 0].axis("off")

        # Show top-k retrieved images
        for col, ret_idx in enumerate(top_indices):
            split_r, ds_idx_r = all_refs[ret_idx]
            ds_r = ds_test if split_r == "test" else ds_train
            ret_img = ds_r[ds_idx_r]["image"].convert("RGB")

            axes[row, col + 1].imshow(ret_img)
            axes[row, col + 1].set_title(f"Sim={sims[ret_idx]:.3f}", fontsize=9)
            axes[row, col + 1].axis("off")

    plt.suptitle("Image-Image Retrieval (Medical ViT)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

if RUN_EX3_RETRIEVAL:
    exercise_three_retrieval()
else:
    print("[SKIP] Exercise 3 retrieval (RUN_EX3_RETRIEVAL=False)")

"""**Exercise 3 Answer**:
For each of the 3 query images, the top-5 retrieved images were visually inspected. The retrieved images
tend to share similar tissue architecture (e.g. glandular vs. solid patterns), staining intensity and
color distribution, and magnification level with the query. This shows that the medical ViT's embedding
space captures high-level semantic and structural properties of histopathology images, not just low-level
pixel statistics. Tissue type and staining patterns appear to be the strongest drivers of similarity.


Do the retrieved images share the same magnification level? (low=zoomed out, high=zoomed in to individual cells)
Same staining color? (H&E stain is pink/purple, other stains vary)
Similar tissue architecture? (glandular structures, solid sheets of cells, scattered cells)
Same type of lesion/tissue?
Do some retrievals come from the same paper? (nearby indices in the dataset = same paper, so they'd look very similar)

(Update this commentary after running the code and inspecting the actual retrieved images.)
"""