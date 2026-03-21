"""
Side-by-side image viewer for comparing benign vs melanoma vs Spitz.
Shows pairs of images. Navigate with keyboard:
  Left/Right arrow: previous/next pair
  Q or Escape: quit
"""

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

IMG_DIR = Path(__file__).parent / "sample_images"

def load_images():
    benign = sorted(IMG_DIR.glob("benign_*.png"))
    melanoma = sorted(IMG_DIR.glob("melanoma_*.png"))
    spitz = sorted(IMG_DIR.glob("spitz_*.png"))

    pairs = []
    # Benign vs Melanoma
    for b, m in zip(benign, melanoma):
        pairs.append((b, m, "BENIGN (easy)", "MELANOMA"))
    # Spitz vs Melanoma
    for s, m in zip(spitz, melanoma):
        pairs.append((s, m, "SPITZ (hard mimic)", "MELANOMA"))

    return pairs


def main():
    pairs = load_images()
    if not pairs:
        print("No images found in", IMG_DIR)
        return

    idx = [0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.canvas.manager.set_window_title("Benign vs Melanoma Comparison")

    def show_pair(i):
        p = pairs[i % len(pairs)]
        img1, img2 = Image.open(p[0]), Image.open(p[1])

        ax1.clear()
        ax2.clear()
        ax1.imshow(img1)
        ax2.imshow(img2)
        ax1.set_title(f"{p[2]}\n{p[0].stem}", fontsize=12, fontweight="bold", color="green" if "BENIGN" in p[2] else "orange")
        ax2.set_title(f"{p[3]}\n{p[1].stem}", fontsize=12, fontweight="bold", color="red")
        ax1.axis("off")
        ax2.axis("off")
        fig.suptitle(f"Pair {i+1}/{len(pairs)}  |  ←/→ navigate  |  Q quit", fontsize=10, color="gray")
        fig.tight_layout()
        fig.canvas.draw()

    def on_key(event):
        if event.key in ("right", " "):
            idx[0] = (idx[0] + 1) % len(pairs)
            show_pair(idx[0])
        elif event.key == "left":
            idx[0] = (idx[0] - 1) % len(pairs)
            show_pair(idx[0])
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    show_pair(0)
    plt.show()


if __name__ == "__main__":
    main()
