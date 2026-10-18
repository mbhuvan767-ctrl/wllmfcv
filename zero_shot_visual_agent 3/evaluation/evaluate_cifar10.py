"""Benchmark: zero-shot CLIP vs. a supervised ImageNet ResNet-50 on CIFAR-10.

CLIP never sees CIFAR labels (pure zero-shot).  ResNet-50 is an ImageNet classifier, so its 1000
outputs are mapped onto the 10 CIFAR classes by summing the probabilities of related ImageNet
classes - an approximate baseline, reported as such.

Usage (from the project root):
    python evaluation/evaluate_cifar10.py                       # downloads CIFAR-10 via torchvision
    python evaluation/evaluate_cifar10.py --n 2000
    python evaluation/evaluate_cifar10.py --kaggle-dir /kaggle/input/competitions/cifar-10
Outputs go to outputs/cifar10_*.{json,csv,png}
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from agent import config as C  # noqa: E402
from agent.models import ModelHub  # noqa: E402

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CLIP_NAMES = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
TEMPLATES = ["a photo of a {}.", "a blurry photo of a {}.", "a low resolution photo of a {}."]

# ImageNet-1k indices related to each CIFAR-10 class (approximate mapping).
IMAGENET_GROUPS = {
    "airplane": [404, 895, 908],
    "automobile": [436, 468, 511, 609, 627, 656, 661, 751, 817],
    "bird": list(range(7, 25)) + list(range(80, 101)) + list(range(127, 147)),
    "cat": [281, 282, 283, 284, 285],
    "deer": [351, 352, 353],
    "dog": list(range(151, 269)),
    "frog": [30, 31, 32],
    "horse": [339],
    "ship": [484, 510, 554, 625, 628, 724, 780, 814, 833, 871, 914],
    "truck": [555, 569, 675, 717, 864, 867],
}


def load_torchvision(root, n, seed):
    from torchvision.datasets import CIFAR10
    ds = CIFAR10(root=str(root), train=False, download=True)
    return _stratified(list(range(len(ds))), [ds.targets[i] for i in range(len(ds))], n, seed,
                       lambda i: (Image.fromarray(ds.data[i]), ds.targets[i]))


def load_kaggle(kaggle_dir, n, seed):
    import pandas as pd
    kaggle_dir = Path(kaggle_dir)
    labels_csv = next(kaggle_dir.rglob("trainLabels.csv"), None)
    if labels_csv is None:
        sys.exit(f"trainLabels.csv not found under {kaggle_dir}. Use the default torchvision source instead.")
    df = pd.read_csv(labels_csv)
    train_dir = next((p for p in kaggle_dir.rglob("train") if p.is_dir()), labels_csv.parent / "train")
    ids, y = df["id"].tolist(), [CLASSES.index(c) for c in df["label"]]
    return _stratified(list(range(len(ids))), y, n, seed,
                       lambda i: (Image.open(train_dir / f"{ids[i]}.png").convert("RGB"), y[i]))


def _stratified(indices, targets, n, seed, getter):
    rng = np.random.default_rng(seed)
    per_class = max(1, n // 10)
    chosen = []
    for c in range(10):
        pool = [i for i in indices if targets[i] == c]
        chosen += list(rng.choice(pool, size=min(per_class, len(pool)), replace=False))
    return [getter(int(i)) for i in chosen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="total images (balanced across 10 classes)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--kaggle-dir", default=None)
    ap.add_argument("--no-resnet", action="store_true")
    args = ap.parse_args()

    print("Loading data ...")
    samples = load_kaggle(args.kaggle_dir, args.n, args.seed) if args.kaggle_dir \
        else load_torchvision(args.data_root, args.n, args.seed)
    print(f"{len(samples)} images")

    hub = ModelHub()
    if not hub.ensure("clip"):
        sys.exit(f"CLIP failed to load: {hub.error('clip')}")
    use_resnet = not args.no_resnet and hub.ensure("cnn")

    texts = [t.format(name) for name in CLIP_NAMES for t in TEMPLATES]   # prompt ensemble
    y_true, clip_pred, cnn_pred = [], [], []
    for k, (img, label) in enumerate(samples):
        logits = hub.clip_logits([img], texts)[0].reshape(len(CLIP_NAMES), len(TEMPLATES))
        clip_pred.append(int(np.argmax(logits.mean(axis=1))))
        if use_resnet:
            p = hub.cnn_probs(img)
            cnn_pred.append(int(np.argmax([p[IMAGENET_GROUPS[c]].sum() for c in CLASSES])))
        y_true.append(label)
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(samples)}")

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    out_dir = C.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"n_images": len(samples), "seed": args.seed,
               "clip_zero_shot_accuracy": float(accuracy_score(y_true, clip_pred))}
    print(f"\nCLIP zero-shot accuracy: {summary['clip_zero_shot_accuracy']:.3f}")
    print(classification_report(y_true, clip_pred, target_names=CLASSES, zero_division=0))
    if use_resnet:
        summary["resnet50_imagenet_mapped_accuracy"] = float(accuracy_score(y_true, cnn_pred))
        print(f"ResNet-50 (ImageNet->CIFAR mapped) accuracy: {summary['resnet50_imagenet_mapped_accuracy']:.3f}")
    (out_dir / "cifar10_summary.json").write_text(json.dumps(summary, indent=2))

    import pandas as pd
    cm = confusion_matrix(y_true, clip_pred, labels=range(10))
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out_dir / "cifar10_clip_confusion.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(10), CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title("CLIP zero-shot on CIFAR-10")
        for i in range(10):
            for j in range(10):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "cifar10_clip_confusion.png", dpi=150)
    except Exception as e:
        print("Could not draw confusion matrix:", e)
    print("Saved results to", out_dir)


if __name__ == "__main__":
    main()
