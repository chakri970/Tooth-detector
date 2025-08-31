# scripts/compute_confmat.py
"""
Compute confusion matrix for test set:
- Matches GT <-> predictions by IoU >= 0.5
- Exports CSV (results_confusion_matrix.csv) and PNG (confusion_matrix.png)
Usage:
python scripts/compute_confmat.py --pred_dir runs/predict/labels \
    --gt_dir data/labels/test \
    --img_dir data/images/test \
    --out results
Notes:
- ultralytics predict with save=True and save_txt=True will write predicted
labels to runs/predict/labels/*.txt
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import math

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def xywh_to_xyxy(box, w, h):
    cx, cy, bw, bh = box
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return [x1, y1, x2, y2]


def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(0, (a[2]-a[0])) * max(0, (a[3]-a[1]))
    areaB = max(0, (b[2]-b[0])) * max(0, (b[3]-b[1]))
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0


def load_yolo_txt(txt_path):
    lines = []
    if not txt_path.exists():
        return lines
    for line in txt_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            cls = int(parts[0])
            vals = list(map(float, parts[1:5]))
            lines.append((cls, vals))
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_dir", default="runs/predict/labels")   # predicted txts
    parser.add_argument("--gt_dir", default="data/labels/test")
    parser.add_argument("--img_dir", default="data/images/test")
    parser.add_argument("--out", default="results")
    parser.add_argument("--iou_th", type=float, default=0.5)
    parser.add_argument("--n_classes", type=int, default=32)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    img_dir = Path(args.img_dir)
    out_p = Path(args.out)
    out_p.mkdir(parents=True, exist_ok=True)

    y_true = []
    y_pred = []

    # iterate over GT files
    for gt_file in sorted(gt_dir.glob("*.txt")):
        stem = gt_file.stem
        # find image file
        img_file = None
        for ext in IMG_EXTS:
            candidate = img_dir / (stem + ext)
            if candidate.exists():
                img_file = candidate
                break
        if img_file is None:
            continue
        w, h = Image.open(img_file).size

        gt_lines = load_yolo_txt(gt_file)
        pred_file = pred_dir / (stem + ".txt")
        pred_lines = load_yolo_txt(pred_file)

        gt_boxes = [(cls, xywh_to_xyxy(box, w, h)) for cls, box in gt_lines]
        pred_boxes = [(cls, xywh_to_xyxy(box, w, h))
                      for cls, box in pred_lines]

        matched_pred_idx = set()
        # match each gt to best pred
        for gt_cls, gt_bb in gt_boxes:
            best_iou = 0
            best_idx = None
            best_pred_cls = None
            for i, (p_cls, p_bb) in enumerate(pred_boxes):
                if i in matched_pred_idx:
                    continue
                val = iou(gt_bb, p_bb)
                if val > best_iou:
                    best_iou = val
                    best_idx = i
                    best_pred_cls = p_cls
            if best_iou >= args.iou_th and best_idx is not None:
                matched_pred_idx.add(best_idx)
                y_true.append(gt_cls)
                y_pred.append(best_pred_cls)
            else:
                # false negative
                y_true.append(gt_cls)
                y_pred.append(-1)

        # unmatched predictions -> false positives
        for i, (p_cls, p_bb) in enumerate(pred_boxes):
            if i not in matched_pred_idx:
                y_true.append(-1)
                y_pred.append(p_cls)

    # Build confusion matrix ignoring background (-1) rows for ground-truth
    mask = [t != -1 for t in y_true]
    filtered_true = [t for t, m in zip(y_true, mask) if m]
    filtered_pred = [p for p, m in zip(y_pred, mask) if m]

    labels = list(range(args.n_classes))
    cm = confusion_matrix(filtered_true, filtered_pred, labels=labels)
    df = pd.DataFrame(cm, index=[str(i) for i in labels], columns=[
                      str(i) for i in labels])
    df.to_csv(out_p / "results_confusion_matrix.csv")
    print(f"Saved CSV -> {out_p / 'results_confusion_matrix.csv'}")

    # Plot and save
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix (GT rows -> Pred columns)")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_p / "confusion_matrix.png", dpi=200)
    print(f"Saved PNG -> {out_p / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
