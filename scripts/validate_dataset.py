# scripts/validate_dataset.py

import os
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

ROOT = Path("dataset")
IMAGES = ROOT / "images"
LABELS = ROOT / "labels"


def main():
    missing_labels = []
    missing_images = []
    empty_label_files = []
    bad_class_lines = []

    for img in IMAGES.iterdir():
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lbl = LABELS / (img.stem + ".txt")
        if not lbl.exists():
            missing_labels.append(img.name)
        else:
            text = lbl.read_text().strip()
            if text == "":
                empty_label_files.append(lbl.name)
            else:
                for i, line in enumerate(text.splitlines(), 1):
                    parts = line.split()
                    if len(parts) < 5:
                        bad_class_lines.append((lbl.name, i, "format"))
                        break
                    try:
                        cls = int(parts[0])
                        if cls < 0 or cls > 31:
                            bad_class_lines.append((lbl.name, i, cls))
                            break
                    except:
                        bad_class_lines.append((lbl.name, i, "nonint"))
                        break

    for lbl in LABELS.iterdir():
        if lbl.suffix.lower() != ".txt":
            continue
        img_found = False
        for ext in IMG_EXTS:
            if (IMAGES / (lbl.stem + ext)).exists():
                img_found = True
                break
        if not img_found:
            missing_images.append(lbl.name)

    print("Summary:")
    print("Missing labels for images:", len(missing_labels))
    print("Missing images for labels:", len(missing_images))
    print("Empty label files:", len(empty_label_files))
    print("Bad label lines:", len(bad_class_lines))
    if missing_labels:
        print("Examples missing label:", missing_labels[:10])
    if missing_images:
        print("Examples missing image:", missing_images[:10])
    if empty_label_files:
        print("Empty labels examples:", empty_label_files[:10])
    if bad_class_lines:
        print("Bad label examples:", bad_class_lines[:10])


if __name__ == '__main__':
    main