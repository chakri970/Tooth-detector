# scripts/postprocess_anatomy.py
"""
Post-processing for anatomical consistency:
- For each predicted detection, compute quadrant from center (based on image midlines)
- If predicted class FDI quadrant mismatches derived quadrant, attempt to remap to same tooth position in correct quadrant
- Save annotated corrected images to runs/postprocessed
Usage:
python scripts/postprocess_anatomy.py --weights runs\tooth_yolo\weights\best.pt --source data\images\test --out runs\postprocessed
"""
import argparse
from ultralytics import YOLO
import cv2
from pathlib import Path

CLASS_TO_FDI = {
    0: "13", 1: "23", 2: "33", 3: "43", 4: "21", 5: "41", 6: "31", 7: "11", 8: "16", 9: "26",
    10: "36", 11: "46", 12: "14", 13: "34", 14: "44", 15: "24", 16: "22", 17: "32", 18: "42", 19: "12",
    20: "17", 21: "27", 22: "37", 23: "47", 24: "15", 25: "25", 26: "35", 27: "45", 28: "18", 29: "28",
    30: "38", 31: "48"
}
# reverse map
FDI_TO_CLASS = {v: int(k) for k, v in CLASS_TO_FDI.items()}


def fdi_quadrant(fdi): return int(fdi[0])


def derived_quadrant(cx, cy, w, h):
    top = cy < h/2
    left = cx < w/2
    if top and not left:
        return 1
    if top and left:
        return 2
    if (not top) and left:
        return 3
    return 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/tooth_yolo/weights/best.pt")
    parser.add_argument("--source", default="data/images/test")
    parser.add_argument("--out", default="runs/postprocessed")
    args = parser.parse_args()

    model = YOLO(args.weights)
    out_p = Path(args.out)
    out_p.mkdir(parents=True, exist_ok=True)

    results = model.predict(source=args.source, device="cpu", save=False)
    for r in results:
        img = r.orig_img.copy()  # BGR numpy
        h, w = img.shape[:2]
        for box in r.boxes:
            cls = int(box.cls.cpu().numpy())
            # pixels if model.predict with default
            xywh = box.xywh[0].cpu().numpy()
            cx, cy, bw, bh = xywh.tolist()
            dq = derived_quadrant(cx, cy, w, h)
            orig_fdi = CLASS_TO_FDI.get(cls, None)
            if orig_fdi is None:
                continue
            orig_q = fdi_quadrant(orig_fdi)
            new_cls = cls
            if orig_q != dq:
                # find candidate with same second digit and desired quadrant
                pos = orig_fdi[1]
                for fdi, c in FDI_TO_CLASS.items():
                    pass
                # search for candidate fdi
                for candidate_class, candidate_fdi in CLASS_TO_FDI.items():
                    if candidate_fdi[1] == pos and fdi_quadrant(candidate_fdi) == dq:
                        new_cls = candidate_class
                        break
            # draw corrected box
            x1 = int(cx - bw/2)
            y1 = int(cy - bh/2)
            x2 = int(cx + bw/2)
            y2 = int(cy + bh/2)
            label = f"{new_cls}:{CLASS_TO_FDI[new_cls]}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, max(10, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out_file = out_p / Path(r.path).name
        cv2.imwrite(str(out_file), img)
    print(f"Saved postprocessed images to {out_p}")


if __name__ == "__main__":
    main()
