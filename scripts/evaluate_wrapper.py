# scripts/evaluate_wrapper.py
"""
Evaluate trained model on val and test splits, save metrics and a JSON summary.
Usage (from project root):
python scripts/evaluate_wrapper.py --weights runs/tooth_yolo/weights/best.pt \
    --data data.yaml
"""
import argparse
import json
from ultralytics import YOLO
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/tooth_yolo/weights/best.pt")
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--out", default="results/metrics")
    args = parser.parse_args()

    out_p = Path(args.out)
    out_p.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

    # Validate on val split
    val_metrics = model.val(data=args.data, split="val",
                            device="cpu", save_json=True)
    # Evaluate on test split
    test_metrics = model.val(data=args.data, split="test",
                             device="cpu", save_json=True)

    summary = {
        "val": {
            # human readable; ultralytics returns object
            "metrics": str(val_metrics)
        },
        "test": {
            "metrics": str(test_metrics)
        }
    }

    with open(out_p / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved metrics summary to {out_p / 'metrics_summary.json'}")
    print("Val result object:", val_metrics)
    print("Test result object:", test_metrics)


if __name__ == "__main__":
    main()
