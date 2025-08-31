# scripts/predict_wrapper.py
from ultralytics import YOLO
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", default="runs/detect/tooth_yolo/weights/best.pt")
    # path to images or folder
    parser.add_argument("--source", default="dataset/images/test")
    args = parser.parse_args()

    # Load trained model
    model = YOLO(args.weights)

    # Run prediction
    results = model.predict(
        source=args.source,
        imgsz=640,
        conf=0.25,
        save=True,
        device="cpu"
    )

    print("✅ Predictions saved in runs/predict.")


if __name__ == "__main__":
    main()
