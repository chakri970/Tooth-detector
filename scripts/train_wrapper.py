# scripts/train_wrapper.py
from ultralytics import YOLO
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    # Load model
    model = YOLO(args.model)

    # Train (force CPU with device="cpu")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs",
        name="tooth_yolo",
        device="cpu"
    )

    print("✅ Training finished. Check runs/detect/tooth_yolo")


if __name__ == "__main__":
    main()
