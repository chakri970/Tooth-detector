# 🦷 OralVis AI Research – Tooth Detection & Analysis  

This project is an **AI-based dental image analysis system** built with **YOLOv10** for tooth detection, classification, and evaluation.  
It includes training, prediction, evaluation, confusion matrix computation, and post-processing scripts.  

---

## ⚙️ Installation  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```
### 2️⃣ Create and activate a virtual environment
```bash
python -m venv olovenv
olovenv\Scripts\activate   # On Windows
# source olovenv/bin/activate   # On Linux/Mac
```
### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```
## Dataset Setup

Place your dataset inside data/ folder with the following structure:
```
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Update data.yaml with the correct dataset paths and number of classes.

## 🚀 Usage
🔹 1. Split Dataset
```
python scripts/split_dataset.py
```

🔹 2. Validate Dataset
```
python scripts/validate_dataset.py
```

🔹 3. Train Model
```
python scripts/train_wrapper.py
```

🔹 4. Run Predictions
```
python scripts/predict_wrapper.py
```

🔹 5. Evaluate Model
```
python scripts/evaluate_wrapper.py
```
🔹 6. Compute Confusion Matrix
```
python scripts/compute_confmat.py --pred_dir runs\detect\predict\labels --gt_dir data\labels\test --img_dir data\images\test --out results --n_classes 32
  ```
🔹 7. Postprocess Anatomy
```
python scripts/postprocess_anatomy.py --weights runs\tooth_yolo10\weights\best.pt --source data\images\test --out runs\postprocessed
```
## Requirements

The main dependencies are:

```
Python 3.9+

torch / torchvision

ultralytics (YOLOv8/YOLOv10)

scikit-learn

opencv-python

matplotlib

pandas

python-docx
```
Install all via:
```
pip install -r requirements.txt
```
## 📈 Outputs
```

Training results → runs/train/

Predictions → runs/detect/

Confusion Matrix → results/confusion_matrix.png

Postprocessed images → runs/postprocessed/
```
# 👨‍💻 Author

Developed by Korsa Chakri
SVNIT SURAT
chakrifavofvd@gmail.com

## ⭐ Contributing

Feel free to fork this repo, open issues, or submit pull requests. Contributions are welcome!



