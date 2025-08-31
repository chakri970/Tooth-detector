import os
import shutil
import random

# Paths
base_dir = "dataset"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

# New YOLO-style dataset folders
output_dir = "data"
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

# Train/Val/Test split ratios
train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1

all_images = [f for f in os.listdir(
    images_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
random.shuffle(all_images)

n_total = len(all_images)
n_train = int(n_total * train_ratio)
n_val = int(n_total * val_ratio)

train_files = all_images[:n_train]
val_files = all_images[n_train:n_train+n_val]
test_files = all_images[n_train+n_val:]


def move_files(file_list, split):
    for img_file in file_list:
        label_file = img_file.replace(".jpg", ".txt").replace(
            ".png", ".txt").replace(".jpeg", ".txt")
        shutil.copy(os.path.join(images_dir, img_file),
                    os.path.join(output_dir, "images", split, img_file))
        if os.path.exists(os.path.join(labels_dir, label_file)):
            shutil.copy(os.path.join(labels_dir, label_file),
                        os.path.join(output_dir, "labels", split, label_file))


move_files(train_files, "train")
move_files(val_files, "val")
move_files(test_files, "test")

print("Dataset split complete!")
