import os
import shutil
from glob import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

RAW_DATASET_PATH = "./mrleyedataset"
PROCESSED_DATA_PATH = "./processed_dataset"
IMG_SIZE = 128
RANDOM_STATE = 42
class_names = sorted(os.listdir(RAW_DATASET_PATH))
image_paths = []
labels = []

for class_name in tqdm(class_names):
    class_dir = os.path.join(RAW_DATASET_PATH, class_name)
    paths = glob(os.path.join(class_dir, "*.png"))
    image_paths.extend(paths)
    labels.extend([0 if "Open" in class_name else 1] * len(paths))

print(f" found {len(image_paths)} images in {len(class_names)} classes")

image_paths = np.array(image_paths)
labels = np.array(labels)

X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths,
    labels,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=labels,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)

print(f" Training set: {len(X_train)} images")
print(f" Validation set: {len(X_val)} images")
print(f" Test set: {len(X_test)} images")
shutil.rmtree(PROCESSED_DATA_PATH, ignore_errors=True)

def process_and_save_split(paths, labels, split_name, augment=False):
    split_dir = os.path.join(PROCESSED_DATA_PATH, split_name)
    dirs = {
        0: os.path.join(split_dir, "Open-Eyes"),
        1: os.path.join(split_dir, "Close-Eyes"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    print(f"\n processing '{split_name}' set")
    for path, label in tqdm(list(zip(paths, labels))):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if augment:
            # horizontal flip
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)

            # random rotation
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2),np.random.uniform(-10, 10),1.0)
            img = cv2.warpAffine(img, M, (w, h))

            # brightness shift
            brightness = np.random.randint(-30, 30)
            img = np.clip(img.astype(np.int16) + brightness, 0, 255).astype(np.uint8)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        save_dir = dirs[int(label)]
        filename = os.path.basename(path)
        cv2.imwrite(os.path.join(save_dir, filename), img)

process_and_save_split(X_train, y_train, "train", augment=True)
process_and_save_split(X_val, y_val, "val", augment=False)
process_and_save_split(X_test, y_test, "test", augment=False)