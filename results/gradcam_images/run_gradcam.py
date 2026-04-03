import os
import multiprocessing
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from pytorch_grad_cam import GradCAM
from tuning import ConvNet

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TUNING_RESULTS_PATH = os.path.join(PROJECT_ROOT, "tuning_results")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "processed_dataset")
MODEL_PATH = os.path.join(TUNING_RESULTS_PATH, "best_hyper_model.pth")
TUNING_XLSX = os.path.join(TUNING_RESULTS_PATH, "tuning_metrics.xlsx")
OUTPUT_PATH = os.path.join(TUNING_RESULTS_PATH, "gradcam_images")

IMG_SIZE = 128
NUM_SAMPLES_PER_CLASS = 5

def preprocess_image(img_path, device):
    image = Image.open(img_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)

def load_best_hparams(excel_path):
    df = pd.read_excel(excel_path)
    best = df.loc[df["Best Val Loss"].idxmin()]
    return float(best.get("Dropout Rate", 0.5)), int(best.get("Dense Units", 128))

def discover_checkpoint(preferred_path, folder):
    if os.path.exists(preferred_path):
        return preferred_path
    for f in os.listdir(folder):
        if f.lower().endswith(".pth"):
            return os.path.join(folder, f)

def find_target_layer(model):
    if hasattr(model, "conv_final"):
        container = model.conv_final
        for m in reversed(list(container.modules())):
            if isinstance(m, torch.nn.Conv2d):
                return m

    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv2d):
            return m

def overlay_heatmap_cv2(img_path, heatmap):
    original = cv2.imread(img_path, cv2.IMREAD_COLOR)
    original = cv2.resize(original, (IMG_SIZE, IMG_SIZE))

    hm = heatmap - heatmap.min()
    hm = hm / (hm.max() + 1e-8)

    heatmap_u8 = np.uint8(255 * hm)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

def collect_images(folder, limit):
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".png")
    ]
    return files[:limit]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    dr, du = load_best_hparams(TUNING_XLSX)
    model = ConvNet(dropout_rate=dr, dense_units=du).to(device)

    ckpt_path = discover_checkpoint(MODEL_PATH, TUNING_RESULTS_PATH)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    target_layer = find_target_layer(model)
    open_dir = os.path.join(PROCESSED_DATA_PATH, "test", "Open-Eyes")
    closed_dir = os.path.join(PROCESSED_DATA_PATH, "test", "Close-Eyes")
    test_open = collect_images(open_dir, NUM_SAMPLES_PER_CLASS)
    test_closed = collect_images(closed_dir, NUM_SAMPLES_PER_CLASS)
    all_images = test_open + test_closed

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        targets = None
        for img_path in all_images:
            print(f"processing: {img_path}")
            input_tensor = preprocess_image(img_path, device)
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            heatmap = grayscale_cam[0]
            overlay = overlay_heatmap_cv2(img_path, heatmap)
            save_path = os.path.join(OUTPUT_PATH, f"gradcam_{os.path.basename(img_path)}")
            cv2.imwrite(save_path, overlay)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()