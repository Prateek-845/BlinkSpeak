import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import itertools
from tqdm import tqdm

PROCESSED_DATA_PATH = "./processed_dataset"
TUNING_RESULTS_PATH = "./tuning_results"
IMG_SIZE = 128
EPOCHS = 50 
BEST_MODEL_NAME = "best_hyper_model.pth" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" using device: {device}")

LEARNING_RATES = [0.0005, 0.0001, 0.00005, 0.00001]
DROPOUT_RATES = [0.4, 0.5]
DENSE_UNITS = [64, 128]
BATCH_SIZES = [16, 32, 48, 64]

tuning_combinations = list(itertools.product(LEARNING_RATES, DROPOUT_RATES, DENSE_UNITS, BATCH_SIZES))

shutil.rmtree(TUNING_RESULTS_PATH)
os.makedirs(TUNING_RESULTS_PATH)

class ConvNet(nn.Module):
    def __init__(self, dropout_rate, dense_units):
        super(ConvNet, self).__init__()
        # PyTorch: (in_channels, out_channels, kernel_size, padding)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # kernel_size=2, stride=2
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_final = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, dense_units),
            nn.ReLU(),
            nn.BatchNorm1d(dense_units),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, 1) 
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv_final(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

def save_plots(history_dict, model_name):
    model_dir = TUNING_RESULTS_PATH
    plt.figure()
    plt.plot(history_dict['train_acc'], label='Training Accuracy')
    plt.plot(history_dict['val_acc'], label='Validation Accuracy')
    plt.title(f'Accuracy Curve - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(model_dir, f'{model_name}_accuracy_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(history_dict['train_loss'], label='Training Loss')
    plt.plot(history_dict['val_loss'], label='Validation Loss')
    plt.title(f'Loss Curve - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(model_dir, f'{model_name}_loss_curve.png'))
    plt.close()

def save_roc_auc_curve(y_true, y_pred_probs, model_name):
    model_dir = TUNING_RESULTS_PATH
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(model_dir, f'{model_name}_roc_auc_curve.png'))
    plt.close()
    return roc_auc

def save_classification_results(y_true, y_pred_classes, model_name, class_indices):
    model_dir = TUNING_RESULTS_PATH
    cm = confusion_matrix(y_true, y_pred_classes)
    class_names = list(class_indices.keys())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label'); plt.ylabel('True Label')
    plt.savefig(os.path.join(model_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()
    
    report = classification_report(y_true, y_pred_classes, target_names=class_names)
    with open(os.path.join(model_dir, f'{model_name}_classification_report.txt'), 'w') as f:
        f.write(report)
    
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes)
    recall = recall_score(y_true, y_pred_classes)
    f1 = f1_score(y_true, y_pred_classes)
    return accuracy, precision, recall, f1

results_df = pd.DataFrame(columns=[
    'Model Name', 'Learning Rate', 'Dropout Rate', 'Dense Units', 'Batch Size', 
    'Best Epoch', 'Best Val Loss', 'Best Val Acc', 'Test Accuracy', 
    'Test AUC', 'Test Precision', 'Test Recall', 'Test F1-Score'
])
best_overall_val_loss = float('inf')
best_history_dict = None
best_model_params = None
best_model_class_indices = None

data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor() # scales to [0, 1]
])

test_dataset = datasets.ImageFolder(os.path.join(PROCESSED_DATA_PATH, 'test'), transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
y_true = np.array(test_dataset.targets)
if best_model_class_indices is None:
    best_model_class_indices = test_dataset.class_to_idx

for lr, dr, du, bs in tuning_combinations:
    model_name = f"model_lr-{lr}_dr-{dr}_du-{du}_bs-{bs}"
    print(f"\n training {model_name}\n")

    train_dataset = datasets.ImageFolder(os.path.join(PROCESSED_DATA_PATH, 'train'), transform=data_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(PROCESSED_DATA_PATH, 'val'), transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=4)

    model = ConvNet(dropout_rate=dr, dense_units=du).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss() # stability
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    best_epoch_index = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, total_train = 0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds == labels).sum().item()
            total_train += labels.size(0)
            
        avg_train_loss = train_loss / total_train
        avg_train_acc = train_correct / total_train
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)

        model.eval()
        val_loss, val_correct, total_val = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = val_loss / total_val
        avg_val_acc = val_correct / total_val
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            print(f"Val loss improved ({best_val_loss:.6f} to {avg_val_loss:.6f}) saving model")
            best_val_loss = avg_val_loss
            best_epoch_index = epoch
            torch.save(model.state_dict(), os.path.join(TUNING_RESULTS_PATH, "temp_best.pth"))
            patience_counter = 0 
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(torch.load(os.path.join(TUNING_RESULTS_PATH, "temp_best.pth")))
    model.eval()
    
    all_pred_probs = []
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_pred_probs.extend(probs)
            
    y_pred_probs = np.array(all_pred_probs).flatten()
    y_pred_classes = (y_pred_probs > 0.5).astype(int)

    if len(y_pred_classes) > len(y_true):
        y_pred_classes = y_pred_classes[:len(y_true)]
    if len(y_pred_probs) > len(y_true):
        y_pred_probs = y_pred_probs[:len(y_true)]
        
    test_accuracy = accuracy_score(y_true, y_pred_classes)
    test_precision = precision_score(y_true, y_pred_classes)
    test_recall = recall_score(y_true, y_pred_classes)
    test_f1 = f1_score(y_true, y_pred_classes)
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    test_auc = auc(fpr, tpr)
    
    new_row = {
        'Model Name': model_name, 'Learning Rate': lr, 'Dropout Rate': dr, 'Dense Units': du, 'Batch Size': bs, 
        'Best Epoch': best_epoch_index + 1,
        'Best Val Loss': best_val_loss,
        'Best Val Acc': history['val_acc'][best_epoch_index],
        'Test Accuracy': test_accuracy, 'Test AUC': test_auc, 'Test Precision': test_precision,
        'Test Recall': test_recall, 'Test F1-Score': test_f1
    }
    results_df = results_df.append(new_row, ignore_index=True)
    results_df.to_excel(os.path.join(TUNING_RESULTS_PATH, "tuning_metrics.xlsx"), index=False)
    
    if best_val_loss < best_overall_val_loss:
        print(f"\n best model (Val Loss: {best_val_loss:.6f}) saving as {BEST_MODEL_NAME}")
        best_overall_val_loss = best_val_loss
        best_history_dict = history
        best_model_params = (dr, du)
        shutil.copyfile(os.path.join(TUNING_RESULTS_PATH, "temp_best.pth"), os.path.join(TUNING_RESULTS_PATH, BEST_MODEL_NAME))

if os.path.exists(os.path.join(TUNING_RESULTS_PATH, "temp_best.pth")):
    os.remove(os.path.join(TUNING_RESULTS_PATH, "temp_best.pth"))

if best_history_dict is not None:
    save_plots(best_history_dict, "best_hyper_model")
    dr, du = best_model_params
    best_model = ConvNet(dropout_rate=dr, dense_units=du).to(device)
    best_model.load_state_dict(torch.load(os.path.join(TUNING_RESULTS_PATH, BEST_MODEL_NAME)))
    best_model.eval()

    all_pred_probs = []
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Final Test"):
            inputs = inputs.to(device)
            outputs = best_model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_pred_probs.extend(probs)
            
    y_pred_probs = np.array(all_pred_probs).flatten()
    y_pred_classes = (y_pred_probs > 0.5).astype(int)

    if len(y_pred_classes) > len(y_true):
        y_pred_classes = y_pred_classes[:len(y_true)]
    if len(y_pred_probs) > len(y_true):
        y_pred_probs = y_pred_probs[:len(y_true)]

    auc_score = save_roc_auc_curve(y_true, y_pred_probs, "best_hyper_model")
    accuracy, precision, recall, f1 = save_classification_results(y_true, y_pred_classes, "best_hyper_model", best_model_class_indices)

    print("\n Best Model Test Set Performance")
    print(f" Test Accuracy: {accuracy:.4f}")
    print(f" Test AUC: {auc_score:.4f}")
    print(f" Test Precision: {precision:.4f}")
    print(f" Test Recall: {recall:.4f}")
    print(f" Test F1-Score: {f1:.4f}")

else:
    print("\n no model was successfully trained")

print("\n hyperparameter tuning complete")