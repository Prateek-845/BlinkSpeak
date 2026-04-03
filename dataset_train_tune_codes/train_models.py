import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

PROCESSED_DATA_PATH = "./processed_dataset"
RESULTS_PATH = "./training_results"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 25

if os.path.exists(RESULTS_PATH):
    shutil.rmtree(RESULTS_PATH)
os.makedirs(RESULTS_PATH)

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    os.path.join(PROCESSED_DATA_PATH, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = datagen.flow_from_directory(
    os.path.join(PROCESSED_DATA_PATH, 'val'),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_generator = datagen.flow_from_directory(
    os.path.join(PROCESSED_DATA_PATH, 'test'),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

def build_model(num_conv_layers):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    current_conv_layers = 1
    filters = 64
    
    while current_conv_layers < num_conv_layers:
        model.add(Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        current_conv_layers += 1

        if current_conv_layers % 2 == 0:
            current_shape = model.output_shape
            height, width = current_shape[1], current_shape[2]

            if height > 2 and width > 2:
                model.add(MaxPooling2D(pool_size=(2, 2)))
                if filters < 512:
                    filters *= 2
           
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    actual_conv_layers = len([layer for layer in model.layers if isinstance(layer, Conv2D)])
    print(f"Built model with {actual_conv_layers} actual convolutional layers for target of {num_conv_layers}.")
    return model

def save_plots(history, model_name):
    model_dir = os.path.join(RESULTS_PATH, model_name)
    plt.figure()
    plt.plot(history.history['acc'], label='Training Accuracy')
    plt.plot(history.history['val_acc'], label='Validation Accuracy')
    plt.title(f'Accuracy Curve - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'accuracy_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss Curve - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'loss_curve.png'))
    plt.close()

def save_roc_auc_curve(y_true, y_pred_probs, model_name):
    model_dir = os.path.join(RESULTS_PATH, model_name)
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(model_dir, 'roc_auc_curve.png'))
    plt.close()
    return roc_auc

def save_classification_results(y_true, y_pred_classes, model_name):
    model_dir = os.path.join(RESULTS_PATH, model_name)
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_generator.class_indices, yticklabels=train_generator.class_indices)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))
    plt.close()

    report = classification_report(y_true, y_pred_classes, target_names=train_generator.class_indices.keys())
    with open(os.path.join(model_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes)
    recall = recall_score(y_true, y_pred_classes)
    f1 = f1_score(y_true, y_pred_classes)
    return accuracy, precision, recall, f1

results_df = pd.DataFrame(columns=[
    'Model Name', 'Actual Conv Layers', 'Best Epoch', 'Test Accuracy', 'AUC', 'Precision', 'Recall', 'F1-Score',
    'Best Val Loss', 'Best Val Acc', 'Corresponding Train Loss', 'Corresponding Train Acc'
])

for i in range(11):
    num_conv_layers = 10 + i
    model_name = f"Model_{num_conv_layers}_Conv_Layers"
    print(f"\nTraining {model_name}")

    model_dir = os.path.join(RESULTS_PATH, model_name)
    os.makedirs(model_dir)

    model = build_model(num_conv_layers)
    optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

    checkpoint_path = os.path.join(model_dir, "best_model.h5")
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping]
    )
    model.load_weights(checkpoint_path)
    save_plots(history, model_name)
    
    y_pred_probs = model.predict_generator(test_generator, steps=test_generator.samples // BATCH_SIZE + 1)
    y_pred_classes = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = test_generator.classes

    if len(y_pred_classes) > len(y_true):
        y_pred_classes = y_pred_classes[:len(y_true)]
    if len(y_pred_probs) > len(y_true):
        y_pred_probs = y_pred_probs[:len(y_true)]

    auc_score = save_roc_auc_curve(y_true, y_pred_probs, model_name)
    accuracy, precision, recall, f1 = save_classification_results(y_true, y_pred_classes, model_name)
    
    best_epoch_index = np.argmin(history.history['val_loss'])
    best_val_loss = history.history['val_loss'][best_epoch_index]
    best_val_acc = history.history['val_acc'][best_epoch_index]
    corresponding_train_loss = history.history['loss'][best_epoch_index]
    corresponding_train_acc = history.history['acc'][best_epoch_index]

    actual_conv_layers = len([layer for layer in model.layers if isinstance(layer, Conv2D)])

    new_row = {
        'Model Name': model_name,
        'Actual Conv Layers': actual_conv_layers,
        'Best Epoch': best_epoch_index + 1,
        'Test Accuracy': accuracy,
        'AUC': auc_score,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Best Val Loss': best_val_loss,
        'Best Val Acc': best_val_acc,
        'Corresponding Train Loss': corresponding_train_loss,
        'Corresponding Train Acc': corresponding_train_acc
    }
    results_df = results_df.append(new_row, ignore_index=True)
    results_df.to_excel(os.path.join(RESULTS_PATH, "cnn_model_metrics.xlsx"), index=False)
    os.remove(checkpoint_path)