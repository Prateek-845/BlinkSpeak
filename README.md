# BlinkSpeak: Blink to Text Assistive Interface

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv&logoColor=white)

> **"Giving a voice to those who can only speak with their eyes."**

**BlinkSpeak** is a real time assistive technology designed for individuals with motor impairments (such as **ALS** or **Locked-in Syndrome**). It serves as an affordable, software based alternative to expensive eye tracking hardware.

Using a standard webcam, BlinkSpeak employs a custom **11 layer Convolutional Neural Network (CNN)** to detect voluntary eye blinks and a deterministic state machine to translate them into **Morse Code** which gets converted into text.


---

## Key Features

* **Hardware Agnostic:** Works on any standard laptop webcam; no specialized sensors required.
* **High Accuracy:** Powered by a custom CNN trained on **85,000+ images** from the MRL Eye Dataset, achieving **99.06% validation accuracy**.
* **Robust Detection:** Uses **MediaPipe Face Mesh** for precise landmark tracking, handling head movements and tilts effectively.
* **Smart Morse Logic:** Distinguishes between "involuntary blinks" (noise) and intentional signals.
* **Real Time Feedback:** Live dashboard displays current eye state, decoding buffer, and suggested actions.

---

## System Architecture

The system follows a five stage pipeline:
1. **Input Stream:** Captures video frames at 30 FPS.
2. **Preprocessing:** Grayscale conversion and resizing (128x128).
3. **Deep Learning:** CNN Binary Classifier (Open vs. Closed).
4. **Time Encoding:** Converts blink duration into Dots/Dashes.
5. **Text Synthesis:** Maps sequences to the alphanumeric dictionary.

---

## Installation & Setup

### Prerequisites
* **Python 3.10** (Crucial for MediaPipe compatibility)
* Webcam

### 1. Clone the Repository
```bash
git clone https://github.com/Prateek-845/BlinkSpeak.git
cd BlinkSpeak
```
### 2. Create a Virtual Environment
It is highly recommended to use a virtual environment to avoid version conflicts.

#### Windows:

```bash
py -3.10 -m venv venv
.\venv\Scripts\activate
```

#### Linux/Mac:
```bash
python3.10 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Usage
Run the inference script to start the application:
```bash
python blink_inference.py
```
---

### Controls
The system distinguishes signals based on blink duration:
| Action | Blink Duration | Description |
| :--- | :--- | :--- |
| **DOT (.)** | 0.2s - 0.7s | A definitive, intentional blink. |
| **DASH (-)** | > 0.7s | A longer, deliberate eye closure. |
| **SPACE** | 4.0s Pause | Keep eyes open to complete a word. |
| **BACKSPACE**| ---- | Blink 4 long dashes to delete the last character. |
| **CLEAR** | ----- | Blink 5 long dashes to clear the screen. |

---

### Project Structure
```
BlinkSpeak/
├── blink_inference.py   # Main application for real-time inference
├── morse_dict.py        # Morse code dictionary & helper functions
├── preprocess.py        # Data pipeline for MRL Eye Dataset
├── train_models.py      # Architecture search script (Phase 1)
├── tuning.py            # Hyperparameter tuning script (Phase 2)
├── requirements.txt     # Project dependencies
└── tuning_results/      # Stores the trained model (.pth)
```
---
## Model Performance & Evaluation

The core of **BlinkSpeak** is a custom **11 Layer Convolutional Neural Network (CNN)** optimized for binary eye state classification. The model was trained on the **MRL Eye Dataset** (85,000+ near infrared images) and achieved exceptional performance metrics, ensuring reliability for real time communication.

### Key Metrics
| Metric | Score | Description |
| :--- | :--- | :--- |
| **Test Accuracy** | **99.00%** | Correctly classified 99% of unseen images. |
| **Validation Accuracy** | **99.06%** | Peak accuracy during hyperparameter tuning. |
| **AUC Score** | **0.9994** | Near-perfect class separability (1.0 is perfect). |
| **F1-Score** | **0.9901** | Balanced precision and recall for both open/closed states. |
| **Inference Speed** | **~30 FPS** | Optimized for real-time CPU performance. |

---

### Hyperparameter Optimization
I conducted an extensive Grid Search to fine tune the model. The best configuration significantly reduced overfitting:

* **Architecture:** Custom 11-Layer CNN (Feature Extraction + Classification Head).
* **Optimizer:** Adam
* **Learning Rate:** `0.0001`
* **Dropout Rate:** `0.4` (Prevents overfitting)
* **Batch Size:** `64`

---

### Confusion Matrix Analysis
The model demonstrates high resilience to "flickering" (false state changes). Out of **8,490 test samples**, the model made only **85 errors**:

| | **Predicted Closed** | **Predicted Open** |
| :--- | :---: | :---: |
| **Actual Closed** | **4136** (True Negatives) | 58 (False Positives) |
| **Actual Open** | 27 (False Negatives) | **4269** (True Positives) |

> **Insight:** The False Negative rate is extremely low (27/4296), meaning the system rarely misses an intentional blink, which is critical for accurate Morse code typing.

---

### Classification Report
The model shows no bias towards either class, performing equally well for both open and closed eyes.

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Closed Eyes** | 0.99 | 0.99 | 0.99 | 4194 |
| **Open Eyes** | 0.99 | 0.99 | 0.99 | 4296 |
| **Weighted Avg** | **0.99** | **0.99** | **0.99** | **8490** |

---

### Performance Visualizations

| Accuracy Curve | Loss Curve |
| :---: | :---: |
| ![Accuracy](tuning_results/best_hyper_model_accuracy_curve.png) | ![Loss](tuning_results/best_hyper_model_loss_curve.png) |

> **Note:** The convergence of Training (Blue) and Validation (Orange) lines indicates robust generalization with minimal overfitting.

---

### License
This project is open source and available for educational and assistive purposes.
