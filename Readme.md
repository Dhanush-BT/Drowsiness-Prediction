
---

# 🚗 Driver Drowsiness Detection using Deep Learning (LSTM)

An **AI-powered Driver Drowsiness Detection System** that analyzes facial behavior from video sequences to detect driver fatigue.
The system extracts **eye movement, mouth movement, head pose, and fatigue indicators** from facial landmarks and uses a **Long Short-Term Memory (LSTM) neural network** to classify whether a driver is **Alert or Drowsy**.

The project focuses on **temporal behavioral analysis**, making it more reliable than traditional frame-based detection systems.

---

# 📌 Overview

Driver fatigue is one of the major causes of road accidents worldwide. Traditional drowsiness detection methods rely on **static image classification**, which fails to capture behavioral patterns that evolve over time.

This project introduces a **sequence-based deep learning approach** where facial behavior across multiple frames is analyzed using an **LSTM network**.

The system detects drowsiness based on:

* Eye closure patterns
* Yawning detection
* Head movement patterns
* Eye closure duration (PERCLOS)

---

# 🧠 Key Features

✅ Real-time facial landmark tracking using **MediaPipe Face Mesh**
✅ Behavioral feature extraction from videos
✅ Temporal sequence modeling using **LSTM networks**
✅ Automatic dataset generation from video frames
✅ Multiple fatigue indicators for robust prediction
✅ Model training with **PyTorch**
✅ Early stopping and learning rate scheduling
✅ Visualization of training performance

---

# 🏗 System Architecture

```
Video Dataset
      │
      ▼
Face Landmark Detection (MediaPipe)
      │
      ▼
Feature Extraction
(EAR, MAR, Head Pose, PERCLOS)
      │
      ▼
Frame-Level Dataset Creation
      │
      ▼
Sequence Generation (30 frames)
      │
      ▼
Feature Normalization
      │
      ▼
LSTM Deep Learning Model
      │
      ▼
Drowsy / Alert Prediction
```

---

# 📊 Features Extracted

The model uses **6 behavioral features** extracted from facial landmarks.

| Feature | Description                            |
| ------- | -------------------------------------- |
| EAR     | Eye Aspect Ratio (detects eye closure) |
| MAR     | Mouth Aspect Ratio (detects yawning)   |
| Pitch   | Head up/down movement                  |
| Yaw     | Head left/right movement               |
| Roll    | Head tilt                              |
| PERCLOS | Percentage of eye closure over time    |

These features are computed using **MediaPipe Face Mesh landmarks**.

---

# 👁 Eye Aspect Ratio (EAR)

EAR is used to measure eye openness.

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

If EAR falls below a threshold, the eye is considered **closed**.

---

# 😮 Mouth Aspect Ratio (MAR)

MAR detects yawning behavior by measuring the distance between upper and lower lips.

```
MAR = vertical_mouth_distance / horizontal_mouth_distance
```

Large MAR values often indicate **yawning**, which is a sign of fatigue.

---

# 🧭 Head Pose Estimation

Head pose is estimated using **Perspective-n-Point (PnP)** solving with facial landmarks.

The system computes:

* **Pitch**
* **Yaw**
* **Roll**

These angles help detect **driver head drooping or distraction**.

---

# 😴 PERCLOS (Percentage of Eye Closure)

PERCLOS measures the **percentage of time the driver's eyes remain closed**.

```
PERCLOS = Closed Frames / Total Frames
```

It is one of the **most reliable fatigue indicators** used in driver monitoring systems.

---

# 📂 Dataset

The system processes the **UTA-RLDD (UTA Real-Life Drowsiness Dataset)**.

Dataset structure:

```
UTA-RLDD
   ├── Subject1
   │      ├── 0 (Alert)
   │      └── 10 (Drowsy)
   │
   ├── Subject2
   │      ├── 0
   │      └── 10
```

Labels:

| Label | Meaning |
| ----- | ------- |
| 0     | Alert   |
| 1     | Drowsy  |

---

# 🔄 Data Processing Pipeline

### Step 1 — Face Landmark Detection

MediaPipe Face Mesh extracts **468 facial landmarks** from each frame.

### Step 2 — Feature Extraction

For every frame the system computes:

* EAR
* MAR
* Head pose angles
* PERCLOS

### Step 3 — Frame Dataset Creation

The extracted features are stored in:

```
frame_level_dataset.csv
```

---

# 🔗 Sequence Creation

The system converts frame data into **temporal sequences**.

Sequence length:

```
SEQ_LEN = 30 frames
```

Each training sample becomes:

```
[EAR, MAR, pitch, yaw, roll, PERCLOS] × 30 frames
```

Resulting tensor shape:

```
X shape = (samples, 30, 6)
```

---

# 📏 Feature Normalization

Features are normalized for stable model training.

| Feature | Normalization       |
| ------- | ------------------- |
| EAR     | / 1.0               |
| MAR     | / 1.5               |
| Pitch   | / 180               |
| Yaw     | / 90                |
| Roll    | / 180               |
| PERCLOS | Already between 0–1 |

---

# 🧠 Deep Learning Model

The system uses a **two-layer LSTM architecture**.

### Model Architecture

```
Input: (30 timesteps × 6 features)

LSTM Layer (64 units)
↓
Dropout (0.4)

LSTM Layer (32 units)
↓
Dropout (0.4)

Fully Connected Layer (16)
↓
Sigmoid Output
```

Output:

```
0 → Alert
1 → Drowsy
```

---

# ⚙️ Training Configuration

| Parameter               | Value                |
| ----------------------- | -------------------- |
| Framework               | PyTorch              |
| Optimizer               | Adam                 |
| Learning Rate           | 0.001                |
| Loss Function           | Binary Cross Entropy |
| Batch Size              | 256                  |
| Epochs                  | 50                   |
| Early Stopping          | Enabled              |
| Learning Rate Scheduler | ReduceLROnPlateau    |

---

# 📈 Model Performance

The LSTM model was trained for **50 epochs** using the extracted behavioral features.

### 🎯 Final Accuracy

**Current Model Accuracy:**
**91% at 50 epochs**

Example training output:

```
Epoch 50/50 | Train Loss: 0.18 | Val Loss: 0.21 | Accuracy: 0.91
```

---

# 📊 Evaluation Metrics

The model evaluation includes:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Example evaluation output:

```
Classification Report
Confusion Matrix
```

These metrics confirm the model's ability to reliably detect drowsy behavior.

---

# 📉 Training Visualization

The training process plots:

* Training Loss
* Validation Loss

This helps monitor model convergence and detect overfitting.

---

# 🧪 Project Structure

```
Driver-Drowsiness-Detection
│
├── dataset
│
├── frame_level_dataset.csv
│
├── X_sequences.npy
├── y_sequences.npy
│
├── X_norm.npy
├── y_norm.npy
│
├── best_drowsiness_model.pth
│
├── dataset_builder.py
├── sequence_generator.py
├── train_lstm.py
│
└── README.md
```

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

### Step 1 — Build dataset

```
python dataset_builder.py
```

### Step 2 — Generate sequences

```
python sequence_generator.py
```

### Step 3 — Train model

```
python train_lstm.py
```

---

# 🔮 Future Improvements

* Real-time webcam drowsiness detection
* Integration with vehicle safety systems
* Edge AI deployment
* Vision Transformer models
* Multi-modal fatigue detection (EEG + camera)

---

# 🌍 Applications

* Driver monitoring systems
* Smart vehicles
* Fleet safety monitoring
* Autonomous vehicle safety
* Transport industry safety systems

---

# 👨‍💻 Author

**Dhanush BT**

AI Developer | Computer Vision Enthusiast

Interests:

* Artificial Intelligence
* Computer Vision
* Deep Learning
* Smart Surveillance Systems

---

⭐ If you find this project useful, consider **starring the repository**.
