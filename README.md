

# 🚨 AI Agent for Video Anomaly Detection in Smart City Surveillance

## 📌 Overview

This project focuses on building an **AI Agent-based system** for detecting abnormal behavior in urban surveillance videos.

The system integrates:

* Deep Learning (CNN, LSTM)
* Graph Neural Networks (GCN/GAT)
* AI Agent (adaptive control for efficient inference)

### 🎯 Objective

* Detect abnormal events (e.g., fighting, accidents, unusual motion)
* Model interactions between objects (via Graph)
* Reduce computational cost using an **Adaptive AI Agent**

---

## 🧠 System Architecture

```
Video Input
    │
    ▼
[1] Object Detection (YOLO - pretrained)
    │
    ▼
[2] Tracking (SORT / DeepSORT)
    │
    ▼
[3] Feature Extraction (CNN - pretrained)
    │
    ▼
[4] Graph Builder  ← (core module)
    │
    ▼
[5] GCN / Temporal Model (trainable)
    │
    ▼
[6] AI Agent (adaptive escalation)
    │
    ▼
[7] Output (Anomaly Score / Alert)
```

---

## 📂 Project Structure

```
project/
│
├── data/
│   ├── train/
│   │   ├── normal/
│   │   └── anomaly/
│   ├── test/
│
├── datasets/
│   └── video_dataset.py
│
├── models/
│   ├── cnn_encoder.py
│   ├── lstm_model.py
│   ├── gcn_model.py
│
├── graph/
│   └── graph_builder.py
│
├── agent/
│   └── adaptive_agent.py
│
├── train.py
├── evaluate.py
└── README.md
```

---

## 📊 Dataset

### Recommended Dataset

* UCF-Crime (primary)
* ShanghaiTech (optional)

### Data Format

```
train/
  normal/
    video1.mp4
  anomaly/
    video2.mp4
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install torch torchvision opencv-python numpy
```

---

### 2. Prepare dataset

* Download dataset
* Organize into:

```
dataset/train/normal/
dataset/train/anomaly/
```

---

### 3. Run DataLoader

```bash
python train.py
```

---

## 🔧 Core Modules (IMPORTANT)

---

### 1. Video Dataset

Implement a PyTorch Dataset that:

* Reads video files
* Samples frame sequences (T frames)
* Outputs tensor: `(T, C, H, W)`

---

### 2. Graph Builder (CORE CONTRIBUTION)

Convert detected objects into a graph.

#### Node features:

* position (x, y)
* bounding box size
* velocity (vx, vy)
* appearance embedding

#### Edge features:

* distance
* relative position
* relative velocity

#### Output:

```
Graph = (nodes, edges, edge_features)
```

---

### 3. GCN / GAT Model

Input:

* Graph from Graph Builder

Output:

* anomaly score

---

### 4. Temporal Modeling

Options:

* LSTM
* Transformer

Purpose:

* capture temporal patterns in video

---

### 5. AI Agent (IMPORTANT)

The AI Agent controls computation dynamically.

#### Responsibilities:

* Decide whether to:

  * stop processing
  * run GCN
  * escalate to heavy model (e.g., VLM)

#### Example logic:

```python
if score < T1:
    return NORMAL
elif score < T2:
    run_gcn()
else:
    run_heavy_model()
```

#### Adaptive Threshold:

```python
T = f(time, location, history)
```

---

## 🧪 Training Pipeline

1. Load dataset
2. Extract features (CNN)
3. Train baseline model (LSTM)
4. Train GCN model
5. Integrate AI Agent
6. Evaluate performance

---

## 📈 Evaluation Metrics

* AUC (Primary)
* ROC Curve
* Accuracy
* Inference Time
* Compute Cost (number of model calls)

---

## 📊 Expected Results

| Model    | AUC  | Compute |
| -------- | ---- | ------- |
| Baseline | 0.xx | Low     |
| + GCN    | ↑    | Medium  |
| + Agent  | ≈    | ↓↓↓     |

---

## 🔥 Research Contributions

1. Graph-based interaction modeling for anomaly detection
2. Adaptive AI Agent for efficient inference
3. End-to-end intelligent surveillance system

---

## 🛠 TODO (For Copilot / Development)

* [ ] Implement VideoDataset
* [ ] Implement CNN feature extractor (ResNet)
* [ ] Implement LSTM baseline model
* [ ] Implement Graph Builder
* [ ] Implement GCN/GAT model
* [ ] Implement AI Agent (adaptive thresholds)
* [ ] Training loop
* [ ] Evaluation script

---

## 🎤 Example Explanation (for interview)

> This project proposes an AI agent-based architecture for anomaly detection in surveillance videos.
> It combines graph neural networks to model interactions between objects and an adaptive agent to optimize computational efficiency.

---

## ⚠️ Notes

* Do NOT train YOLO from scratch
* Use pretrained models for perception
* Focus on:

  * Graph modeling
  * AI Agent design

---

## 🚀 Future Work

* Integrate Vision-Language Models (VLM)
* Improve explainability
* Deploy on edge devices

---

## 👨‍💻 Author

* AI Research / Computer Vision / Graph Learning
* Focus: Efficient AI Systems for Smart Cities

---

---

# 🚀 Cách dùng README này với Copilot

Bạn làm:

1. Tạo repo → paste README này
2. Tạo file trống (graph_builder.py, gcn_model.py…)
3. Gõ comment kiểu:

```python
# build graph from detections using distance + velocity
```
