# Google Colab Notebook — Training on Free GPU
# 
# Copy this entire file content into a Google Colab notebook
# or upload it as a .py file and convert to notebook.
#
# This script is designed to run on Google Colab with free T4 GPU.
# It handles: dataset download, training (all phases), evaluation.

COLAB_SETUP = """
# ============================================================
# CELL 1: Setup & Install Dependencies
# ============================================================
# Run this cell first in Google Colab

!pip install torch torchvision torchaudio
!pip install torch-geometric
!pip install ultralytics
!pip install opencv-python numpy scipy scikit-learn
!pip install matplotlib seaborn tqdm tensorboard
!pip install deep-sort-realtime
!pip install gdown  # for downloading from Google Drive

# Clone your repo (replace with your repo URL)
# !git clone https://github.com/YOUR_USERNAME/final_project.git
# %cd final_project

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
"""

COLAB_DOWNLOAD_DATASET = """
# ============================================================
# CELL 2: Download UCF-Crime Dataset (subset)
# ============================================================
# Full UCF-Crime is ~128GB. We download a manageable subset.

import os
import gdown

# Option A: Download pre-extracted features (recommended — much smaller)
# Many researchers share pre-extracted I3D/C3D features for UCF-Crime
# 
# Option B: Download a small subset of videos for demo
#
# Option C: Use ShanghaiTech (smaller dataset)

# Create data directories
os.makedirs("data/train/normal", exist_ok=True)
os.makedirs("data/train/anomaly", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

print("📁 Data directories created")
print()
print("⬇️  To download UCF-Crime features:")
print("   Visit: https://www.crcv.ucf.edu/projects/real-world/")
print("   Or use pre-extracted features from:")
print("   https://github.com/WaqasSultani/AnomalyDetectionCVPR2018")
print()
print("For quick testing, we'll generate synthetic data first.")
"""

COLAB_TRAIN_BASELINE = """
# ============================================================
# CELL 3: Train Baseline Model (Phase 1)
# ============================================================

import sys
sys.path.append('.')

import torch
import config

# Override config for Colab
config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
config.BATCH_SIZE = 16
config.NUM_EPOCHS = 30
config.NUM_WORKERS = 2

print(f"Device: {config.DEVICE}")

# If you have real data:
# from train import train_baseline
# from datasets.video_dataset import get_dataloaders
# loaders = get_dataloaders()
# cnn, lstm = train_baseline(loaders["train"], loaders["test"])

# For demo with synthetic features:
from models.cnn_encoder import CNNEncoder
from models.lstm_model import LSTMAnomaly, RankingLoss

device = config.DEVICE

# Create models
cnn = CNNEncoder(backbone="resnet50", pretrained=True, freeze_backbone=True).to(device)
lstm = LSTMAnomaly(input_dim=2048, hidden_dim=512).to(device)

# Synthetic training data
import numpy as np

num_samples = 200
T = 16

# Generate features (as if extracted from CNN)
normal_features = torch.randn(num_samples // 2, T, 2048) * 0.5
anomaly_features = torch.randn(num_samples // 2, T, 2048) * 0.5 + 0.3

features = torch.cat([normal_features, anomaly_features], dim=0)
labels = torch.cat([
    torch.zeros(num_samples // 2),
    torch.ones(num_samples // 2),
])

# Shuffle
perm = torch.randperm(num_samples)
features = features[perm]
labels = labels[perm]

# Split
train_feat = features[:160].to(device)
train_labels = labels[:160].to(device)
test_feat = features[160:].to(device)
test_labels = labels[160:].to(device)

# Training
optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-4)
criterion = torch.nn.BCELoss()

lstm.train()
for epoch in range(30):
    # Mini-batch training
    for i in range(0, len(train_feat), 16):
        batch_feat = train_feat[i:i+16]
        batch_labels = train_labels[i:i+16].unsqueeze(1)
        
        scores = lstm(batch_feat)
        loss = criterion(scores, batch_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        lstm.eval()
        with torch.no_grad():
            test_scores = lstm(test_feat)
            test_loss = criterion(test_scores, test_labels.unsqueeze(1))
            preds = (test_scores.squeeze() > 0.5).float()
            acc = (preds == test_labels).float().mean()
        lstm.train()
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | "
              f"Val Loss: {test_loss.item():.4f} | Val Acc: {acc.item():.3f}")

# Save
torch.save({
    "lstm_state": lstm.state_dict(),
    "cnn_state": cnn.state_dict(),
}, "checkpoints/baseline_best.pth")
print("\\n✅ Baseline training complete!")
"""

COLAB_TRAIN_GCN = """
# ============================================================
# CELL 4: Train GCN/GAT Model (Phase 2)
# ============================================================

from models.gcn_model import GATAnomaly
from graph.graph_builder import GraphBuilder, Detection
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create GAT model
gat = GATAnomaly(
    node_feat_dim=8,
    hidden_dim=128,
    output_dim=64,
    dropout=0.3,
).to(device)

graph_builder = GraphBuilder(use_appearance=False)

# Generate synthetic graph data
def generate_training_graphs(num_samples=200):
    graphs = []
    labels = []
    
    for i in range(num_samples):
        is_anomaly = i >= num_samples // 2
        num_objects = np.random.randint(2, 8)
        
        dets = []
        for j in range(num_objects):
            cx = np.random.uniform(50, 600)
            cy = np.random.uniform(50, 400)
            det = Detection(
                bbox=(cx-25, cy-50, cx+25, cy+50),
                class_id=0,
                confidence=np.random.uniform(0.5, 0.95),
                track_id=j,
            )
            if is_anomaly:
                det.vx = np.random.uniform(5, 20)
                det.vy = np.random.uniform(-10, 10)
            else:
                det.vx = np.random.uniform(-2, 2)
                det.vy = np.random.uniform(-1, 1)
            dets.append(det)
        
        graph = graph_builder.build(dets)
        graphs.append(graph)
        labels.append(1 if is_anomaly else 0)
    
    return graphs, labels

print("Generating training graphs...")
graphs, labels = generate_training_graphs(400)

# Shuffle
perm = np.random.permutation(len(graphs))
graphs = [graphs[i] for i in perm]
labels = [labels[i] for i in perm]

# Train
optimizer = torch.optim.Adam(gat.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = torch.nn.BCELoss()

gat.train()
for epoch in range(50):
    total_loss = 0
    correct = 0
    
    for g, label in zip(graphs[:320], labels[:320]):
        if g["num_nodes"] == 0:
            continue
            
        score = gat(
            g["node_features"].to(device),
            g["edge_index"].to(device),
            g["num_nodes"],
        )
        
        target = torch.tensor([[float(label)]], device=device)
        loss = criterion(score, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = 1 if score.item() > 0.5 else 0
        correct += (pred == label)
    
    if (epoch + 1) % 10 == 0:
        # Validation
        gat.eval()
        val_correct = 0
        with torch.no_grad():
            for g, label in zip(graphs[320:], labels[320:]):
                if g["num_nodes"] == 0:
                    continue
                score = gat(
                    g["node_features"].to(device),
                    g["edge_index"].to(device),
                    g["num_nodes"],
                )
                pred = 1 if score.item() > 0.5 else 0
                val_correct += (pred == label)
        gat.train()
        
        print(f"Epoch {epoch+1:3d} | Loss: {total_loss/320:.4f} | "
              f"Train Acc: {correct/320:.3f} | Val Acc: {val_correct/80:.3f}")

torch.save({"model_state": gat.state_dict()}, "checkpoints/gat_best.pth")
print("\\n✅ GAT training complete!")
"""

COLAB_EVALUATE = """
# ============================================================
# CELL 5: Evaluate & Generate Results
# ============================================================

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Run demo pipeline
exec(open("demo.py").read())

# Or run agent simulation
exec(open("simulate_agent.py").read())

print("\\n✅ All results generated!")
print("📁 Check 'figures/' folder for paper-ready plots")
"""

COLAB_DOWNLOAD_RESULTS = """
# ============================================================
# CELL 6: Download Results to Local Machine
# ============================================================

# Zip all figures and checkpoints
!zip -r results.zip figures/ checkpoints/ 2>/dev/null

# Download
from google.colab import files
files.download('results.zip')

print("✅ Results downloaded!")
"""


def generate_colab_notebook():
    """Generate a .py file that can be uploaded to Colab."""
    
    cells = [
        ("# Setup", COLAB_SETUP),
        ("# Download Dataset", COLAB_DOWNLOAD_DATASET),
        ("# Train Baseline", COLAB_TRAIN_BASELINE),
        ("# Train GCN", COLAB_TRAIN_GCN),
        ("# Evaluate", COLAB_EVALUATE),
        ("# Download Results", COLAB_DOWNLOAD_RESULTS),
    ]
    
    content = '"""Google Colab Notebook for AI Agent Video Anomaly Detection"""\n\n'
    
    for title, code in cells:
        content += f"\n# {'='*60}\n{title}\n# {'='*60}\n"
        content += code.strip() + "\n\n"
    
    return content


if __name__ == "__main__":
    content = generate_colab_notebook()
    with open("colab_training.py", "w") as f:
        f.write(content)
    print("✅ Generated colab_training.py — upload to Google Colab")
