"""
Configuration for AI Agent Video Anomaly Detection System.
AN AI AGENT INTEGRATING DEEP LEARNING AND VIDEO UNDERSTANDING
FOR ABNORMAL BEHAVIOR DETECTION IN SMART CITY SURVEILLANCE
"""

import os

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# ============================================================
# Dataset
# ============================================================
DATASET = "UCF-Crime"  # "UCF-Crime" | "ShanghaiTech" | "UCSD-Ped2"
SEQUENCE_LENGTH = 16       # number of frames per clip (T)
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
FRAME_STRIDE = 2           # sample every N-th frame
NUM_WORKERS = 4

# ============================================================
# Model — CNN Encoder
# ============================================================
CNN_BACKBONE = "resnet50"   # "resnet50" | "resnet18" | "i3d"
CNN_FEATURE_DIM = 2048      # output dim of backbone (resnet50=2048)
CNN_PRETRAINED = True
FREEZE_CNN = True           # freeze backbone weights

# ============================================================
# Model — LSTM (baseline temporal)
# ============================================================
LSTM_HIDDEN_DIM = 512
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_BIDIRECTIONAL = False

# ============================================================
# Model — GCN / GAT
# ============================================================
GCN_NODE_FEAT_DIM = 256     # node feature dimension
GCN_HIDDEN_DIM = 128
GCN_OUTPUT_DIM = 64
GCN_NUM_LAYERS = 3
GCN_DROPOUT = 0.3
GCN_MODEL_TYPE = "GAT"     # "GCN" | "GAT"
GAT_NUM_HEADS = 4

# ============================================================
# Graph Builder
# ============================================================
GRAPH_DISTANCE_THRESHOLD = 200.0   # max pixel distance for edge
GRAPH_USE_VELOCITY = True
GRAPH_USE_APPEARANCE = True
GRAPH_NODE_FEATURE_DIM = 128       # appearance embedding dim

# ============================================================
# Object Detection (pretrained — NOT trainable)
# ============================================================
YOLO_MODEL = "yolov8n.pt"          # nano model for speed
YOLO_CONFIDENCE = 0.5
YOLO_CLASSES = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck

# ============================================================
# AI Agent
# ============================================================
AGENT_T1 = 0.3             # threshold: below → NORMAL (skip)
AGENT_T2 = 0.7             # threshold: above → escalate to heavy model
AGENT_FRAME_SKIP_RATIO = 0.8   # skip 80% frames when scene is normal
AGENT_ADAPTIVE_THRESHOLD = True
AGENT_USE_ROI = True            # region-of-interest filtering
AGENT_CASCADE_LEVELS = ["yolo", "gcn", "vlm"]

# Adaptive threshold schedule (hour → threshold_modifier)
AGENT_TIME_SCHEDULE = {
    "night": {"hours": (0, 6), "modifier": -0.1},    # lower threshold at night
    "morning": {"hours": (6, 12), "modifier": 0.0},
    "afternoon": {"hours": (12, 18), "modifier": 0.05},
    "evening": {"hours": (18, 24), "modifier": -0.05},
}

# ============================================================
# Training
# ============================================================
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
SCHEDULER = "cosine"        # "cosine" | "step"
LOSS_TYPE = "ranking"       # "bce" | "ranking" (weakly supervised)
RANKING_MARGIN = 1.0

# ============================================================
# Evaluation
# ============================================================
EVAL_METRICS = ["auc", "roc", "accuracy", "inference_time", "compute_cost"]

# ============================================================
# Device
# ============================================================
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
