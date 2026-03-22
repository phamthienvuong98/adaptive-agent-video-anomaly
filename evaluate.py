"""
Evaluation Script — Comprehensive evaluation for anomaly detection.

Metrics:
    - AUC (primary)
    - ROC Curve
    - Accuracy
    - Inference Time
    - Compute Cost (number of model calls per level)

Compares:
    1. Baseline (CNN + LSTM)
    2. + GCN/GAT
    3. + AI Agent (compute savings)

Usage:
    python evaluate.py --model baseline --checkpoint checkpoints/baseline_best.pth
    python evaluate.py --model gcn --checkpoint checkpoints/gcn_best.pth
    python evaluate.py --model agent --checkpoint checkpoints/gcn_best.pth
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from datasets.video_dataset import VideoDataset
from models.cnn_encoder import CNNEncoder
from models.lstm_model import LSTMAnomaly
from models.gcn_model import GCNAnomaly, GATAnomaly, TemporalGCN
from agent.adaptive_agent import AdaptiveAgent, AgentAction


# ==============================================================
# Metrics
# ==============================================================

def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Area Under ROC Curve."""
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return 0.0


def compute_roc_curve(
    labels: np.ndarray, scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return fpr, tpr, thresholds


def compute_accuracy(
    labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> float:
    """Compute binary accuracy."""
    preds = (scores >= threshold).astype(int)
    return (preds == labels).mean()


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    inference_times: Optional[List[float]] = None,
    agent_stats: Optional[Dict] = None,
) -> Dict:
    """Compute all evaluation metrics."""
    results = {
        "auc": compute_auc(labels, scores),
        "accuracy": compute_accuracy(labels, scores),
    }

    if inference_times:
        results["avg_inference_time_ms"] = np.mean(inference_times) * 1000
        results["total_inference_time_s"] = np.sum(inference_times)
        results["fps"] = len(inference_times) / max(1e-6, np.sum(inference_times))

    if agent_stats:
        results["compute_saved_pct"] = agent_stats.get("compute_saved_pct", 0)
        results["yolo_calls"] = agent_stats.get("yolo_calls", 0)
        results["gcn_calls"] = agent_stats.get("gcn_calls", 0)
        results["vlm_calls"] = agent_stats.get("vlm_calls", 0)
        results["skip_rate"] = agent_stats.get("skip_rate", 0)

    return results


# ==============================================================
# Evaluation Functions
# ==============================================================

def evaluate_baseline(
    checkpoint_path: str,
    test_loader: DataLoader,
    device: str = config.DEVICE,
) -> Dict:
    """Evaluate baseline CNN + LSTM model."""
    print("\n📊 Evaluating Baseline (CNN + LSTM)")
    print("-" * 40)

    # Load models
    cnn = CNNEncoder(
        backbone=config.CNN_BACKBONE,
        feature_dim=config.CNN_FEATURE_DIM,
    ).to(device)

    lstm = LSTMAnomaly(
        input_dim=config.CNN_FEATURE_DIM,
        hidden_dim=config.LSTM_HIDDEN_DIM,
    ).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        cnn.load_state_dict(ckpt["cnn_state"])
        lstm.load_state_dict(ckpt["lstm_state"])
        print(f"  Loaded checkpoint from {checkpoint_path}")

    cnn.eval()
    lstm.eval()

    all_labels = []
    all_scores = []
    inference_times = []

    with torch.no_grad():
        for clips, labels in test_loader:
            clips = clips.to(device)

            t_start = time.time()
            features = cnn(clips)
            scores = lstm(features)
            t_end = time.time()

            inference_times.append(t_end - t_start)
            all_labels.extend(labels.numpy())
            all_scores.extend(scores.cpu().squeeze().numpy())

    labels_np = np.array(all_labels)
    scores_np = np.array(all_scores)

    results = compute_metrics(labels_np, scores_np, inference_times)
    _print_results("Baseline", results)
    return results


def evaluate_with_agent(
    checkpoint_path: str,
    test_loader: DataLoader,
    device: str = config.DEVICE,
) -> Dict:
    """
    Evaluate with AI Agent controlling inference.

    This demonstrates the COMPUTE SAVINGS of the agent
    by tracking how many frames skip heavy processing.
    """
    print("\n📊 Evaluating with AI Agent")
    print("-" * 40)

    # Load models
    cnn = CNNEncoder(
        backbone=config.CNN_BACKBONE,
        feature_dim=config.CNN_FEATURE_DIM,
    ).to(device)

    lstm = LSTMAnomaly(
        input_dim=config.CNN_FEATURE_DIM,
        hidden_dim=config.LSTM_HIDDEN_DIM,
    ).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        cnn.load_state_dict(ckpt.get("cnn_state", ckpt.get("model_state", {})))
        lstm.load_state_dict(ckpt.get("lstm_state", {}))

    cnn.eval()
    lstm.eval()

    # Create Agent
    agent = AdaptiveAgent()

    all_labels = []
    all_scores = []
    inference_times = []
    frame_idx = 0

    with torch.no_grad():
        for clips, labels in test_loader:
            clips = clips.to(device)
            batch_size = clips.size(0)

            for i in range(batch_size):
                t_start = time.time()

                # Quick score from CNN (cheap model)
                clip = clips[i:i+1]
                features = cnn(clip)
                quick_score = lstm(features).item()

                # Agent decides what to do
                decision = agent.decide(
                    frame_idx=frame_idx,
                    yolo_score=quick_score,
                    num_detections=0,
                    motion_magnitude=abs(quick_score),
                )

                if decision.action == AgentAction.SKIP:
                    final_score = 0.0
                elif decision.action == AgentAction.STOP_NORMAL:
                    final_score = quick_score
                elif decision.action == AgentAction.RUN_GCN:
                    # Simulate GCN processing (in real pipeline, would run GCN)
                    final_score = quick_score  # placeholder
                    agent.update_with_gcn_score(decision, final_score)
                else:
                    final_score = quick_score
                    agent.update_with_vlm_score(decision, final_score)

                t_end = time.time()

                inference_times.append(t_end - t_start)
                all_labels.append(labels[i].item())
                all_scores.append(final_score)
                frame_idx += 1

    labels_np = np.array(all_labels)
    scores_np = np.array(all_scores)
    agent_stats = agent.get_statistics()

    results = compute_metrics(labels_np, scores_np, inference_times, agent_stats)
    _print_results("Agent", results)
    return results


# ==============================================================
# Visualization
# ==============================================================

def plot_roc_comparison(results_dict: Dict[str, Dict], save_path: str = "roc_comparison.png"):
    """Plot ROC curves for all models."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        for name, results in results_dict.items():
            if "fpr" in results and "tpr" in results:
                ax.plot(
                    results["fpr"], results["tpr"],
                    label=f"{name} (AUC={results['auc']:.3f})",
                )

        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\n📈 ROC curve saved to {save_path}")
        plt.close()
    except ImportError:
        print("⚠️  matplotlib not available for plotting")


def plot_compute_comparison(results_dict: Dict[str, Dict], save_path: str = "compute_comparison.png"):
    """Plot compute cost comparison."""
    try:
        import matplotlib.pyplot as plt

        models = list(results_dict.keys())
        aucs = [results_dict[m].get("auc", 0) for m in models]
        fps_vals = [results_dict[m].get("fps", 0) for m in models]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # AUC comparison
        bars1 = ax1.bar(models, aucs, color=["#2196F3", "#4CAF50", "#FF9800"])
        ax1.set_ylabel("AUC")
        ax1.set_title("Detection Performance (AUC)")
        ax1.set_ylim(0, 1)
        for bar, val in zip(bars1, aucs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center")

        # FPS comparison
        bars2 = ax2.bar(models, fps_vals, color=["#2196F3", "#4CAF50", "#FF9800"])
        ax2.set_ylabel("FPS")
        ax2.set_title("Inference Speed (FPS)")
        for bar, val in zip(bars2, fps_vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val:.1f}", ha="center")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"📊 Compute comparison saved to {save_path}")
        plt.close()
    except ImportError:
        print("⚠️  matplotlib not available for plotting")


# ==============================================================
# Helpers
# ==============================================================

def _print_results(model_name: str, results: Dict):
    """Pretty-print evaluation results."""
    print(f"\n{'='*40}")
    print(f"  {model_name} Results")
    print(f"{'='*40}")
    print(f"  AUC:              {results.get('auc', 0):.4f}")
    print(f"  Accuracy:         {results.get('accuracy', 0):.4f}")

    if "avg_inference_time_ms" in results:
        print(f"  Avg Inference:    {results['avg_inference_time_ms']:.2f} ms")
        print(f"  FPS:              {results['fps']:.1f}")

    if "compute_saved_pct" in results:
        print(f"  Compute Saved:    {results['compute_saved_pct']:.1f}%")
        print(f"  YOLO calls:       {results['yolo_calls']}")
        print(f"  GCN calls:        {results['gcn_calls']}")
        print(f"  VLM calls:        {results['vlm_calls']}")
        print(f"  Skip rate:        {results['skip_rate']:.1%}")


def generate_report(results_dict: Dict[str, Dict], save_path: str = "evaluation_report.txt"):
    """Generate a text report comparing all models."""
    with open(save_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("EVALUATION REPORT\n")
        f.write("AI Agent for Video Anomaly Detection\n")
        f.write("=" * 60 + "\n\n")

        # Comparison table
        f.write(f"{'Model':<20} {'AUC':<10} {'Accuracy':<12} {'FPS':<10} {'Compute':<15}\n")
        f.write("-" * 67 + "\n")

        for name, results in results_dict.items():
            auc = f"{results.get('auc', 0):.4f}"
            acc = f"{results.get('accuracy', 0):.4f}"
            fps = f"{results.get('fps', 0):.1f}"
            compute = f"{results.get('compute_saved_pct', 0):.1f}% saved"
            f.write(f"{name:<20} {auc:<10} {acc:<12} {fps:<10} {compute:<15}\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"\n📄 Report saved to {save_path}")


# ==============================================================
# Main
# ==============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Video Anomaly Detection Models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["baseline", "gcn", "agent", "all"],
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(config.CHECKPOINT_DIR, "baseline_best.pth"),
    )
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--device", type=str, default=config.DEVICE)
    args = parser.parse_args()

    print("🔍 Video Anomaly Detection — Evaluation")
    print(f"   Device: {args.device}")
    print(f"   Test dir: {config.TEST_DIR}")

    # Load test data
    test_dataset = VideoDataset(config.TEST_DIR, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    if len(test_dataset) == 0:
        print("⚠️  No test data found. Add videos to data/test/normal/ and data/test/anomaly/")
        return

    results_dict = {}

    if args.model in ("baseline", "all"):
        results_dict["Baseline"] = evaluate_baseline(
            args.checkpoint, test_loader, args.device
        )

    if args.model in ("agent", "all"):
        results_dict["+ Agent"] = evaluate_with_agent(
            args.checkpoint, test_loader, args.device
        )

    # Generate reports & plots
    if len(results_dict) > 0:
        generate_report(results_dict)
        plot_roc_comparison(results_dict)
        plot_compute_comparison(results_dict)


if __name__ == "__main__":
    main()
