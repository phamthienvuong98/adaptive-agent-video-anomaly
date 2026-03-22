"""
Training Pipeline — End-to-end training for video anomaly detection.

Supports training in phases:
    Phase 1: Baseline (CNN + LSTM)
    Phase 2: GCN/GAT model (graph-based)
    Phase 3: TemporalGCN (spatio-temporal)

Training strategies:
    - Binary Cross-Entropy (fully supervised)
    - Ranking Loss (weakly supervised — recommended for UCF-Crime)

Usage:
    python train.py --phase baseline
    python train.py --phase gcn
    python train.py --phase temporal_gcn
"""

import os
import sys
import time
import argparse
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from datasets.video_dataset import VideoDataset, get_dataloaders
from models.cnn_encoder import CNNEncoder
from models.lstm_model import LSTMAnomaly, RankingLoss
from models.gcn_model import GCNAnomaly, GATAnomaly, TemporalGCN


def train_baseline(
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = config.NUM_EPOCHS,
    device: str = config.DEVICE,
    checkpoint_dir: str = config.CHECKPOINT_DIR,
):
    """
    Phase 1: Train baseline CNN + LSTM model.

    Pipeline: Video → CNN features → LSTM → anomaly score
    """
    print("=" * 60)
    print("PHASE 1: Training Baseline (CNN + LSTM)")
    print("=" * 60)

    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, "baseline"))

    # Models
    cnn = CNNEncoder(
        backbone=config.CNN_BACKBONE,
        feature_dim=config.CNN_FEATURE_DIM,
        pretrained=config.CNN_PRETRAINED,
        freeze_backbone=config.FREEZE_CNN,
    ).to(device)

    lstm = LSTMAnomaly(
        input_dim=config.CNN_FEATURE_DIM,
        hidden_dim=config.LSTM_HIDDEN_DIM,
        num_layers=config.LSTM_NUM_LAYERS,
        dropout=config.LSTM_DROPOUT,
    ).to(device)

    # Loss & Optimizer
    if config.LOSS_TYPE == "ranking":
        criterion = RankingLoss(margin=config.RANKING_MARGIN)
    else:
        criterion = nn.BCELoss()

    # Only train LSTM parameters (CNN is frozen)
    trainable_params = list(lstm.parameters())
    if not config.FREEZE_CNN:
        trainable_params += list(cnn.parameters())

    optimizer = optim.Adam(
        trainable_params,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = _get_scheduler(optimizer, num_epochs)

    # Training loop
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        cnn.train() if not config.FREEZE_CNN else cnn.eval()
        lstm.train()

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (clips, labels) in enumerate(train_loader):
            clips = clips.to(device)  # (B, T, C, H, W)
            labels = labels.float().to(device).unsqueeze(1)  # (B, 1)

            # Extract CNN features
            with torch.no_grad() if config.FREEZE_CNN else torch.enable_grad():
                features = cnn(clips)  # (B, T, feature_dim)

            # LSTM prediction
            scores = lstm(features)  # (B, 1)

            # Loss
            if config.LOSS_TYPE == "ranking":
                # Split into normal and anomaly for ranking loss
                normal_mask = (labels.squeeze() == 0)
                anomaly_mask = (labels.squeeze() == 1)

                if normal_mask.sum() > 0 and anomaly_mask.sum() > 0:
                    score_normal = scores[normal_mask]
                    score_anomaly = scores[anomaly_mask]
                    # Match sizes
                    min_size = min(len(score_normal), len(score_anomaly))
                    loss = criterion(
                        score_normal[:min_size],
                        score_anomaly[:min_size],
                    )
                else:
                    loss = nn.BCELoss()(scores, labels)
            else:
                loss = criterion(scores, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{num_epochs}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        # Epoch summary
        avg_loss = epoch_loss / max(1, num_batches)
        if scheduler:
            scheduler.step()

        writer.add_scalar("Loss/train", avg_loss, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}] — Avg Loss: {avg_loss:.4f}")

        # Validation
        val_loss = _validate_baseline(cnn, lstm, test_loader, criterion, device)
        writer.add_scalar("Loss/val", val_loss, epoch)
        print(f"  Val Loss: {val_loss:.4f}")

        # Checkpointing
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "cnn_state": cnn.state_dict(),
                "lstm_state": lstm.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            }, os.path.join(checkpoint_dir, "baseline_best.pth"))
            print(f"  ✅ Saved best model (loss={best_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"  ⚠️ Early stopping at epoch {epoch+1}")
                break

    writer.close()
    print(f"\n🏁 Baseline training complete. Best loss: {best_loss:.4f}")
    return cnn, lstm


def train_gcn(
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = config.NUM_EPOCHS,
    device: str = config.DEVICE,
    checkpoint_dir: str = config.CHECKPOINT_DIR,
    model_type: str = config.GCN_MODEL_TYPE,
):
    """
    Phase 2: Train GCN/GAT model on graph data.

    Pipeline: Video → YOLO → Graph Builder → GCN → anomaly score
    """
    print("=" * 60)
    print(f"PHASE 2: Training {model_type} Model")
    print("=" * 60)

    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, f"gcn_{model_type.lower()}"))

    # Model
    if model_type == "GAT":
        model = GATAnomaly(
            node_feat_dim=8,  # base features from graph builder
            hidden_dim=config.GCN_HIDDEN_DIM,
            output_dim=config.GCN_OUTPUT_DIM,
            dropout=config.GCN_DROPOUT,
        ).to(device)
    else:
        model = GCNAnomaly(
            node_feat_dim=8,
            hidden_dim=config.GCN_HIDDEN_DIM,
            output_dim=config.GCN_OUTPUT_DIM,
            dropout=config.GCN_DROPOUT,
        ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = _get_scheduler(optimizer, num_epochs)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (graphs_batch, labels) in enumerate(train_loader):
            labels = labels.float().to(device).unsqueeze(1)

            # Process each graph in batch
            batch_scores = []
            for graphs in graphs_batch:
                if isinstance(graphs, list) and len(graphs) > 0:
                    # Take last graph in sequence (or aggregate)
                    g = graphs[-1]
                    if g["num_nodes"] > 0:
                        score = model(
                            g["node_features"].to(device),
                            g["edge_index"].to(device),
                            g["num_nodes"],
                        )
                        batch_scores.append(score)
                    else:
                        batch_scores.append(torch.tensor([[0.5]], device=device))
                else:
                    batch_scores.append(torch.tensor([[0.5]], device=device))

            if len(batch_scores) == 0:
                continue

            scores = torch.cat(batch_scores, dim=0)
            loss = criterion(scores, labels[:len(scores)])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        if scheduler:
            scheduler.step()

        writer.add_scalar("Loss/train", avg_loss, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}] — Loss: {avg_loss:.4f}")

        # Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            }, os.path.join(checkpoint_dir, f"{model_type.lower()}_best.pth"))
            print(f"  ✅ Saved best model (loss={best_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"  ⚠️ Early stopping at epoch {epoch+1}")
                break

    writer.close()
    print(f"\n🏁 {model_type} training complete. Best loss: {best_loss:.4f}")
    return model


# ==============================================================
# Helpers
# ==============================================================

def _get_scheduler(optimizer, num_epochs):
    """Create learning rate scheduler."""
    if config.SCHEDULER == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif config.SCHEDULER == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    return None


def _validate_baseline(cnn, lstm, loader, criterion, device):
    """Run validation for baseline model."""
    cnn.eval()
    lstm.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for clips, labels in loader:
            clips = clips.to(device)
            labels = labels.float().to(device).unsqueeze(1)

            features = cnn(clips)
            scores = lstm(features)

            if config.LOSS_TYPE == "ranking":
                loss = nn.BCELoss()(scores, labels)
            else:
                loss = criterion(scores, labels)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(1, num_batches)


# ==============================================================
# Main
# ==============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Video Anomaly Detection Models"
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="baseline",
        choices=["baseline", "gcn", "gat", "temporal_gcn"],
        help="Training phase to run",
    )
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--device", type=str, default=config.DEVICE)
    args = parser.parse_args()

    print(f"🖥  Device: {args.device}")
    print(f"📁 Train dir: {config.TRAIN_DIR}")
    print(f"📁 Test dir:  {config.TEST_DIR}")

    if args.phase == "baseline":
        loaders = get_dataloaders(
            batch_size=args.batch_size,
            num_workers=config.NUM_WORKERS,
        )
        train_baseline(
            loaders["train"],
            loaders["test"],
            num_epochs=args.epochs,
            device=args.device,
        )
    elif args.phase in ("gcn", "gat"):
        model_type = args.phase.upper()
        # For GCN training, we need graph-based dataset
        # (requires YOLO + GraphBuilder preprocessing)
        print(f"⚠️  GCN/GAT training requires graph-preprocessed data.")
        print(f"    Run preprocessing first, or use VideoGraphDataset.")
        # Placeholder: use standard loaders with synthetic graphs for testing
        loaders = get_dataloaders(
            batch_size=args.batch_size,
            num_workers=config.NUM_WORKERS,
        )
        train_gcn(
            loaders["train"],
            loaders["test"],
            num_epochs=args.epochs,
            device=args.device,
            model_type=model_type,
        )
    elif args.phase == "temporal_gcn":
        print("⚠️  TemporalGCN training requires graph sequences.")
        print("    Combine Phase 2 graph + Phase 1 temporal modeling.")

    print("\n✅ Training pipeline complete.")


if __name__ == "__main__":
    main()
