"""
scripts/colab_train.py
──────────────────────
Training MobileNetV3-Small Student CNN trên UCF-Crime.

Chiến lược: CLIP-guided Knowledge Distillation
  Teacher: CLIP ViT-B/16 (frozen, pre-trained) → soft anomaly scores
  Student: MobileNetV3-Small → học approximate CLIP nhưng nhanh hơn ~10x

Loss = α * KL(student ‖ clip_soft) + (1-α) * BCE(student, hard_label)

Chạy trên Google Colab (GPU T4 / A100):
  !git clone <your_repo>
  %cd final_project
  !pip install -r requirements.txt
  !python scripts/colab_train.py --drive-dir /content/drive/MyDrive/VAD_Checkpoint

Thời gian ước tính:
  - Download + extract frames: ~30-60 phút
  - CLIP labeling: ~20 phút (T4 GPU)
  - Training 30 epochs: ~45-90 phút (T4 GPU)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ─── Đảm bảo project root trong sys.path ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Setup & Imports
# ══════════════════════════════════════════════════════════════════════════════

def setup_colab():
    """Cài packages và mount Google Drive nếu đang trong Colab."""
    try:
        import google.colab  # noqa: F401
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB:
        print("=== Đang chạy trong Google Colab ===")
        print("Mounting Google Drive...")
        from google.colab import drive
        drive.mount("/content/drive")

        print("Installing packages...")
        os.system("pip install -q open-clip-torch torchvision loguru scikit-learn tqdm gdown")
    else:
        print("=== Chạy local ===")

    return IN_COLAB


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Download UCF-Crime Dataset
# ══════════════════════════════════════════════════════════════════════════════

UCF_CRIME_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
]

NORMAL_CATEGORY = "Normal_Videos_event"


def download_ucf_crime(data_dir: Path) -> Path:
    """
    Download UCF-Crime dataset.

    UCF-Crime yêu cầu đăng ký tại:
    https://www.crcv.ucf.edu/projects/real-world/

    Nếu đã có dataset, đặt vào: data_dir/UCF_Crime/
    Structure mong đợi:
        UCF_Crime/
          Abuse/
            video1.mp4
            video2.mp4
          ...
          Normal_Videos_event/
            video1.mp4
    """
    ucf_dir = data_dir / "UCF_Crime"

    if ucf_dir.exists() and any(ucf_dir.iterdir()):
        print(f"UCF-Crime đã có tại: {ucf_dir}")
        return ucf_dir

    print("""
╔══════════════════════════════════════════════════════════════╗
║  UCF-Crime Dataset Setup                                       ║
╠══════════════════════════════════════════════════════════════╣
║  Option 1 (Kaggle - dễ nhất):                                  ║
║    kaggle datasets download -d odins0n/ucf-crime-dataset       ║
║                                                                ║
║  Option 2 (Official - cần đăng ký):                            ║
║    https://www.crcv.ucf.edu/projects/real-world/               ║
║                                                                ║
║  Option 3 (Subset nhanh - 5 categories, ~2GB):                 ║
║    python scripts/colab_train.py --subset --drive-dir ...      ║
╚══════════════════════════════════════════════════════════════╝
""")

    use_kaggle = input("Dùng Kaggle API? (y/n): ").strip().lower() == "y"
    if use_kaggle:
        _download_via_kaggle(ucf_dir)
    else:
        print("Hãy upload dataset thủ công vào Google Drive rồi chạy lại.")
        sys.exit(0)

    return ucf_dir


def _download_via_kaggle(ucf_dir: Path):
    """Download UCF-Crime qua Kaggle API."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        os.system("pip install -q kaggle")

    ucf_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading UCF-Crime từ Kaggle...")
    os.system(
        f"kaggle datasets download -d odins0n/ucf-crime-dataset "
        f"--path {ucf_dir} --unzip -q"
    )
    print(f"Download xong: {ucf_dir}")


def create_mini_dataset(data_dir: Path, num_anomaly_videos: int = 50, num_normal_videos: int = 50) -> Path:
    """
    Tạo mini dataset từ subset UCF-Crime để test nhanh.
    Chỉ dùng khi không có full dataset.
    """
    import random
    import shutil

    full_dir = data_dir / "UCF_Crime"
    mini_dir = data_dir / "UCF_Crime_mini"

    if mini_dir.exists():
        return mini_dir

    mini_dir.mkdir(parents=True)
    random.seed(42)

    # Copy anomaly videos
    for cat in UCF_CRIME_CATEGORIES:
        src = full_dir / cat
        if not src.exists():
            continue
        videos = list(src.glob("*.mp4")) + list(src.glob("*.avi"))
        selected = random.sample(videos, min(num_anomaly_videos // len(UCF_CRIME_CATEGORIES), len(videos)))
        dst = mini_dir / cat
        dst.mkdir(exist_ok=True)
        for v in selected:
            shutil.copy(v, dst / v.name)

    # Copy normal videos
    src_normal = full_dir / NORMAL_CATEGORY
    if src_normal.exists():
        videos = list(src_normal.glob("*.mp4")) + list(src_normal.glob("*.avi"))
        selected = random.sample(videos, min(num_normal_videos, len(videos)))
        dst_normal = mini_dir / NORMAL_CATEGORY
        dst_normal.mkdir(exist_ok=True)
        for v in selected:
            shutil.copy(v, dst_normal / v.name)

    return mini_dir


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Frame Extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_frames(
    ucf_dir: Path,
    frames_dir: Path,
    sample_fps: float = 1.0,
    max_frames_per_video: int = 150,
    img_size: int = 224,
) -> dict:
    """
    Trích xuất frames từ videos, lưu theo structure:
        frames_dir/
          anomaly/
            video_name_0000.jpg
          normal/
            video_name_0000.jpg

    Returns: metadata dict {split: {frame_path: label}}
    """
    import cv2
    from tqdm import tqdm

    (frames_dir / "anomaly").mkdir(parents=True, exist_ok=True)
    (frames_dir / "normal").mkdir(parents=True, exist_ok=True)

    metadata = {"anomaly": [], "normal": []}

    all_videos = []
    for cat in UCF_CRIME_CATEGORIES:
        cat_dir = ucf_dir / cat
        if cat_dir.exists():
            for v in cat_dir.glob("*"):
                if v.suffix.lower() in (".mp4", ".avi", ".mkv"):
                    all_videos.append((v, "anomaly"))

    normal_dir = ucf_dir / NORMAL_CATEGORY
    if normal_dir.exists():
        for v in normal_dir.glob("*"):
            if v.suffix.lower() in (".mp4", ".avi", ".mkv"):
                all_videos.append((v, "normal"))

    print(f"\nExtract frames từ {len(all_videos)} videos (sample_fps={sample_fps})...")

    for video_path, label in tqdm(all_videos, desc="Extracting"):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = max(1, int(orig_fps / sample_fps))
        stem = video_path.stem

        frame_idx = 0
        saved = 0

        while saved < max_frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                frame_resized = cv2.resize(frame, (img_size, img_size))
                out_name = f"{stem}_{saved:04d}.jpg"
                out_path = frames_dir / label / out_name
                cv2.imwrite(str(out_path), frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
                metadata[label].append(str(out_path))
                saved += 1

            frame_idx += 1

        cap.release()

    # Save metadata
    meta_path = frames_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    print(f"  Anomaly frames: {len(metadata['anomaly'])}")
    print(f"  Normal frames:  {len(metadata['normal'])}")
    print(f"  Saved metadata: {meta_path}")

    return metadata


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CLIP Teacher: Tạo Soft Labels
# ══════════════════════════════════════════════════════════════════════════════

ANOMALY_TEXT_QUERIES = [
    "a person fighting or being violent",
    "a robbery or theft in progress",
    "a car accident or crash",
    "suspicious behavior on the street",
    "someone running away from danger",
    "vandalism or property destruction",
    "an explosion or fire emergency",
    "abnormal crowd behavior",
]

NORMAL_TEXT_QUERIES = [
    "people walking normally on the street",
    "normal traffic on the road",
    "everyday activities in public",
    "pedestrians at a crosswalk",
    "people standing and talking",
    "normal daily urban scene",
]


def compute_clip_soft_labels(
    frame_paths: list[str],
    batch_size: int = 64,
    device: str = "cuda",
) -> dict[str, float]:
    """
    Dùng CLIP ViT-B/16 để tính anomaly score cho từng frame.
    Score = max(anomaly_similarities) - max(normal_similarities)
    → Normalized về [0, 1] bằng sigmoid.

    Returns: {frame_path: soft_score}
    """
    import open_clip
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from tqdm import tqdm

    print("\nLoading CLIP ViT-B/16...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    model = model.to(device).eval()

    # Encode text queries
    with torch.no_grad():
        anomaly_tokens = tokenizer(ANOMALY_TEXT_QUERIES).to(device)
        normal_tokens = tokenizer(NORMAL_TEXT_QUERIES).to(device)
        anomaly_feats = F.normalize(model.encode_text(anomaly_tokens), dim=-1)
        normal_feats = F.normalize(model.encode_text(normal_tokens), dim=-1)

    soft_labels = {}

    for i in tqdm(range(0, len(frame_paths), batch_size), desc="CLIP labeling"):
        batch_paths = frame_paths[i : i + batch_size]
        images = []
        valid_paths = []

        for p in batch_paths:
            try:
                img = preprocess(Image.open(p).convert("RGB"))
                images.append(img)
                valid_paths.append(p)
            except Exception:
                continue

        if not images:
            continue

        batch = torch.stack(images).to(device)
        with torch.no_grad():
            img_feats = F.normalize(model.encode_image(batch), dim=-1)
            # Similarity với mỗi text group
            anom_sims = (img_feats @ anomaly_feats.T)  # (B, num_anomaly_queries)
            norm_sims = (img_feats @ normal_feats.T)   # (B, num_normal_queries)
            # Score: max anomaly sim - max normal sim
            raw_scores = anom_sims.max(dim=1).values - norm_sims.max(dim=1).values
            # Sigmoid → [0, 1]
            scores = torch.sigmoid(raw_scores * 3.0).cpu().tolist()

        for path, score in zip(valid_paths, scores):
            soft_labels[path] = score

    return soft_labels


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PyTorch Dataset
# ══════════════════════════════════════════════════════════════════════════════

def build_datasets(
    metadata: dict,
    soft_labels: dict[str, float],
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
):
    """Tạo train/val/test splits từ metadata và soft labels."""
    import random
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
    from PIL import Image

    random.seed(42)

    all_samples = []
    for path in metadata.get("anomaly", []):
        hard_label = 1.0
        soft = soft_labels.get(path, 0.7)
        all_samples.append((path, hard_label, soft))
    for path in metadata.get("normal", []):
        hard_label = 0.0
        soft = soft_labels.get(path, 0.1)
        all_samples.append((path, hard_label, soft))

    random.shuffle(all_samples)
    n = len(all_samples)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_set = all_samples[:n_test]
    val_set = all_samples[n_test : n_test + n_val]
    train_set = all_samples[n_test + n_val :]

    print(f"\nDataset splits: train={len(train_set)} | val={len(val_set)} | test={len(test_set)}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class FrameDataset(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, hard_label, soft_label = self.samples[idx]
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            return img, torch.tensor(hard_label, dtype=torch.float32), torch.tensor(soft_label, dtype=torch.float32)

    return (
        FrameDataset(train_set, train_transform),
        FrameDataset(val_set, eval_transform),
        FrameDataset(test_set, eval_transform),
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def train(
    train_ds,
    val_ds,
    checkpoint_dir: Path,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 3e-4,
    kd_alpha: float = 0.7,
    kd_temperature: float = 4.0,
    device: str = "cuda",
    backbone: str = "mobilenet_v3_small",
):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score
    from tqdm import tqdm

    from models.lightweight_cnn import LightweightCNN

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Model — dùng pretrained ImageNet weights, không dùng checkpoint cũ
    model = LightweightCNN(backbone=backbone, checkpoint=None, device=device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    best_auc = 0.0
    history = []

    print(f"\n{'='*60}")
    print(f"Training {backbone} | {epochs} epochs | lr={lr} | α={kd_alpha}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for imgs, hard_labels, soft_labels in tqdm(train_loader, desc=f"Epoch {epoch:02d} [train]", leave=False):
            imgs = imgs.to(device)
            hard_labels = hard_labels.to(device)
            soft_labels = soft_labels.to(device)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    student_out = model(imgs).squeeze(1)
                    loss = _distillation_loss(student_out, soft_labels, hard_labels, kd_alpha, kd_temperature)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                student_out = model(imgs).squeeze(1)
                loss = _distillation_loss(student_out, soft_labels, hard_labels, kd_alpha, kd_temperature)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # ── Validation ─────────────────────────────────────────────────
        val_auc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss={train_loss:.4f} | "
            f"val_auc={val_auc:.4f} | "
            f"lr={current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc, "lr": current_lr})

        # Save best checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            ckpt_path = checkpoint_dir / "student_cnn_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "config": {
                    "backbone": backbone,
                    "kd_alpha": kd_alpha,
                    "kd_temperature": kd_temperature,
                },
            }, ckpt_path)
            print(f"  ★ Best checkpoint saved: {ckpt_path} (AUC={best_auc:.4f})")

        # Save latest checkpoint (để resume nếu Colab timeout)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_auc": val_auc,
            "history": history,
        }, checkpoint_dir / "student_cnn_latest.pth")

    # Save training history
    with open(checkpoint_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n Training hoàn thành! Best Val AUC: {best_auc:.4f}")
    return best_auc, history


def _distillation_loss(
    student_out: "torch.Tensor",
    soft_labels: "torch.Tensor",
    hard_labels: "torch.Tensor",
    alpha: float,
    temperature: float,
) -> "torch.Tensor":
    """
    Combined KD loss:
    - Hard: BCE(student, ground_truth_label)
    - Soft: MSE(student_scaled, clip_soft_label)
    """
    import torch.nn.functional as F
    import torch

    # Hard label loss
    ce = F.binary_cross_entropy(student_out, hard_labels)

    # Soft label loss (temperature scaling)
    student_soft = torch.sigmoid(torch.logit(student_out.clamp(1e-6, 1 - 1e-6)) / temperature)
    teacher_soft = soft_labels
    kd = F.mse_loss(student_soft, teacher_soft) * (temperature ** 2)

    return alpha * kd + (1 - alpha) * ce


def evaluate(model, loader, device: str) -> float:
    """Tính AUC-ROC trên một dataloader."""
    import torch
    from sklearn.metrics import roc_auc_score

    model.eval()
    all_scores, all_labels = [], []

    with torch.no_grad():
        for imgs, hard_labels, _ in loader:
            imgs = imgs.to(device)
            scores = model(imgs).squeeze(1).cpu().tolist()
            all_scores.extend(scores)
            all_labels.extend(hard_labels.tolist())

    try:
        return roc_auc_score(all_labels, all_scores)
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Final Test Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def final_evaluation(checkpoint_dir: Path, test_ds, device: str, backbone: str):
    """Load best checkpoint và evaluate trên test set."""
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score, classification_report
    import numpy as np

    from models.lightweight_cnn import LightweightCNN

    ckpt_path = checkpoint_dir / "student_cnn_best.pth"
    if not ckpt_path.exists():
        print("Không tìm thấy checkpoint để evaluate.")
        return

    model = LightweightCNN(backbone=backbone, checkpoint=None, device=device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)
    all_scores, all_labels = [], []

    with torch.no_grad():
        for imgs, hard_labels, _ in test_loader:
            imgs = imgs.to(device)
            scores = model(imgs).squeeze(1).cpu().tolist()
            all_scores.extend(scores)
            all_labels.extend(hard_labels.tolist())

    auc = roc_auc_score(all_labels, all_scores)
    preds = [1 if s >= 0.5 else 0 for s in all_scores]

    print("\n" + "="*50)
    print(f"TEST SET RESULTS")
    print("="*50)
    print(f"AUC-ROC: {auc:.4f}")
    print(classification_report(all_labels, preds, target_names=["Normal", "Anomaly"]))

    results = {"test_auc": auc, "checkpoint": str(ckpt_path)}
    with open(checkpoint_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return auc


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Copy checkpoint to Google Drive
# ══════════════════════════════════════════════════════════════════════════════

def save_to_drive(checkpoint_dir: Path, drive_dir: str):
    """Copy checkpoints lên Google Drive để không mất khi Colab reset."""
    import shutil

    drive_path = Path(drive_dir)
    drive_path.mkdir(parents=True, exist_ok=True)

    for f in checkpoint_dir.glob("*.pth"):
        dst = drive_path / f.name
        shutil.copy(f, dst)
        print(f"  Copied {f.name} → {dst}")

    for f in checkpoint_dir.glob("*.json"):
        dst = drive_path / f.name
        shutil.copy(f, dst)

    print(f"\nCheckpoints đã lưu vào Drive: {drive_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train Student CNN trên UCF-Crime")
    p.add_argument("--data-dir", default="data", help="Thư mục chứa dataset")
    p.add_argument("--drive-dir", default="/content/drive/MyDrive/VAD_Checkpoint",
                   help="Thư mục Google Drive để lưu checkpoint")
    p.add_argument("--checkpoint-dir", default="checkpoints", help="Thư mục lưu checkpoint local")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--kd-alpha", type=float, default=0.7, help="Trọng số soft label (CLIP teacher)")
    p.add_argument("--sample-fps", type=float, default=1.0, help="FPS để sample frames")
    p.add_argument("--max-frames", type=int, default=150, help="Max frames mỗi video")
    p.add_argument("--backbone", default="mobilenet_v3_small",
                   choices=["mobilenet_v3_small", "mobilenet_v3_large"])
    p.add_argument("--subset", action="store_true", help="Dùng mini dataset (test nhanh)")
    p.add_argument("--skip-extract", action="store_true", help="Bỏ qua extract nếu đã có frames")
    p.add_argument("--skip-clip", action="store_true", help="Bỏ qua CLIP labeling nếu đã có")
    p.add_argument("--resume", action="store_true", help="Resume từ latest checkpoint")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA không có, dùng CPU (sẽ chậm hơn nhiều).")
        device = "cpu"

    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Setup ──────────────────────────────────────────────────────────────
    in_colab = setup_colab()
    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    frames_dir = data_dir / "frames"
    soft_labels_path = data_dir / "clip_soft_labels.json"
    meta_path = frames_dir / "metadata.json"

    # ── Download dataset ───────────────────────────────────────────────────
    ucf_dir = download_ucf_crime(data_dir)
    if args.subset:
        ucf_dir = create_mini_dataset(data_dir)

    # ── Extract frames ─────────────────────────────────────────────────────
    if args.skip_extract and meta_path.exists():
        print(f"Dùng frames đã có: {frames_dir}")
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        metadata = extract_frames(
            ucf_dir, frames_dir,
            sample_fps=args.sample_fps,
            max_frames_per_video=args.max_frames,
        )

    all_paths = metadata.get("anomaly", []) + metadata.get("normal", [])

    # ── CLIP Soft Labels ───────────────────────────────────────────────────
    if args.skip_clip and soft_labels_path.exists():
        print(f"Dùng CLIP labels đã có: {soft_labels_path}")
        with open(soft_labels_path) as f:
            soft_labels = json.load(f)
    else:
        soft_labels = compute_clip_soft_labels(all_paths, device=device)
        with open(soft_labels_path, "w") as f:
            json.dump(soft_labels, f)
        print(f"Saved CLIP soft labels: {soft_labels_path}")

    # ── Build Datasets ─────────────────────────────────────────────────────
    train_ds, val_ds, test_ds = build_datasets(metadata, soft_labels)

    # ── Train ──────────────────────────────────────────────────────────────
    best_auc, history = train(
        train_ds, val_ds,
        checkpoint_dir=checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        kd_alpha=args.kd_alpha,
        device=device,
        backbone=args.backbone,
    )

    # ── Final Evaluation ───────────────────────────────────────────────────
    test_auc = final_evaluation(checkpoint_dir, test_ds, device, args.backbone)

    # ── Save to Drive ──────────────────────────────────────────────────────
    if in_colab or Path(args.drive_dir).exists():
        save_to_drive(checkpoint_dir, args.drive_dir)

    print(f"\n{'='*50}")
    print(f"DONE | Best Val AUC: {best_auc:.4f} | Test AUC: {test_auc:.4f}")
    print(f"Checkpoint: {checkpoint_dir}/student_cnn_best.pth")
    print(f"Dùng trong pipeline:")
    print(f"  lightweight_cnn:")
    print(f"    checkpoint: {checkpoint_dir}/student_cnn_best.pth")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
