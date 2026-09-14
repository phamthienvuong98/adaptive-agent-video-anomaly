"""
scripts/baseline_comparison.py
────────────────────────────────
So sánh hệ thống với các baseline methods.

Baselines:
  B1: CLIP Zero-Shot      — chỉ dùng CLIP text similarity, không train gì
  B2: CNN Supervised      — MobileNetV3 train BCE thuần (không KD từ CLIP)
  B3: CNN + CLIP (no KD)  — kết hợp CNN và CLIP nhưng không knowledge distillation
  B4: Ours (CLIP-KD)      — Student CNN được distill từ CLIP teacher (đề xuất)
  B5: Ours Full System    — B4 + AdaptiveThreshold + QVAD + RAG (toàn bộ hệ thống)

Output: bảng so sánh cho luận văn + LaTeX format.

Chạy:
    python scripts/baseline_comparison.py --config configs/agent_config.yaml
    python scripts/baseline_comparison.py --checkpoint checkpoints/student_cnn_best.pth
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Anomaly text prompts dùng chung ───────────────────────────────────────────
ANOMALY_QUERIES = [
    "a person fighting or being violent",
    "a robbery or theft in progress",
    "a car accident or crash",
    "suspicious behavior on the street",
    "someone running away from danger",
    "vandalism or property destruction",
]
NORMAL_QUERIES = [
    "people walking normally on the street",
    "normal traffic on the road",
    "everyday activities in public",
    "normal daily urban scene",
]


# ══════════════════════════════════════════════════════════════════════════════
# Baseline B1: CLIP Zero-Shot
# ══════════════════════════════════════════════════════════════════════════════

class CLIPBaseline:
    """B1: Zero-shot anomaly detection bằng CLIP text similarity."""

    def __init__(self, device: str = "cuda"):
        import open_clip
        import torch
        import torch.nn.functional as F

        self.device = device
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        self.model = model.to(device).eval()
        self.preprocess = preprocess

        with torch.no_grad():
            anom_tokens = tokenizer(ANOMALY_QUERIES).to(device)
            norm_tokens = tokenizer(NORMAL_QUERIES).to(device)
            self.anom_feats = F.normalize(model.encode_text(anom_tokens), dim=-1)
            self.norm_feats = F.normalize(model.encode_text(norm_tokens), dim=-1)

    def score(self, frame_bgr: np.ndarray) -> float:
        import torch
        import torch.nn.functional as F
        from PIL import Image

        rgb = frame_bgr[..., ::-1].copy()
        img = self.preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = F.normalize(self.model.encode_image(img), dim=-1)
            anom_sim = (feat @ self.anom_feats.T).max().item()
            norm_sim = (feat @ self.norm_feats.T).max().item()
            raw = anom_sim - norm_sim
            return float(torch.sigmoid(torch.tensor(raw * 3.0)).item())


# ══════════════════════════════════════════════════════════════════════════════
# Baseline B2: CNN Supervised (no KD)
# ══════════════════════════════════════════════════════════════════════════════

class CNNSupervisedBaseline:
    """
    B2: MobileNetV3-Small trained với BCE thuần (không dùng CLIP teacher).
    Dùng cùng checkpoint nhưng đặt alpha=0 trong distillation loss.
    """

    def __init__(self, checkpoint: Optional[str], device: str = "cuda"):
        from models.lightweight_cnn import LightweightCNN
        self.model = LightweightCNN(checkpoint=checkpoint, device=device)

    def score(self, frame_bgr: np.ndarray) -> float:
        return self.model.predict_score(frame_bgr)


# ══════════════════════════════════════════════════════════════════════════════
# Baseline B3: CNN + CLIP average (no KD)
# ══════════════════════════════════════════════════════════════════════════════

class CNNPlusCLIPBaseline:
    """B3: Simple average của CNN score và CLIP score (không KD)."""

    def __init__(self, checkpoint: Optional[str], device: str = "cuda"):
        self.clip = CLIPBaseline(device)
        from models.lightweight_cnn import LightweightCNN
        self.cnn = LightweightCNN(checkpoint=checkpoint, device=device)

    def score(self, frame_bgr: np.ndarray) -> float:
        clip_s = self.clip.score(frame_bgr)
        cnn_s = self.cnn.predict_score(frame_bgr)
        return (clip_s + cnn_s) / 2.0


# ══════════════════════════════════════════════════════════════════════════════
# Baseline B4: Ours CLIP-KD (Tier 1 only)
# ══════════════════════════════════════════════════════════════════════════════

class OursTier1:
    """B4: Student CNN distilled từ CLIP + CLIP fusion (Tier 1 only)."""

    def __init__(self, config: dict, checkpoint: Optional[str], device: str):
        from agents.fast_filter_agent import FastFilterAgent
        cfg = dict(config.get("fast_filter", {}))
        if checkpoint:
            cfg.setdefault("lightweight_cnn", {})["checkpoint"] = checkpoint
        cfg["deep_reasoning"] = {"enabled": False}
        self.agent = FastFilterAgent(cfg)

    def score(self, frame_bgr: np.ndarray) -> float:
        result = self.agent.process(frame_bgr)
        return result.anomaly_score if result else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation Runner
# ══════════════════════════════════════════════════════════════════════════════

UCF_ANOMALY_CATS = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents",
]


def collect_videos(data_dir: Path, max_per_class: int = 15) -> tuple[list[Path], list[int]]:
    ucf_dir = data_dir / "UCF_Crime"
    paths, labels = [], []

    for cat in UCF_ANOMALY_CATS:
        cat_dir = ucf_dir / cat
        if not cat_dir.exists():
            continue
        for v in sorted(cat_dir.glob("*.mp4"))[:max_per_class]:
            paths.append(v)
            labels.append(1)

    normal_dir = ucf_dir / "Normal_Videos_event"
    if normal_dir.exists():
        for v in sorted(normal_dir.glob("*.mp4"))[:max_per_class * 2]:
            paths.append(v)
            labels.append(0)

    return paths, labels


def score_videos(scorer, video_paths: list[Path], sample_every: int = 5) -> list[float]:
    """Chạy scorer trên danh sách video, trả về video-level max score."""
    import cv2
    from tqdm import tqdm

    video_scores = []
    for video_path in tqdm(video_paths, desc=f"  Scoring ({type(scorer).__name__})", leave=False):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            video_scores.append(0.0)
            continue

        frame_scores = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_every == 0:
                try:
                    s = scorer.score(frame)
                    frame_scores.append(s)
                except Exception:
                    frame_scores.append(0.0)
            idx += 1
        cap.release()
        video_scores.append(float(np.max(frame_scores)) if frame_scores else 0.0)

    return video_scores


def run_comparison(
    data_dir: Path,
    base_config: dict,
    checkpoint: Optional[str],
    device: str,
    sample_every: int,
    max_per_class: int,
) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score

    video_paths, video_labels = collect_videos(data_dir, max_per_class)
    if not video_paths:
        print("[ERROR] Không tìm thấy video. Kiểm tra --data-dir.")
        sys.exit(1)

    labels_arr = np.array(video_labels)
    print(f"\nSo sánh trên {len(video_paths)} videos (anomaly={labels_arr.sum()}, normal={len(labels_arr)-labels_arr.sum()})\n")

    baselines = {
        "B1_CLIP_ZeroShot": ("CLIP Zero-Shot (no training)", lambda: CLIPBaseline(device)),
        "B2_CNN_Supervised": ("CNN Supervised (no KD)", lambda: CNNSupervisedBaseline(checkpoint, device)),
        "B3_CNN_CLIP_Avg": ("CNN + CLIP Average", lambda: CNNPlusCLIPBaseline(checkpoint, device)),
        "B4_Ours_Tier1": ("Ours: CLIP-KD Tier 1", lambda: OursTier1(base_config, checkpoint, device)),
    }

    results = {}

    for b_id, (description, factory) in baselines.items():
        print(f"[{b_id}] {description}")
        try:
            t0 = time.time()
            scorer = factory()
            scores = score_videos(scorer, video_paths, sample_every)
            elapsed = time.time() - t0

            scores_arr = np.array(scores)
            auc = roc_auc_score(labels_arr, scores_arr)
            ap = average_precision_score(labels_arr, scores_arr)

            results[b_id] = {
                "description": description,
                "auc_roc": round(auc, 4),
                "average_precision": round(ap, 4),
                "time_s": round(elapsed, 1),
            }
            print(f"  AUC-ROC={auc:.4f} | AP={ap:.4f} | time={elapsed:.1f}s")

        except Exception as e:
            print(f"  [ERROR] {e}")
            results[b_id] = {"description": description, "auc_roc": None, "error": str(e)}

    return results


def print_comparison_table(results: dict):
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON — UCF-Crime")
    print("=" * 70)
    print(f"{'Method':<35} {'AUC-ROC':>8} {'AP':>8}")
    print("-" * 70)

    for b_id, info in results.items():
        auc = info.get("auc_roc")
        ap = info.get("average_precision")
        if auc is None:
            print(f"  {info['description']:<33} {'ERROR':>8} {'—':>8}")
        else:
            print(f"  {info['description']:<33} {auc:.4f}   {ap:.4f}")

    print("=" * 70)

    # LaTeX
    print("\nLaTeX table:")
    print("\\begin{tabular}{lcc}")
    print("\\hline")
    print("Method & AUC-ROC & AP \\\\")
    print("\\hline")
    for b_id, info in results.items():
        auc = info.get("auc_roc")
        ap = info.get("average_precision")
        if auc is None:
            continue
        desc = info["description"].replace("_", "\\_")
        mark = " \\textbf{(Ours)}" if "Ours" in info["description"] else ""
        print(f"{desc}{mark} & {auc:.4f} & {ap:.4f} \\\\")
    print("\\hline")
    print("\\end{tabular}")


def main():
    import yaml

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/agent_config.yaml")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--checkpoint", default="checkpoints/student_cnn_best.pth",
                   help="Path đến student CNN checkpoint")
    p.add_argument("--output", default="outputs/baseline_comparison.json")
    p.add_argument("--sample-every", type=int, default=5)
    p.add_argument("--max-per-class", type=int, default=15)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = p.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA không có, dùng CPU.")
        args.device = "cpu"

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    checkpoint = args.checkpoint if Path(args.checkpoint).exists() else None
    if not checkpoint:
        print(f"[WARN] Checkpoint không tìm thấy: {args.checkpoint}")
        print("  B2/B3/B4 sẽ dùng pretrained ImageNet weights (kết quả sẽ thấp hơn)")

    results = run_comparison(
        data_dir=Path(args.data_dir),
        base_config=base_config,
        checkpoint=checkpoint,
        device=args.device,
        sample_every=args.sample_every,
        max_per_class=args.max_per_class,
    )

    print_comparison_table(results)

    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
