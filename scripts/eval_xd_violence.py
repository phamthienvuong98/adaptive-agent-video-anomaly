"""
scripts/eval_xd_violence.py
────────────────────────────
Evaluation trên XD-Violence Dataset.
Metric: AP (Average Precision) — chuẩn của benchmark XD-Violence.

Dataset structure mong đợi:
    data/XD_Violence/
      videos/
        A.Beautiful.Mind.2001__#1-00-01-00_00-02-01_label_A.mp4
        ...
      labels.csv   (hoặc annotations.txt)

Format labels.csv:
    id,label
    A.Beautiful.Mind...,A     ← A=anomaly, N=normal
    ...

Chạy:
    python scripts/eval_xd_violence.py --config configs/agent_config.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_xd_labels(data_dir: Path) -> dict[str, int]:
    """
    Load XD-Violence labels.
    Returns: {video_name: label}  (label: 0=normal, 1=anomaly)

    Hỗ trợ 2 format:
    1. labels.csv: id,label (A=anomaly, N=normal)
    2. Tự suy ra từ tên file: '_label_A' = anomaly, '_label_N' = normal
    """
    labels = {}

    # Format 1: CSV
    csv_path = data_dir / "XD_Violence" / "labels.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("id", "").strip()
                label_str = row.get("label", "N").strip()
                labels[name] = 1 if label_str == "A" else 0
        return labels

    # Format 2: Tên file chứa '_label_A' hoặc '_label_N'
    video_dir = data_dir / "XD_Violence" / "videos"
    if video_dir.exists():
        for f in video_dir.glob("*"):
            if f.suffix.lower() in (".mp4", ".avi", ".mkv"):
                label = 1 if "_label_A" in f.name else 0
                labels[f.stem] = label

    return labels


def score_video(video_path: Path, fast_filter, sample_every: int = 5) -> list[float]:
    """Chạy Fast Filter Agent lấy anomaly score list."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    scores = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every == 0:
            try:
                result = fast_filter.process(frame)
                scores.append(result.anomaly_score if result else 0.0)
            except Exception:
                scores.append(0.0)
        idx += 1
    cap.release()
    return scores


def evaluate_xd_violence(
    data_dir: Path,
    config: dict,
    sample_every: int = 5,
    max_videos: Optional[int] = None,
) -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    from tqdm import tqdm
    from agents.fast_filter_agent import FastFilterAgent

    video_dir = data_dir / "XD_Violence" / "videos"
    if not video_dir.exists():
        print(f"[ERROR] Không tìm thấy: {video_dir}")
        sys.exit(1)

    labels_map = load_xd_labels(data_dir)
    fast_filter = FastFilterAgent(config.get("fast_filter", {}))

    all_scores, all_labels = [], []
    processed = 0

    videos = sorted(video_dir.glob("*"))
    for video_path in tqdm(videos, desc="XD-Violence"):
        if video_path.suffix.lower() not in (".mp4", ".avi", ".mkv"):
            continue
        if max_videos and processed >= max_videos:
            break

        # Lấy label
        label = labels_map.get(video_path.stem, labels_map.get(video_path.name, -1))
        if label == -1:
            # Fallback: từ tên file
            label = 1 if "_label_A" in video_path.name else 0

        scores = score_video(video_path, fast_filter, sample_every)
        if not scores:
            continue

        # Video-level score = max frame score (convention cho XD-Violence)
        video_score = float(np.max(scores))
        all_scores.append(video_score)
        all_labels.append(label)
        processed += 1

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    ap = average_precision_score(all_labels, all_scores)
    auc_roc = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.0

    return {
        "dataset": "XD-Violence",
        "num_videos": processed,
        "auc_roc": round(auc_roc, 4),
        "average_precision": round(ap, 4),
        "anomaly_ratio": round(all_labels.mean(), 4),
    }


def main():
    import yaml

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/agent_config.yaml")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output", default="outputs/eval_xd_violence.json")
    p.add_argument("--sample-every", type=int, default=5)
    p.add_argument("--max-videos", type=int, default=None)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results = evaluate_xd_violence(
        data_dir=Path(args.data_dir),
        config=config,
        sample_every=args.sample_every,
        max_videos=args.max_videos,
    )

    print("\n" + "=" * 50)
    print("XD-VIOLENCE EVALUATION RESULTS")
    print("=" * 50)
    print(f"  AUC-ROC:            {results['auc_roc']:.4f}")
    print(f"  Average Precision:  {results['average_precision']:.4f}")
    print(f"  Videos evaluated:   {results['num_videos']}")
    print("=" * 50)

    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
