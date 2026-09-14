"""
scripts/eval_ucf_crime.py
──────────────────────────
Evaluation trên UCF-Crime Dataset.
Metric: Frame-level AUC-ROC (chuẩn của benchmark).

Dataset structure mong đợi:
    data/UCF_Crime/
      Videos/
        Abuse/  Arrest/ ... Normal_Videos_event/
      Temporal_Anomaly_Annotation_all.txt

Chạy:
    python scripts/eval_ucf_crime.py --config configs/agent_config.yaml
    python scripts/eval_ucf_crime.py --config configs/agent_config.yaml --tier1-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# Annotation Parser
# ══════════════════════════════════════════════════════════════════════════════

def load_temporal_annotations(anno_path: Path) -> dict[str, list[tuple[int, int]]]:
    """
    Parse UCF-Crime temporal annotation file.
    Format: VideoName  StartFrame  EndFrame  [StartFrame2  EndFrame2]
    Returns: {video_name: [(start, end), ...]}
    Normal videos → empty list []
    """
    annotations = {}
    if not anno_path.exists():
        print(f"[WARN] Annotation file not found: {anno_path}")
        print("  → Dùng video-level labels thay thế (anomaly videos = all frames anomaly)")
        return {}

    with open(anno_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            name = parts[0]
            segs = []
            for i in range(1, len(parts) - 1, 2):
                try:
                    s, e = int(parts[i]), int(parts[i + 1])
                    if s > 0 and e > 0:
                        segs.append((s, e))
                except (ValueError, IndexError):
                    pass
            annotations[name] = segs

    return annotations


def get_frame_labels(
    video_name: str,
    num_frames: int,
    annotations: dict,
    category: str,
) -> list[int]:
    """
    Tạo frame-level labels (0=normal, 1=anomaly).
    - Normal video: tất cả 0
    - Anomaly video với annotation: 1 trong [start, end], 0 ngoài
    - Anomaly video không có annotation: tất cả 1
    """
    if category == "Normal":
        return [0] * num_frames

    segs = annotations.get(video_name, [])
    if not segs:
        # Video-level label: all frames anomaly
        return [1] * num_frames

    labels = [0] * num_frames
    for start, end in segs:
        for i in range(min(start - 1, num_frames), min(end, num_frames)):
            labels[i] = 1
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# Score Extractor
# ══════════════════════════════════════════════════════════════════════════════

UCF_CRIME_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
]


def score_video_tier1(video_path: Path, fast_filter, sample_every: int = 1) -> list[float]:
    """
    Chạy Tier 1 (Fast Filter) trên video, trả về list anomaly scores.
    sample_every=1: score mỗi frame; =5: score mỗi 5 frame.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    scores = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            try:
                result = fast_filter.process(frame)
                scores.append(result.anomaly_score if result else 0.0)
            except Exception:
                scores.append(0.0)
        frame_idx += 1

    cap.release()
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation Runner
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_ucf_crime(
    data_dir: Path,
    config: dict,
    tier1_only: bool = True,
    sample_every: int = 5,
    max_videos: Optional[int] = None,
) -> dict:
    """
    Chạy evaluation toàn bộ UCF-Crime test set.
    Returns: {auc_roc, ap, num_videos, ...}
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    from tqdm import tqdm

    from agents.fast_filter_agent import FastFilterAgent

    ucf_dir = data_dir / "UCF_Crime"
    anno_path = data_dir / "UCF_Crime" / "Temporal_Anomaly_Annotation_all.txt"

    annotations = load_temporal_annotations(anno_path)

    # Init Fast Filter Agent
    fast_filter = FastFilterAgent(config.get("fast_filter", {}))

    all_scores, all_labels = [], []
    video_results = []
    processed = 0

    # Process anomaly videos
    for cat in tqdm(UCF_CRIME_CATEGORIES, desc="Anomaly categories"):
        cat_dir = ucf_dir / cat
        if not cat_dir.exists():
            continue

        for video_path in sorted(cat_dir.glob("*")):
            if video_path.suffix.lower() not in (".mp4", ".avi", ".mkv"):
                continue
            if max_videos and processed >= max_videos:
                break

            scores = score_video_tier1(video_path, fast_filter, sample_every)
            if not scores:
                continue

            labels = get_frame_labels(video_path.name, len(scores), annotations, cat)
            all_scores.extend(scores)
            all_labels.extend(labels)

            video_auc = 0.0
            if len(set(labels)) > 1:
                video_auc = roc_auc_score(labels, scores)

            video_results.append({
                "video": video_path.name,
                "category": cat,
                "num_frames": len(scores),
                "auc": video_auc,
            })
            processed += 1

    # Process normal videos
    normal_dir = ucf_dir / "Normal_Videos_event"
    if normal_dir.exists():
        for video_path in tqdm(sorted(normal_dir.glob("*")), desc="Normal videos"):
            if video_path.suffix.lower() not in (".mp4", ".avi", ".mkv"):
                continue
            if max_videos and processed >= max_videos:
                break

            scores = score_video_tier1(video_path, fast_filter, sample_every)
            if not scores:
                continue

            labels = [0] * len(scores)
            all_scores.extend(scores)
            all_labels.extend(labels)
            processed += 1

    # Compute metrics
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    auc_roc = roc_auc_score(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)

    results = {
        "dataset": "UCF-Crime",
        "num_videos": processed,
        "num_frames": len(all_labels),
        "auc_roc": round(auc_roc, 4),
        "average_precision": round(ap, 4),
        "anomaly_ratio": round(all_labels.mean(), 4),
        "tier": "Tier1-only" if tier1_only else "Full (Tier1+Tier2)",
        "per_video": video_results,
    }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import yaml

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/agent_config.yaml")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output", default="outputs/eval_ucf_crime.json")
    p.add_argument("--tier1-only", action="store_true", help="Chỉ dùng Tier 1 (nhanh hơn)")
    p.add_argument("--sample-every", type=int, default=5, help="Lấy 1 frame mỗi N frame")
    p.add_argument("--max-videos", type=int, default=None, help="Giới hạn số video (debug)")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"\nEvaluating on UCF-Crime | sample_every={args.sample_every}")
    print(f"Config: {args.config}\n")

    results = evaluate_ucf_crime(
        data_dir=Path(args.data_dir),
        config=config,
        tier1_only=args.tier1_only,
        sample_every=args.sample_every,
        max_videos=args.max_videos,
    )

    # Print summary
    print("\n" + "=" * 50)
    print("UCF-CRIME EVALUATION RESULTS")
    print("=" * 50)
    print(f"  AUC-ROC:            {results['auc_roc']:.4f}")
    print(f"  Average Precision:  {results['average_precision']:.4f}")
    print(f"  Videos evaluated:   {results['num_videos']}")
    print(f"  Total frames:       {results['num_frames']}")
    print(f"  Mode:               {results['tier']}")
    print("=" * 50)

    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
