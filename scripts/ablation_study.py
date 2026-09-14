"""
scripts/ablation_study.py
──────────────────────────
Ablation study: so sánh 6 variants routing trên UCF-Crime test set.

V1: Fixed cascade baseline (motion only, hardcoded threshold)
V2: Routing motion only
V3: Routing motion + person count
V4: Routing motion + person + scene complexity
V5: Full routing (motion + person + complexity + context) — PLAN.md system
V6: Full routing + per-camera FAISS memory

Metrics: Frame-level AUC-ROC, Tier-1 recall, throughput (FPS)

Chạy:
  python scripts/ablation_study.py --data data/UCF_Crime/ --subset 200
  python scripts/ablation_study.py --data data/UCF_Crime/ --variants V1,V5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from agents.routing_agent import RoutingAgent, RoutingDecision
from features.optical_flow import OpticalFlowExtractor
from features.scene_embedding import SceneEmbedder
from features.pose_detector import PoseDetector
from utils.video_preprocessor import VideoPreprocessor


# ─── Variant configurations ──────────────────────────────────────────────────

VARIANT_CONFIGS = {
    "V1": {
        "name": "Fixed Cascade (Baseline)",
        "description": "Motion score với hardcoded threshold 0.5 — mimics Cerberus",
        "routing_config": {
            "weights": {"motion": 1.0, "person": 0.0, "complexity": 0.0, "risk": 0.0},
            "tier_thresholds": [0.5, 0.9],  # Fixed threshold
            "night_multiplier": 1.0,          # Không dùng night context
        },
        "use_memory": False,
        "use_pose": False,
        "use_scene": False,
    },
    "V2": {
        "name": "Motion Only",
        "description": "Routing chỉ dùng optical flow signal",
        "routing_config": {
            "weights": {"motion": 1.0, "person": 0.0, "complexity": 0.0, "risk": 0.0},
            "tier_thresholds": [0.35, 0.70],
            "night_multiplier": 1.3,
        },
        "use_memory": False,
        "use_pose": False,
        "use_scene": False,
    },
    "V3": {
        "name": "Motion + Person",
        "description": "Routing motion + person count",
        "routing_config": {
            # Normalize weights để sum = 1.0
            "weights": {"motion": 0.58, "person": 0.42, "complexity": 0.0, "risk": 0.0},
            "tier_thresholds": [0.35, 0.70],
            "night_multiplier": 1.3,
        },
        "use_memory": False,
        "use_pose": True,
        "use_scene": False,
    },
    "V4": {
        "name": "Motion + Person + Complexity",
        "description": "Routing motion + person + scene complexity",
        "routing_config": {
            "weights": {"motion": 0.47, "person": 0.33, "complexity": 0.20, "risk": 0.0},
            "tier_thresholds": [0.35, 0.70],
            "night_multiplier": 1.3,
        },
        "use_memory": False,
        "use_pose": True,
        "use_scene": True,
    },
    "V5": {
        "name": "Full Routing (System)",
        "description": "Tất cả 4 signals — PLAN.md baseline system",
        "routing_config": {
            "weights": {"motion": 0.35, "person": 0.25, "complexity": 0.20, "risk": 0.20},
            "tier_thresholds": [0.35, 0.70],
            "night_multiplier": 1.3,
        },
        "use_memory": False,
        "use_pose": True,
        "use_scene": True,
    },
    "V6": {
        "name": "Full Routing + Per-Camera Memory",
        "description": "V5 + per-camera FAISS memory cho Tier 3 context",
        "routing_config": {
            "weights": {"motion": 0.35, "person": 0.25, "complexity": 0.20, "risk": 0.20},
            "tier_thresholds": [0.35, 0.70],
            "night_multiplier": 1.3,
        },
        "use_memory": True,
        "use_pose": True,
        "use_scene": True,
    },
}


# ─── Ablation runner ─────────────────────────────────────────────────────────

class AblationRunner:
    def __init__(self, data_dir: str, subset: Optional[int] = None, device: str = "cpu"):
        self.data_dir = Path(data_dir)
        self.subset = subset
        self.device = device

        self.flow_extractor = OpticalFlowExtractor()
        self.scene_embedder = SceneEmbedder(device=device)
        self.pose_detector = PoseDetector()
        self.preprocessor = VideoPreprocessor({"strategy": "uniform", "sample_rate": 5})

    def extract_features(self, video_path: str, variant_cfg: dict) -> list[dict]:
        """Trích xuất features từ video cho 1 variant."""
        features_list = []
        prev_frame = None

        try:
            for frame_data in self.preprocessor.stream(video_path):
                flow = self.flow_extractor.compute(prev_frame, frame_data.frame)
                prev_frame = frame_data.frame

                features = {
                    "motion_mean": flow["mean"],
                    "person_count": 0,
                    "scene_complexity": 0.5,
                    "camera_id": "ucf_cam_default",
                    "timestamp": frame_data.timestamp,
                }

                if variant_cfg.get("use_pose"):
                    pose = self.pose_detector.detect(frame_data.frame)
                    features["person_count"] = pose["person_count"]

                if variant_cfg.get("use_scene"):
                    features["scene_complexity"] = self.scene_embedder.scene_complexity(
                        frame_data.frame
                    )

                features_list.append(features)

        except Exception as e:
            logger.warning(f"Lỗi xử lý {video_path}: {e}")

        return features_list

    def run_variant(self, variant_id: str, video_paths: list[str], gt_labels: dict) -> dict:
        """Chạy 1 variant trên toàn bộ videos, trả về metrics."""
        cfg = VARIANT_CONFIGS[variant_id]
        agent = RoutingAgent(config=cfg["routing_config"])

        all_scores = []
        all_labels = []
        tier_counts = {1: 0, 2: 0, 3: 0}
        total_time = 0.0
        total_frames = 0

        for video_path in video_paths:
            video_name = Path(video_path).stem
            labels = gt_labels.get(video_name, [])

            t0 = time.perf_counter()
            features_list = self.extract_features(video_path, cfg)
            total_time += time.perf_counter() - t0
            total_frames += len(features_list)

            for i, features in enumerate(features_list):
                decision: RoutingDecision = agent.route(features)
                tier_counts[decision.tier] += 1

                # routing_score làm proxy anomaly score cho AUC computation
                all_scores.append(decision.routing_score)

                gt = labels[i] if i < len(labels) else 0
                all_labels.append(gt)

        return self._compute_metrics(
            scores=all_scores,
            labels=all_labels,
            tier_counts=tier_counts,
            total_frames=total_frames,
            total_time=total_time,
            variant_name=cfg["name"],
        )

    def _compute_metrics(
        self,
        scores: list,
        labels: list,
        tier_counts: dict,
        total_frames: int,
        total_time: float,
        variant_name: str,
    ) -> dict:
        auc = 0.0
        if len(set(labels)) > 1:
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(labels, scores)
            except Exception:
                pass

        fps = total_frames / max(total_time, 1e-6)
        tier1_rate = tier_counts[1] / max(total_frames, 1)

        return {
            "variant": variant_name,
            "auc_roc": round(auc * 100, 2),
            "fps": round(fps, 1),
            "tier1_rate": round(tier1_rate * 100, 1),
            "tier2_rate": round(tier_counts[2] / max(total_frames, 1) * 100, 1),
            "tier3_rate": round(tier_counts[3] / max(total_frames, 1) * 100, 1),
            "total_frames": total_frames,
        }


def load_ucf_crime_videos(data_dir: Path, subset: Optional[int]) -> tuple[list, dict]:
    """Load video paths và ground truth labels từ UCF-Crime directory."""
    video_paths = []
    gt_labels: dict = {}

    for pattern in ["**/*.mp4", "**/*.avi"]:
        video_paths.extend(data_dir.glob(pattern))

    if subset:
        video_paths = video_paths[:subset]

    # Load ground truth nếu có annotation file
    anno_file = data_dir / "Temporal_Anomaly_Annotation.txt"
    if anno_file.exists():
        with open(anno_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    name = parts[0]
                    gt_labels[name] = [int(x) for x in parts[2:]]

    return [str(p) for p in video_paths], gt_labels


def print_results_table(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS — VAD Agent Routing")
    print("=" * 80)
    print(f"{'Variant':<30} {'AUC-ROC':>8} {'FPS':>7} {'T1%':>6} {'T2%':>6} {'T3%':>6}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['variant']:<30} "
            f"{r['auc_roc']:>7.2f}% "
            f"{r['fps']:>6.1f} "
            f"{r['tier1_rate']:>5.1f}% "
            f"{r['tier2_rate']:>5.1f}% "
            f"{r['tier3_rate']:>5.1f}%"
        )
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Ablation study: V1-V6 routing variants")
    parser.add_argument("--data", default="data/UCF_Crime/", help="UCF-Crime data directory")
    parser.add_argument("--subset", type=int, default=None, help="Số videos để test (None = tất cả)")
    parser.add_argument(
        "--variants",
        default="V1,V2,V3,V4,V5,V6",
        help="Comma-separated variants (e.g. V1,V5)",
    )
    parser.add_argument("--device", default="cpu", help="cuda | cpu")
    parser.add_argument("--output", default="outputs/ablation_results.json")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",")]
    data_dir = Path(args.data)

    if not data_dir.exists():
        logger.error(f"Data directory không tồn tại: {data_dir}")
        logger.info("Download UCF-Crime: https://www.crcv.ucf.edu/research/real-world-anomaly-detection/")
        return

    logger.info(f"Loading videos từ {data_dir}...")
    video_paths, gt_labels = load_ucf_crime_videos(data_dir, args.subset)
    logger.info(f"Tìm thấy {len(video_paths)} videos. GT labels: {len(gt_labels)} entries.")

    if not video_paths:
        logger.error("Không tìm thấy videos. Kiểm tra data directory.")
        return

    runner = AblationRunner(args.data, args.subset, device=args.device)
    all_results = []

    for variant_id in variants:
        if variant_id not in VARIANT_CONFIGS:
            logger.warning(f"Variant không hợp lệ: {variant_id}. Bỏ qua.")
            continue

        logger.info(f"\nChạy {variant_id}: {VARIANT_CONFIGS[variant_id]['name']}...")
        result = runner.run_variant(variant_id, video_paths, gt_labels)
        result["variant_id"] = variant_id
        all_results.append(result)
        logger.info(f"  AUC-ROC: {result['auc_roc']:.2f}% | FPS: {result['fps']:.1f}")

    print_results_table(all_results)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Kết quả lưu tại: {args.output}")


if __name__ == "__main__":
    main()
