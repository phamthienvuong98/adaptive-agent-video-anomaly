"""
scripts/efficiency_analysis.py
────────────────────────────────
Đo efficiency của cascade architecture vs các baseline:
- FLOPs per frame
- Latency (ms/frame)
- Throughput (FPS)
- Compute savings ratio

Chạy trên CPU để dễ reproduce, không cần GPU.
Output: bảng so sánh cho Chương 4 luận văn.

Usage:
    python scripts/efficiency_analysis.py
    python scripts/efficiency_analysis.py --trigger-rate 0.05  # 5% frames pass Tier 1
"""

from __future__ import annotations

import argparse
import time

import numpy as np


# ─── FLOPs ước tính ─────────────────────────────────────────────────────────

# Nguồn: torchprofile / paper benchmarks
FLOPS = {
    "mobilenet_v3_small": 56e6,         # 56M FLOPs — student CNN (Tier 1)
    "clip_vit_b16_encode": 35e9,        # 35G FLOPs — CLIP image encoder
    "clip_vit_b16_text_sim": 0.5e6,     # 0.5M FLOPs — text similarity (nhỏ)
    "qwen_vl_7b_per_frame": 14e9,       # ~14G FLOPs — VLM (Tier 2, 1 frame)
    "qwen3_4b_llm": 8e9,                # ~8G FLOPs — LLM (1 inference)
}

# Latency trung bình (ms) đo trên GPU T4 (theo paper benchmarks)
LATENCY_MS = {
    "tier1_cnn_only": 1.5,
    "tier1_clip_encode": 3.0,
    "tier1_full": 5.0,                  # CNN + CLIP + threshold
    "tier2_vlm_1round": 450.0,          # VLM 1 round trên T4
    "tier2_vlm_3rounds": 1200.0,        # VLM 3 rounds (max)
    "tier2_llm": 120.0,                 # LLM per call
}


def compute_cascade_efficiency(trigger_rate: float = 0.05) -> dict:
    """
    Tính efficiency của cascade system với trigger_rate = % frames qua Tier 1.

    Returns dict với FLOPs, latency, throughput cho từng method.
    """
    results = {}

    # ── Baseline 1: Naive Full-VLM (mọi frame đều qua VLM) ─────────────────
    # Đây là cách tiếp cận naïve, không dùng cascade
    flops_naive = FLOPS["clip_vit_b16_encode"] + FLOPS["qwen_vl_7b_per_frame"] * 3 + FLOPS["qwen3_4b_llm"] * 3
    latency_naive = LATENCY_MS["tier1_clip_encode"] + LATENCY_MS["tier2_vlm_3rounds"] + LATENCY_MS["tier2_llm"] * 3
    results["Naive Full-VLM"] = {
        "flops_per_frame": flops_naive,
        "latency_ms": latency_naive,
        "fps": 1000.0 / latency_naive,
        "flops_ratio": 1.0,
    }

    # ── Baseline 2: CLIP Zero-Shot only ─────────────────────────────────────
    flops_clip_zs = FLOPS["clip_vit_b16_encode"] + FLOPS["clip_vit_b16_text_sim"]
    latency_clip_zs = LATENCY_MS["tier1_clip_encode"]
    results["CLIP Zero-Shot only"] = {
        "flops_per_frame": flops_clip_zs,
        "latency_ms": latency_clip_zs,
        "fps": 1000.0 / latency_clip_zs,
        "flops_ratio": flops_clip_zs / flops_naive,
    }

    # ── Baseline 3: CNN Supervised (no cascade, no VLM) ─────────────────────
    flops_cnn = FLOPS["mobilenet_v3_small"]
    latency_cnn = LATENCY_MS["tier1_cnn_only"]
    results["CNN Supervised only"] = {
        "flops_per_frame": flops_cnn,
        "latency_ms": latency_cnn,
        "fps": 1000.0 / latency_cnn,
        "flops_ratio": flops_cnn / flops_naive,
    }

    # ── Ours: Cascade (Tier 1 + Tier 2 only trigger_rate of frames) ─────────
    # Average FLOPs = Tier1_flops + trigger_rate * Tier2_flops
    flops_tier1 = FLOPS["mobilenet_v3_small"] + FLOPS["clip_vit_b16_encode"] + FLOPS["clip_vit_b16_text_sim"]
    flops_tier2_avg = LATENCY_MS["tier2_vlm_3rounds"] / LATENCY_MS["tier2_vlm_1round"] * FLOPS["qwen_vl_7b_per_frame"] + FLOPS["qwen3_4b_llm"] * 3
    flops_cascade = flops_tier1 + trigger_rate * flops_tier2_avg

    # Average latency = Tier1 + trigger_rate * Tier2 (parallel processing assumed)
    latency_cascade = LATENCY_MS["tier1_full"] + trigger_rate * (LATENCY_MS["tier2_vlm_3rounds"] + LATENCY_MS["tier2_llm"] * 3)

    results[f"Ours: Cascade ({trigger_rate*100:.0f}% trigger)"] = {
        "flops_per_frame": flops_cascade,
        "latency_ms": latency_cascade,
        "fps": 1000.0 / latency_cascade,
        "flops_ratio": flops_cascade / flops_naive,
    }

    return results


def run_latency_microbenchmark(n_runs: int = 50) -> dict:
    """
    Đo latency thực của các CPU-only operations.
    Dùng numpy mock để không cần GPU/model thật.
    """
    timings = {}

    # Mock CNN forward (simulate MobileNetV3-small on CPU)
    dummy_frame = np.random.rand(224, 224, 3).astype(np.float32)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        # Simulate: normalize + reshape + matmul (cheap version of CNN)
        x = (dummy_frame - 0.5) / 0.5
        x = x.reshape(1, -1)
        _ = np.dot(x, np.random.rand(x.shape[1], 1))
    timings["cnn_mock_ms"] = (time.perf_counter() - t0) * 1000 / n_runs

    # Mock threshold check
    t0 = time.perf_counter()
    scores = np.random.rand(n_runs)
    window = []
    for s in scores:
        window.append(s)
        if len(window) > 16:
            window.pop(0)
        _ = np.mean(window) > 0.45
    timings["threshold_mock_ms"] = (time.perf_counter() - t0) * 1000 / n_runs

    return timings


def print_efficiency_table(results: dict) -> None:
    print("\n" + "=" * 80)
    print("EFFICIENCY ANALYSIS — Cascade vs Baselines")
    print("=" * 80)
    print(f"{'Method':<40} {'FLOPs/frame':>12} {'Latency(ms)':>12} {'FPS':>8} {'vs Naive':>10}")
    print("-" * 80)

    for method, metrics in results.items():
        flops_str = f"{metrics['flops_per_frame'] / 1e9:.1f}G"
        savings_str = f"{metrics['flops_ratio']:.4f}×"
        print(
            f"{method:<40} {flops_str:>12} {metrics['latency_ms']:>10.1f}ms "
            f"{metrics['fps']:>7.1f} {savings_str:>10}"
        )

    naive_fps = list(results.values())[0]["fps"]
    cascade_fps = list(results.values())[-1]["fps"]
    speedup = cascade_fps / naive_fps
    print("-" * 80)
    print(f"  Speedup vs Naive Full-VLM: {speedup:.0f}×")
    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(description="Efficiency analysis for cascade VAD")
    parser.add_argument("--trigger-rate", type=float, default=0.05,
                        help="Fraction of frames that pass Tier 1 (default: 0.05 = 5%%)")
    parser.add_argument("--microbenchmark", action="store_true",
                        help="Run CPU microbenchmark (no GPU needed)")
    args = parser.parse_args()

    print(f"\nEfficiency analysis with trigger_rate = {args.trigger_rate * 100:.0f}%")
    results = compute_cascade_efficiency(trigger_rate=args.trigger_rate)
    print_efficiency_table(results)

    if args.microbenchmark:
        print("Running CPU microbenchmark...")
        timings = run_latency_microbenchmark()
        print(f"  Mock CNN: {timings['cnn_mock_ms']:.3f}ms/frame")
        print(f"  Threshold check: {timings['threshold_mock_ms']:.4f}ms/frame")

    print("Ghi chú:")
    print("  - FLOPs là ước tính từ paper benchmarks (T4 GPU)")
    print("  - Tier 2 latency bao gồm VLM 3 rounds + LLM 3 calls")
    print("  - Trigger rate thực tế phụ thuộc vào dataset và threshold calibration")
    print()


if __name__ == "__main__":
    main()
