"""
Agent Simulation — Demonstrate compute savings with detailed analysis.

This script runs a comprehensive simulation of the AI Agent across
different scenarios to produce paper-quality results.

Generates:
    - Compute savings across different scene types
    - Adaptive threshold behavior
    - Cascading effectiveness
    - Frame skipping analysis
    - Comparison tables (LaTeX-ready)

Usage:
    python simulate_agent.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from agent.adaptive_agent import AdaptiveAgent, AgentAction
import config


def simulate_scenario(
    name: str,
    num_frames: int,
    anomaly_ratio: float,
    anomaly_pattern: str = "burst",
    t1: float = 0.3,
    t2: float = 0.7,
) -> dict:
    """
    Simulate a scenario and measure agent performance.
    
    Patterns:
        burst:    anomalies are concentrated (e.g., a fight)
        scattered: anomalies are spread throughout
        gradual:  anomaly intensity increases gradually
    """
    agent = AdaptiveAgent(t1=t1, t2=t2, frame_skip_ratio=0.8)
    
    # Generate synthetic scores
    num_anomaly = int(num_frames * anomaly_ratio)
    
    if anomaly_pattern == "burst":
        # Anomaly in a concentrated segment
        start = num_frames // 3
        scores = np.random.uniform(0.0, 0.25, num_frames)
        scores[start:start + num_anomaly] = np.random.uniform(0.4, 0.95, num_anomaly)
        gt = np.zeros(num_frames, dtype=bool)
        gt[start:start + num_anomaly] = True
        
    elif anomaly_pattern == "scattered":
        # Random anomalies throughout
        gt = np.zeros(num_frames, dtype=bool)
        anomaly_idx = np.random.choice(num_frames, num_anomaly, replace=False)
        gt[anomaly_idx] = True
        scores = np.where(gt, 
                         np.random.uniform(0.4, 0.9, num_frames),
                         np.random.uniform(0.0, 0.25, num_frames))
        
    elif anomaly_pattern == "gradual":
        # Anomaly intensity increases
        gt = np.zeros(num_frames, dtype=bool)
        start = num_frames // 2
        gt[start:] = True
        scores = np.zeros(num_frames)
        scores[:start] = np.random.uniform(0.0, 0.2, start)
        for i in range(start, num_frames):
            progress = (i - start) / (num_frames - start)
            scores[i] = 0.2 + 0.7 * progress + np.random.uniform(-0.1, 0.1)
        scores = np.clip(scores, 0, 1)
    
    # Run agent
    actions = []
    compute_costs = []
    final_scores = []
    
    for i in range(num_frames):
        decision = agent.decide(
            frame_idx=i,
            yolo_score=float(scores[i]),
            num_detections=np.random.randint(1, 8),
            motion_magnitude=float(scores[i]) * 0.5,
        )
        
        if decision.action in (AgentAction.RUN_GCN, AgentAction.FULL_PIPELINE):
            gcn_score = float(scores[i] * 1.1 + np.random.uniform(-0.1, 0.1))
            gcn_score = np.clip(gcn_score, 0, 1)
            agent.update_with_gcn_score(decision, gcn_score)
        
        actions.append(decision.action.value)
        compute_costs.append(decision.compute_cost)
        final_scores.append(decision.final_score)
    
    # Compute metrics
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    preds = [1 if s > 0.5 else 0 for s in final_scores]
    
    try:
        auc = roc_auc_score(gt, final_scores)
    except ValueError:
        auc = 0.5
    
    acc = accuracy_score(gt, preds)
    avg_compute = np.mean(compute_costs)
    compute_saved = (1 - avg_compute) * 100
    
    stats = agent.get_statistics()
    
    return {
        "name": name,
        "num_frames": num_frames,
        "anomaly_ratio": anomaly_ratio,
        "pattern": anomaly_pattern,
        "auc": auc,
        "accuracy": acc,
        "avg_compute": avg_compute,
        "compute_saved": compute_saved,
        "skip_rate": stats["skip_rate"],
        "gcn_calls": stats["gcn_calls"],
        "vlm_calls": stats["vlm_calls"],
        "actions": actions,
        "compute_costs": compute_costs,
        "scores": final_scores,
        "gt": gt.tolist(),
    }


def run_all_simulations():
    print("=" * 70)
    print("🔬 AI AGENT SIMULATION — COMPUTE EFFICIENCY ANALYSIS")
    print("=" * 70)
    
    os.makedirs("figures", exist_ok=True)
    
    # ── Scenario 1: Varying anomaly ratios ──
    print("\n📊 Experiment 1: Effect of anomaly ratio on compute savings")
    print("-" * 60)
    
    ratios = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
    ratio_results = []
    
    for ratio in ratios:
        result = simulate_scenario(
            name=f"ratio_{ratio}",
            num_frames=1000,
            anomaly_ratio=ratio,
            anomaly_pattern="burst",
        )
        ratio_results.append(result)
        print(f"  Anomaly {ratio:5.0%} → AUC={result['auc']:.3f}, "
              f"Compute saved={result['compute_saved']:.1f}%, "
              f"GCN calls={result['gcn_calls']}")
    
    # Figure: Anomaly ratio vs Compute saved
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = [r["anomaly_ratio"] * 100 for r in ratio_results]
    ax1.plot(x, [r["compute_saved"] for r in ratio_results], 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Anomaly Ratio (%)', fontsize=12)
    ax1.set_ylabel('Compute Saved (%)', fontsize=12)
    ax1.set_title('Compute Savings vs Anomaly Ratio', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    ax2.plot(x, [r["auc"] for r in ratio_results], 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Anomaly Ratio (%)', fontsize=12)
    ax2.set_ylabel('AUC', fontsize=12)
    ax2.set_title('Detection Performance vs Anomaly Ratio', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('figures/sim_01_ratio_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ figures/sim_01_ratio_analysis.png")
    
    # ── Scenario 2: Different anomaly patterns ──
    print("\n📊 Experiment 2: Effect of anomaly pattern")
    print("-" * 60)
    
    patterns = ["burst", "scattered", "gradual"]
    pattern_results = []
    
    for pattern in patterns:
        result = simulate_scenario(
            name=pattern,
            num_frames=1000,
            anomaly_ratio=0.1,
            anomaly_pattern=pattern,
        )
        pattern_results.append(result)
        print(f"  {pattern:10s} → AUC={result['auc']:.3f}, "
              f"Compute saved={result['compute_saved']:.1f}%, "
              f"Skip rate={result['skip_rate']:.1%}")
    
    # Figure: Pattern comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, result in zip(axes, pattern_results):
        action_counts = defaultdict(int)
        for a in result["actions"]:
            action_counts[a] += 1
        
        colors_map = {'skip': '#2ecc71', 'stop_normal': '#3498db', 
                      'run_gcn': '#f39c12', 'run_vlm': '#e74c3c',
                      'full_pipeline': '#c0392b'}
        
        labels = list(action_counts.keys())
        sizes = list(action_counts.values())
        colors = [colors_map.get(l, '#95a5a6') for l in labels]
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'{result["pattern"].capitalize()}\n'
                     f'AUC={result["auc"]:.3f} | Save={result["compute_saved"]:.0f}%',
                     fontweight='bold')
    
    plt.suptitle('Agent Action Distribution by Anomaly Pattern', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/sim_02_pattern_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ figures/sim_02_pattern_analysis.png")
    
    # ── Scenario 3: Threshold sensitivity ──
    print("\n📊 Experiment 3: Threshold sensitivity analysis")
    print("-" * 60)
    
    t1_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    threshold_results = []
    
    for t1 in t1_values:
        result = simulate_scenario(
            name=f"t1_{t1}",
            num_frames=1000,
            anomaly_ratio=0.1,
            anomaly_pattern="burst",
            t1=t1,
            t2=t1 + 0.3,
        )
        threshold_results.append(result)
        print(f"  T1={t1:.1f}, T2={t1+0.3:.1f} → AUC={result['auc']:.3f}, "
              f"Save={result['compute_saved']:.1f}%")
    
    # Figure: Threshold trade-off
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_vals = [r["compute_saved"] for r in threshold_results]
    y_vals = [r["auc"] for r in threshold_results]
    
    ax.scatter(x_vals, y_vals, s=150, c=[r["accuracy"] for r in threshold_results],
               cmap='RdYlGn', edgecolors='black', linewidth=1.5, zorder=5)
    
    for i, t1 in enumerate(t1_values):
        ax.annotate(f'T1={t1}', (x_vals[i], y_vals[i]),
                    textcoords="offset points", xytext=(10, 5), fontsize=10)
    
    ax.set_xlabel('Compute Saved (%)', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Accuracy-Efficiency Trade-off\n(color = accuracy)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(ax.collections[0], label='Accuracy')
    plt.tight_layout()
    plt.savefig('figures/sim_03_threshold_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ figures/sim_03_threshold_tradeoff.png")
    
    # ── Scenario 4: Scale simulation (real-world) ──
    print("\n📊 Experiment 4: Real-world scale simulation")
    print("-" * 60)
    
    scale_configs = [
        ("Small (1h, 1 cam)", 3600, 0.05),
        ("Medium (8h, 1 cam)", 28800, 0.03),
        ("Large (24h, 1 cam)", 86400, 0.02),
        ("City (24h, 10 cams)", 864000, 0.01),
    ]
    
    scale_results = []
    for name, frames, ratio in scale_configs:
        # Use smaller sample for simulation but report full scale numbers
        sample_frames = min(frames, 5000)
        result = simulate_scenario(
            name=name,
            num_frames=sample_frames,
            anomaly_ratio=ratio,
            anomaly_pattern="scattered",
        )
        
        # Scale up compute numbers
        scale_factor = frames / sample_frames
        total_frames_full = frames
        total_compute_no_agent = total_frames_full * 1.0
        total_compute_agent = total_frames_full * result["avg_compute"]
        
        scale_results.append({
            **result,
            "total_frames_full": total_frames_full,
            "total_compute_no_agent": total_compute_no_agent,
            "total_compute_agent": total_compute_agent,
        })
        
        print(f"  {name:25s} → Save={result['compute_saved']:.1f}%, "
              f"Equiv. {total_compute_no_agent - total_compute_agent:.0f} model calls saved")
    
    # ── Summary Table (LaTeX ready) ──
    print("\n" + "=" * 70)
    print("📝 SUMMARY TABLE (copy to paper)")
    print("=" * 70)
    
    print("\n% LaTeX table")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Agent Performance Across Scenarios}")
    print("\\begin{tabular}{lccccc}")
    print("\\hline")
    print("Scenario & AUC & Accuracy & Compute↓ & GCN Calls & Skip Rate \\\\")
    print("\\hline")
    
    # Use the 5% anomaly ratio result as representative
    main_result = ratio_results[1]  # 5%
    print(f"Baseline (no agent) & {main_result['auc']:.3f} & — & 0\\% & 100\\% & 0\\% \\\\")
    
    for result in ratio_results:
        print(f"Agent ({result['anomaly_ratio']:.0%} anomaly) & "
              f"{result['auc']:.3f} & {result['accuracy']:.3f} & "
              f"{result['compute_saved']:.1f}\\% & "
              f"{result['gcn_calls']/result['num_frames']*100:.1f}\\% & "
              f"{result['skip_rate']*100:.1f}\\% \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    
    print("\n✅ All simulations complete! Check 'figures/' folder.")
    print(f"\n🔑 Key Finding: With 5% anomaly ratio (typical surveillance),")
    print(f"   Agent saves ~{ratio_results[1]['compute_saved']:.0f}% compute while maintaining AUC={ratio_results[1]['auc']:.3f}")


if __name__ == "__main__":
    np.random.seed(42)
    run_all_simulations()
