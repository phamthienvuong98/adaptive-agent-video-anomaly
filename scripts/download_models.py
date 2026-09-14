"""
scripts/download_models.py
───────────────────────────
Download model weights cần thiết cho VAD Agent pipeline.
HuggingFace Hub được dùng cho VLM/LLM; open-clip download CLIP tự động.

Cách dùng:
    python scripts/download_models.py               # Download tất cả
    python scripts/download_models.py --tier cpu    # Chỉ download CLIP (CPU mode)
    python scripts/download_models.py --tier mid    # CLIP + LLaVA-1.6 + Qwen3-1.7B
    python scripts/download_models.py --tier full   # CLIP + Qwen2.5-VL-7B + Qwen3-4B
"""

import argparse
import os
import sys
from pathlib import Path


MODELS = {
    "clip": {
        "description": "CLIP ViT-B/16 (Fast Filter, ~300MB)",
        "tier": ["cpu", "mid", "full"],
        "download_fn": "download_clip",
    },
    "sentence_transformer": {
        "description": "all-MiniLM-L6-v2 (RAG Memory, ~80MB)",
        "tier": ["cpu", "mid", "full"],
        "download_fn": "download_sentence_transformer",
    },
    "llava": {
        "description": "LLaVA-1.6-mistral-7b (VLM, ~14GB)",
        "tier": ["mid"],
        "hf_model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "download_fn": "download_hf_model",
        "cache_dir": "checkpoints/vlm/",
    },
    "qwen_vlm": {
        "description": "Qwen2.5-VL-7B-Instruct (VLM, ~15GB)",
        "tier": ["full"],
        "hf_model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "download_fn": "download_hf_model",
        "cache_dir": "checkpoints/vlm/",
    },
    "qwen_llm_small": {
        "description": "Qwen3-1.7B (LLM Orchestrator, ~3.5GB)",
        "tier": ["mid"],
        "hf_model_id": "Qwen/Qwen3-1.7B",
        "download_fn": "download_hf_model",
        "cache_dir": "checkpoints/llm/",
    },
    "qwen_llm": {
        "description": "Qwen3-4B (LLM Orchestrator, ~8GB)",
        "tier": ["full"],
        "hf_model_id": "Qwen/Qwen3-4B",
        "download_fn": "download_hf_model",
        "cache_dir": "checkpoints/llm/",
    },
}


def download_clip():
    print("  Downloading CLIP ViT-B/16 qua open-clip-torch...")
    try:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        print("  CLIP đã sẵn sàng (cached bởi open_clip).")
    except ImportError:
        print("  [ERROR] open-clip-torch chưa cài. Chạy: pip install open-clip-torch")
        sys.exit(1)


def download_sentence_transformer():
    print("  Downloading all-MiniLM-L6-v2...")
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("  Sentence Transformer đã sẵn sàng.")
    except ImportError:
        print("  [ERROR] sentence-transformers chưa cài. Chạy: pip install sentence-transformers")
        sys.exit(1)


def download_hf_model(hf_model_id: str, cache_dir: str):
    print(f"  Downloading {hf_model_id} → {cache_dir}")
    try:
        from huggingface_hub import snapshot_download
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        token = os.environ.get("HF_TOKEN")
        local_dir = snapshot_download(
            repo_id=hf_model_id,
            local_dir=cache_dir + hf_model_id.replace("/", "--"),
            token=token,
        )
        print(f"  Đã lưu tại: {local_dir}")
    except ImportError:
        print("  [ERROR] huggingface_hub chưa cài. Chạy: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"  [ERROR] Download thất bại: {e}")
        if "401" in str(e):
            print("  Hint: Model này cần xác thực. Set biến môi trường HF_TOKEN=<your_token>")
        sys.exit(1)


def run_downloads(tier: str):
    print(f"\n=== Download models cho tier: {tier.upper()} ===\n")

    for name, info in MODELS.items():
        if tier not in info["tier"]:
            continue
        print(f"[{name}] {info['description']}")

        fn_name = info["download_fn"]
        if fn_name == "download_clip":
            download_clip()
        elif fn_name == "download_sentence_transformer":
            download_sentence_transformer()
        elif fn_name == "download_hf_model":
            download_hf_model(info["hf_model_id"], info["cache_dir"])

        print()

    print("=== Hoàn thành! ===")
    print("\nBước tiếp theo:")
    if tier == "cpu":
        print("  python main.py --source data/sample_video.mp4 --mode demo")
    else:
        print("  python main.py --source data/sample_video.mp4 --mode demo --show")


def main():
    parser = argparse.ArgumentParser(description="Download model weights cho VAD Agent")
    parser.add_argument(
        "--tier",
        choices=["cpu", "mid", "full"],
        default="mid",
        help=(
            "cpu: CLIP + SentenceTransformer (CPU-only mode)\n"
            "mid: + LLaVA-1.6 + Qwen3-1.7B (GPU 8GB)\n"
            "full: + Qwen2.5-VL-7B + Qwen3-4B (GPU 16GB+)"
        ),
    )
    args = parser.parse_args()

    print("""
VAD Agent — Model Downloader
─────────────────────────────
Lưu ý:
  - Cần kết nối internet tốt (các model VLM/LLM có thể > 10GB)
  - Đặt HF_TOKEN nếu dùng model cần xác thực HuggingFace
  - CLIP và SentenceTransformer tự động cache, không cần HF_TOKEN
""")

    run_downloads(args.tier)


if __name__ == "__main__":
    main()
