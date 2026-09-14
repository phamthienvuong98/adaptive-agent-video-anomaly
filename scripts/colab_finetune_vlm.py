"""
scripts/colab_finetune_vlm.py
──────────────────────────────
[OPTIONAL] Fine-tune VLM (LLaVA-1.6 hoặc Qwen2.5-VL) với LoRA
trên anomaly Q&A dataset.

MỤC TIÊU: VLM nhận diện chính xác hơn các hành vi bất thường
trong ngữ cảnh đô thị Việt Nam/châu Á (không chỉ Western datasets).

THỜI GIAN: ~3-4 giờ trên Colab A100 (GPU 40GB).
YÊU CẦU: GPU >= 16GB (A100 hoặc T4 với 4-bit quantization).

CHÚ Ý: Script này chuẩn bị sẵn cho tương lai.
Chỉ chạy khi:
  1. Đã có student_cnn_best.pth (từ colab_train.py)
  2. Đã có kết quả ablation/baseline
  3. Muốn cải thiện thêm Tier 2 accuracy

Chạy:
    python scripts/colab_finetune_vlm.py --model llava --epochs 3
    python scripts/colab_finetune_vlm.py --model qwen_vl --epochs 2 --use-4bit
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Synthetic Q&A Dataset Generator
# ══════════════════════════════════════════════════════════════════════════════

ANOMALY_CLASSES = {
    "fighting": {
        "description": "người đánh nhau, xô xát bạo lực",
        "questions": [
            "What type of violence or aggressive behavior do you observe?",
            "Are there any signs of physical altercation between people?",
            "Describe the body language of the people in the scene.",
        ],
        "answers": [
            "I can see people engaged in physical fighting with aggressive body movements.",
            "There are visible signs of physical altercation - people pushing, hitting, or grappling.",
            "The body language shows extreme aggression with raised arms, physical contact, and confrontational postures.",
        ],
    },
    "robbery": {
        "description": "cướp giật, trộm cắp",
        "questions": [
            "Is there any suspicious exchange or forceful taking of property?",
            "Describe any unusual interaction between people involving belongings.",
        ],
        "answers": [
            "There appears to be a forced taking of property with the victim showing distress.",
            "One person is aggressively grabbing or taking belongings from another person without consent.",
        ],
    },
    "road_accident": {
        "description": "tai nạn giao thông",
        "questions": [
            "Is there evidence of a vehicle accident or collision?",
            "Describe the state of vehicles and people in the scene.",
        ],
        "answers": [
            "I can see a vehicle collision with damaged vehicles and people in distress.",
            "There is visible vehicle damage consistent with a traffic accident.",
        ],
    },
    "vandalism": {
        "description": "phá hoại tài sản",
        "questions": [
            "Is there any property damage or vandalism occurring?",
            "Describe any destructive behavior visible in the scene.",
        ],
        "answers": [
            "Someone appears to be intentionally damaging property.",
            "There is visible destructive behavior directed at property.",
        ],
    },
    "normal": {
        "description": "hoạt động bình thường",
        "questions": [
            "Is there any suspicious or abnormal activity in this scene?",
            "Describe the general activity level and behavior of people.",
        ],
        "answers": [
            "The scene appears completely normal with people going about their daily activities.",
            "No suspicious behavior is observed. People are moving normally and calmly.",
        ],
    },
}


def generate_qa_dataset(
    frames_dir: Path,
    output_path: Path,
    num_samples: int = 500,
) -> list[dict]:
    """
    Tạo synthetic Q&A dataset từ extracted frames.
    Format: LLaVA instruction-tuning format.

    Args:
        frames_dir: thư mục chứa frames từ colab_train.py
        output_path: nơi lưu dataset
        num_samples: số Q&A pairs cần tạo
    """
    random.seed(42)
    dataset = []

    # Collect available frames
    anomaly_frames = list((frames_dir / "anomaly").glob("*.jpg")) if (frames_dir / "anomaly").exists() else []
    normal_frames = list((frames_dir / "normal").glob("*.jpg")) if (frames_dir / "normal").exists() else []

    if not anomaly_frames and not normal_frames:
        print("[WARN] Không có frames. Tạo template dataset không có ảnh.")
        return _generate_text_only_dataset(num_samples)

    # Balanced sampling
    n_anomaly = num_samples * 2 // 3
    n_normal = num_samples - n_anomaly

    selected_anomaly = random.choices(anomaly_frames, k=min(n_anomaly, len(anomaly_frames)))
    selected_normal = random.choices(normal_frames, k=min(n_normal, len(normal_frames)))

    # Create Q&A pairs for anomaly frames
    anomaly_classes = [k for k in ANOMALY_CLASSES if k != "normal"]

    for frame_path in selected_anomaly:
        # Randomize anomaly class (without GT we assume some distribution)
        anomaly_class = random.choice(anomaly_classes)
        cls_info = ANOMALY_CLASSES[anomaly_class]

        q = random.choice(cls_info["questions"])
        a = random.choice(cls_info["answers"])

        # LLaVA instruction format
        dataset.append({
            "id": f"anomaly_{len(dataset)}",
            "image": str(frame_path),
            "conversations": [
                {"from": "human", "value": f"<image>\n{q}"},
                {"from": "gpt", "value": a},
            ],
            "label": anomaly_class,
        })

    # Create Q&A pairs for normal frames
    for frame_path in selected_normal:
        cls_info = ANOMALY_CLASSES["normal"]
        q = random.choice(cls_info["questions"])
        a = random.choice(cls_info["answers"])

        dataset.append({
            "id": f"normal_{len(dataset)}",
            "image": str(frame_path),
            "conversations": [
                {"from": "human", "value": f"<image>\n{q}"},
                {"from": "gpt", "value": a},
            ],
            "label": "normal",
        })

    random.shuffle(dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Dataset tạo: {len(dataset)} Q&A pairs → {output_path}")
    return dataset


def _generate_text_only_dataset(num_samples: int) -> list[dict]:
    """Fallback: tạo text-only dataset (không có ảnh)."""
    dataset = []
    classes = list(ANOMALY_CLASSES.keys())

    for i in range(num_samples):
        cls = random.choice(classes)
        cls_info = ANOMALY_CLASSES[cls]
        q = random.choice(cls_info["questions"])
        a = random.choice(cls_info["answers"])

        dataset.append({
            "id": f"sample_{i}",
            "conversations": [
                {"from": "human", "value": q},
                {"from": "gpt", "value": a},
            ],
            "label": cls,
        })

    return dataset


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LoRA Fine-tuning (LLaVA)
# ══════════════════════════════════════════════════════════════════════════════

def finetune_llava_lora(
    dataset_path: Path,
    output_dir: Path,
    model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 2e-5,
    use_4bit: bool = True,
):
    """
    Fine-tune LLaVA với LoRA. Yêu cầu GPU >= 16GB (hoặc 4-bit quantization).
    """
    print(f"\nFine-tuning LLaVA: {model_id}")
    print(f"Dataset: {dataset_path} | Epochs: {epochs} | 4-bit: {use_4bit}")

    try:
        import torch
        from transformers import (
            LlavaNextProcessor,
            LlavaNextForConditionalGeneration,
            BitsAndBytesConfig,
            TrainingArguments,
            Trainer,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from torch.utils.data import Dataset
        from PIL import Image
    except ImportError as e:
        print(f"[ERROR] Missing package: {e}")
        print("Cài: pip install peft bitsandbytes transformers[torch] accelerate")
        return

    # ── Load Model ──────────────────────────────────────────────────────────
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    print("Loading model...")
    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # ── LoRA Config ─────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ─────────────────────────────────────────────────────────────
    with open(dataset_path) as f:
        data = json.load(f)

    split = int(len(data) * 0.9)
    train_data = data[:split]
    val_data = data[split:]

    class VLMDataset(Dataset):
        def __init__(self, samples, proc):
            self.samples = samples
            self.proc = proc

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            convs = sample["conversations"]
            user_msg = convs[0]["value"]
            assistant_msg = convs[1]["value"]

            img = None
            if "image" in sample and Path(sample["image"]).exists():
                img = Image.open(sample["image"]).convert("RGB")

            prompt = f"USER: {user_msg} ASSISTANT: {assistant_msg}"
            inputs = self.proc(
                text=prompt,
                images=img,
                return_tensors="pt",
                padding="max_length",
                max_length=512,
                truncation=True,
            )
            inputs["labels"] = inputs["input_ids"].clone()
            return {k: v.squeeze(0) for k, v in inputs.items()}

    train_ds = VLMDataset(train_data, processor)
    val_ds = VLMDataset(val_data, processor)

    # ── Training ─────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("\nBắt đầu training...")
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    print(f"\nLoRA adapter saved: {output_dir / 'final'}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LoRA Fine-tuning (Qwen2.5-VL)
# ══════════════════════════════════════════════════════════════════════════════

def finetune_qwen_vl_lora(
    dataset_path: Path,
    output_dir: Path,
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    epochs: int = 2,
    use_4bit: bool = True,
):
    """Fine-tune Qwen2.5-VL với LoRA. Yêu cầu GPU >= 16GB."""
    print(f"\nFine-tuning Qwen2.5-VL: {model_id}")

    try:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        print(f"[ERROR] Missing: {e}")
        print("Cài: pip install peft bitsandbytes transformers qwen-vl-utils")
        return

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    bnb_config = None
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Sử dụng cùng VLMDataset class như LLaVA
    with open(dataset_path) as f:
        data = json.load(f)

    print(f"Dataset: {len(data)} samples")
    print("Note: Xem LLaVA training loop để adapt Qwen-VL dataloader nếu cần.")
    print("Cấu trúc tương tự, chỉ khác format prompt của Qwen.")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Fine-tune VLM với LoRA cho anomaly detection")
    p.add_argument("--model", choices=["llava", "qwen_vl"], default="llava",
                   help="Model để fine-tune")
    p.add_argument("--frames-dir", default="data/frames",
                   help="Thư mục frames từ colab_train.py")
    p.add_argument("--dataset-path", default="data/vlm_qa_dataset.json",
                   help="Path dataset Q&A (sẽ tạo nếu chưa có)")
    p.add_argument("--output-dir", default="checkpoints/vlm_lora",
                   help="Nơi lưu LoRA adapter")
    p.add_argument("--drive-dir", default="/content/drive/MyDrive/VAD_Checkpoint/vlm_lora")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-qa-samples", type=int, default=500)
    p.add_argument("--use-4bit", action="store_true", default=True,
                   help="Dùng 4-bit quantization (cho GPU < 24GB)")
    p.add_argument("--generate-only", action="store_true",
                   help="Chỉ tạo dataset, không train")
    args = p.parse_args()

    # Step 1: Generate Q&A dataset
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print("Generating Q&A dataset...")
        generate_qa_dataset(
            frames_dir=Path(args.frames_dir),
            output_path=dataset_path,
            num_samples=args.num_qa_samples,
        )
    else:
        print(f"Dataset đã có: {dataset_path}")

    if args.generate_only:
        print("--generate-only: Dừng sau khi tạo dataset.")
        return

    # Step 2: Fine-tune
    output_dir = Path(args.output_dir)

    if args.model == "llava":
        finetune_llava_lora(
            dataset_path=dataset_path,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_4bit=args.use_4bit,
        )
    else:
        finetune_qwen_vl_lora(
            dataset_path=dataset_path,
            output_dir=output_dir,
            epochs=args.epochs,
            use_4bit=args.use_4bit,
        )

    # Step 3: Copy to Drive
    try:
        import shutil
        drive_path = Path(args.drive_dir)
        drive_path.mkdir(parents=True, exist_ok=True)
        if (output_dir / "final").exists():
            shutil.copytree(output_dir / "final", drive_path / "final", dirs_exist_ok=True)
            print(f"\nLoRA adapter saved to Drive: {drive_path}")
    except Exception as e:
        print(f"[WARN] Không copy được lên Drive: {e}")

    print("\nDone! Để dùng LoRA adapter trong pipeline:")
    print(f"  Sửa configs/agent_config.yaml:")
    print(f"    vlm_reasoner:")
    print(f"      lora_adapter: {output_dir}/final")


if __name__ == "__main__":
    main()
