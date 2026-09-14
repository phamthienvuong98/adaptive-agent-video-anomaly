# Google Colab Training — Hướng dẫn từng bước

## Tổng quan
Chỉ cần train **1 model nhỏ**: MobileNetV3-Small Student CNN (~3ms/frame).  
VLM, LLM, CLIP đều dùng pre-trained, không cần train.

**Thời gian ước tính (T4 GPU miễn phí):**
| Bước | Thời gian |
|---|---|
| Setup + clone repo | ~3 phút |
| Download dataset UCF-Crime | ~30-60 phút |
| Extract frames (1 FPS) | ~20 phút |
| CLIP soft labeling | ~20 phút |
| Training 30 epochs | ~60-90 phút |
| **Tổng** | **~2.5-3 giờ** |

---

## Cell 1 — Kiểm tra GPU
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## Cell 2 — Mount Google Drive và Clone Repo
```python
from google.colab import drive
drive.mount('/content/drive')

# Clone repo (thay bằng repo của bạn)
!git clone https://github.com/YOUR_USERNAME/final_project.git
%cd final_project
```

## Cell 3 — Cài dependencies
```python
!pip install -q open-clip-torch torchvision loguru scikit-learn tqdm gdown sentence-transformers
!pip install -r requirements.txt
```

## Cell 4 — Download UCF-Crime Dataset

### Option A: Kaggle (khuyên dùng)
```python
# Upload kaggle.json credentials vào Colab trước
from google.colab import files
files.upload()  # Upload file kaggle.json

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d odins0n/ucf-crime-dataset --path data/ --unzip -q
```

### Option B: Từ Google Drive (nếu đã có)
```python
import shutil
# Copy từ Drive vào Colab
shutil.copytree('/content/drive/MyDrive/UCF_Crime', 'data/UCF_Crime')
```

## Cell 5 — Extract Frames (1 FPS, max 150 frames/video)
```python
!python scripts/colab_train.py \
    --data-dir data \
    --checkpoint-dir checkpoints \
    --drive-dir /content/drive/MyDrive/VAD_Checkpoint \
    --sample-fps 1.0 \
    --max-frames 150 \
    --epochs 0  # Chỉ extract, không train (sẽ báo lỗi cuối — bình thường)
```

> **Tip**: Sau khi extract xong, dùng `--skip-extract` để bỏ qua bước này.

## Cell 6 — Chạy training đầy đủ
```python
!python scripts/colab_train.py \
    --data-dir data \
    --checkpoint-dir checkpoints \
    --drive-dir /content/drive/MyDrive/VAD_Checkpoint \
    --epochs 30 \
    --batch-size 64 \
    --lr 3e-4 \
    --kd-alpha 0.7 \
    --sample-fps 1.0 \
    --skip-extract \
    --device cuda
```

## Cell 7 — Nếu Colab timeout, Resume training
```python
# Checkpoint latest được save tự động sau mỗi epoch
# Copy từ Drive về rồi resume:
!cp /content/drive/MyDrive/VAD_Checkpoint/student_cnn_latest.pth checkpoints/

!python scripts/colab_train.py \
    --skip-extract \
    --skip-clip \
    --resume \
    --epochs 30 \
    --device cuda
```

## Cell 8 — Kiểm tra kết quả
```python
import json
with open('checkpoints/test_results.json') as f:
    results = json.load(f)
print(f"Test AUC-ROC: {results['test_auc']:.4f}")

# Xem training history
with open('checkpoints/train_history.json') as f:
    history = json.load(f)
    
import matplotlib.pyplot as plt
aucs = [h['val_auc'] for h in history]
losses = [h['train_loss'] for h in history]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(aucs); ax1.set_title('Val AUC-ROC'); ax1.set_xlabel('Epoch')
ax2.plot(losses); ax2.set_title('Train Loss'); ax2.set_xlabel('Epoch')
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/VAD_Checkpoint/training_curve.png')
plt.show()
```

## Cell 9 — Copy checkpoint vào project để dùng trong pipeline
```python
# File cần copy vào: checkpoints/student_cnn.pth
# (configs/agent_config.yaml đang dùng path này)
import shutil
shutil.copy(
    '/content/drive/MyDrive/VAD_Checkpoint/student_cnn_best.pth',
    'checkpoints/student_cnn.pth'
)
print("Ready! Giờ có thể chạy: python main.py --source video.mp4 --mode demo")
```

---

## Kết quả kỳ vọng

| Metric | Mục tiêu luận văn | Baseline CLIP-only |
|---|---|---|
| UCF-Crime AUC-ROC | ≥ 87% | ~82% |
| Val AUC (training) | ≥ 85% | — |

**Nếu kết quả thấp hơn mục tiêu:**
1. Tăng `--epochs 50`
2. Giảm `--lr 1e-4`  
3. Tăng `--kd-alpha 0.8` (học nhiều hơn từ CLIP teacher)
4. Dùng `--backbone mobilenet_v3_large`

---

## Giải thích kiến trúc training cho luận văn

```
UCF-Crime Video
      ↓ (1 FPS sampling)
  Frames (224×224)
      ↓
  ┌──────────────────────────────────┐
  │  CLIP ViT-B/16 (TEACHER - frozen)│
  │  8 anomaly text queries          │
  │  6 normal text queries           │
  │  → soft_score ∈ [0,1]           │
  └──────────────────────────────────┘
      ↓ soft labels (Knowledge Distillation)
  ┌──────────────────────────────────┐
  │  MobileNetV3-Small (STUDENT)     │
  │  + Anomaly Head                  │
  │  Loss = 0.7 × KL(soft) +        │
  │         0.3 × BCE(hard)          │
  └──────────────────────────────────┘
      ↓
  student_cnn_best.pth (~10MB)
  → Deploy vào Fast Filter Agent (Tier 1)
  → 3-5ms/frame trên GPU
```

**Đây là đóng góp kỹ thuật**: Dùng CLIP zero-shot knowledge làm teacher signal để distill model nhỏ, không cần labeled dataset thủ công hay train teacher riêng.
