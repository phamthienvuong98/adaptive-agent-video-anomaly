# VAD Agent — Kế hoạch tổng thể (Luận văn Thạc sĩ)

> Tổng hợp từ research + conversation design session  
> Cập nhật: 2026-05-31  
> Deadline target: 8 tuần

---

## 1. Tên đề tài

**Phát hiện hành vi bất thường trong giám sát đô thị thông minh bằng AI Agent tích hợp học sâu và hiểu video**

---

## 2. Core Idea — Một câu

> Thay vì cascade cứng (nếu score > threshold thì gọi model nặng), hệ thống dùng một **routing agent** quan sát nhiều đặc tính của frame (motion, pose, scene complexity, context) để **dynamically chọn model tier phù hợp** — đảm bảo resource tối ưu mà không sacrifice recall.

---

## 3. Tại sao approach này — Motivation

### Vấn đề với cascade cứng (Cerberus, SlowFastVAD, QVAD)
- Cascade cứng dùng **một tín hiệu duy nhất** (thường là CLIP score) để quyết định có gọi model nặng không
- Không phân biệt được: cùng một motion score cao, fighting khác hoàn toàn với crowd bình thường giờ tan tầm
- Fixed threshold không adapt theo scene — camera sân bay và camera bãi biển cần threshold khác nhau

### Insight then chốt từ design session
- Anomaly không có pattern cố định → không thể hardcode rule "loại này dùng model đó"
- Nhưng **đặc tính của frame** (bao nhiêu người, motion mạnh cỡ nào, scene phức tạp thế nào, bối cảnh gì) có thể cho biết **cần model nặng đến mức nào**
- Agent đọc multi-signal → decision động → đây là gap thực sự so với SotA

---

## 4. Architecture Final

```
Video Stream (camera)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│           FEATURE EXTRACTION (~15ms total)           │
│                                                       │
│  Optical Flow     MobileNetV3    MoveNet Lightning   Metadata
│  (Farneback)      (embedding)    (pose/count)        (time, cam)
│  ~2ms             ~8ms           ~5ms                0ms
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│              ROUTING AGENT                           │
│                                                       │
│  routing_score = (                                   │
│    0.35 × motion_magnitude_norm                      │
│    0.25 × person_count_norm                          │
│    0.20 × scene_complexity_norm                      │
│    0.20 × risk_context_numeric                       │
│  ) × night_multiplier (1.3 if night else 1.0)        │
│                                                       │
│  score < 0.35  → Tier 1  (~94% frames)               │
│  score 0.35–0.7 → Tier 2 (~5% frames)               │
│  score > 0.7   → Tier 3  (~1% frames)               │
└─────────────────────────────────────────────────────┘
        │              │              │
        ▼              ▼              ▼
   TIER 1          TIER 2         TIER 3
   MobileNetV3     CLIP           Qwen2.5-VL-7B
   anomaly score   ViT-B/16       (4-bit quantized)
   ~10ms           ~50ms          ~500ms
        │              │              │
        └──────────────┴──────────────┘
                       │
                       ▼
            ┌─────────────────┐
            │  ALERT ENGINE    │
            │  conf < 0.6 drop │
            │  cooldown 10s    │
            │  log JSON        │
            └─────────────────┘
                       │
               [Per-camera FAISS]
               Memory chỉ nhận
               frame khi score < 0.35
               (anomaly gate)
```

---

## 5. Chi tiết từng component

### 5.1 Feature Extraction

#### Optical Flow — Farneback
- **Library**: `cv2.calcOpticalFlowFarneback()`
- **Output**: flow magnitude map → `mean`, `max`, `std`, `spatial_pattern`
- **Latency**: ~2ms/frame CPU
- **Lý do**: Free, no model, cho biết motion magnitude và distribution

```python
flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
motion_score = {
    "mean": magnitude.mean(),
    "max": magnitude.max(),
    "std": magnitude.std(),
    "coverage": (magnitude > threshold).mean()  # % pixels có motion
}
```

#### MobileNetV3-Small — Scene Embedding
- **Library**: `torchvision.models.mobilenet_v3_small(pretrained=True)`
- **Output**: 576-dim embedding vector → entropy làm proxy scene complexity
- **Latency**: ~8ms GPU (Colab T4)
- **Lý do**: Lightweight, capture scene-level semantics, không cần fine-tune

```python
# Dùng features trước classifier head
model = mobilenet_v3_small(pretrained=True)
model.classifier = nn.Identity()
embedding = model(frame_tensor)  # [576]
scene_complexity = scipy.stats.entropy(softmax(embedding))
```

#### MoveNet Lightning — Pose/Person Count
- **Library**: TensorFlow Hub `movenet/singlepose/lightning`
- **Output**: keypoints + confidence → person count, pose roughness
- **Latency**: ~5ms GPU
- **Lý do**: Nhẹ hơn YOLO 3×, đủ để biết "bao nhiêu người, tư thế thô"
- **Note**: Literature NeurIPS 2024 xác nhận skeleton là lightweight alternative hiệu quả cho VAD, đặc biệt human-related anomalies (fighting, assault, robbery chiếm ~60% UCF-Crime)

```python
# MoveNet multi-pose cho nhiều người
detector = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
outputs = detector(frame_tensor)
# Đếm người dựa trên keypoint confidence > 0.3
person_count = sum(
    1 for person in outputs["output_0"][0]
    if person[..., 2].mean() > 0.3
)
```

#### Metadata
- `camera_id` → lookup risk_tag từ config (`low/medium/high`)
- `timestamp` → `is_night = hour < 6 or hour > 22`
- Cost: 0ms

---

### 5.2 Routing Agent

#### Rule-based (Phase 1 — Week 3-4)

```python
class RoutingAgent:
    def __init__(self, config):
        self.weights = {
            "motion": 0.35,
            "person": 0.25,
            "complexity": 0.20,
            "risk": 0.20,
        }
        self.night_multiplier = 1.3
        self.tier_thresholds = [0.35, 0.70]

    def compute_score(self, features: dict) -> float:
        score = (
            self.weights["motion"] * normalize(features["motion_mean"]) +
            self.weights["person"] * normalize(features["person_count"]) +
            self.weights["complexity"] * normalize(features["scene_complexity"]) +
            self.weights["risk"] * features["risk_numeric"]
        )
        if features["is_night"]:
            score *= self.night_multiplier
        return min(score, 1.0)

    def route(self, features: dict) -> int:
        score = self.compute_score(features)
        if score < self.tier_thresholds[0]:
            return 1
        elif score < self.tier_thresholds[1]:
            return 2
        else:
            return 3
```

#### Learned Upgrade (Phase 2 — nếu còn thời gian)
- Thu thập routing decisions từ rule-based làm pseudo-label
- Train MLP nhỏ (3 layers, 64 hidden) trên features → tier label
- So sánh accuracy với rule-based → nếu tốt hơn thì swap

---

### 5.3 Model Tiers

#### Tier 1 — MobileNetV3 Anomaly Scorer
- Fine-tune trên UCF-Crime normal vs anomaly clips
- Output: anomaly_score [0,1]
- Nếu score > 0.5 → escalate lên Tier 2 (safety net trong tier)

#### Tier 2 — CLIP ViT-B/16
- Zero-shot hoặc few-shot với text prompts:
```python
ANOMALY_PROMPTS = [
    "people fighting or engaged in violence",
    "a robbery or theft in progress",
    "vandalism or property destruction",
    "a person falling or collapsing",
    "a crowd panic or stampede",
]
NORMAL_PROMPTS = [
    "people walking normally in a public space",
    "a normal street scene",
    "a quiet surveillance scene",
]
# Score = max(anomaly_sims) - max(normal_sims)
```

#### Tier 3 — Qwen2.5-VL-7B (4-bit)
- Inference-only, không fine-tune (Colab T4 đủ VRAM với 4-bit)
- Input: frame + context từ per-camera FAISS memory
- Output: anomaly class + confidence + natural language explanation
- QVAD-style dialogue loop (tối đa 2 vòng để giữ latency):
  - LLM tạo câu hỏi → VLM trả lời từ ảnh → LLM đánh giá

---

### 5.4 Per-Camera FAISS Memory

```python
class PerCameraMemory:
    def __init__(self):
        self.indices = {}  # camera_id → faiss.IndexFlatL2

    def add(self, camera_id: str, embedding: np.ndarray,
            routing_score: float):
        # Anomaly gate: chỉ lưu frame bình thường
        if routing_score > 0.35:
            return  # Không lưu frame suspicious
        if camera_id not in self.indices:
            self.indices[camera_id] = faiss.IndexFlatL2(512)
        self.indices[camera_id].add(embedding.reshape(1, -1))

    def retrieve(self, camera_id: str, query: np.ndarray,
                 k: int = 5) -> list:
        if camera_id not in self.indices:
            return []  # Cold start: return empty
        D, I = self.indices[camera_id].search(query.reshape(1, -1), k)
        return I[0].tolist()
```

**Cold start handling**: camera mới → memory rỗng → Tier 3 chạy không có RAG context → vẫn hoạt động, chỉ kém hơn

---

## 6. Benchmark & Evaluation

### Dataset: UCF-Crime
- 1,900 videos, 13 anomaly categories
- Train: 1,610 videos | Test: 290 videos
- Metric: **Frame-level AUC-ROC** (standard)
- Download: [https://www.crcv.ucf.edu/research/real-world-anomaly-detection/](https://www.crcv.ucf.edu/research/real-world-anomaly-detection/)

### 13 Categories và nhóm
| Nhóm | Categories | Routing signal chính |
|------|-----------|----------------------|
| Violence/Action | Assault, Fighting, Robbery, Shooting, Stealing | Motion cao + person count cao |
| Accident | Car accident, Explosion, Fire | Motion spike + scene change |
| Property crime | Arson, Burglary, Vandalism, Shoplifting | Motion vừa + single person |

### Ablation study (bắt buộc cho luận văn)

| Variant | Mô tả |
|---------|-------|
| V1 | Fixed cascade (baseline, giống Cerberus) |
| V2 | Routing chỉ dùng motion |
| V3 | Routing motion + person |
| V4 | Routing motion + person + complexity |
| V5 | Full routing (motion + person + complexity + context) |
| V6 | Full routing + per-camera memory |

→ So sánh AUC-ROC và throughput (FPS) của từng variant

### Target numbers để defend
- **Tier-1 recall** ≥ 0.95 (anomaly không bị drop ở bước routing)
- **End-to-end AUC-ROC** ≥ 82% trên UCF-Crime (LAVAD baseline là 80.28%)
- **Throughput** ≥ 25 FPS (Tier 1 path, Colab T4)

---

## 7. Novelty Claim — Cách frame cho hội đồng

### Đóng góp chính
> *"Hệ thống đầu tiên sử dụng multi-signal routing agent (optical flow + skeleton pose + scene embedding + contextual metadata) để dynamically select model tier trong VAD — thay vì cascade ngưỡng cứng một chiều như Cerberus, SlowFastVAD, và QVAD."*

### So sánh với SotA

| Paper | Approach | Gap so với hệ thống này |
|-------|----------|------------------------|
| Cerberus (2025) | CLIP score → fixed top-k filter | 1 signal, không adapt theo context |
| SlowFastVAD (2025) | Entropy-based intervention | Không dùng pose/metadata |
| QVAD (2026) | Fixed dialogue loop, no routing | Không có dynamic model selection |
| LAVAD (CVPR 2024) | Training-free, global memory | Không per-camera, không routing |

### Câu trả lời cho hội đồng

**Q: Khác gì Cerberus?**  
A: Cerberus dùng CLIP score một chiều để filter. Hệ thống này dùng 4 signals độc lập (motion + pose + scene + context) → routing động. Ablation V1 vs V5 chứng minh delta.

**Q: Tại sao không chỉ dùng CLIP score như Cerberus?**  
A: CLIP score nhìn vào content của frame — không biết frame đó đến từ sân bay ban đêm hay bãi biển ban ngày. Context signal (metadata + pose count) bổ sung information mà CLIP không capture được.

**Q: Tier-1 recall là bao nhiêu?**  
A: [Điền số từ ablation — phải ≥ 0.95]

**Q: Routing agent học từ đâu?**  
A: Rule-based với weights được chọn dựa trên UCF-Crime category distribution. Weights có thể được upgrade lên learned MLP nếu có thêm thời gian.

---

## 8. Papers tham khảo chính

| Paper | Venue | Arxiv | Relevance |
|-------|-------|-------|-----------|
| LAVAD | CVPR 2024 | 2404.01014 | Training-free VLM VAD baseline |
| VERA | CVPR 2025 | 2412.01095 | VLM verbalized learning, 86.55% UCF-Crime |
| Cerberus | Preprint 2025 | 2510.16290 | Cascade VLM, closest competitor |
| QVAD | Preprint 2026 | 2604.03040 | Dialogue loop, Tier 3 design |
| SlowFastVAD | Preprint 2025 | 2504.10320 | RAG + slow/fast cascade |
| Flashback | Preprint 2025 | 2505.15205 | Memory-driven zero-shot VAD |
| TCVADS | Preprint 2024 | 2412.20201 | KD + MobileNet + CLIP |
| VadCLIP | AAAI 2024 | 2308.11681 | Weakly-supervised CLIP baseline |
| NeurIPS VAD Survey | NeurIPS 2024 | — | Skeleton as lightweight alternative |

### SOTA numbers để compare

| Method | Paradigm | UCF-Crime AUC | XD-Violence |
|--------|----------|---------------|-------------|
| LAVAD | Training-free | 80.28% | 85.36% AUC |
| VERA (InternVL2-8B) | Training-free | 86.55% | 88.26% AUC |
| Flashback | Zero-shot | ~87.3% | 75.1% AP |
| QVAD | Agentic | ~86.5%* | ~82.5% AP* |
| VadCLIP | Weakly-sup | 88.02% | 84.51% AP |

*Provisional — verify từ PDF gốc trước khi cite

**Cảnh báo metric**: XD-Violence có cả AUC và AP — không mix hai metric này trong cùng một bảng

---

## 9. File Structure đề xuất

```
final_project/
├── agents/
│   ├── routing_agent.py          ← Core novelty
│   ├── deep_reasoning_agent.py   ← Tier 3 QVAD loop
│   └── orchestrator.py
├── models/
│   ├── tier1_mobilenet.py        ← Anomaly scorer
│   ├── tier2_clip.py             ← CLIP zero-shot
│   └── tier3_vlm.py              ← Qwen2.5-VL wrapper
├── features/
│   ├── optical_flow.py           ← Farneback extractor
│   ├── scene_embedding.py        ← MobileNetV3 embedder
│   └── pose_detector.py          ← MoveNet Lightning
├── utils/
│   ├── video_preprocessor.py
│   ├── per_camera_memory.py      ← FAISS + anomaly gate
│   └── alert_engine.py
├── scripts/
│   ├── train_tier1.py            ← Fine-tune MobileNetV3
│   ├── ablation_study.py         ← V1–V6 comparison
│   ├── efficiency_analysis.py    ← FPS + FLOPs table
│   └── evaluate_ucfcrime.py      ← AUC-ROC computation
├── configs/
│   ├── routing_config.yaml       ← Weights, thresholds
│   └── camera_config.yaml        ← Camera risk tags
├── tests/
│   └── test_routing_agent.py
├── PLAN.md                       ← File này
└── README.md
```

---

## 10. 8-Week Timeline

```
Week 1  │ UCF-Crime download + EDA
        │ feature_extractor pipeline (optical flow + MobileNetV3)
        │ Baseline AUC với fixed threshold
        
Week 2  │ MoveNet Lightning integration
        │ Metadata pipeline (camera config, timestamp)
        │ Unit tests cho toàn bộ feature extraction

Week 3  │ routing_agent.py — rule-based implementation
        │ Tier 1 (MobileNetV3) fine-tune trên Colab T4
        │ Routing + Tier 1 end-to-end test

Week 4  │ Tier 2 (CLIP) integration
        │ Ablation V1 vs V2 vs V3 (motion only → motion+person)
        │ Measure Tier-1 recall → phải ≥ 0.95

Week 5  │ Tier 3 Qwen2.5-VL-7B (4-bit) trên Colab T4
        │ QVAD-style dialogue loop (max 2 turns)
        │ Per-camera FAISS memory + anomaly gate

Week 6  │ Full pipeline end-to-end
        │ Ablation V4 vs V5 vs V6
        │ Debug + optimize bottlenecks

Week 7  │ Full ablation study (V1–V6) trên UCF-Crime test set
        │ Efficiency analysis: FPS + latency bảng
        │ Compare vs Cerberus (cite paper numbers)

Week 8  │ Final AUC-ROC numbers
        │ Write Chương 3 (map với code)
        │ Điền số liệu vào Chương 4
```

---

## 11. Risk Flags & Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Qwen2.5-VL OOM trên Colab T4 | Cao | Dùng 4-bit quantization (bitsandbytes), giảm max_new_tokens |
| MoveNet multi-pose chậm hơn expected | Vừa | Fallback: chỉ dùng person count từ optical flow blob detection |
| UCF-Crime download chậm | Thấp | Dùng subset 200 videos để develop, full set để final eval |
| Tier-1 recall < 0.95 | Vừa | Giảm tier threshold từ 0.35 xuống 0.25, accept more Tier 2 calls |
| Routing weights không optimal | Vừa | Grid search trên validation set: motion [0.2–0.5], person [0.1–0.4] |

---

## 12. Compute Requirements

| Task | Hardware | Estimated Time |
|------|----------|---------------|
| Feature extraction pipeline | CPU local | Không giới hạn |
| Tier 1 fine-tune (MobileNetV3) | Colab T4 | ~1.5h |
| Tier 2 CLIP (inference) | Colab T4 | ~0.5h/eval |
| Tier 3 Qwen2.5-VL 4-bit | Colab T4 | ~3h/eval |
| Full ablation (6 variants) | Colab T4 | ~8h total |

**Colab session limit**: 12h liên tục → chia ablation thành 2 session

---

*Generated from: research session 2026-05-31 + conversation design session*  
*Architecture version: 1.0 — Adaptive Routing VAD*
