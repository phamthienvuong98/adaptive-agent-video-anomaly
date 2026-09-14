# VAD Agent — Trạng thái dự án & Research Status

**Luận văn Thạc sĩ**: *Phát hiện hành vi bất thường trong giám sát đô thị thông minh bằng AI Agent tích hợp học sâu và hiểu video*

> Tài liệu này tổng hợp **trạng thái thực tế của codebase** + **đánh giá phản biện về hướng nghiên cứu**, dùng làm điểm xuất phát để tiếp tục research.
>
> - Cập nhật: 2026-08-22
> - PLAN.md gốc: 2026-05-31 (đã 3 tháng — xem §6 về rủi ro lỗi thời)
> - Chi tiết kiến trúc đầy đủ: xem [PLAN.md](PLAN.md)

---

## 1. Core claim của luận văn

> Thay vì cascade cứng (score > threshold → gọi model nặng), hệ thống dùng một **routing agent** đọc nhiều tín hiệu của frame (motion, pose, scene complexity, context) để **chọn động model tier phù hợp** — tối ưu tài nguyên mà không hy sinh recall.

Novelty claim đăng ký (PLAN.md §7):

> *"Hệ thống đầu tiên dùng multi-signal routing agent (optical flow + skeleton pose + scene embedding + contextual metadata) để dynamically select model tier trong VAD — thay vì cascade ngưỡng cứng một chiều như Cerberus, SlowFastVAD, QVAD."*

⚠️ **Claim này đang có vấn đề** — xem §4.

---

## 2. Kiến trúc hiện tại (đã implement)

```
Video Stream
     │
     ▼
┌──────────────────────────────────────────────────────────┐
│  FEATURE EXTRACTION  (~15ms)                             │
│  Optical Flow    MobileNetV3     MoveNet        Metadata │
│  (Farneback)     (embedding)     (pose/count)   (time,cam)│
│  ~2ms            ~8ms            ~5ms           0ms      │
└──────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────┐
│  ROUTING AGENT                                            │
│  score = (0.35·motion + 0.25·person                       │
│         + 0.20·complexity + 0.20·risk) × night_mult(1.3)  │
└──────────────────────────────────────────────────────────┘
     │
     ├─ score < 0.35 ──→ Tier 1: MobileNetV3      ~94% frames, ~10ms
     │                    └─ escalate nếu score > 0.5
     ├─ 0.35–0.70 ─────→ Tier 2: CLIP ViT-B/16    ~5% frames,  ~50ms
     │                    └─ escalate nếu score > 0.65
     └─ score > 0.70 ──→ Tier 3: Qwen2.5-VL-7B    ~1% frames,  ~500ms
                          └─ QVAD dialogue loop (max 2 vòng)
                          └─ Per-camera FAISS memory
                                │
                                ▼
                          Alert Engine
```

### Bản đồ file

| Thư mục | File | Vai trò | Trạng thái |
|---|---|---|---|
| `agents/` | `routing_agent.py` | **Core novelty** — multi-signal routing | ✅ Xong, 21 test pass |
| | `orchestrator.py` | Điều phối 3-tier | ✅ Xong, chưa chạy e2e |
| | `deep_reasoning_agent.py` | Tier 3 VLM↔LLM loop | ⚠️ Có bug chặn (§5) |
| | `fast_filter_agent.py` | Kiến trúc cũ 2-tier | ❌ Dead code, nên xóa |
| `features/` | `optical_flow.py` | Farneback → dict 4 giá trị | ✅ |
| | `scene_embedding.py` | MobileNetV3 pretrained → entropy | ✅ |
| | `pose_detector.py` | MoveNet multipose + fallback blob | ✅ |
| `models/` | `tier1_mobilenet.py` | Student CNN (knowledge distillation) | ✅ Chưa train |
| | `tier2_clip.py` | CLIP zero-shot scorer | ✅ |
| | `tier3_vlm.py` | Qwen2.5-VL wrapper | ✅ Chưa test GPU |
| | `llm_orchestrator.py` | Qwen3-4B — sinh & refine câu hỏi | ✅ |
| `utils/` | `per_camera_memory.py` | FAISS + anomaly gate (score < 0.35) | ✅ |
| | `alert_engine.py`, `anomaly_scorer.py`, `video_preprocessor.py` | | ✅ |
| `configs/` | `routing_config.yaml`, `camera_config.yaml`, `agent_config.yaml` | | ✅ |
| `scripts/` | `ablation_study.py` | 6 variants V1–V6 | ✅ Chưa chạy (thiếu data) |
| | `eval_ucf_crime.py`, `eval_xd_violence.py`, `efficiency_analysis.py`, `baseline_comparison.py`, `train_tier1.py` | | ✅ Chưa chạy |

Tổng: **5.478 dòng Python**.

### Test status

```
21 passed  — tests/test_routing_agent.py (routing logic, boundary, night multiplier)
28 failed  — tests/test_fast_filter.py + test_deep_reasoning.py
```

Nguyên nhân 28 fail: (a) thiếu `cv2`, `torchvision` trong env local — **không phải lỗi logic**; (b) các test này viết cho kiến trúc 2-tier cũ đã bị xóa (`AdaptiveThreshold`, `TemporalScoreBuffer`). Cần viết lại hoặc xóa.

**Chưa có kết quả thực nghiệm nào** — chưa download UCF-Crime, chưa train Tier 1, chưa chạy ablation.

---

## 3. Đánh giá phản biện: yếu tố "Agent" ⚠️

Đây là điểm yếu lớn nhất hiện tại. Hội đồng gần như chắc chắn sẽ hỏi.

### `RoutingAgent` **không phải** agent

Bóc code ra chỉ có 2 phép tính:

- [`routing_agent.py:157-165`](agents/routing_agent.py#L157-L165) — tổng có trọng số tuyến tính, weights là hằng số trong YAML
- [`routing_agent.py:176-182`](agents/routing_agent.py#L176-L182) — `if/elif/else` trên 2 ngưỡng cố định

Đặc tính:

- **Stateless** — `route()` không nhớ gì về frame trước
- **Không học** — weights không bao giờ thay đổi
- **Không feedback** — kết quả Tier 2/3 không quay lại ảnh hưởng routing sau đó
- **Escalation hardcode** — [`orchestrator.py:106-108`](agents/orchestrator.py#L106-L108) fix cứng `0.5` / `0.65`

Về học thuật, đây là **static gating function / heuristic router**, không phải agent. PLAN.md:318 cũng tự thừa nhận: *"Rule-based với weights được chọn dựa trên UCF-Crime category distribution."*

### `DeepReasoningAgent` **là** agent thật — nhưng chỉ ~1% hệ thống

[`deep_reasoning_agent.py:89-118`](agents/deep_reasoning_agent.py#L89-L118): LLM sinh câu hỏi → VLM trả lời → LLM đánh giá → **tự quyết định** dừng sớm hay refine câu hỏi. Luồng điều khiển do model quyết định, không phải do code. Cộng RAG retrieval làm context. Đây là vòng perception → reasoning → action selection → termination thật sự.

Hạn chế: `max_rounds = 2`, chỉ chạy trên ~1% frame.

### Đối chiếu tiêu chí agent chuẩn

| Thuộc tính | Hiện có? | Ở đâu |
|---|---|---|
| Perception (đa tín hiệu) | ✅ | 4 signals |
| Tool / model selection | ⚠️ | Có, nhưng bằng công thức cố định |
| Memory, state theo thời gian | ❌ | Router stateless; chỉ Tier-3 RAG có |
| Planning | ❌ | — |
| Learning / adaptation | ❌ | — |
| Reflection, self-correction | ⚠️ | Chỉ trong vòng lặp Tier 3 |
| Closed-loop feedback | ❌ | Escalation một chiều |

**Kết luận**: novelty hiện tại thực chất là **feature engineering cho một cascade**, không phải agency. Tiêu đề luận văn có chữ "AI Agent" nhưng phần được gọi là agent lại là phần ít agentic nhất.

---

## 4. Rủi ro với novelty claim

Ba vấn đề cần giải quyết trước khi bảo vệ:

1. **"Multi-signal" là novelty yếu.** Thêm tín hiệu vào một hàm tuyến tính là feature engineering, không phải đóng góp thuật toán. Phản biện dễ thấy: *"Sao không chỉ train một MLP nhỏ trên 4 feature đó?"* — và câu trả lời trung thực là *"làm được, và có lẽ tốt hơn."*

2. **Chữ "first" rất rủi ro.** Cần literature sweep mới để xác nhận (§6).

3. **Ablation V1–V6 không đo được agency.** Sáu variant hiện tại chỉ thay đổi *dùng signal nào* → chứng minh feature engineering. Muốn chứng minh đóng góp "agent" phải có variant thêm state/feedback/adaptation.

---

## 5. Nợ kỹ thuật cần xử lý

| # | Vấn đề | File | Mức độ |
|---|---|---|---|
| 1 | `filter_result` không tồn tại — `NameError` ngay frame Tier-3 đầu tiên | [`deep_reasoning_agent.py:103`](agents/deep_reasoning_agent.py#L103) | 🔴 Chặn |
| 2 | `tier1_escalate_threshold` / `tier2_escalate_threshold` trong `routing_config.yaml:30,33` **không chỗ nào đọc** — config chết, giá trị thật hardcode trong orchestrator | `routing_config.yaml`, `orchestrator.py:106-108` | 🟡 |
| 3 | `fast_filter_agent.py` là dead code của kiến trúc cũ | `agents/` | 🟡 |
| 4 | 28 test fail — viết cho kiến trúc đã xóa | `tests/` | 🟡 |
| 5 | **Target không nhất quán**: PLAN.md:287 ghi AUC ≥ 82%, README cũ ghi ≥ 87%; FPS ≥ 25 vs ≥ 30 | | 🟡 Phải chốt |
| 6 | Chưa có data, chưa train, chưa có số thực nghiệm nào | | 🔴 |

---

## 6. Câu hỏi research cần verify

Đây là phần chính để tiếp tục nghiên cứu. PLAN.md viết 2026-05-31, **hiện đã 2026-08-22 — cách gần 3 tháng**, lĩnh vực VAD/agentic đang chạy rất nhanh.

### 6.1 Tính toàn vẹn của trích dẫn 🔴

PLAN.md:346 đã tự đánh dấu *"Provisional — verify từ PDF gốc trước khi cite"*. **Chưa ai verify.** Cần đọc PDF gốc từng paper:

| Paper | arXiv ID cần check | Số đang cite | Rủi ro |
|---|---|---|---|
| QVAD | 2604.03040 | ~86.5% UCF, ~82.5% AP XD | ⚠️ Cả ID lẫn số đều chưa verify |
| Cerberus | 2510.16290 | "138× speedup" | ⚠️ Chưa verify |
| LAVAD | 2404.01014 | 80.28% UCF | Nhiều khả năng đúng |
| VERA | 2412.01095 | 86.55% UCF | Nhiều khả năng đúng |
| Flashback | 2505.15205 | ~87.3% UCF | ⚠️ |
| SlowFastVAD | 2504.10320 | — | ⚠️ |

Cite sai số là lỗi chết người ở luận văn. Ưu tiên số 1.

⚠️ **Bẫy metric** (PLAN.md:348): XD-Violence có cả AUC và AP — tuyệt đối không trộn hai metric trong cùng một bảng.

### 6.2 Novelty còn đứng vững không? 🔴

Cần sweep literature từ 2026-05 đến nay, tập trung:

- Có paper nào đã làm **learned/adaptive routing** cho VAD chưa?
- Các hướng liên quan: MoE-style tier selection, dynamic/adaptive inference, early-exit networks áp dụng vào video
- "Agentic VAD" 2026 — QVAD có follow-up không?
- Có ai đã dùng skeleton pose làm **routing signal** (không phải làm feature detection) chưa?

Từ khoá gợi ý: `adaptive routing video anomaly detection`, `dynamic model selection VAD`, `agentic video anomaly detection 2026`, `mixture-of-experts video surveillance`, `early-exit VLM cascade`, `resource-aware video understanding`.

### 6.3 Định vị lại đóng góp

Nếu §6.2 cho thấy "multi-signal routing" đã có người làm, cần pivot. Ba lựa chọn (chi tiết §7):

- Đóng góp = **học** routing policy (không phải multi-signal)
- Đóng góp = **closed-loop adaptation** giữa các tier
- Đóng góp = **per-camera personalization** (memory + risk context) — hướng này ít bị đụng hàng hơn

### 6.4 Target có thực tế không?

- AUC ≥ 82% (PLAN) so với LAVAD 80.28% — biên độ chỉ ~1.7pp, dễ bị nói là noise. VERA đã 86.55%.
- Câu hỏi thật: hệ thống này bán **accuracy** hay bán **efficiency**? Nếu là efficiency thì phải đóng khung là *"đạt AUC tương đương với X% compute"* và bảng chính phải là accuracy-vs-FLOPs, không phải AUC đơn thuần.

### 6.5 Khả thi kỹ thuật

- Qwen2.5-VL-7B 4-bit có thật sự chạy được trong VRAM Colab T4 (16GB) không?
- ~500ms/frame Tier 3 có đạt trên T4 không?
- MoveNet (TensorFlow) + PyTorch cùng process — xung đột CUDA/memory?

---

## 7. Hướng nâng cấp để "agent" thành thật

Xếp theo tỉ lệ giá trị / công sức:

### Hướng 1 — State + feedback loop (~1 ngày) ⭐ khuyến nghị

Router giữ state per-camera: lịch sử tier gần đây, kết quả escalation, một mức *vigilance*.

- Tier 2/3 xác nhận anomaly → vigilance camera đó tăng → boost routing score N frame kế tiếp
- Escalation liên tục trả về "normal" → vigilance giảm

Cho: **internal state + closed-loop adaptation** (đủ điều kiện agent). Quan trọng hơn: **có khả năng tăng AUC thật**, vì anomaly trong UCF-Crime kéo dài nhiều frame liên tiếp — đây là cải tiến có cơ sở thực nghiệm, không phải trang trí học thuật.

### Hướng 2 — Budget-aware control (~nửa ngày)

Router có mục tiêu FPS/compute budget, tự điều chỉnh ngưỡng để giữ trong budget. Đây là *goal-directed behavior under constraint* — thuộc tính agent kinh điển, dễ bảo vệ ("resource-rational agent"). Rẻ và gắn trực tiếp với câu chuyện efficiency ở §6.4.

### Hướng 3 — LLM meta-controller (~1–2 ngày)

Mỗi N frame, một LLM đọc thống kê routing (phân bố tier, tỉ lệ escalation trúng, FPS) rồi xuất weights/thresholds điều chỉnh. "Agentic" nhất theo nghĩa đen, compute không đáng kể (1 lần / ~1000 frame), rất dễ demo. Rủi ro: nếu không gắn được với cải thiện metric thì bị coi là gimmick.

### Lợi ích phụ: ablation mới đo được agency

Thêm Hướng 1 + 2 cho V7/V8, lúc đó ablation mới thực sự tách bạch đóng góp "agent" khỏi feature engineering:

| Variant | Nội dung |
|---|---|
| V1 | Fixed cascade (motion, threshold cứng) — baseline kiểu Cerberus |
| V2 | Routing motion only |
| V3 | + person count |
| V4 | + scene complexity |
| V5 | Full 4 signals (hệ thống PLAN.md) |
| V6 | V5 + per-camera FAISS memory |
| **V7** | **V6 + feedback state/vigilance** ← đo agency |
| **V8** | **V7 + budget-aware control** ← đo agency |

### Phương án không đụng code

Đổi framing: gọi đúng tên là *"adaptive multi-signal routing policy"*, bỏ chữ agent khỏi claim. An toàn về mặt học thuật, nhưng tiêu đề luận văn đang là *"bằng AI Agent"* nên sẽ hụt.

---

## 8. Benchmark & bối cảnh SOTA

⚠️ Toàn bộ số dưới đây **chưa được verify từ PDF gốc** — xem §6.1.

| Method | Paradigm | UCF-Crime AUC | XD-Violence |
|---|---|---|---|
| LAVAD | Training-free | 80.28% | 85.36% AUC |
| VERA (InternVL2-8B) | Training-free | 86.55% | 88.26% AUC |
| Flashback | Zero-shot | ~87.3% | 75.1% AP |
| QVAD | Agentic | ~86.5%* | ~82.5% AP* |
| VadCLIP | Weakly-supervised | 88.02% | 84.51% AP |
| **Hệ thống này** | **Agentic routing** | **mục tiêu ≥ 82%** | — |

Mục tiêu defend (PLAN.md:285-288):

- Tier-1 recall ≥ 0.95 (anomaly không bị drop ở bước routing)
- End-to-end AUC-ROC ≥ 82% trên UCF-Crime
- Throughput ≥ 25 FPS (Tier 1 path, Colab T4)

---

## 9. Việc cần làm tiếp

**Ưu tiên P0 — research (làm trước khi code thêm)**

1. Verify toàn bộ số & arXiv ID từ PDF gốc (§6.1)
2. Literature sweep 2026-05 → nay, xác nhận novelty còn đứng (§6.2)
3. Chốt lại đóng góp và target metric dựa trên kết quả 1–2 (§6.3, §6.4)

**Ưu tiên P0 — code**

4. Sửa bug `filter_result` ([`deep_reasoning_agent.py:103`](agents/deep_reasoning_agent.py#L103))
5. Download UCF-Crime + chạy `scripts/ablation_study.py --subset 50 --variants V1,V5` để có số đầu tiên
6. Train Tier 1 (`scripts/train_tier1.py`), đo Tier-1 recall

**Ưu tiên P1**

7. Implement Hướng 1 + 2 (§7) → V7/V8
8. Dọn nợ kỹ thuật (§5, mục 2–4)

---

## 10. Cài đặt

```bash
conda create -n vad_agent python=3.10
conda activate vad_agent
pip install -r requirements.txt
python scripts/download_models.py

# Chạy pipeline
python main.py --source data/sample_video.mp4 --config configs/agent_config.yaml

# Ablation
python scripts/ablation_study.py --data data/UCF_Crime/ --subset 200

# Test
python -m pytest tests/test_routing_agent.py -q
```

UCF-Crime: https://www.crcv.ucf.edu/research/real-world-anomaly-detection/
