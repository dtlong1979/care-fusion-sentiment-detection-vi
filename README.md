# CARE-Fusion — Regime-Aware Fusion of Text & Affective Markers for Vietnamese Emotion Classification

Mã nguồn thực nghiệm cho mô hình **CARE-Fusion** trên tập **ViGoEmotions** (6.813 mẫu, 6 nhóm cảm xúc: `positive, sadness, anger, fear, interest, neutral`). Repo phục vụ một bài báo khoa học — ưu tiên **tái lập được** và **chống rò rỉ** (mọi tài nguyên dẫn xuất chỉ dựng từ tập *train*).

> Toàn bộ thiết kế thuật toán ở [`docs/CARE-Fusion_Experimental_Protocol.md`](docs/CARE-Fusion_Experimental_Protocol.md). README này tập trung vào *cách chạy*.

## Kiến trúc (tóm tắt)

```
PhoBERT(text_clean) ──► p_text ─┐
                                 ├─► JSD(δ) ─► Router r_j ─► Regime fusion z
markers (emoji/emoticon) ─► e_j ─┘   (redundancy / complementarity / conflict)
                                            │
                          PMI graph (GCN) ──► z̃ ─► classifier ─► ŷ
```

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| `src/care_fusion/markers.py` | Trích emoji + emoticon (longest-match, đếm lặp) — Part A2 |
| `src/care_fusion/preprocess.py` | NFC, word-segment (VnCoreNLP), tokenize — Part A |
| `src/care_fusion/resources.py` | `q_j`, weak regime labels, PMI graph — Part B (chỉ train) |
| `src/care_fusion/model.py` | Kiến trúc CARE-Fusion — Part C *(Phase 2)* |
| `src/care_fusion/losses.py` | Focal class-balanced + routing + counterfactual — Part D1 *(Phase 2)* |
| `src/care_fusion/train.py` | Vòng huấn luyện multi-seed, early-stop macro-F1 — Part D *(Phase 2)* |
| `src/care_fusion/baselines.py` | B0–B4 — Part E *(Phase 2)* |
| `src/care_fusion/evaluate.py` | Macro-F1 phân tầng, causal, faithfulness, bootstrap — Part F–G *(Phase 3)* |
| `configs/default.yaml` | Toàn bộ siêu tham số (Part H) |
| `notebooks/` | Notebook chạy 1-click trên Colab/Kaggle |
| `data/raw/` | CSV gốc (đã kèm) — dữ liệu dẫn xuất nằm trong `.gitignore` |

## 1) Chạy cục bộ (Windows, dev / smoke-test)

Yêu cầu: **Python 3.11**, **Java ≥ 8** (cho VnCoreNLP — đặt `JAVA_HOME`).

```powershell
# tạo môi trường
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# torch KHÔNG nằm trong requirements (để không phá bản GPU có sẵn của Colab/Kaggle).
# Cài torch trước cho máy local — chọn 1:
pip install torch --index-url https://download.pytorch.org/whl/cu124   # GPU NVIDIA
# pip install torch --index-url https://download.pytorch.org/whl/cpu   # chỉ CPU

pip install -r requirements.txt
pip install -e .
$env:JAVA_HOME = "C:\Program Files\Java\jdk-21"   # trỏ tới JDK của bạn

# Part A — tiền xử lý (tải model VnCoreNLP ở lần đầu)
python -m care_fusion.preprocess --config configs/default.yaml

# Part B — dựng tài nguyên (q_j + PMI graph; chỉ từ train)
python -m care_fusion.resources --config configs/default.yaml --steps q,pmi

# unit test
pytest -q
```

> RTX 3050 6GB chỉ đủ *smoke-test* (1 seed/1 epoch). Huấn luyện thật → dùng Colab/Kaggle (mục 2).

## 2) Chạy trên GPU (Google Colab / Kaggle) — khuyến nghị cho thực nghiệm thật

### Colab

1. Mở [`notebooks/CARE_Fusion_Colab.ipynb`](notebooks/CARE_Fusion_Colab.ipynb) trên Colab (badge ở đầu notebook).
2. **Runtime → Change runtime type → GPU**. Free = T4 (16GB); Pro chọn **A100/L4** để nhanh hơn.
3. Chạy lần lượt các cell: clone repo → cài deps → cài Java → preprocess → resources → train → evaluate.
4. Checkpoint được lưu ra **Google Drive** (`/content/drive/MyDrive/care-fusion/`) để không mất khi session ngắt.

### Kaggle

- Tạo Notebook mới, **Settings → Accelerator → GPU (T4×2 / P100)**, Internet **On**.
- Dán nội dung notebook Colab (Kaggle có sẵn Java). 30h GPU/tuần, session tới ~12h → hợp chạy nhiều seed.

### Ước lượng thời gian & chi phí (full protocol)

| Hạ tầng | Thời gian máy | Ghi chú |
|---|---|---|
| Colab **A100** (Pro) | ~7–13h | ~90–170 compute units ≈ **$10–20** (gói CU $0.10/CU) |
| Colab **T4** / Kaggle | ~20–28h | miễn phí, cần checkpoint vì hay ngắt |
| RTX 3050 cục bộ | — | chỉ smoke-test |

Mẹo giảm chi phí: chạy ablation **3 seed**, chỉ CARE-Fusion + baseline mạnh nhất chạy đủ **5 seed** cho kiểm định thống kê.

## 3) Tái lập (Part H)

- Cố định seed (`torch`, `numpy`, `random`, `cudnn.deterministic`).
- Mọi siêu tham số trong `configs/*.yaml`; ghi version thư viện vào `requirements.txt`.
- Tài nguyên dẫn xuất (`q_table.json`, `pmi_graph.json`, weak labels) tạo lại được bằng `resources.py` — **chỉ từ train**.
- Trích dẫn ViGoEmotions gốc; ánh xạ 27→6 nhóm + quy tắc ưu tiên mô tả trong protocol.

## Tiến độ

- [x] **Phase 1** — preprocess (A), resources `q_j` + PMI (B1, B3), unit test, smoke-test.
- [x] **Phase 2** — model CARE-Fusion (C), losses (D1), engine + train (D), baselines B0–B4 + OOF→weak labels (B2), ablation flags (−routing/−δ/−L_cf/−c_j/−GCN), harness `experiments.py` chạy cả ma trận. Smoke-test mọi variant + pilot trên 3050.
- [x] **Phase 3** — `evaluate.py`: F1 (macro/per-class/confusion/5-lớp), F2 phân tầng theo chế độ, F3 causal, F4 faithfulness, G (bootstrap CI + McNemar + Wilcoxon).
- [ ] **Chạy thật A100** — full data, max_len 256, 5 seed, dò λ → bảng số cho bài báo.

### Lệnh chạy đầy đủ (A100)
```bash
python -m care_fusion.preprocess --config configs/default.yaml
python -m care_fusion.resources  --config configs/default.yaml --steps q,pmi
python -m care_fusion.baselines  --config configs/default.yaml --emit-oof
python -m care_fusion.resources  --config configs/default.yaml --steps weak --ptext artifacts/p_text_oof.json
python -m care_fusion.experiments --config configs/default.yaml --out artifacts/checkpoints   # B0–B4 + CARE + ablations
python -m care_fusion.evaluate    --config configs/default.yaml --care-ckpt artifacts/checkpoints/CARE_full_seed13.pt --baseline B1_text --preds-dir artifacts/checkpoints/preds
```

## Limitations (minh bạch cho bài báo)
- **F5 (độ nhạy quy tắc tie-break 27→6):** đã hiện thực trong `scripts/f5_sensitivity.py`, dùng 3 file gốc ViGoEmotions (nhãn 27) để dựng lại nhãn theo quy tắc thay thế và đánh giá lại từ predictions đã lưu (không train lại). **31,69% mẫu là đa nhóm** (chịu tác động quy tắc ưu tiên) — xem `scripts/compute_multigroup.py`.
- **Cross-platform:** dataset không có trường nền tảng → không đánh giá cross-platform (đưa vào future work).
- **`neutral` cực thưa** (val 4 / test 6): báo cáo riêng + dùng macro-F1 5 lớp đông làm kết luận chính phụ trợ.
- **`−q` (lexicon phương Tây):** cần một emoji-sentiment lexicon ngoài để thay `q_j`; hiện để ngỏ (hook qua `q_table` thay thế).

## License & trích dẫn

Mã nguồn: (TBD). Vui lòng trích dẫn bài báo CARE-Fusion và dataset ViGoEmotions gốc khi sử dụng.
