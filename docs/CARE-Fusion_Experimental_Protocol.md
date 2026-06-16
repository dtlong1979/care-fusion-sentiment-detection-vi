# CARE-Fusion — Thiết kế thuật toán thử nghiệm (bản triển khai)

Tài liệu này mô tả toàn bộ quy trình thực nghiệm để hiện thực hóa mô hình **CARE-Fusion** trên tập dữ liệu dẫn xuất từ ViGoEmotions (6.813 mẫu, 6 nhóm cảm xúc). Mọi tài nguyên (phân bố marker, nhãn chế độ yếu, đồ thị PMI) được dựng **chỉ từ tập train** để tránh rò rỉ. Ký hiệu mô hình theo file *Methods*; tài liệu này tập trung vào *cách làm*.

Nhãn: `{positive, sadness, anger, fear, interest, neutral}` → `C = 6`. Lưu ý độ thưa của `neutral` (49 mẫu) được xử lý riêng ở Phần F.

---

## Phần A — Tiền xử lý dữ liệu

Đầu vào: `vigoemotions_text_emoji_6groups.csv` (cột `source_split, source_id, text, emoji_type, emotion_group`). Giữ nguyên split gốc.

**A1. Chuẩn hóa Unicode.** Áp dụng NFC cho mọi `text` (sửa các dòng bị tách dấu kiểu "thâ ̣ thi ̀"). Chuẩn hóa khoảng trắng, hạ dấu lặp ký tự kéo dài tùy chọn (giữ lại vì có thể mang cường độ).

**A2. Trích xuất marker (emoji + emoticon).** Tách tín hiệu phi ngôn ngữ khỏi text, **giữ thứ tự xuất hiện** và **đếm lặp**:
- *Emoji Unicode*: bắt bằng dải Unicode emoji (`\U0001F000-\U0001FAFF`, `\u2600-\u27BF`, `\u2764`, ...). Dùng thư viện `emoji` để chuẩn.
- *Emoticon văn bản*: dùng danh sách + **longest-match** (khớp dài trước ngắn) để tránh `:))` bị cắt thành `:)`. Danh sách tối thiểu: `:))`, `=))`, `:)`, `:((`, `:(`, `:v`, `:3`, `<3`, `^^`, `T_T`, `:'(`, `:D`, `:P`, `;)`, `:|`, `>:(`. Bổ sung biến thể kéo dài (`:)))`, `:))))`) bằng regex nhóm lặp.

Kết quả mỗi mẫu:
- `text_clean`: text đã gỡ marker (đưa vào PhoBERT).
- `marker_seq`: list marker theo thứ tự, ví dụ `[":))", ":))"]`.
- `marker_intensity`: dict đếm lặp, ví dụ `{":))": 2}`; cường độ `c_j = log(1 + count_j)`.
- `n_marker`.

> Lưu ý: ~27% mẫu là emoticon → **không** phụ thuộc lexicon Unicode phương Tây (xem Phần B1).

**A3. Word segmentation cho PhoBERT (bắt buộc).** PhoBERT yêu cầu văn bản đã *tách từ*. Dùng `py_vncorenlp` (RDRSegmenter) hoặc VnCoreNLP để segment `text_clean` trước khi tokenize. Bỏ qua bước này là lỗi triển khai phổ biến làm tụt mạnh hiệu năng.

**A4. Tokenize & cắt độ dài.** Tokenize bằng tokenizer của `vinai/phobert-base`, `max_length = 256` (text dài tới 906 ký tự nên cần truncation), padding theo batch.

**A5. Encode nhãn.** Ánh xạ 6 nhóm → chỉ số `0..5`. Lưu `label2id`/`id2label`.

**A6. Kiểm tra sạch.** Loại mẫu sau khi gỡ marker mà `text_clean` rỗng (nếu có) hoặc ghi cờ `text_empty` để phân tích riêng. Không xóa mẫu khỏi test một cách tùy tiện.

---

## Phần B — Dựng tài nguyên (CHỈ từ train)

**B1. Phân bố cảm xúc thực nghiệm của marker `q_j`.** Thay cho ESR/Emoji-Dis (không hợp tiếng Việt, không phủ emoticon), ước lượng trực tiếp:
```
for mỗi marker m trong train:
    đếm count(m, e) = số mẫu train chứa m và có nhãn e
    q_m[e] = (count(m, e) + α) / (Σ_e count(m, e) + α·C)     # Laplace smoothing, α=1
```
- Marker có tần suất train < τ (vd τ=5): gán `q_m` = phân bố nhãn toàn cục của train (back-off) và đánh cờ `low_freq` để model dùng nhánh embedding học được thay vì tin vào `q_m`.
- Lưu `q` thành bảng tra; marker ngoài bảng (chỉ xuất hiện ở val/test) → back-off toàn cục.

**B2. Nhãn chế độ yếu (weak regime labels) cho train.** Cần `p_text` (xác suất cảm xúc *chỉ từ text*). Để tránh lạc quan, tạo `p_text` bằng **dự đoán out-of-fold**: chia train thành 5 fold, huấn luyện baseline text-only PhoBERT trên 4 fold, dự đoán fold còn lại → softmax `p_text` cho mọi mẫu train. (Mô hình text-only này dùng lại làm baseline ở Phần E.)

Với mỗi marker `j` trong mẫu:
```
δ_j = JSD(p_text ‖ q_j)
pol(p_text) = argmax-polarity theo nhóm:  POS={positive,interest}, NEG={sadness,anger,fear}, NEU={neutral}
pol(q_j)    = tương tự trên q_j
gán nhãn chế độ yếu r̃_j:
    nếu δ_j < δ_low                          → redundancy
    nếu δ_j ≥ δ_high và pol(p_text)=NEU/yếu  → complementarity
    nếu δ_j ≥ δ_high và pol(p_text) ≠ pol(q_j) (đối cực) → conflict
    còn lại                                   → complementarity (mặc định)
```
Ngưỡng `δ_low, δ_high` chọn theo phân vị của phân phối `δ` trên train (vd 33% và 66%). Đây là *nhãn yếu*, chỉ giám sát nhánh routing, không phải nhãn cuối.

**B3. Đồ thị PMI marker–cảm xúc (chống rò rỉ).** Node = {các marker} ∪ {6 nhóm cảm xúc}. Cạnh = PPMI trên đồng xuất hiện *train*:
```
PPMI(a,b) = max(0, log( P(a,b) / (P(a)·P(b)) ))
```
Khởi tạo embedding node: marker → `q_m` chiếu tuyến tính; nhóm cảm xúc → embedding học được. GCN 2 lớp (công thức chuẩn hóa Laplacian trong file Methods). Đồ thị là tham số mô hình, **không chứa mẫu test**.

---

## Phần C — Forward pass (mức triển khai)

Theo đúng kiến trúc Methods, thay "emoji" bằng "affective marker":
1. `H = PhoBERT(text_clean_segmented)` (giữ chuỗi token), `p_text = softmax(W_t·attn_pool(H))`.
2. Mỗi marker `j`: `e_j = W_e[ℓ_j ‖ c_j]` với `ℓ_j` = `q_j` (hoặc embedding học nếu `low_freq`); lấy `q_j` cho `δ_j = JSD(p_text‖q_j)`.
3. `g_j = LayerNorm(e_j + GatedCrossAttention(e_j, H))`.
4. `r_j = softmax(W_r[g_j ‖ e_j ‖ δ_j ‖ c_j]) ∈ Δ²` (redundancy/complementarity/conflict).
5. `z_j = r_j^ρ F_ρ + r_j^κ F_κ + r_j^χ F_χ`; `z = Σ_j c_j z_j / Σ_j c_j` (hoặc `z_∅` nếu `n_marker=0`).
6. `z̃ = GCN_enrich(z, A)`; `ŷ = softmax(W_o z̃ + b_o)`.

Gợi ý chiều: `d_t=768` (PhoBERT-base), `d_e=128`, `d=256`. Multi-head cross-attention: 4 heads.

---

## Phần D — Thuật toán huấn luyện

**D1. Hàm mất mát.**
- Phân loại: **class-balanced focal loss** thay cho CE thường (xử lý mất cân bằng): trọng số lớp `β_c = (1−γ)/(1−γ^{n_c})` (Cui et al.) hoặc nghịch tần suất, kết hợp focal `(1−p)^2`.
- Routing (giám sát yếu): `L_route = mean_j CE(r_j, r̃_j)`.
- Phản thực: với mỗi mẫu tạo counterfactual (xóa marker và/hoặc đảo cực marker), tính `ŷ'`; `L_cf = ‖ KL(ŷ‖ŷ') − s_j ‖²` với `s_j` là độ dịch kỳ vọng theo chế độ (redundancy→~0, complementarity/conflict→lớn).
- Tổng: `L = L_cls + λ1·L_route + λ2·L_cf`. Dò `λ1, λ2 ∈ {0.1, 0.3, 0.5, 1.0}` trên val.

**D2. Tối ưu.** AdamW, **discriminative LR**: PhoBERT `2e-5`, các head/nhánh mới `1e-3`. Warmup 10% bước, linear decay, weight decay `0.01`, grad clip `1.0`. Batch 16–32, tối đa 20 epoch, **early stopping theo macro-F1 trên val** (patience 3). Mixed precision (fp16) tùy chọn.

**D3. Nhiều seed.** Chạy lại toàn bộ với **≥5 seed** `{13, 42, 123, 2024, 7}`. Lưu mọi checkpoint tốt nhất theo val. Báo cáo mean ± std (Phần G).

**D4. Pseudo-code vòng huấn luyện.**
```
build q (B1), weak labels r̃ (B2), PMI graph A (B3)   # train only
for seed in SEEDS:
  set_seed(seed)
  for epoch in 1..E:
    for batch in train_loader:
      H, p_text = text_encoder(batch.text)
      markers    = encode_markers(batch, q)           # e_j, c_j, δ_j
      g, r       = cross_attn(markers, H), router(...)
      z          = regime_fusion(r, H, g)
      z_tilde    = gcn_enrich(z, A)
      y_hat      = classifier(z_tilde)
      cf         = make_counterfactual(batch)          # xóa/đảo marker
      y_hat_cf   = forward(cf)
      loss = focal_cb(y_hat, y) + λ1*ce(r, r̃) + λ2*cf_loss(y_hat, y_hat_cf, r)
      loss.backward(); clip; opt.step(); sched.step()
    evaluate(val); early_stop_on(macro_F1_val)
  evaluate(test) with best checkpoint
```

---

## Phần E — Baseline & Ablation (cùng dataset, cùng split, cùng seed)

**Baseline (tái cài đặt, không lấy số từ bài khác):**
| # | Mô hình | Mục đích |
|---|---------|----------|
| B0 | Majority class | sàn dưới |
| B1 | PhoBERT text-only | đo đóng góp của marker |
| B2 | PhoBERT + marker concat (không trọng số) | baseline fusion cũ |
| B3 | Scalar gated fusion (α vô hướng — "flexible weighting" cũ) | chứng minh routing > gating |
| B4 | Cross-attention fusion (không regime routing) | tách đóng góp của routing |
| B5 | (tùy chọn) LLM few-shot | tham chiếu thời LLM |
| ★ | **CARE-Fusion (đầy đủ)** | mô hình đề xuất |

**Ablation của CARE-Fusion:**
- − regime routing (gộp 1 toán tử fusion duy nhất)
- − đầu vào congruence `δ`
- − loss phản thực `L_cf`
- − `q_j` thực nghiệm (thay bằng lexicon phương Tây) → chứng minh việc thích nghi tiếng Việt là cần thiết
- − cường độ `c_j` (bỏ thông tin lặp marker)
- − đồ thị PMI/GCN
- chỉ-emoji vs chỉ-emoticon (kiểm tra mô hình có khai thác được cả hai)

---

## Phần F — Giao thức đánh giá

**F1. Metric chính.** **Macro-F1** (vì mất cân bằng nặng; *không* dùng accuracy làm chính — đoán toàn `positive` đã ~48%). Báo kèm: weighted-F1, **F1 từng lớp**, confusion matrix. Với `neutral` (49 mẫu): báo cáo *riêng* kèm cảnh báo low-power; đề xuất metric chính phụ trợ là **macro-F1 trên 5 lớp đông** để kết luận không bị nhiễu bởi lớp gần như rỗng.

**F2. Đánh giá phân tầng theo chế độ (kết quả chủ lực — bằng chứng novelty).** Gán mỗi mẫu test một chế độ:
- *Proxy tự động*: dùng `δ` (B2) trên test để chia 3 tầng redundancy/complementarity/conflict.
- *Tập xác minh người*: gán tay chế độ cho ~300 mẫu test (đặc biệt nhóm conflict/mỉa mai như `:))`, `:v`) để có một sub-evaluation đã kiểm chứng; báo cáo độ khớp proxy–người.
Báo **macro-F1 theo từng tầng**. Kỳ vọng: CARE-Fusion *vượt rõ baseline ở tầng conflict & complementarity*, ngang ở tầng redundancy. Đây là bảng thuyết phục reviewer rằng cải thiện đến từ đúng cơ chế.

**F3. Probing nhân quả (causal).** Với mỗi mẫu test: sinh phiên bản (xóa marker / đảo cực marker / tăng cường độ). Đo tỉ lệ đổi dự đoán và *liệu `r_j` mô hình suy ra có dịch đúng hướng*. Định nghĩa **Causal Sensitivity Score**; so CARE-Fusion với B3 (gating vô hướng) → kỳ vọng phản ứng có cấu trúc hơn.

**F4. Độ trung thực của trọng số (faithfulness, kiểu ERASER).** *Comprehensiveness*: xóa marker được gán trọng số/`r` cao → độ tụt prob của lớp dự đoán. *Sufficiency*: chỉ giữ marker đó → prob còn lại. Trọng số "thật sự được dùng" nếu comprehensiveness cao.

**F5. Phân tích độ nhạy quy tắc tiền xử lý (bắt buộc cho Phương án 1).**
- *Tỉ lệ đa nhóm*: báo cáo % mẫu bị quy tắc ưu tiên `anger>sadness>fear>positive>interest>neutral` kích hoạt (mẫu có ≥2 nhóm sau ánh xạ).
- *Tie-break thay thế*: dựng lại nhãn bằng một quy tắc khác (vd ưu tiên theo nhãn confidence cao nhất của bộ gốc, hoặc loại bỏ mẫu đa cực), chạy lại CARE-Fusion + baseline mạnh nhất → chứng minh thứ hạng mô hình **ổn định**, kết luận không phụ thuộc lựa chọn ưu tiên.

> Ghi chú: dataset không có trường nền tảng (platform) → *không* chạy được đánh giá cross-platform; nêu rõ trong Limitations và để vào future work.

---

## Phần G — Kiểm định thống kê

- Mỗi cấu hình chạy **5 seed**; báo cáo **mean ± std** cho mọi metric.
- **Khoảng tin cậy 95%** bằng bootstrap (resample test 1.000 lần) cho macro-F1.
- **Kiểm định ý nghĩa** CARE-Fusion vs baseline mạnh nhất:
  - giữa các seed: Wilcoxon signed-rank trên macro-F1;
  - trên nhãn dự đoán (một test cố định): **McNemar's test** hoặc bootstrap paired test;
  - báo cáo p-value và **effect size**.
- Mọi tuyên bố "vượt" phải kèm CI + p-value; chênh lệch < ~1% không có ý nghĩa thống kê thì trình bày trung thực là "tương đương".

---

## Phần H — Cấu hình & tái lập

**Bảng siêu tham số đề xuất (chốt sau khi dò trên val):**
| Tham số | Giá trị |
|---|---|
| PhoBERT | `vinai/phobert-base` |
| max_length | 256 |
| d_t / d_e / d | 768 / 128 / 256 |
| cross-attn heads | 4 |
| LR (PhoBERT / head) | 2e-5 / 1e-3 |
| batch size | 16–32 |
| epochs / patience | ≤20 / 3 |
| optimizer | AdamW, wd=0.01, warmup 10%, clip 1.0 |
| loss weights λ1, λ2 | dò {0.1,0.3,0.5,1.0} |
| δ_low / δ_high | phân vị 33% / 66% train |
| smoothing α (q_j) | 1.0 |
| seeds | 13, 42, 123, 2024, 7 |

**Tái lập:** cố định seed (torch, numpy, random, `cudnn.deterministic`); lưu file cấu hình YAML; ghi version thư viện (transformers, torch, py_vncorenlp, emoji); công bố code + tài nguyên dẫn xuất (`q`, weak labels, PMI graph) trên GitHub/Zenodo; trích dẫn ViGoEmotions gốc và mô tả minh bạch ánh xạ 27→6 + quy tắc ưu tiên.

---

## Trình tự triển khai gợi ý (thứ tự code)

1. Module tiền xử lý (A1–A6) + unit test trích marker (kiểm `:))`, `:v`, `<3`, emoji).
2. Baseline B1 text-only (đồng thời tạo `p_text` out-of-fold cho B2 weak labels).
3. Dựng tài nguyên B1–B3, lưu ra đĩa.
4. Model CARE-Fusion (Phần C) + loss (D1).
5. Vòng train/eval (D2–D4), early stopping theo macro-F1 val.
6. Baseline B0,B2,B3,B4 (+B5 tùy chọn) + ablation (Phần E).
7. Đánh giá F1–F5, kiểm định G.
8. Đóng gói tái lập (Phần H).
