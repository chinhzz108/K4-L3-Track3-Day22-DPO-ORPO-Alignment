# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

- **Tên:** Trần Trọng Chinh
- **Mã học viên:** 2A202602720
- **Khoá:** K4 · Track 3
- **Tier đã chạy:** Kaggle T4 × 2 được cấp; lab dùng GPU 0
- **Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

**Trạng thái bản nháp:** NB0, NB1 và NB2 đã hoàn thành. NB3 đang huấn luyện DPO; NB4 và phần đóng gói kết quả sẽ cập nhật khi run Kaggle kết thúc. Các mục ghi “chờ log” chưa có số liệu cuối và không được suy đoán.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Tesla T4 × 2 được cấp; Unsloth xác nhận model vừa trên `cuda:0`, tối đa hiển thị 14.562 GB |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned`, 1.000 mẫu, 1 epoch, 125/125 bước; loss cuối 1.3603 |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (Vietnamese), 800 train / 100 held-out; không trùng prompt |
| Chosen dài hơn rejected (NB2) | 65,9%; median chosen 94 token, rejected 86 token |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1; đang chạy 100 bước trên 800 cặp |
| Giám khảo | Cấu hình hội đồng local: `Skywork-Reward-V2-Qwen3-4B` và `Skywork-Reward-V2-Llama-3.2-3B`; sanity accuracy chờ NB4 |
| Chi phí | Kaggle notebook GPU, dùng quota của tài khoản; log không ghi số tiền |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | Chờ log hoàn tất |
| VRAM cao nhất | Chờ log hoàn tất |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | Chờ `adapters/dpo/dpo_metrics.json` |
| Độ chính xác reward trên held-out | Chờ `adapters/dpo/dpo_metrics.json` |
| Margin trên held-out | Chờ `adapters/dpo/dpo_metrics.json` |
| Chẩn đoán tự động (`diagnosis`) | Chờ log hoàn tất |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | Chờ NB4 |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

NB3 chưa xuất đường reward cuối tại thời điểm ghi bản nháp. Sau khi train xong, cần đọc riêng `rewards/chosen` và
`rewards/rejected` trên train và held-out trong ảnh `03-dpo-reward-curves.png`. Kết luận sẽ dựa vào việc chosen tăng,
rejected giảm, hay rejected giảm nhanh hơn chosen; chỉ nhìn loss giảm không đủ để kết luận mô hình học đúng sở thích.
Sau đó đối chiếu hai tập để xem xu hướng held-out có đi cùng train hay có dấu hiệu overfit, rồi so với trường
`diagnosis` trong metrics JSON. Phần này sẽ được thay bằng diễn giải số liệu thực tế khi NB3 hoàn tất.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | Chờ NB4 | Chờ NB4 | Chờ NB4 | Chờ NB4 | Chờ NB4 | Chờ NB4 | Chờ NB4 |
| hữu ích — helpfulness (4) | Chờ NB4 | Chờ NB4 | Chờ NB4 | Chờ NB4 | Chờ NB4 | Chờ NB4 | Chờ NB4 |
| an toàn — safety (4) | Chờ NB4 | Chờ NB4 | Chờ NB4 | Chờ NB4 | Chờ NB4 | Chờ NB4 | Chờ NB4 |

Giám khảo, sanity accuracy, score-length Spearman / position consistency: chờ NB4.

NB4 sẽ chấm 50 prompt held-out và 8 prompt cố định bằng hội đồng reward model. Chưa có đầu ra để xác định khoảng tin cậy
có chứa 0.5 hay không, độ tin cậy trên tiếng Việt, thiên vị độ dài, hoặc mức đồng ý giữa Qwen3 và Llama. Sau khi sinh
xong, báo cáo sẽ ghi các thống kê trong `judge_summary.json` và đối chiếu hai ví dụ cụ thể trong `side_by_side.jsonl`:
một prompt hữu ích và một prompt an toàn. Không đưa kết luận thắng/thua trước khi các file này được tạo.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | |
| 0.1 | | | | |
| 0.5 | | | | |

Chưa chạy bonus β-sweep. Dự đoán: β lớn hơn sẽ tăng mức phạt khi policy lệch khỏi reference, nên cập nhật có thể thận trọng hơn.
Vì reward được nhân với β, margin đo được cũng có thể lớn hơn dù hành vi thay đổi ít hơn. Cần so sánh trên cùng tập held-out
và xem cả reward accuracy để tách hiệu ứng thang đo khỏi chất lượng xếp hạng.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

Quyết định quan trọng ở bản chạy này là dùng LoRA trên tier T4 với 800 cặp preference và giữ riêng 100 cặp held-out.
Phương án thay thế là tinh chỉnh toàn bộ mô hình hoặc dùng tier lớn hơn; với bộ nhớ của T4, LoRA giảm số tham số cần
cập nhật và cho phép hoàn thành bài lab trên một GPU. Cấu hình ghi nhận β=0.1, learning rate 5e-6, một epoch, chiều dài
tối đa 768 token. Số liệu hiện có xác nhận pipeline SFT chạy được: 1.000 ví dụ, 125 bước, loss cuối 1.3603; bộ preference
đã chia thành 800/100 prompt không giao nhau. Tập này có chosen dài hơn rejected ở 65,9% cặp, nên khi đọc kết quả DPO
cần kiểm tra xem mô hình có học chất lượng hay chỉ tăng độ dài. Tại thời điểm viết, DPO chưa kết thúc nên chưa thể nói
β hoặc learning rate có cải thiện mô hình; kết luận sẽ được cập nhật từ reward curves, held-out accuracy và hội đồng judge.
Nếu làm lại, trước hết tôi sẽ giữ nguyên held-out để so sánh công bằng, sau đó mới thử β khác trên cùng cấu hình và ghi
riêng từng lần chạy, tránh chọn tham số chỉ vì train loss giảm.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

_Δ nào vượt ~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?_

_Trả lời ở đây._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | | | | |
| RPO | | | | |
| DPO-norm | | | | |
| LD-DPO | | | | |
| ORPO | | | | |

_Biến thể nào thay đổi độ dài nhiều nhất, và vì sao (dựa vào công thức loss)?_

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | _<... / ... (n=...)>_ |
| Sai số chuẩn ≈ √(p(1−p)/n) | _<...>_ |

_Thành phần reward nào tăng trước (đúng định dạng hay đúng đáp án)? Chênh lệch có vượt nhiễu không?_

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

_(Tuỳ chọn, 1–3 câu)_
