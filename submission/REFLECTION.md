# Bài phản tư — Lab 22: căn chỉnh mô hình bằng DPO/ORPO

- **Tên:** Trần Trọng Chinh
- **Mã học viên:** 2A202602720
- **Khoá:** K4 · Track 3
- **Tier:** Kaggle T4 × 2 được cấp; mô hình được chạy trên `cuda:0`
- **Ngày:** 2026-10-08

> Số liệu DPO bên dưới được chép từ log của phiên Kaggle hoàn tất NB3. Notebook dừng ở bước lưu `dpo_metrics.json`: runner dùng tên `result` cho `shell.run_cell(...)`, ghi đè kết quả `trainer.train()`. Vì vậy file metrics và gói kết quả chưa được tạo. NB4 chưa chạy; các ô không có kết quả được ghi rõ, không ước lượng.

## Trạng thái

NB0–NB3 đã chạy. NB0 xác nhận custom DPO loss khởi tạo ở 0.6931 (log 2). NB1 hoàn tất SFT. NB2 lọc và chia dữ liệu preference thành 800 cặp train, 100 cặp held-out theo prompt không trùng nhau. NB3 hoàn tất 100/100 bước DPO và đánh giá held-out cuối. NB4 (sinh câu trả lời và chấm bằng reward model), đóng gói zip và các bonus chưa hoàn tất.

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Kaggle cấp Tesla T4 × 2; Unsloth ghi `Max memory: 14.562 GB` và xác nhận mô hình dùng `cuda:0`. Đây là mức bộ nhớ hiển thị của GPU, không phải số đo peak allocated của phiên train. |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned`; 1.000 mẫu, 1 epoch, 125 bước; loss cuối 1.3603. |
| Dữ liệu preference | `sailor2/sea-ultrafeedback-onpolicy` (lọc tiếng Việt); 800 train / 100 held-out; prompt giữa hai tập không giao nhau. |
| Đặc điểm độ dài NB2 | Chosen dài hơn rejected ở 65,9% cặp; median lần lượt 94 và 86 token. |
| DPO | β = 0.1; learning rate = 5e-6; 1 epoch; 100 bước; max length 768; loss sigmoid. |
| Giám khảo | Đã cấu hình hai reward model Skywork Qwen3-4B và Llama-3.2-3B; NB4 chưa chạy nên chưa có sanity accuracy. |
| Chi phí | Chạy bằng quota Kaggle; log không ghi chi phí tiền tệ. |

## 2. Kết quả DPO

| Chỉ số | Kết quả |
|---|---:|
| Thời gian NB3 | Log ghi 23:06 cho 100 bước train; tiến trình hoàn tất đánh giá cuối ở 24:16. |
| Training loss cuối | 0.6734; loss đầu tiên được log là 0.6931258. |
| Reward chosen cuối trên held-out | 0.443373 |
| Reward rejected cuối trên held-out | 0.353932 |
| Reward margin cuối trên held-out | 0.089441 |
| Reward accuracy cuối trên held-out | 0.730 |
| Chẩn đoán tự động | `INTENDED`; trung bình 3 điểm đánh giá cuối: chosen +0.435, rejected +0.347, margin +0.088. |
| Độ dài câu trả lời SFT → DPO | Chưa đo; NB4 chưa chạy. |
| Peak VRAM | Không có số peak allocated/reserved riêng trong log; chỉ có thông tin GPU 14.562 GB ở §1. |

## 3. Đọc đường reward

Ở lần đánh giá held-out cuối, reward của chosen là 0.443373 và của rejected là 0.353932, tạo margin dương 0.089441; reward accuracy là 0.730. Bộ chẩn đoán lấy trung bình ba điểm đánh giá cuối và báo chosen +0.435, rejected +0.347, margin +0.088, nên cả hai loại câu trả lời đều nhận reward cao hơn mốc tham chiếu nhưng chosen tăng nhiều hơn. Đây phù hợp với nhãn `INTENDED`: mô hình phân biệt chosen tốt hơn rejected. Nó không giống likelihood displacement, vốn có chosen reward âm trong khi rejected giảm nhanh hơn. Tuy nhiên, reward accuracy chỉ đo mức nhất quán với reward model trên held-out, không chứng minh người dùng sẽ thích câu trả lời hơn. Dù NB2 đã tách prompt, mình chưa kết luận về overfit vì runner lỗi trước khi xuất gói kết quả và chưa đối chiếu đầy đủ đường train với held-out trong báo cáo. Cũng chưa thể kiểm tra hiệu ứng độ dài: chosen dài hơn ở 65,9% cặp preference, còn độ dài câu trả lời SFT/DPO chưa được sinh và so sánh ở NB4.

## 4. So sánh SFT và SFT+DPO

NB4 chưa thực thi. Runner dừng khi lưu metrics sau NB3 với lỗi `AttributeError: 'ExecutionResult' object has no attribute 'training_loss'`: vòng lặp bootstrap đặt kết quả `shell.run_cell(...)` vào biến `result`, làm ghi đè biến kết quả huấn luyện cùng tên. Do đó chưa có 8 prompt cố định, 50 câu trả lời held-out, `side_by_side.jsonl`, `judge_summary.json`, sanity accuracy, win rate, khoảng tin cậy, position consistency, score-length Spearman hay ví dụ hữu ích/an toàn. Không có kết luận DPO thắng SFT ở phần đánh giá này.

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate / CI 95% | Cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| Held-out | Chưa chạy | — | — | — | — | — | — |
| Hữu ích (4 prompt cố định) | Chưa chạy | — | — | — | — | — | — |
| An toàn (4 prompt cố định) | Chưa chạy | — | — | — | — | — | — |

## 5. Đánh đổi theo β (bonus)

β-sweep với 0.05, 0.1 và 0.5 chưa chạy; chỉ có cấu hình baseline β = 0.1 ở §1–§2. Dự đoán của mình là β lớn làm policy bị ràng buộc chặt hơn với reference và có thể làm thay đổi hành vi ít hơn. Vì reward/margin cũng phụ thuộc β, margin số học có thể đổi mà không đồng nghĩa chất lượng xếp hạng tăng. Muốn kết luận cần chạy cùng held-out, so reward accuracy và trajectory, đồng thời giữ nguyên dữ liệu, seed, learning rate và số bước.

## 6. Một quyết định quan trọng nhất

Quyết định quan trọng nhất là dùng LoRA trên Kaggle T4 với 800 cặp preference để train và giữ riêng 100 cặp held-out theo prompt. Phương án thay thế là full fine-tuning hoặc xin tier GPU lớn hơn. Với mô hình 4B và bộ nhớ T4, LoRA giảm số tham số cần cập nhật; cấu hình thực tế đã chạy trên `cuda:0`, trong khi vẫn giữ được tập held-out độc lập để theo dõi reward. Kết quả xác nhận pipeline SFT và DPO có thể chạy hết phần train: SFT dùng 1.000 mẫu trong 125 bước với loss cuối 1.3603; DPO dùng β=0.1, learning rate 5e-6, chiều dài tối đa 768 và hoàn thành 100 bước. Held-out reward accuracy đạt 0.730 với margin cuối 0.089441, còn chẩn đoán trên ba lần eval cuối là `INTENDED`. Điều mình chú ý là reward của rejected cũng tăng (+0.347 theo chẩn đoán), thay vì giảm; DPO cải thiện khoảng cách chủ yếu vì chosen được reward cao hơn. Đây mới là kết quả theo reward model, chưa phải bằng chứng chất lượng đối thoại tăng, vì runner lỗi khiến NB4 chưa chạy. Nếu làm lại, mình sẽ sửa bootstrap để tên biến kết quả cell không đè `result` của notebook, ghi metrics ngay sau train và chạy NB4 trước khi thử bonus. Sau đó mới đánh giá độ dài và độ tin cậy của hai reward model trên tiếng Việt.

## 7. Bộ đo chuẩn (bonus NB6)

Chưa chạy. Không có kết quả IFEval, GSM8K hoặc Global-MMLU-vi; vì vậy chưa thể tính sai số chuẩn hay kết luận có alignment tax.

## 8. Biến thể loss (bonus NB3b)

Chưa chạy RPO, DPO-norm, LD-DPO hoặc ORPO. Kết quả trong báo cáo chỉ thuộc baseline DPO sigmoid ở §1–§3.

## 9. GRPO (bonus NB7)

Chưa chạy GRPO; chưa có số đo trước/sau hoặc ước lượng sai số.

## Điều bất ngờ nhất

Reward của rejected cũng tăng ở các điểm eval cuối. Margin vẫn dương vì chosen tăng nhiều hơn, nhưng kết quả này nhắc mình phải xem cả hai đường reward và kiểm tra độ dài câu trả lời trước khi diễn giải DPO là cải thiện hữu ích.

## Bonus chưa thực hiện

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng API khác họ (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md`
