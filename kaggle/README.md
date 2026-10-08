# Chạy Lab 22 trên Kaggle

1. Tải `Lab22_DPO_T4.ipynb` lên Kaggle Notebooks. Nếu cần, có thể nhập bản Jupytext `Lab22_DPO_T4.py`
   qua **File → Import Notebook**.
2. Chọn accelerator GPU T4 (T4×2 cũng được) và bật Internet.
3. Chạy cell từ trên xuống. Pipeline bắt buộc NB0–NB4 huấn luyện SFT, chuẩn bị preference data,
   train DPO, rồi đánh giá SFT và SFT+DPO.
4. Cell cuối ghi gói kết quả nhỏ tại `/kaggle/working/lab22-results.zip`. Tải gói này về và giải nén
   vào repo. Nó chứa adapter metadata và metrics, parquet train/held-out, kết quả judge và bốn ảnh
   bắt buộc; trọng số mô hình và thông tin cá nhân không được đưa vào gói.

Nếu thay đổi `notebooks/*.py` hoặc `lab22/*.py`, chạy `python scripts/build_kaggle.py` để tạo lại
notebook. Chạy `python scripts/build_kaggle.py --check` để kiểm tra notebook có đồng bộ với mã nguồn không.

Nếu dùng bootstrap Python để chạy notebook cell-by-cell, giữ kết quả `shell.run_cell(...)` trong biến riêng như `_cell_result`; không đặt vào `result`, vì cell train DPO dùng tên này cho metrics của trainer.
