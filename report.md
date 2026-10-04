# Báo cáo DeepWeeds — Thực nghiệm huấn luyện và suy luận

> Trạng thái: khung báo cáo. Thay mọi ô `[CHỜ LOG THẬT]` từ `runs/`, `predictions/` và `eval.py`; không điền số giả.

## 1. Tóm tắt

Bài toán phân loại 9 lớp DeepWeeds được dùng để so sánh backbone, công thức huấn luyện và kỹ thuật suy luận. Fold 0 nguyên bản được giữ cố định; train chỉ cập nhật trọng số, validation quyết định toàn bộ cấu hình, test chỉ chạy một lần trên mỗi seed sau khi chốt. Cấu hình tốt nhất và kết quả test: **[CHỜ LOG THẬT]**.

## 2. Dữ liệu và thiết lập

- Số ảnh train/val/test và phân bố lớp: [CHỜ EDA].
- Giao ba cặp split: [CHỜ `split.json`], kỳ vọng đều bằng 0; hợp kỳ vọng 17.509.
- T00: 12 epoch, batch 64, AdamW, LR backbone `1e-4`, LR head `1e-3`, weight decay `0.05`, warmup 1 epoch + cosine.
- Thiết bị, phiên bản thư viện, thời gian/epoch: [CHỜ LOG THẬT].
- Chỉ số chính: macro-F1; kèm top-1, balanced accuracy, ECE và chỉ số theo lớp.

## 3. So sánh backbone

[CHÈN bảng Backbones và biểu đồ F1–latency từ kết quả thật. Nêu hội tụ, overfit, tham số và GMAC.]

Backbone được chọn: **[CHỜ VAL]**, vì **[lý do định lượng]**. Kết quả một seed chỉ là sàng lọc.

## 4. Công thức huấn luyện

[CHÈN bảng ablation: mỗi hàng chỉ đổi một yếu tố, Δ so với T00 và so với std.]

Nếu `|Δ| < std`, kết luận là “không phân biệt được”.

## 5. Suy luận

[CHÈN I00 và ít nhất 4 phương pháp; macro-F1/top-1/ECE/p50/p95/p99/chi phí tương đối.]

Temperature `T=[CHỜ VAL]` được fit duy nhất trên validation và giữ nguyên cho test. Accuracy trước/sau scaling phải không đổi.

## 6. Cấu hình tốt nhất và test

- Cấu hình tái lập: [CHỜ CHỐT TRÊN VAL].
- F01 test, mean ± sample std qua seed 0/1/2: [CHỜ `eval.py score`].
- Baseline T00 cùng seed: [CHỜ `eval.py score`].
- Recall Chinee Apple và Snake Weed: [CHỜ `eval.py`].
- Ma trận nhầm lẫn và ảnh lỗi: [CHỜ TEST].

## 7. Kết luận và khuyến nghị

[CHỜ LOG THẬT]. Với robot có ngân sách p95 30–100 ms, chỉ khuyến nghị cấu hình đã đo thật trên phần cứng đích.

## 8. Hạn chế và việc tiếp theo

Thí nghiệm mới dùng một fold và số seed hữu hạn. Split gốc ngẫu nhiên, không theo địa điểm, nên test có thể lạc quan khi gặp địa điểm, mùa, ánh sáng hoặc góc chụp khác. Cần kiểm chứng thêm bằng split theo địa điểm và dữ liệu thực địa.

## 9. Phụ lục

Danh sách `exp_id`, cấu hình đầy đủ và link notebook: [CHỜ HOÀN THIỆN].
