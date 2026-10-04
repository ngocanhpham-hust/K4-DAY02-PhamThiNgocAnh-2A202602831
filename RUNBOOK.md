# Cách chạy bài làm DeepWeeds

## Chuẩn bị và kiểm tra

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
PYTHONPATH=code:. python -m py_compile code/*.py
```

Đặt ảnh ở `data/images/` và CSV nguyên bản ở `data/labels/`. Không sửa hoặc chia lại CSV. MD5 của `images.zip` phải là `b7b30f96d466fba86016aa5a26606e0f`.

Mở `code/lab_day2.ipynb` để chạy EDA và các vòng thí nghiệm. CLI tương đương:

```bash
python code/train.py --set exp_id=B01 backbone=resnet50 seed=0
```

Kết quả nằm ở `runs/<exp_id>/seed<seed>/`; dự đoán val ở `predictions/`. Test mặc định bị khóa và chỉ tạo khi truyền `save_test_predictions=true` ở bước chung kết.

## Trình tự

1. Đo một epoch cho từng backbone và chốt ngân sách GPU.
2. Chạy ít nhất 5 backbone bằng cùng T00.
3. Chọn 1–2 backbone trên val; chạy ít nhất 3 trục ablation và một cấu hình kết hợp.
4. So sánh ít nhất 4 kỹ thuật inference trên val, fit temperature trên val và đo latency.
5. Chốt hoàn toàn trên val; sau đó chạy F01 và T00 với seed 0, 1, 2 cùng `save_test_predictions=true`.
6. Chạy `eval.py score` và `eval.py grade`, rồi điền workbook/báo cáo bằng log thật.

Không commit dữ liệu hoặc checkpoint lớn. Không dùng test để chọn cấu hình.
