# DATH-AI-Text

Dự án phân loại văn bản tiếng Anh theo 14 nhãn của bộ dữ liệu DBpedia-14
(`fancyzhx/dbpedia_14`). Dự án so sánh ba cách tiếp cận:

- **TF-IDF + Linear SVM**: vectorizer TF-IDF tự cài đặt kết hợp `LinearSVM`.
- **BERT Base**: fine-tune `bert-base-uncased` cho phân loại chuỗi.
- **GPT-2**: fine-tune `openai-community/gpt2` cho phân loại chuỗi.

Các script huấn luyện dùng trường `content` của dữ liệu. Phần đánh giá ghi
metrics, báo cáo theo từng lớp và confusion matrix. Demo web cục bộ cho phép
phân loại văn bản và so sánh các chỉ số đã lưu.

## Cấu trúc thư mục

```text
.
├── app.py                       # Server HTTP cho web demo và API
├── app.js                       # Tương tác giao diện và gọi API
├── index.html                   # Giao diện demo
├── style.css                    # Kiểu dáng giao diện
├── requirements.txt             # Thư viện Python cần cài thêm
├── BERT/
│   ├── load_train.py            # Huấn luyện BERT; lưu checkpoint
│   ├── evaluate_model.py        # Đánh giá checkpoint BERT trên tập test
│   └── results_dbpedia/         # Checkpoint và kết quả đánh giá
├── GPT/
│   ├── load_train_gpt2.py       # Huấn luyện GPT-2; lưu checkpoint và model
│   ├── evaluate_model_gpt2.py   # Đánh giá GPT-2 trên tập test
│   └── results_dbpedia_gpt2/    # Checkpoint và kết quả đánh giá
├── SVM/
│   ├── tfidf_svm.py             # Huấn luyện và đánh giá TF-IDF + Linear SVM
│   ├── test_tfidf_svm.py        # Unit test cho vectorizer và metric
│   └── results_dbpedia_svm/     # Model SVM và kết quả đánh giá
└── test_inputs_by_label/        # Văn bản mẫu theo từng nhãn
```

Các thư mục `results_*` được tạo hoặc cập nhật khi chạy huấn luyện/đánh giá.
Dataset DBpedia-14 và model nền từ Hugging Face có thể cần được tải ở lần chạy
đầu tiên.

## Cài đặt

Yêu cầu Python 3.10 trở lên và driver NVIDIA tương thích nếu dùng CUDA.
Từ thư mục gốc dự án, tạo và kích hoạt môi trường ảo trên PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Cài PyTorch có hỗ trợ CUDA theo cấu hình máy (ví dụ lệnh CUDA 13.4):

```powershell
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu134
```

Sau đó cài các thư viện còn lại:

```powershell
python -m pip install -r requirements.txt
```

Kiểm tra PyTorch nhận GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Nếu kết quả là `False`, kiểm tra driver NVIDIA và phiên bản CUDA/PyTorch đã cài.

## Huấn luyện và lấy thông số đánh giá

Chạy các lệnh từ thư mục gốc dự án. Lần đầu chạy có thể tải DBpedia-14 và model
nền từ Hugging Face.

### TF-IDF + Linear SVM

```powershell
python SVM\tfidf_svm.py
```

Lệnh trên dùng toàn bộ tập train và test. Để chạy thử nhanh trên mẫu nhỏ:

```powershell
python SVM\tfidf_svm.py --train-limit 2000 --test-limit 500
```

### BERT

```powershell
python BERT\load_train.py
python BERT\evaluate_model.py
```

### GPT-2

```powershell
python GPT\load_train_gpt2.py
python GPT\evaluate_model_gpt2.py
```

Các script đánh giá in accuracy, precision, recall và F1 trên tập test; kết quả
được lưu trong thư mục `results_*` tương ứng, gồm `metrics.json`,
`classification_report_test.txt`, `confusion_matrix_test.csv` và
`confusion_matrix_test.png` (tùy script). Huấn luyện Transformer có thể mất
nhiều thời gian và cần GPU cùng bộ nhớ phù hợp.

Chạy unit test của SVM bằng:

```powershell
python -m unittest discover -s SVM -p "test_*.py"
```

## Chạy web demo

Model SVM và checkpoint/model Transformer cần có sẵn trong các thư mục
`results_*`. Nếu checkout mới chưa có model, hãy chạy script huấn luyện tương
ứng trước.

Khởi động server từ thư mục gốc:

```powershell
python app.py
```

Mở [http://127.0.0.1:8000](http://127.0.0.1:8000) trên trình duyệt. Chọn model,
nhập văn bản tiếng Anh hoặc tải tệp `.txt`, rồi nhấn **Phân loại văn bản**.
Tab **So sánh mô hình** hiển thị metrics và confusion matrix đã được lưu.

## Quy tắc lưu model trong Git

`.gitignore` bỏ qua môi trường ảo, bytecode Python, các file kết quả đánh giá
như `metrics.json`, báo cáo phân loại, confusion matrix, toàn bộ thư mục
`checkpoint-*` và model/tokenizer GPT-2 đã xuất trong `GPT/results_dbpedia_gpt2/`.
Các file này vẫn nằm trên máy hiện tại nhưng không được Git thêm vào.
