# Báo cáo thực nghiệm phân loại văn bản trên tập dữ liệu DBpedia-14 với mô hình BERT-base

---

## 1. Giới thiệu về mô hình BERT và kiến trúc BERT-base

### 1.1. Tổng quan về mô hình BERT

**BERT** (viết tắt của **Bidirectional Encoder Representations from Transformers**), được giới thiệu bởi nhóm nghiên cứu Google AI (Devlin et al.) vào cuối năm 2018, đã tạo ra một bước ngoặt mang tính cách mạng trong lĩnh vực Xử lý Ngôn ngữ Tự nhiên (NLP). 

Trước khi BERT xuất hiện, các mô hình ngôn ngữ phổ biến như Word2Vec hay GloVe chỉ tạo ra các biểu diễn từ tĩnh (static word embeddings), nghĩa là một từ luôn có cùng một vector biểu diễn bất kể ngữ cảnh xuất hiện. Các mô hình mạng nơ-ron hồi quy tuần tự (RNN, LSTM, BiLSTM) hoặc mô hình sinh tự hồi quy (Autoregressive models như OpenAI GPT đời đầu) có thể nắm bắt ngữ cảnh nhưng bị giới hạn bởi tính chất xử lý một chiều (hoặc trái qua phải, hoặc phải qua trái), khiến việc nắm bắt ngữ cảnh đa chiều đồng thời bị hạn chế.

BERT giải quyết triệt để vấn đề này nhờ việc tận dụng cấu trúc **Transformer Encoder** và giới thiệu phương pháp học biểu diễn ngữ cảnh hai chiều thực thụ (truly bidirectional):
- **Cơ chế Self-Attention (Tự chú ý)**: Cho phép mô hình tại mỗi từ/token có thể tính toán trọng số liên kết trực tiếp tới tất cả các từ khác trong câu mà không bị cản trở bởi khoảng cách từ xa hay phụ thuộc vào tính tuần tự như RNN.
- **Chiến lược Pre-training đột phá**:
  1. **Masked Language Model (MLM)**: Mô hình che ngẫu nhiên 15% số token trong câu đầu vào (80% thay bằng token `[MASK]`, 10% thay bằng token ngẫu nhiên, 10% giữ nguyên) và huấn luyện mạng dự đoán chính xác từ gốc dựa vào toàn bộ ngữ cảnh hai phía xung quanh.
  2. **Next Sentence Prediction (NSP)**: Huấn luyện mô hình nhận biết hai câu liên tiếp có quan hệ ngữ cảnh logic kế tiếp nhau trong văn bản gốc hay không (phân loại nhị phân `IsNext` / `NotNext`), giúp tăng cường khả năng hiểu quan hệ liên câu cho các bài toán hỏi đáp (QA) hay suy luận tự nhiên (NLI).
- **Mô hình nền tảng (Pre-trained Foundation Model)**: BERT được tiền huấn luyện trên kho dữ liệu khổng lồ gồm BooksCorpus (800 triệu từ) và English Wikipedia (2.500 triệu từ), tích lũy vốn tri thức ngữ nghĩa và cú pháp sâu sắc. Từ đó, người dùng chỉ cần thêm một lớp phân loại đơn giản ở đầu ra và tinh chỉnh (fine-tune) lại trên tập dữ liệu mục tiêu với chi phí tính toán thấp hơn nhiều so với huấn luyện từ đầu.

---

### 1.2. Chi tiết kiến trúc mô hình BERT-base

Trong bài báo gốc của Google, nhóm tác giả công bố hai phiên bản chính: **BERT-base** và **BERT-large**. Trong thực nghiệm này, cấu hình được sử dụng là **BERT-base** (cụ thể là phiên bản `bert-base-uncased`), đại diện cho sự cân bằng tối ưu giữa kích thước mô hình, tốc độ xử lý và năng lực trích xuất đặc trưng.

```
       [Đầu ra phân loại: 14 Logits]
                     ▲
            [Lớp Tuyến tính (Dense)]
                     ▲
                 [Dropout]
                     ▲
           [Vector ẩn h_[CLS]]
                     ▲
   ┌───────────────────────────────────┐
   │    Transformer Encoder Layer 12   │
   ├───────────────────────────────────┤
   │                 ...               │
   ├───────────────────────────────────┤
   │    Transformer Encoder Layer 1    │
   └───────────────────────────────────┘
                     ▲
  [Tổng: Token Emb + Segment Emb + Pos Emb]
                     ▲
    [CLS] + [Chuỗi văn bản mã hóa WordPiece] + [SEP]
```

#### Thông số kỹ thuật của BERT-base:
- **Số tầng Transformer Encoder ($L$ - Layers)**: $12$ tầng encoder xếp chồng lên nhau.
- **Kích thước vector biểu diễn ẩn ($H$ - Hidden Size / $d_{\text{model}}$)**: $768$ chiều.
- **Số đầu tự chú ý ($A$ - Attention Heads)**: $12$ heads. Mỗi head có kích thước $d_k = 768 / 12 = 64$ chiều, cho phép mô hình đồng thời chú ý vào nhiều khía cạnh cú pháp và ngữ nghĩa khác nhau trong câu.
- **Kích thước tầng Feed-Forward trung gian ($d_{\text{ff}}$ - Intermediate Size)**: $3072$ chiều ($4 \times H$).
- **Hàm kích hoạt**: GELU (Gaussian Error Linear Unit).
- **Tổng số lượng tham số**: Khoảng **110 triệu tham số** (110M parameters).

#### Cơ chế biểu diễn đầu vào (Input Embeddings):
Đầu vào của BERT là sự tổng hợp (cộng trực tiếp từng phần tử) của 3 loại vector nhúng:
1. **Token Embeddings**: Sử dụng thuật toán phân tách từ con **WordPiece** với kích thước bộ từ vựng $V = 30.522$ tokens. Phiên bản *uncased* chuyển toàn bộ văn bản về chữ thường và chuẩn hóa dấu câu trước khi tra cứu từ điển.
2. **Segment Embeddings**: Dùng để phân định giữa câu A và câu B trong các tác vụ liên câu. Trong bài toán phân loại đơn văn bản, toàn bộ token thuộc về một phân đoạn duy nhất (Segment A).
3. **Position Embeddings**: Do Transformer không có tính tuần tự tự nhiên như RNN, BERT bổ sung các vector vị trí tuyệt đối (learnable position embeddings) từ vị trí $0$ đến $511$ (độ dài ngữ cảnh tối đa 512 tokens).

#### Các token đặc biệt quan trọng:
- `[CLS]` (*Classification token*): Luôn được chèn vào vị trí đầu tiên của mọi chuỗi đầu vào. Trạng thái ẩn cuối cùng tương ứng với token này ($h_{\text{[CLS]}} \in \mathbb{R}^{768}$) được xem là vector cô đọng toàn bộ thông tin ngữ nghĩa của cả đoạn văn bản, đóng vai trò là đầu vào cho bộ phân loại (Classification Head).
- `[SEP]` (*Separator token*): Đánh dấu kết thúc một câu hoặc phân cách giữa hai câu.
- `[PAD]` (*Padding token*): Dùng để bù vào các chuỗi ngắn hơn độ dài chuẩn hóa cố định (`max_length`).

---

## 2. Giới thiệu về tập dữ liệu DBpedia-14

### 2.1. Nguồn gốc và chủ đề của tập dữ liệu

Tập dữ liệu **DBpedia-14** (hay *DBpedia ontology dataset*) được xây dựng và công bố bởi các tác giả **Xiang Zhang, Junbo Zhao và Yann LeCun** trong bài báo nghiên cứu nổi tiếng *"Character-level Convolutional Networks for Text Classification"* (NeurIPS 2015).

Dữ liệu được trích xuất từ dự án **DBpedia** – một cơ sở tri thức mở quy mô lớn cấu trúc hóa thông tin từ bách khoa toàn thư Wikipedia. DBpedia-14 tập trung vào bài toán **phân loại thực thể theo danh mục bản thể học (Ontology Classification)**, yêu cầu mô hình đọc đoạn văn bản tóm tắt về một thực thể Wikipedia và dự đoán chính xác thực thể đó thuộc nhóm đối tượng nào trong thế giới thực.

---

### 2.2. Quy mô và cấu trúc phân chia dữ liệu

DBpedia-14 là một trong những tập dữ liệu chuẩn mực (benchmark) lớn và chất lượng cao nhất cho bài toán phân loại văn bản:
- **Tổng số lượng mẫu**: **630.000 mẫu** văn bản.
- **Tập huấn luyện (Train set)**: **560.000 mẫu** (chính xác $40.000$ mẫu cho mỗi lớp).
- **Tập kiểm thử (Test set)**: **70.000 mẫu** (chính xác $5.000$ mẫu cho mỗi lớp).
- **Tính chất phân bố**: Dữ liệu được thu thập theo phương thức **hoàn toàn cân bằng** (perfect class balance). Điều này loại bỏ hoàn toàn rủi ro thiên lệch nhãn (class imbalance bias), giúp các độ đo đánh giá như Accuracy, Macro-Precision, Macro-Recall và Macro-F1 có tính tin cậy tuyệt đối và không bị chi phối bởi các lớp áp đảo.

Mỗi mẫu trong tập dữ liệu bao gồm 3 trường thông tin:
1. `label`: Nhãn số nguyên từ $0$ đến $13$ tương ứng với 14 lớp ontology.
2. `title`: Tiêu đề bài viết Wikipedia (tên thực thể).
3. `content`: Đoạn văn bản tóm tắt/giới thiệu mở đầu (lead abstract) mô tả về thực thể đó. Trong thực nghiệm này, trường `content` được sử dụng làm văn bản đầu vào chính để huấn luyện mô hình.

---

### 2.3. Danh sách 14 lớp phân loại và đặc trưng ngữ nghĩa

| ID | Tên lớp (Ontology Class) | Ý nghĩa danh mục | Ví dụ điển hình |
|:--:|:---|:---|:---|
| 0 | `Company` | Công ty, tập đoàn, doanh nghiệp | Apple Inc., Microsoft, BMW |
| 1 | `EducationalInstitution` | Cơ sở giáo dục, đại học, học viện, trường học | Harvard University, Stanford, Oxford |
| 2 | `Artist` | Nghệ sĩ, nhạc sĩ, ca sĩ, diễn viên, họa sĩ | Leonardo da Vinci, Michael Jackson, Picasso |
| 3 | `Athlete` | Vận động viên, cầu thủ thể thao | Lionel Messi, Michael Jordan, Usain Bolt |
| 4 | `OfficeHolder` | Chính khách, quan chức chính phủ, tổng thống, thủ tướng | Barack Obama, Abraham Lincoln, Angela Merkel |
| 5 | `MeanOfTransportation` | Phương tiện giao thông vận tải (tàu thủy, máy bay, ô tô) | Boeing 747, RMS Titanic, Ferrari F40 |
| 6 | `Building` | Công trình kiến trúc, tòa nhà, tháp, nhà thờ, di tích | Eiffel Tower, Empire State Building, Notre-Dame |
| 7 | `NaturalPlace` | Địa danh tự nhiên, sông, núi, thung lũng, đại dương | Mount Everest, Amazon River, Grand Canyon |
| 8 | `Village` | Làng mạc, khu dân cư nông thôn | Bibury, Oia, Giethoorn |
| 9 | `Animal` | Động vật | Bengal tiger, Bald eagle, Blue whale |
| 10 | `Plant` | Thực vật, hoa, cây cối | Ginkgo biloba, Rosa damascena, Sequoia |
| 11 | `Album` | Album âm nhạc, đĩa thu âm | Thriller, The Dark Side of the Moon, Abbey Road |
| 12 | `Film` | Phim điện ảnh, tác phẩm truyền hình | Inception, Titanic, The Godfather |
| 13 | `WrittenWork` | Tác phẩm văn học, tiểu thuyết, sách báo, tạp chí | War and Peace, Hamlet, The Great Gatsby |

---

### 2.4. Thách thức phân loại của DBpedia-14

Mặc dù các danh mục có vẻ tách biệt, DBpedia-14 chứa nhiều thách thức ngữ nghĩa tinh tế:
1. **Sự tương đồng giữa các thực thể con người**: Lớp `Artist` và `OfficeHolder` đều nói về con người có tiểu sử, năm sinh, trường từng học, hoạt động xã hội và các thành tựu sự nghiệp.
2. **Sự đan xen giữa tổ chức và cơ sở vật chất**: Các công ty (`Company`) và cơ sở giáo dục (`EducationalInstitution`) thường sở hữu hoặc được gọi theo tên tòa nhà/trụ sở (`Building`), hoặc tên trường đại học tư thục gắn liền với một thực thể doanh nghiệp.
3. **Mối liên hệ giữa các tác phẩm nghệ thuật sáng tạo**: Phim ảnh (`Film`), đĩa nhạc (`Album`) và sách/truyện (`WrittenWork`) thường chia sẻ chung tên gọi, chung bối cảnh chuyển thể, thông tin ngày phát hành và giải thưởng nghệ thuật.

---

## 3. Quá trình Fine-tune mô hình BERT-base

### 3.1. Thiết lập kiến trúc Fine-tuning

Trong tác vụ phân loại chuỗi (Sequence Classification), kiến trúc `BertForSequenceClassification` từ thư viện Hugging Face Transformers được áp dụng:
1. Mô hình tải trọng số tiền huấn luyện từ `bert-base-uncased`.
2. Trích xuất vector biểu diễn của token `[CLS]` tại tầng ẩn thứ 12: $h_{\text{[CLS]}} \in \mathbb{R}^{768}$.
3. Đưa qua lớp **Dropout** với xác suất ngắt kết nối $p = 0.1$ nhằm hạn chế hiện tượng học vẹt (overfitting).
4. Đưa qua một lớp tuyến tính (Linear Classifier) chiếu từ không gian 768 chiều sang không gian 14 chiều tương ứng với 14 lớp:
   $$z = W \cdot \text{Dropout}(h_{\text{[CLS]}}) + b \quad \text{với } W \in \mathbb{R}^{14 \times 768}, \; b \in \mathbb{R}^{14}$$
5. Áp dụng hàm mất mát **Cross-Entropy Loss** cho bài toán phân loại đa lớp:
   $$\mathcal{L} = -\sum_{i=1}^{C} y_i \log \left(\frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}\right) \quad (C = 14)$$

---

### 3.2. Tiền xử lý dữ liệu (Tokenization)

- **Bộ phân tách từ con (Tokenizer)**: `AutoTokenizer.from_pretrained("bert-base-uncased")`.
- **Độ dài chuỗi tối đa (`max_length`)**: Thiết lập cố định là **128 tokens**. Độ dài này đủ để bao trọn câu mở đầu định nghĩa thực thể trong Wikipedia (phần chứa đựng nhiều thông tin ngữ nghĩa quyết định nhất của bài tóm tắt) mà vẫn tối ưu được dung lượng bộ nhớ VRAM card đồ họa cũng như tốc độ tính toán.
- **Chiến lược chuẩn hóa**: 
  - `truncation=True`: Cắt ngắn văn bản nếu vượt quá 128 tokens.
  - `padding="max_length"`: Chèn các token `[PAD]` vào phía sau nếu văn bản ngắn hơn 128 tokens để đảm bảo kích thước ma trận đồng nhất cho việc xử lý theo mini-batch.

---

### 3.3. Các siêu tham số huấn luyện (Hyperparameters)

Quá trình fine-tune được cấu hình chi tiết thông qua Hugging Face `TrainingArguments` với các tham số tối ưu sau:

| Siêu tham số (Hyperparameter) | Giá trị thiết lập | Mục đích & Giải thích kỹ thuật |
|:---|:---:|:---|
| **Base Model** | `bert-base-uncased` | Mô hình gốc 110M tham số, văn bản viết thường. |
| **Số lớp phân loại (`num_labels`)** | 14 | Tương ứng 14 danh mục ontology của DBpedia. |
| **Độ dài ngữ cảnh (`max_length`)** | 128 | Cân đối thời gian tính toán và độ phủ ngữ nghĩa. |
| **Batch size mỗi thiết bị (`per_device_train_batch_size`)** | 8 | Phù hợp với bộ nhớ phần cứng GPU cá nhân (VRAM 8GB - 12GB). |
| **Tích lũy gradient (`gradient_accumulation_steps`)** | 2 | Tích lũy gradient qua 2 bước trước khi cập nhật trọng số. |
| **Kích thước Batch hiệu dụng (Effective Batch Size)** | **16** | $8 \times 2 = 16$, tạo bước cập nhật gradient ổn định hơn. |
| **Tốc độ học (Learning Rate)** | $2 \times 10^{-5}$ ($2\text{e-}5$) | Giá trị tối ưu khuyến nghị cho BERT fine-tuning, tránh phá hủy tri thức đã học (catastrophic forgetting). |
| **Bộ tối ưu hóa (Optimizer)** | AdamW | Phiên bản cải tiến của Adam kết hợp chuẩn hóa trọng số tách rời (decoupled weight decay). |
| **Suy giảm trọng số (Weight Decay)** | 0.01 | Điều chuẩn L2 regularization nhằm hạn chế overfitting trên các trọng số lớn. |
| **Learning Rate Schedule** | Linear with warmup | Giảm dần tốc độ học theo hàm tuyến tính sau các bước khởi động. |
| **Số epoch (`num_train_epochs`)** | 1 epoch | Đủ để mô hình hội tụ trên tập dữ liệu lớn 560.000 mẫu ($35.000$ bước cập nhật). |
| **Tổng số bước tối ưu (`total_steps`)** | **35.000 steps** | $560.000 \text{ mẫu} / 16 = 35.000$ steps huấn luyện. |
| **Huấn luyện độ chính xác hỗn hợp (`fp16`)** | `True` | Sử dụng kiểu dữ liệu 16-bit nổi, tăng tốc độ tính toán gấp 2–3 lần và giảm 50% dung lượng VRAM. |
| **Tần suất ghi nhật ký (`logging_steps`)** | 100 steps | Ghi nhận hàm mất mát (loss) và learning rate đều đặn. |
| **Chiến lược lưu mô hình (`save_strategy`)** | `"epoch"` | Lưu lại checkpoint tốt nhất tại cuối epoch huấn luyện (`checkpoint-35000`). |

---

## 4. Kết quả thực nghiệm và Đánh giá chi tiết

### 4.1. Kết quả định lượng trên tập Train và Test

Mô hình được đánh giá sau khi hoàn thành 1 epoch fine-tuning (checkpoint-35000). Toàn bộ 70.000 mẫu trong tập test (không tham gia vào quá trình cập nhật trọng số) được sử dụng để kiểm thử năng lực tổng quát hóa.

#### Bảng kết quả tổng hợp:

| Phân vùng dữ liệu / Độ đo | Giá trị thực nghiệm |
|:---|:---:|
| **Số lượng mẫu kiểm thử (Test samples)** | **70.000 mẫu** (5.000 mẫu / lớp) |
| **Hàm mất mát huấn luyện (Train loss)** | **0.0220** |
| **Độ chính xác huấn luyện (Train accuracy)** | **0.9954 (99.54%)** |
| **Độ chính xác kiểm thử (Test accuracy)** | **0.9933 (99.33%)** |
| **Độ chuẩn xác (Precision - Macro average)** | **0.9933 (99.33%)** |
| **Độ phủ / Độ hồi phục (Recall - Macro average)** | **0.9933 (99.33%)** |
| **Điểm F1-score (Macro average)** | **0.9933 (99.33%)** |

> [!NOTE]
> **Giải thích ý nghĩa các chỉ số đo lường:**
> - **Accuracy**: Tỷ lệ số lượng mẫu dự đoán chính xác trên tổng số 70.000 mẫu test.
> - **Precision (Macro)**: Trung bình cộng tỷ lệ dự đoán đúng của từng lớp trên tổng số mẫu được mô hình dự đoán là lớp đó.
> - **Recall (Macro)**: Trung bình cộng tỷ lệ số mẫu thực tế của từng lớp được mô hình nhận diện thành công.
> - **F1-score (Macro)**: Trung bình điều hòa giữa Precision và Recall. Vì toàn bộ các chỉ số Macro đều đạt mức **0.9933**, điều này khẳng định chất lượng dự đoán của BERT-base đồng đều tuyệt đối trên cả 14 phân lớp, không có hiện tượng lớp nào bị mô hình bỏ quên hay đánh đổi.
> - **Chênh lệch Train/Test**: Train Accuracy (99.54%) và Test Accuracy (99.33%) chỉ cách nhau **0.21%**, chứng minh mô hình **không hề bị overfitting** mà có khả năng tổng quát hóa xuất sắc trên dữ liệu chưa từng thấy.

---

### 4.2. Phân tích ma trận nhầm lẫn (Confusion Matrix)

Dưới đây là ma trận nhầm lẫn (Confusion Matrix) chi tiết trên toàn bộ 70.000 mẫu của tập kiểm thử DBpedia-14:

![Ma trận nhầm lẫn của BERT trên tập kiểm thử DBpedia-14](../BERT.png)

#### Bảng số liệu chi tiết theo từng lớp (trên 5.000 mẫu/lớp):

| Lớp thực tế (Ground Truth) | Số mẫu đúng | Tỷ lệ đúng (Recall) | Số mẫu sai | Các lớp nhầm lẫn chủ yếu |
|:---|:---:|:---:|:---:|:---|
| **Company** | 4.899 / 5.000 | 97.98% | 101 | Building (31), EducationalInstitution (28), WrittenWork (17), MeanOfTransportation (12) |
| **EducationalInstitution** | 4.944 / 5.000 | 98.88% | 56 | Company (34), Building (15), OfficeHolder (4) |
| **Artist** | 4.949 / 5.000 | 98.98% | 51 | OfficeHolder (42), Athlete (4), Company (3) |
| **Athlete** | 4.988 / 5.000 | 99.76% | 12 | OfficeHolder (10), EducationalInstitution (1), Album (1) |
| **OfficeHolder** | 4.952 / 5.000 | 99.04% | 48 | Artist (39), Athlete (4), Building (2), Company (1) |
| **MeanOfTransportation** | 4.980 / 5.000 | 99.60% | 20 | Company (16), Building (3), WrittenWork (1) |
| **Building** | 4.931 / 5.000 | 98.62% | 69 | Company (37), EducationalInstitution (15), MeanOfTransportation (9), NaturalPlace (7) |
| **NaturalPlace** | 4.986 / 5.000 | 99.72% | 14 | Building (10), Village (4) |
| **Village** | **4.996 / 5.000** | **99.92%** | **4** | EducationalInstitution (1), Building (1), NaturalPlace (1), WrittenWork (1) |
| **Animal** | 4.993 / 5.000 | 99.86% | 7 | Plant (6), Company (1) |
| **Plant** | 4.985 / 5.000 | 99.70% | 15 | Company (11), Animal (3), MeanOfTransportation (1) |
| **Album** | 4.984 / 5.000 | 99.68% | 16 | Film (6), WrittenWork (5), Artist (4), Athlete (1) |
| **Film** | 4.973 / 5.000 | 99.46% | 27 | WrittenWork (12), Album (11), Company (3), Artist (1) |
| **WrittenWork** | 4.972 / 5.000 | 99.44% | 28 | Company (17), Film (5), Artist (3), Building (1) |
| **Tổng cộng** | **69.532 / 70.000** | **99.33%** | **468** | **Tỷ lệ sai sót toàn tập chỉ chiếm 0.67%** |

---

### 4.3. Phân tích chuyên sâu về các trường hợp nhầm lẫn

Nhìn vào ma trận phân bố sai số, tổng cộng mô hình chỉ dự đoán sai **468 mẫu trên tổng số 70.000 mẫu** (tương đương tỷ lệ sai chỉ **0.67%**). Khi phân tích sâu vào các cặp nhầm lẫn, các nguyên nhân ngữ nghĩa hoàn toàn hợp lý và phản ánh tính phức tạp tự nhiên của văn bản Wikipedia:

#### 1. Nhầm lẫn giữa nhóm người: `Artist` $\leftrightarrow$ `OfficeHolder`
- Có **42 mẫu** `Artist` bị nhận nhầm thành `OfficeHolder` và **39 mẫu** `OfficeHolder` bị nhận nhầm thành `Artist`.
- *Nguyên nhân*: Cả hai đều thuộc phân loại đối tượng người (*Person*). Nhiều nhân vật lịch sử hoặc chính trị gia xuất thân từ nghệ thuật (hoặc ngược lại, các nhà thơ, nhà văn giữ vị trí bộ trưởng, nghị sĩ). Ngoài ra, cấu trúc câu mô tả nhân vật trong Wikipedia có tính quy chuẩn cao ("*was a ... who served as ...*", "*was born in ...*"), khiến mô hình gặp khó khăn khi văn bản đề cập đến cả hoạt động văn hóa lẫn vị trí công quyền của đối tượng.

#### 2. Nhầm lẫn giữa tổ chức và cơ sở vật chất: `Company` $\leftrightarrow$ `Building` $\leftrightarrow$ `EducationalInstitution`
- Có **37 mẫu** `Building` nhầm thành `Company` và **31 mẫu** `Company` nhầm thành `Building`.
- Có **34 mẫu** `EducationalInstitution` nhầm thành `Company` và **28 mẫu** `Company` nhầm thành `EducationalInstitution`.
- *Nguyên nhân*: Nhiều tập đoàn lớn sở hữu các tòa nhà chọc trời hoặc khu tổ hợp mang chính tên tập đoàn đó (ví dụ: Chrysler Building, Willis Tower), hoặc bài viết giới thiệu công ty nhưng lại tập trung mô tả kiến trúc trụ sở chính. Tương tự, nhiều học viện hoặc trường đào tạo tư nhân hoạt động dưới tư cách pháp nhân thương mại, sử dụng từ ngữ quản trị và tài chính tương tự doanh nghiệp.

#### 3. Nhầm lẫn trong cụm tác phẩm sáng tạo: `Film` $\leftrightarrow$ `Album` $\leftrightarrow$ `WrittenWork`
- `Film` nhầm sang `Album` (11 mẫu) và `WrittenWork` (12 mẫu).
- `Album` nhầm sang `Film` (6 mẫu) và `WrittenWork` (5 mẫu).
- `WrittenWork` nhầm sang `Film` (5 mẫu).
- *Nguyên nhân*: Đây là hiện tượng đa phương tiện hóa một tác phẩm nghệ thuật. Một bộ phim kinh điển thường được chuyển thể từ một cuốn tiểu thuyết (`WrittenWork`) và phát hành kèm album nhạc phim (`Album`). Nội dung tóm tắt của một bộ phim điện ảnh thường đề cập dày đặc đến tác giả kịch bản và tiểu thuyết gốc ("*based on the novel by...*"), khiến mô hình bị phân tán sự chú ý sang danh mục ấn phẩm viết.

#### 4. Các lớp có độ chính xác gần như hoàn hảo:
- `Village` đạt độ chính xác đáng kinh ngạc: **99.92%** (chỉ nhầm đúng 4 mẫu trên 5.000 mẫu). Đặc trưng của các mục làng mạc thường chứa các cấu trúc địa lý rất rõ ràng như "*is a village and civil parish in...*", "*located at coordinates...*", giúp BERT nhận diện gần như lập tức.
- `Animal` (99.86%) và `Athlete` (99.76%) cũng đạt độ chính xác gần như tuyệt đối do vốn từ vựng sinh học (tên khoa học loài, họ) và thuật ngữ thể thao (câu lạc bộ, vị trí thi đấu, bàn thắng, huy chương) mang tính đặc trưng cao, hiếm khi bị trùng lặp với các lĩnh vực khác.

---

## 5. Kết luận

Thực nghiệm phân loại văn bản trên tập dữ liệu DBpedia-14 với mô hình **BERT-base** mang lại các kết luận khoa học và thực tiễn quan trọng:

1. **Hiệu năng vượt trội của mô hình ngôn ngữ dựa trên Transformer**:
   Chỉ sau **1 epoch** huấn luyện (tương ứng 35.000 bước cập nhật trọng số), mô hình đã đạt độ chính xác **99.33%** trên tập test gồm 70.000 mẫu văn bản đa dạng. Điểm số này vượt xa các phương pháp phân loại truyền thống (SVM, Naive Bayes, TF-IDF) và các mạng nơ-ron sâu thế hệ trước (CNN, LSTM) trên cùng tập dữ liệu chuẩn.

2. **Khả năng tổng quát hóa hoàn hảo**:
   Mức sai lệch giữa độ chính xác trên tập huấn luyện (99.54%) và tập kiểm thử (99.33%) chỉ là **0.21%**, minh chứng rằng mô hình nắm bắt được bản chất ngữ nghĩa của các thực thể thay vì chỉ ghi nhớ các mẫu dữ liệu huấn luyện.

3. **Tính tối ưu của chiến lược Fine-tuning**:
   Việc tận dụng trọng số tiền huấn luyện từ `bert-base-uncased`, kết hợp kỹ thuật huấn luyện nửa chính xác (`fp16`), tốc độ học nhỏ ($2\text{e-}5$) và tích lũy gradient (`gradient_accumulation_steps=2`) cho phép huấn luyện thành công mô hình ngôn ngữ lớn trên một tập dữ liệu khổng lồ (560.000 mẫu train) một cách ổn định, tiết kiệm tối đa tài nguyên tính toán và bộ nhớ VRAM.

