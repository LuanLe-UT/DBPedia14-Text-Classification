# Báo cáo thực nghiệm phân loại văn bản trên tập dữ liệu DBpedia-14 với mô hình GPT-2

---

## 1. Giới thiệu về mô hình GPT và kiến trúc GPT-2

### 1.1. Tổng quan về họ mô hình GPT

**GPT** (viết tắt của **Generative Pre-trained Transformer**) là dòng mô hình ngôn ngữ đột phá được phát triển bởi OpenAI, khởi đầu từ GPT-1 (2018), tiếp nối bởi GPT-2 (2019), GPT-3 (2020) và các thế hệ LLM hiện đại. 

Khác với nhánh mô hình **Encoder-only** (tiêu biểu là BERT) học biểu diễn ngữ cảnh hai chiều đồng thời, GPT thuộc nhánh **Decoder-only Transformer** hoạt động theo nguyên lý **sinh tự hồi quy (Autoregressive Language Modeling)**. Mục tiêu cốt lõi của GPT trong giai đoạn tiền huấn luyện là tối đa hóa hàm hợp lý logarit (log-likelihood) để dự đoán từ tiếp theo trong chuỗi dựa vào toàn bộ các từ đã xuất hiện ở quá khứ:

$$P(X) = \prod_{i=1}^{n} P(x_i \mid x_1, x_2, \dots, x_{i-1})$$

Để thực hiện điều này mà không làm rò rỉ thông tin tương lai, GPT sử dụng cơ chế **Masked Multi-Head Self-Attention (Causal Attention Masking)**. Ma trận chú ý được áp đặt một mặt nạ tam giác trên ($-\infty$ tại các vị trí $j > i$), buộc mỗi token tại bước thứ $i$ chỉ được phép chú ý đến chính nó và các token đứng trước ($j \le i$).

```
        [Dự đoán phân loại hoặc Token tiếp theo]
                          ▲
            [Lớp Tuyến tính (Dense Head)]
                          ▲
             [Layer Normalization cuối]
                          ▲
      ┌────────────────────────────────────────┐
      │      Transformer Decoder Layer 12      │
      │  ┌──────────────────────────────────┐  │
      │  │  Feed-Forward Network (MLP)      │  │
      │  ├──────────────────────────────────┤  │
      │  │  Masked Self-Attention (Causal)  │  │
      │  └──────────────────────────────────┘  │
      ├────────────────────────────────────────┤
      │                  ...                   │
      ├────────────────────────────────────────┤
      │      Transformer Decoder Layer 1       │
      └────────────────────────────────────────┘
                          ▲
       [Tổng: Token Embedding + Position Embedding]
                          ▲
           [Chuỗi đầu vào mã hóa Byte-level BPE]
```

Mặc dù xuất phát điểm của kiến trúc Decoder-only là sinh văn bản (Text Generation), các nghiên cứu của Radford et al. đã chỉ ra rằng: việc huấn luyện mô hình dự đoán từ tiếp theo trên quy mô dữ liệu khổng lồ giúp mạng nơ-ron tự động hấp thụ các đặc trưng ngữ pháp, cú pháp, tri thức sự kiện và khả năng lập luận ngữ cảnh sâu sắc. Nhờ vậy, GPT có thể thích ứng tuyệt vời cho các bài toán phân loại chuỗi (Sequence Classification) khi được gắn thêm một đầu phân loại phù hợp.

---

### 1.2. Đi sâu vào chi tiết kiến trúc GPT-2

Mô hình **GPT-2** được công bố trong bài báo nghiên cứu *"Language Models are Unsupervised Multitask Learners"* (Radford et al., 2019). Trong thực nghiệm này, phiên bản được sử dụng là **GPT-2 Small** (checkpoint chuẩn `openai-community/gpt2` với 124 triệu tham số).

#### 1.2.1. Cấu hình phần cứng và siêu tham số kiến trúc của GPT-2 Small:
- **Số tầng Transformer Decoder ($N_{\text{layer}}$)**: $12$ tầng decoder đồng nhất.
- **Kích thước biểu diễn ẩn ($d_{\text{model}} / n_{\text{embd}}$)**: $768$ chiều.
- **Số đầu tự chú ý ($n_{\text{head}}$)**: $12$ heads. Mỗi head có kích thước $d_k = 768 / 12 = 64$ chiều.
- **Chiều dài ngữ cảnh tối đa ($n_{\text{ctx}} / n_{\text{positions}}$)**: $1024$ tokens.
- **Hàm kích hoạt**: `gelu_new` (phiên bản chuẩn hóa xấp xỉ liên tục của Gaussian Error Linear Unit).
- **Tổng số lượng tham số**: Khoảng **124 triệu tham số** (124M parameters).

#### 1.2.2. Điểm cải tiến cấu trúc then chốt của GPT-2:
1. **Pre-Layer Normalization (Pre-LN)**: Khác với kiến trúc Post-LN truyền thống (đặt LayerNorm sau khối cộng phần dư), GPT-2 đưa LayerNorm lên **trước** mỗi khối Masked Self-Attention và trước khối MLP, đồng thời bổ sung thêm một lớp LayerNorm ở sau tầng decoder cuối cùng. Cải tiến này giúp dòng gradient truyền thẳng qua các nhánh kết nối tắt (residual connections), cho phép mô hình huấn luyện ổn định và hội tụ nhanh hơn.
2. **Khởi tạo trọng số có điều chỉnh tỷ lệ**: Các trọng số của lớp chiếu đầu ra trong các khối residual được chia theo hệ số $\frac{1}{\sqrt{2 \times N_{\text{layer}}}}$ nhằm kiểm soát độ lệch chuẩn kích hoạt không bị bùng nổ khi số tầng tăng lên.

---

### 1.3. Kỹ thuật Tokenizer: Byte-Pair Encoding ở cấp độ Byte (Byte-level BPE)

Một trong những đóng góp mang tính kỹ thuật quan trọng nhất của GPT-2 là bộ tách từ **Byte-level Byte-Pair Encoding (Byte-level BPE)**:
- **Nguyên lý hoạt động**: Các thuật toán BPE truyền thống (hoặc WordPiece trong BERT) thường bắt đầu từ các ký tự Unicode và có thể gặp lỗi từ ngoài từ điển (out-of-vocabulary, token `<unk>`) khi bắt gặp các ký tự lạ, emoji hoặc chuỗi đa ngôn ngữ không có trong bộ dữ liệu huấn luyện. Ngược lại, Byte-level BPE của GPT-2 sử dụng **256 giá trị byte cơ bản** của mã hóa UTF-8 làm bảng chữ cái ban đầu.
- **Không bao giờ phát sinh lỗi `<unk>`**: Nhờ xây dựng từ cấp độ byte, bộ tokenizer của GPT-2 có khả năng phân tách và biểu diễn **mọi chuỗi ký tự trên thế giới** mà không cần token thay thế không xác định (`<unk>`).
- **Quy mô từ vựng ($V$)**: GPT-2 sở hữu bộ từ vựng gồm **50.257 tokens**, bao gồm:
  - 256 ký tự byte cơ bản.
  - 50.000 quy tắc hợp nhất cặp byte phổ biến nhất được học từ ngữ liệu WebText.
  - 1 token đặc biệt `<|endoftext|>` (Token ID: `50256`) đánh dấu điểm kết thúc của tài liệu.
- **Quy tắc tiền phân đoạn biểu thức chính quy (Regex Splitting)**: GPT-2 áp dụng biểu thức chính quy để ngăn không cho các cặp byte hợp nhất xuyên qua ranh giới giữa dấu câu, khoảng trắng và chữ cái (ví dụ: không ghép dấu phẩy hay dấu chấm vào từ đứng trước nó), từ đó duy trì tính nhất quán ngữ nghĩa của từ vựng.

---

### 1.4. Kỹ thuật Embedding trong GPT-2

Không gian biểu diễn đầu vào của GPT-2 là sự kết hợp của hai thành phần:

$$h_0 = W_{\text{te}}[x] + W_{\text{pe}}[\text{pos}]$$

1. **Token Embeddings ($W_{\text{te}}$)**: Ma trận tra cứu kích thước $50.257 \times 768$, ánh xạ mỗi ID token thành một vector 768 chiều.
2. **Positional Embeddings ($W_{\text{pe}}$)**: GPT-2 sử dụng **Learned Absolute Positional Embeddings** kích thước $1024 \times 768$. Không sử dụng hàm sin/cos cố định như Transformer gốc, GPT-2 tự động học các vector vị trí tối ưu từ $0$ đến $1023$ trong suốt quá trình huấn luyện.
3. **Không sử dụng Segment Embeddings**: Do GPT-2 là mô hình sinh đơn chuỗi tự hồi quy tự nhiên, nó không cần các vector phân đoạn câu A / câu B như cấu trúc song ngữ cảnh của BERT.

---

### 1.5. Cơ chế Pooling cho bài toán Phân loại văn bản: Last Token Pooling

Vì GPT-2 vốn là mô hình sinh tự hồi quy, việc chuyển đổi sang bài toán phân loại chuỗi đòi hỏi cách xử lý hoàn toàn khác biệt so với BERT:
- **Tại sao không thể dùng token đầu tiên như BERT?** Trong BERT, cơ chế Self-Attention là hai chiều (bidirectional), token đầu tiên `[CLS]` có thể quan sát toàn bộ câu. Nhưng trong GPT-2, do có Causal Mask, **token đầu tiên chỉ nhìn thấy chính nó** và hoàn toàn mù thông tin về phần còn lại của câu!
- **Cơ chế Last Token Pooling**: Token hợp lệ cuối cùng của văn bản là token duy nhất đã tích lũy toàn bộ thông tin ngữ cảnh của tất cả các từ đứng trước nó trong chuỗi. Do đó, lớp `GPT2ForSequenceClassification` sẽ trích xuất vector trạng thái ẩn tại vị trí **token không phải PAD cuối cùng (last non-pad token)** để đưa vào bộ phân loại.
- **Tái sử dụng EOS làm PAD token**: GPT-2 nguyên bản không định nghĩa token `[PAD]` riêng biệt. Để có thể ghép các chuỗi có độ dài khác nhau vào cùng một mini-batch cố định (`max_length=128`), token kết thúc văn bản `<|endoftext|>` (ID: `50256`) được gán làm `pad_token`. Mô hình sử dụng `attention_mask` để vô hiệu hóa sự ảnh hưởng của các token đệm này trong quá trình tính toán.

---

## 2. Giới thiệu về tập dữ liệu DBpedia-14

### 2.1. Nguồn gốc và chủ đề

Tập dữ liệu **DBpedia-14** (DBpedia ontology classification dataset) được công bố bởi nhóm nghiên cứu **Xiang Zhang, Junbo Zhao và Yann LeCun** (2015) trong bài báo nghiên cứu nền tảng *"Character-level Convolutional Networks for Text Classification"*.

DBpedia là dự án trích xuất thông tin có cấu trúc từ Wikipedia. Bộ dữ liệu DBpedia-14 được thiết kế nhằm chuẩn hóa bài toán **phân loại thực thể bách khoa theo danh mục bản thể học (Ontology Classification)**: từ đoạn tóm tắt mở đầu của một bài viết Wikipedia, mô hình phải suy luận và gán nhãn thực thể đó vào đúng 1 trong 14 nhóm khái niệm trong thế giới thực.

---

### 2.2. Quy mô và cấu trúc phân chia dữ liệu

- **Tổng số lượng mẫu**: **630.000 mẫu** văn bản tiếng Anh chất lượng cao.
- **Tập huấn luyện (Train set)**: **560.000 mẫu** (chính xác $40.000$ mẫu cho mỗi lớp).
- **Tập kiểm thử (Test set)**: **70.000 mẫu** (chính xác $5.000$ mẫu cho mỗi lớp).
- **Phân bố cân bằng tuyệt đối (Perfect Class Balance)**: Cả tập train và tập test đều có số lượng mẫu đồng đều tuyệt đối giữa 14 phân lớp ($1/14 \approx 7.14\%$ mỗi lớp). Đặc điểm này giúp loại trừ triệt để hiện tượng thiên lệch nhãn đa số (majority class bias), đảm bảo độ đo Accuracy và Macro-Average phản ánh chính xác 100% năng lực tổng quát hóa của mô hình.

Mỗi mẫu dữ liệu bao gồm 3 trường:
- `label`: Nhãn số nguyên từ $0$ đến $13$.
- `title`: Tiêu đề bài viết Wikipedia (tên thực thể cần định danh).
- `content`: Đoạn văn bản tóm tắt trích dẫn (lead paragraph) mô tả chi tiết về thực thể. Trong thực nghiệm, trường `content` được đưa qua bộ tách từ làm đầu vào huấn luyện.

---

### 2.3. Danh sách 14 lớp bản thể học (Ontology Classes)

| ID | Tên nhãn (Ontology Class) | Lĩnh vực khái niệm | Ví dụ thực thể tiêu biểu |
|:--:|:---|:---|:---|
| 0 | `Company` | Doanh nghiệp, tập đoàn, công ty thương mại | Microsoft, Toyota, Sony |
| 1 | `EducationalInstitution` | Viện nghiên cứu, trường đại học, học viện | MIT, Cambridge University, Sorbonne |
| 2 | `Artist` | Nghệ sĩ, nhạc sĩ, ca sĩ, diễn viên, đạo diễn | Wolfgang Amadeus Mozart, Adele, Akira Kurosawa |
| 3 | `Athlete` | Vận động viên, cầu thủ, tay đua chuyên nghiệp | Cristiano Ronaldo, Roger Federer, Serena Williams |
| 4 | `OfficeHolder` | Chính khách, lãnh đạo nhà nước, nghị sĩ | Winston Churchill, Nelson Mandela, Shinzo Abe |
| 5 | `MeanOfTransportation` | Phương tiện giao thông (tàu lửa, máy bay, xe cơ giới) | Airbus A380, HMS Victory, Boeing 777 |
| 6 | `Building` | Công trình xây dựng, tòa nhà chọc trời, cung điện, nhà thờ | Burj Khalifa, Colosseum, Taj Mahal |
| 7 | `NaturalPlace` | Thắng cảnh tự nhiên, ngọn núi, dòng sông, thung lũng | Mount Fuji, Nile River, Sahara Desert |
| 8 | `Village` | Làng mạc, khu định cư nông thôn | Hallstatt, Reine, Shirakawa-go |
| 9 | `Animal` | Thế giới động vật (loài thú, chim, bò sát, côn trùng) | Panthera leo, Orcinus orca, Apis mellifera |
| 10 | `Plant` | Thế giới thực vật (cây thân gỗ, hoa, dược liệu) | Sequoia sempervirens, Nelumbo nucifera |
| 11 | `Album` | Album âm nhạc, đĩa thu âm phòng thu | The Wall, Back in Black, 21 |
| 12 | `Film` | Tác phẩm điện ảnh, phim tài liệu, series truyền hình | Pulp Fiction, Interstellar, Spirited Away |
| 13 | `WrittenWork` | Tác phẩm văn học, tiểu thuyết, thi ca, báo chí | Don Quixote, The Iliad, The New York Times |

---

### 2.4. Thách thức ngữ nghĩa đối với mô hình Decoder-only

DBpedia-14 chứa đựng nhiều thử thách phức tạp về mặt ngữ cảnh:
1. **Sự giao thoa ngữ cảnh nhân vật con người**: Các thực thể `Artist`, `Athlete` và `OfficeHolder` đều có cùng mô thức mở đầu tiểu sử trong Wikipedia ("*is an American...*", "*born in...*", "*attended school at...*").
2. **Sự nhập nhằng giữa doanh nghiệp và kiến trúc**: Nhiều công ty thương mại (`Company`) sở hữu các tòa tháp thương mại hoặc cơ sở hạ tầng (`Building`) mang tên mình.
3. **Mối quan hệ đa phương tiện trong nghệ thuật**: Nhiều bộ phim (`Film`) được chuyển thể từ tiểu thuyết (`WrittenWork`) và phát hành đĩa nhạc phim (`Album`) với tiêu đề và dàn tác giả trùng lặp.
4. **Thử thách từ cơ chế Causal Masking**: Với GPT-2, vì mô hình chỉ được nhìn theo một chiều từ trái sang phải, việc nắm bắt ngữ cảnh phân loại đòi hỏi mô hình phải nén và lưu trữ liên tục thông tin từ đầu câu đến token cuối cùng trước khi đưa ra dự đoán.

---

## 3. Quá trình Fine-tune mô hình GPT-2

### 3.1. Thiết lập kiến trúc phân loại chuỗi

Trong thực nghiệm, kiến trúc `GPT2ForSequenceClassification` từ thư viện Hugging Face Transformers được áp dụng:
1. Nạp trọng số tiền huấn luyện từ `openai-community/gpt2`.
2. Thay thế đầu sinh ngôn ngữ tự do (Causal LM Head) bằng một lớp chiếu tuyến tính (Score/Classification Head) kích thước $768 \rightarrow 14$.
3. Trích xuất trạng thái ẩn tại token không phải PAD cuối cùng của mỗi mẫu văn bản:
   $$h_{\text{last}} = h[\text{batch\_idx}, \text{sequence\_lengths} - 1] \in \mathbb{R}^{768}$$
4. Áp dụng Dropout với xác suất $p = 0.1$ nhằm chống hiện tượng ghi nhớ mẫu (overfitting).
5. Tính toán 14 giá trị logit qua biến đổi tuyến tính:
   $$z = W_{\text{cls}} \cdot \text{Dropout}(h_{\text{last}}) \quad \text{với } W_{\text{cls}} \in \mathbb{R}^{14 \times 768}$$
6. Tối ưu hóa mô hình với hàm mất mát **Cross-Entropy Loss**:
   $$\mathcal{L} = -\sum_{c=0}^{13} y_c \log \left(\frac{e^{z_c}}{\sum_{j=0}^{13} e^{z_j}}\right)$$

---

### 3.2. Tiền xử lý dữ liệu và cấu hình Tokenizer

- **Bộ tách từ**: `AutoTokenizer.from_pretrained("openai-community/gpt2")`.
- **Cấu hình PAD token**: GPT-2 không có token padding mặc định, do đó token `<|endoftext|>` được chỉ định làm padding:
  ```python
  tokenizer.pad_token = tokenizer.eos_token
  model.config.pad_token_id = tokenizer.pad_token_id
  ```
- **Độ dài ngữ cảnh tối đa (`max_length`)**: Thiết lập cố định ở mức **128 tokens**, đi kèm các cờ `truncation=True` và `padding="max_length"`. Độ dài 128 tokens vừa vặn để chứa đựng toàn bộ các câu cốt lõi định nghĩa thực thể Wikipedia, đồng thời tối ưu hóa tài nguyên bộ nhớ VRAM card đồ họa.

---

### 3.3. Bảng tổng hợp các siêu tham số huấn luyện (Hyperparameters)

Quá trình tinh chỉnh mô hình được kiểm soát bởi các tham số tối ưu sau:

| Siêu tham số (Hyperparameter) | Giá trị thiết lập | Phân tích vai trò kỹ thuật |
|:---|:---:|:---|
| **Kiến trúc mô hình gốc** | `openai-community/gpt2` | GPT-2 Small, 124M tham số, Decoder-only Transformer. |
| **Số lượng nhãn (`num_labels`)** | 14 | Tương ứng 14 lớp ontology DBpedia. |
| **Độ dài ngữ cảnh (`max_length`)** | 128 tokens | Đảm bảo bao phủ thông tin thực thể mà vẫn tiết kiệm VRAM. |
| **Batch size mỗi thiết bị (`per_device_train_batch_size`)** | 8 | Tối ưu hóa cho bộ nhớ GPU cá nhân (8GB - 12GB VRAM). |
| **Tích lũy gradient (`gradient_accumulation_steps`)** | 2 | Tích lũy gradient qua 2 micro-batch trước khi cập nhật trọng số. |
| **Kích thước Batch hiệu dụng (Effective Batch Size)** | **16** | $8 \times 2 = 16$, tạo bước cập nhật gradient ổn định. |
| **Tốc độ học (Learning Rate)** | $2 \times 10^{-5}$ ($2\text{e-}5$) | Tốc độ học nhỏ giúp bảo toàn tri thức pre-trained, tránh hiện tượng phá vỡ trọng số. |
| **Bộ tối ưu hóa (Optimizer)** | AdamW | Thuật toán Adam kết hợp phân rã trọng số độc lập (weight decay). |
| **Suy giảm trọng số (Weight Decay)** | 0.01 | Điều chuẩn L2 regularization kiểm soát độ lớn ma trận trọng số. |
| **Lịch trình học (Learning Rate Scheduler)** | Linear with warmup | Tăng dần trong giai đoạn đầu rồi giảm đều theo hàm tuyến tính. |
| **Số epoch huấn luyện (`num_train_epochs`)** | 1 epoch | Với 560.000 mẫu train, 1 epoch đã tương đương 35.000 bước tối ưu. |
| **Tổng số bước tối ưu (`total_steps`)** | **35.000 steps** | $560.000 \text{ mẫu} / 16 = 35.000$ steps cập nhật gradient. |
| **Huấn luyện độ chính xác hỗn hợp (`fp16`)** | `True` | Sử dụng kiểu số thực 16-bit nổi, tăng tốc Tensor Core và giảm 50% dung lượng VRAM. |
| **Tần suất ghi log (`logging_steps`)** | 100 steps | Theo dõi loss và learning rate định kỳ. |
| **Chiến lược lưu mô hình (`save_strategy`)** | `"epoch"` | Lưu lại checkpoint hoàn chỉnh tại bước 35.000 (`results_dbpedia_gpt2`). |

---

### 3.4. So sánh kỹ thuật Fine-tuning: GPT-2 (Decoder-only) vs BERT-base (Encoder-only)

| Tiêu chí kỹ thuật | BERT-base (`bert-base-uncased`) | GPT-2 (`openai-community/gpt2`) |
|:---|:---|:---|
| **Loại kiến trúc** | Encoder-only Transformer | Decoder-only Transformer |
| **Cơ chế Attention** | Bidirectional Self-Attention (2 chiều) | Masked Causal Self-Attention (1 chiều trái sang phải) |
| **Cơ chế Pooling phân loại** | Lấy trạng thái ẩn token đầu tiên `[CLS]` | Lấy trạng thái ẩn token hợp lệ cuối cùng (*Last token*) |
| **Xử lý PAD Token** | Có token `[PAD]` riêng biệt | Không có token PAD riêng; tái sử dụng `<|endoftext|>` |
| **Độ dài ngữ cảnh tối đa** | 512 tokens | 1024 tokens |
| **Kích thước từ điển** | 30.522 (WordPiece, uncased) | 50.257 (Byte-level BPE, case-sensitive) |
| **Xử lý từ ngoài từ vựng** | Có thể sinh token `[UNK]` | Không bao giờ sinh `[UNK]` nhờ xử lý cấp độ byte |
| **Số lượng tham số** | ~110 triệu tham số | ~124 triệu tham số |

---

## 4. Kết quả thực nghiệm và Đánh giá chi tiết

### 4.1. Kết quả định lượng trên tập Train và Test

Mô hình sau khi hoàn thành chu trình fine-tune 1 epoch (checkpoint-35000) được đánh giá toàn diện trên cả tập huấn luyện (560.000 mẫu) và tập kiểm thử độc lập (70.000 mẫu).

#### Bảng tổng hợp kết quả:

| Chỉ số đo lường (Metrics) | Tập Huấn luyện (Train) | Tập Kiểm thử (Test) |
|:---|:---:|:---:|
| **Số lượng mẫu đánh giá** | 560.000 mẫu | 70.000 mẫu |
| **Hàm mất mát (Loss)** | **0.0341** | - |
| **Độ chính xác (Accuracy)** | **0.9936 (99.36%)** | **0.9918 (99.18%)** |
| **Độ chuẩn xác (Precision - Macro)** | - | **0.9918 (99.18%)** |
| **Độ hồi phục (Recall - Macro)** | - | **0.9918 (99.18%)** |
| **Điểm F1-score (Macro)** | - | **0.9918 (99.18%)** |

> [!NOTE]
> **Nhận xét về độ lệch Train/Test và năng lực tổng quát hóa:**
> - Độ chính xác trên tập kiểm thử đạt mức **99.18%**, chỉ kém tập huấn luyện đúng **0.18%** (từ 99.36% xuống 99.18%). Sự chênh lệch cực kỳ nhỏ này khẳng định mô hình **hoàn toàn không bị hiện tượng overfitting (học vẹt)**.
> - Các chỉ số Macro-Precision, Macro-Recall và Macro-F1 đều đạt mức cân bằng tuyệt đối **0.9918**, minh chứng rằng năng lực phân loại của GPT-2 đồng đều xuất sắc trên toàn bộ 14 phân lớp, không bị thiên vị hay sụt giảm hiệu năng ở bất kỳ danh mục cụ thể nào.

---

### 4.2. So sánh đối chiếu kết quả giữa GPT-2 và BERT-base

| Mô hình kiểm nghiệm | Kiến trúc | Train Loss | Train Acc | Test Acc | Macro Precision | Macro Recall | Macro F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BERT-base** | Encoder-only | **0.0220** | **99.54%** | **99.33%** | **0.9933** | **0.9933** | **0.9933** |
| **GPT-2** | Decoder-only | 0.0341 | 99.36% | 99.18% | 0.9918 | 0.9918 | 0.9918 |

**Nhận định chuyên môn**:
- Cả hai mô hình đều đạt kết quả xuất sắc trên ngưỡng 99.1%.
- BERT-base nhỉnh hơn GPT-2 khoảng **0.15%** độ chính xác trên tập test (99.33% so với 99.18%) và có Train Loss thấp hơn (0.0220 so với 0.0341). Kết quả này hoàn toàn phù hợp với lý thuyết xử lý ngôn ngữ tự nhiên: **Cơ chế Self-Attention hai chiều của BERT** cho phép mọi từ trong câu tương tác tự do với cả từ đứng trước và từ đứng sau, tạo ra biểu diễn ngữ cảnh trọn vẹn hơn cho bài toán hiểu ngôn ngữ (NLU/Phân loại chuỗi) so với **cơ chế Causal Masking một chiều của GPT-2**.
- Mặc dù vậy, kết quả 99.18% của GPT-2 khẳng định rằng các mô hình sinh tự hồi quy (Autoregressive Decoder-only) khi được tinh chỉnh đúng cách hoàn toàn có thể đảm đương các tác vụ phân loại chuỗi với độ chính xác tiệm cận mô hình Encoder chuyên biệt.

---

### 4.3. Phân tích ma trận nhầm lẫn (Confusion Matrix) của GPT-2

Dưới đây là ma trận nhầm lẫn chi tiết được trích xuất từ quá trình đánh giá trên toàn bộ 70.000 mẫu của tập kiểm thử DBpedia-14:

![Ma trận nhầm lẫn của GPT-2 trên tập kiểm thử DBpedia-14](../GPT-2.png)

#### Bảng số liệu thống kê chi tiết theo từng lớp (trên 5.000 mẫu/lớp):

| Lớp thực tế (Ground Truth) | Số mẫu đúng | Tỷ lệ đúng (Recall) | Số mẫu sai | Các lớp dự đoán nhầm lẫn phổ biến nhất |
|:---|:---:|:---:|:---:|:---|
| **Company** | 4.882 / 5.000 | 97.64% | 118 | Building (34), EducationalInstitution (30), WrittenWork (18), MeanOfTransportation (16) |
| **EducationalInstitution** | 4.929 / 5.000 | 98.58% | 71 | Company (39), Building (24), OfficeHolder (5) |
| **Artist** | 4.934 / 5.000 | 98.68% | 66 | OfficeHolder (54), Athlete (5), Company (4) |
| **Athlete** | 4.984 / 5.000 | 99.68% | 16 | OfficeHolder (11), Artist (5) |
| **OfficeHolder** | 4.944 / 5.000 | 98.88% | 56 | Artist (41), Athlete (7), Building (4), Company (2) |
| **MeanOfTransportation** | 4.983 / 5.000 | 99.66% | 17 | Company (14), Building (2), WrittenWork (1) |
| **Building** | 4.911 / 5.000 | 98.22% | 89 | Company (43), EducationalInstitution (22), MeanOfTransportation (10), NaturalPlace (9) |
| **NaturalPlace** | 4.982 / 5.000 | 99.64% | 18 | Building (9), Village (7), Company (1), Artist (1) |
| **Village** | **4.996 / 5.000** | **99.92%** | **4** | NaturalPlace (2), EducationalInstitution (1), WrittenWork (1) |
| **Animal** | 4.989 / 5.000 | 99.78% | 11 | Plant (9), Athlete (1), WrittenWork (1) |
| **Plant** | 4.977 / 5.000 | 99.54% | 23 | Company (11), Animal (10), MeanOfTransportation (1), Building (1) |
| **Album** | 4.982 / 5.000 | 99.64% | 18 | Film (6), Artist (5), WrittenWork (5), Company (2) |
| **Film** | 4.965 / 5.000 | 99.30% | 35 | WrittenWork (15), Album (13), Company (4), Artist (3) |
| **WrittenWork** | 4.966 / 5.000 | 99.32% | 34 | Company (16), Film (8), Building (3), Artist (2), OfficeHolder (2), Album (2) |
| **Toàn bộ tập test** | **69.424 / 70.000** | **99.18%** | **576** | **Tỷ lệ sai sót toàn tập chỉ chiếm 0.82%** |

---

### 4.4. Phân tích chuyên sâu về các nguyên nhân gây sai số

Trên toàn bộ 70.000 mẫu kiểm thử, GPT-2 dự đoán chính xác **69.424 mẫu** và chỉ mắc lỗi ở **576 mẫu** (tương ứng tỷ lệ lỗi chỉ **0.82%**). Khi mổ xẻ các trường hợp sai lệch, nguyên nhân chủ yếu bắt nguồn từ các điểm nghẽn ngữ nghĩa thực tế:

#### 1. Nhầm lẫn nhân vật: `Artist` $\leftrightarrow$ `OfficeHolder`
- Có **54 mẫu** `Artist` bị nhầm thành `OfficeHolder` và **41 mẫu** `OfficeHolder` bị nhầm thành `Artist`.
- *Bản chất ngôn ngữ*: Cả hai đều mô tả thực thể người (*Person*). Nhiều nghệ sĩ, nhà văn tích cực tham gia chính trường và nắm giữ các cương vị nhà nước; ngược lại nhiều chính khách nổi tiếng là nhà văn hoặc tác giả hồi ký. Đoạn tóm tắt Wikipedia về hai nhóm đối tượng này chia sẻ chung nhiều từ vựng về chức vụ, giải thưởng và tiểu sử.

#### 2. Nhầm lẫn tổ chức và kiến trúc: `Company` $\leftrightarrow$ `Building` $\leftrightarrow$ `EducationalInstitution`
- Có **43 mẫu** `Building` nhầm thành `Company` và **34 mẫu** `Company` nhầm thành `Building`.
- Có **39 mẫu** `EducationalInstitution` nhầm thành `Company` và **30 mẫu** `Company` nhầm thành `EducationalInstitution`.
- *Bản chất ngôn ngữ*: Các bài viết Wikipedia về trụ sở chính công ty thường miêu tả sâu về thiết kế kiến trúc, diện tích sàn và lịch sử xây dựng tòa nhà. Tương tự, nhiều học viện đào tạo và viện nghiên cứu hoạt động dưới hình thức doanh nghiệp tư nhân, khiến văn bản chứa các thuật ngữ pháp lý và tài chính gần như tương đồng với các công ty thương mại.

#### 3. Nhầm lẫn giữa các tác phẩm nghệ thuật: `Film` $\leftrightarrow$ `Album` $\leftrightarrow$ `WrittenWork`
- `Film` bị nhầm sang `WrittenWork` (15 mẫu) và `Album` (13 mẫu).
- `WrittenWork` bị nhầm sang `Film` (8 mẫu).
- `Album` bị nhầm sang `Film` (6 mẫu) và `WrittenWork` (5 mẫu).
- *Bản chất ngôn ngữ*: Tính chất chuyển thể đa nền tảng trong văn hóa đại chúng. Một bộ phim chiếu rạp kinh điển thường đi kèm album nhạc phim (*soundtrack*) và được chuyển thể từ tiểu thuyết gốc. Nội dung tóm tắt phim thường liệt kê tên cuốn sách, tác giả kịch bản và nhà sản xuất âm nhạc, tạo nên sự phân tán tín hiệu chú ý của mô hình.

#### 4. Các nhóm đối tượng đạt độ chính xác gần như tuyệt đối:
- **`Village` (99.92% - 4.996/5.000 mẫu đúng, chỉ sai 4 mẫu)**: Tương tự như trên BERT, các mục về làng xã sở hữu cấu trúc định vị địa lý cực kỳ chuẩn tắc và bất biến ("*is a village and civil parish...*", tọa độ địa lý, dân số), giúp mô hình nhận diện dễ dàng.
- **`Animal` (99.78%)**, **`Athlete` (99.68%)**, **`MeanOfTransportation` (99.66%)**: Các lĩnh vực này có vốn từ vựng đặc thù (danh pháp sinh học, tên phương tiện kỹ thuật hàng không/hàng hải, thuật ngữ giải đấu thể thao) gần như không bị trùng lặp với các danh mục khác.

---

## 5. Kết luận

Thực nghiệm fine-tune mô hình **GPT-2** trên tập dữ liệu phân loại bản thể học **DBpedia-14** mang lại những bài học giá trị:

1. **Hiệu năng ấn tượng của kiến trúc Decoder-only**:
   Mặc dù được thiết kế ban đầu cho bài toán sinh tự hồi quy một chiều, GPT-2 Small (124M tham số) chứng minh năng lực hiểu ngữ nghĩa cực kỳ mạnh mẽ khi đạt độ chính xác **99.18%** trên 70.000 mẫu kiểm thử DBpedia-14 chỉ sau **1 epoch** huấn luyện.

2. **Kỹ thuật thích ứng hiệu quả**:
   Việc tái sử dụng `<|endoftext|>` làm PAD token, kết hợp cơ chế lấy đặc trưng tại token cuối cùng (*Last Token Pooling*) và tối ưu hóa với hàm mất mát Cross-Entropy đã biến đổi thành công một mô hình sinh ngôn ngữ thành một bộ phân loại chuỗi chính xác cao.

3. **Tổng quát hóa vững chắc**:
   Độ lệch giữa độ chính xác tập huấn luyện (99.36%) và tập kiểm thử (99.18%) chỉ là **0.18%**, khẳng định chiến lược tiền huấn luyện trên kho ngữ liệu quy mô lớn WebText giúp mô hình có khả năng khái quát hóa vượt trội trên các phân phối dữ liệu mới.

