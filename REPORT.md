# Building Machine Translation English - Vietnamese

## Đề Bài (Problem Statement)
- **Task**: Building Machine Translation English - Vietnamese
- **Reimplementation with 2 cases**:
  1. **Google Colab**: `demo_transformer.ipynb` (Transformer Architecture)
  2. **GitHub NMT**: [tensorflow/nmt](https://github.com/tensorflow/nmt) (Seq2Seq + Luong Attention Architecture)
- **Comparison Corpus**: English - Vietnamese Parallel Corpus (**IWSLT 2015**)

> Ghi chú: Case NMT (Seq2Seq) được **reimplement bằng PyTorch** theo kiến trúc/hparams của `tensorflow/nmt` (IWSLT15), không chạy TensorFlow 1.x gốc. Case Transformer reimplement từ `demo_transformer.ipynb`.

---

## 1. Tổng Quan Bộ Dữ Liệu (IWSLT 2015 Corpus)
- **Tập Train (`train.en`, `train.vi`)**: 133,317 cặp câu (lọc độ dài ≤ 80 khi train)
- **Tập Validation (`tst2012`)**: 1,553 cặp câu
- **Tập Test (`tst2013`)**: 1,268 cặp câu
- **Vocabulary (build từ train, min_freq=2, max_size=30k)**: EN: 28145 | VI: 12234

---

## 2. Reimplementation Case 1: GitHub NMT (Seq2Seq + Luong Attention)
Dựa trên kiến trúc `tensorflow/nmt` (hparams `iwslt15.json`):
- **Encoder**: 2-layer Bidirectional LSTM (hidden mỗi hướng = 256, concat = 512).
- **Decoder**: 2-layer Unidirectional LSTM (hidden = 512).
- **Attention**: Scaled Luong Attention + Input Feeding.
- **Embedding**: 512
- **Tổng tham số**: **36.1M**
- **Train**: 20 epochs | train loss cuối 2.6823 | val loss tốt nhất-track 3.3851
- **Thời gian train**: 6410.72s

---

## 3. Reimplementation Case 2: Google Colab (`demo_transformer.ipynb`)
Dựa trên notebook `demo_transformer.ipynb` (PyTorch):
- **Kiến trúc**: Transformer Encoder-Decoder.
- **Cấu hình thực tế**: $N=4$ layers, $d_{model}=512$, $d_{ff}=2048$, $h=8$ heads.
- **Positional Encoding**: Sinusoidal.
- **Tổng tham số**: **50.1M**
- **Train**: 25 epochs | train loss cuối 0.8640 | val loss cuối 1.1684
- **Thời gian train**: 1778.74s

---

## 4. Bảng So Sánh Kết Quả (Fair Comparison)

Cả hai model được đánh giá lại trên **cùng** `tst2013`, **cùng tokenizer**, với **cùng chế độ decode**.

| Tiêu chí | Case 1: Seq2Seq + Luong Attn | Case 2: Transformer |
|---|---|---|
| **Mô hình** | 2-layer Bi-LSTM + Luong | 4-layer Multi-Head Transformer |
| **Tổng tham số** | **36.1M** | **50.1M** |
| **Số Epochs train** | 20 | 25 |
| **Train Loss cuối** | 2.6823 | 0.8640 |
| **Val Loss cuối (`tst2012`)** | 3.3851 | 1.1684 |
| **BLEU greedy (`tst2013`)** | **27.49** | **28.81** |
| **BLEU beam=5 (`tst2013`)** | **28.71** | **30.00** |
| **Thời gian huấn luyện** | 6410.72s | **1778.74s** |

*Primary decode mode for qualitative samples: **beam**.

---

## 5. Ví Dụ Dịch Thực Tế Trên `tst2013` (decode=beam)

### Ví dụ 1:
- **Input (EN)**: `When I was little , I thought my country was the best on the planet , and I grew up singing a song called &quot; Nothing To Envy . &quot;`
- **Reference (VI)**: `Khi tôi còn nhỏ , Tôi nghĩ rằng BắcTriều Tiên là đất nước tốt nhất trên thế giới và tôi thường hát bài &quot; Chúng ta chẳng có gì phải ghen tị . &quot;`
- **Case 1 (Seq2Seq)**: `khi tôi còn nhỏ , tôi nghĩ đất nước mình là người giỏi nhất hành tinh , và tôi lớn lên hát một bài hát tên là &quot không có gì để ghen tị . &quot`
- **Case 2 (Transformer)**: `khi tôi còn nhỏ , tôi nghĩ đất nước tôi là người giỏi nhất trên hành tinh , và tôi lớn lên hát một bài hát có tên là &quot không có gì để ghen tị &quot .`

### Ví dụ 2:
- **Input (EN)**: `And I was very proud .`
- **Reference (VI)**: `Tôi đã rất tự hào về đất nước tôi .`
- **Case 1 (Seq2Seq)**: `và tôi đã rất tự hào .`
- **Case 2 (Transformer)**: `và tôi rất tự hào .`

### Ví dụ 3:
- **Input (EN)**: `In school , we spent a lot of time studying the history of Kim Il-Sung , but we never learned much about the outside world , except that America , South Korea , Japan are the enemies .`
- **Reference (VI)**: `Ở trường , chúng tôi dành rất nhiều thời gian để học về cuộc đời của chủ tịch Kim II- Sung , nhưng lại không học nhiều về thế giới bên ngoài , ngoại trừ việc Hoa Kỳ , Hàn Quốc và Nhật Bản là kẻ thù của chúng tôi .`
- **Case 1 (Seq2Seq)**: `ở trường , chúng tôi đã dành rất nhiều thời gian nghiên cứu lịch sử của kim <unk> , nhưng chúng tôi chưa bao giờ học được nhiều về thế giới bên ngoài , ngoại trừ nước mỹ , nam hàn , nhật bản là kẻ thù .`
- **Case 2 (Transformer)**: `ở trường , chúng tôi đã dành rất nhiều thời gian nghiên cứu lịch sử của kim <unk> , nhưng chúng tôi chưa bao giờ học được nhiều về thế giới bên ngoài , ngoại trừ nước mỹ , hàn quốc , nhật bản là những kẻ thù .`

### Ví dụ 4:
- **Input (EN)**: `Although I often wondered about the outside world , I thought I would spend my entire life in North Korea , until everything suddenly changed .`
- **Reference (VI)**: `Mặc dù tôi đã từng tự hỏi không biết thế giới bên ngoài kia như thế nào , nhưng tôi vẫn nghĩ rằng mình sẽ sống cả cuộc đời ở BắcTriều Tiên , cho tới khi tất cả mọi thứ đột nhiên thay đổi .`
- **Case 1 (Seq2Seq)**: `mặc dù tôi thường tự hỏi về thế giới bên ngoài , tôi nghĩ tôi sẽ dành cả đời mình ở bắc triều tiên , cho đến mọi thứ đột ngột thay đổi .`
- **Case 2 (Transformer)**: `mặc dù tôi thường tự hỏi về thế giới bên ngoài , tôi nghĩ mình sẽ dành toàn bộ cuộc sống của mình ở bắc triều tiên , cho đến khi mọi thứ đột nhiên thay đổi .`

### Ví dụ 5:
- **Input (EN)**: `When I was seven years old , I saw my first public execution , but I thought my life in North Korea was normal .`
- **Reference (VI)**: `Khi tôi lên 7 , tôi chứng kiến cảnh người ta xử bắn công khai lần đầu tiên trong đời , nhưng tôi vẫn nghĩ cuộc sống của mình ở đây là hoàn toàn bình thường .`
- **Case 1 (Seq2Seq)**: `khi tôi bảy tuổi , tôi thấy cuộc hành trình công cộng đầu tiên của mình , nhưng tôi nghĩ cuộc sống của mình ở bắc triều tiên là bình thường .`
- **Case 2 (Transformer)**: `khi tôi 7 tuổi , tôi thấy cuộc hành hình công cộng đầu tiên của mình , nhưng tôi nghĩ cuộc sống của tôi ở bắc triều tiên là bình thường .`

---

## 6. Kết Luận
1. **Chất lượng dịch (fair, beam)**: Transformer cao hơn (**30.00 vs 28.71**, Δ = **1.30 BLEU**).
2. **Cùng chiều hướng với decode greedy**: Transformer 28.81 vs Seq2Seq 27.49 (Δ = +1.32).
3. **Tốc độ train**: Transformer nhanh hơn khoảng **3.6×** (29.6 phút vs 106.8 phút).
4. **Fairness**: So sánh BLEU đã dùng cùng corpus IWSLT15, cùng split, cùng tokenizer và cùng chế độ decode (greedy và/hoặc beam).
