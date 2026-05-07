# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Trí Cao - 2A202600223
**Submission option**: B (HF Hub)

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples (180 train + 20 eval)
- **max_seq_length**: 1024 (p95 ≈ 562, rounded up to power of 2)
- **GPU**: Tesla T4, 16 GB VRAM
- **Training cost**: $0.07 (~12.2 phút @ $0.35/hr)
- **HF Hub link**: https://huggingface.co/Kao1412/qwen2.5-3b-vi-lab21-r16

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8    | 1,843,200       | 4.00 min   | 7.22 GB   | 1.5577    | 4.7479     |
| 16   | 3,686,400       | 4.26 min   | 6.62 GB   | 1.5161    | 4.5544     |
| 64   | 14,745,600      | 3.99 min   | 8.00 GB   | 1.4768    | 4.3790     |
| **16 (All)** | **29,933,568** | **3.97 min** | **12.77 GB** | **1.5057** | **4.5074** |
| Base | -               | -          | -         | 1.6214    | 5.0581     |

## 3. Loss Curve Analysis
- **Quan sát**: Loss giảm ổn định qua 3 epoch cho cả 3 cấu hình rank. Không có dấu hiệu overfitting rõ rệt do tập dữ liệu nhỏ và mô hình 3B có khả năng tổng quát hóa tốt.
- **Lý do**: Việc sử dụng Unsloth giúp ổn định quá trình hội tụ, và mức rank cao hơn (r=64) cho phép mô hình học được nhiều sắc thái ngôn ngữ tiếng Việt hơn từ tập Alpaca.

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.
**Base**: Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu...
**Fine-tuned (r=16)**: Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI...
**Nhận xét**: Improved. Câu trả lời của mô hình FT chuyên nghiệp và súc tích hơn.

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.
**Base**: Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp. [Code snippet cơ bản]
**Fine-tuned (r=16)**: Để tính số Fibonacci thứ n, bạn có thể viết một đoạn code Python như sau: [Code snippet có xử lý ValueError và tối ưu hơn]
**Nhận xét**: Improved. Bản FT có thêm các bước kiểm tra đầu vào (input validation).

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.
**Base**: 1. Thân thiện với người dùng... sắp xếp bố cục, màu sắc...
**Fine-tuned (r=16)**: 1. Chuyển đổi: hướng tới hành động. 2. Thích ứng: đa thiết bị. 3. Đơn giản...
**Nhận xét**: Improved. Các nguyên tắc được liệt kê có tính chuyên môn cao hơn (conversion, adaptive).

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.
**Base**: LoRA và QLoRA là hai phương pháp cải thiện hiệu năng mô hình NLU...
**Fine-tuned (r=16)**: LoRA (Layer-wise Adaptive...) và QLoRA là hai phương pháp regularization/quantization được phát triển để cải thiện hiệu quả...
**Nhận xét**: Improved. Giải thích rõ ràng hơn về cơ chế quantization của QLoRA.

### Example 5
**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.
**Base**: Prompt engineering, RAG, và fine-tuning là ba cách khác nhau để cải thiện hiệu suất...
**Fine-tuned (r=16)**: Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau... Prompt engineering tập trung vào xây dựng câu lệnh...
**Nhận xét**: Same/Slightly Improved. Cấu trúc câu văn mạch lạc hơn.

## 5. Bonus Section: Target ALL Layers
- **Thực nghiệm**: Huấn luyện thêm 1 adapter với `rank=16` nhưng tác động vào tất cả các lớp tuyến tính (`q, k, v, o, gate, up, down`).
- **Kết quả**:
    - **Perplexity** giảm từ **4.5544** (chỉ target q,v) xuống **4.5074**.
    - **Trainable Parameters** tăng từ ~3.6M lên ~29.9M (gấp ~8 lần).
    - **Peak VRAM** tăng từ **6.62 GB** lên **12.77 GB**.
- **Nhận xét**: Việc target toàn bộ các lớp mang lại độ chính xác cao hơn so với việc chỉ tăng rank trên 2 lớp (q,v). Tuy nhiên, chi phí bộ nhớ VRAM tăng vọt, gần chạm ngưỡng giới hạn của T4 (16GB). Đây là một sự đánh đổi đáng cân nhắc khi tài nguyên dư dả.

## 6. Conclusion về Rank Trade-off

- **Rank ROI tốt nhất**: **r=64**. Mặc dù số lượng tham số tăng gấp 4 lần so với r=16, nhưng mức tiêu thụ VRAM chỉ tăng thêm khoảng 1.4 GB (vẫn nằm an toàn trong giới hạn 16GB của T4). Đổi lại, Perplexity giảm đáng kể xuống 4.379, mang lại câu trả lời chất lượng nhất.
- **Hiệu quả của All Layers**: Thí nghiệm Bonus cho thấy việc target nhiều lớp hơn có hiệu quả tương đương với việc tăng rank cao. Cụ thể, bản `r=16 (All Layers)` đạt Perplexity gần tiệm cận bản `r=64` nhưng tốn VRAM hơn nhiều.
- **Recommendation**: Nếu deploy production trên hạ tầng có giới hạn như T4, tôi khuyến nghị chọn **r=16** hoặc **r=32** và target **All Layers** nếu VRAM cho phép. Nếu cần tối ưu tốc độ và bộ nhớ, chỉ target `q, v` với **r=64** là đủ tốt.

## 7. What I Learned
- **Sức mạnh của Unsloth**: Khả năng tối ưu VRAM thực sự ấn tượng, giúp việc fine-tune "All Layers" vẫn khả thi trên Tesla T4.
- **Ảnh hưởng của Rank & Layers**: Rank ảnh hưởng đến độ sâu kiến thức, trong khi việc target nhiều lớp giúp mô hình phối hợp các thành phần (Attention & MLP) nhịp nhàng hơn.
- **Manual Evaluation**: Kỹ năng quan trọng để kiểm soát quá trình đánh giá khi gặp giới hạn phần cứng.
