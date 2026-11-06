# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603919
- **Họ và tên:** Trần Duy Thanh
- **SEED:** 5067

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑  |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑  |


## 2. Kết quả chính

| Cách làm | F1 test | F1 SUB 40 | p95 (ms/tin) |
| --- | ---: | ---: | ---: | ---: |
| BoW + LogReg | 0.928 | 0.974 | 0.108 |
| BoW + MLP | 0.974 | 0.974 | 0.379 |
| Encoder + kNN | 0.936 | 1.000 | 54.964 |
| Encoder + LogReg | 0.980 | 0.950 | 32.865 |
| Encoder + MLP | 0.985 | 0.976 | 33.319 |
| LLM nhỏ zero-shot | — | 0.974 | 1526.029 |
| LLM lớn zero-shot | — | 1.000 | 1232.793 |
| LLM nhỏ few-shot | — | 1.000 | 1604.817 |

LLM chỉ được đánh giá trên SUB 40. Encoder + MLP có F1 test cao nhất trong các cách đạt p95 dưới 50 ms, còn LLM khá chậm.

## 3. Dự đoán và thực tế

- **BoW + LogReg:**  đoán F1 khoảng 0.85–0.90 nhưng thực tế đạt **0.928**. Có lẽ những từ khóa trong tập test vẫn khá giống dữ liệu train nên mô hình làm tốt hơn tôi nghĩ.
- **BoW + MLP:** mình nghĩ F1 sẽ tương đương hoặc thấp hơn LogReg do dễ overfit, nhưng thực tế **0.974**, cao hơn LogReg (**0.928**). Như vậy thêm lớp phi tuyến vẫn có ích cho dữ liệu này.
- **Stress test:** Tôi đoán LLM đúng khoảng 75–90%, nhưng thực tế đạt **100%** trên bộ case đã thử.

## 4. Phân tích 50 case khó

Đã thêm 5 case nên lần chạy cuối có **55 case**. **BoW + MLP tụt nhiều nhất**, F1 từ **0.974** xuống **0.341**. Mô hình này đạt **0% đúng** ở nhóm spam giọng tự nhiên và né từ khóa.

Hai ví dụ trong bộ test:
- **Case #11:** “Hey it's Jess from the gym ... 3 months for the price of 1 ...”. Đây là spam quảng cáo nhưng viết như người quen nên BoW + MLP đoán sai.
- **Case #29:** “C0ngr4tulat10ns! Y0u h4ve w0n ...”. Chữ bị thay bằng số, không khớp từ khóa quen thuộc nên BoW + MLP cũng đoán sai.


## 5. Thử thách cá nhân

**Đề #4 — Embedding:** Tôi chạy `knn_predict` với `k = 1, 5, 25` trên tập test:

| k | F1 |
| ---: | ---: |
| 1 | 0.9109 |
| 5 | **0.9360** |
| 25 | 0.8932 |

Trong ba giá trị thì **k=5 tốt nhất**. K quá nhỏ nhạy với nhiễu hoặc nhãn sai; k quá lớn dễ bị lớp ham đông hơn lấn át nên bỏ sót spam.

## 6. Phương án cho sản phẩm thật

chọn **Encoder + MLP** nhưng thông lượng đo được mới khoảng **12 tin/s**, thấp hơn mức trung bình khoảng **116 tin/s**, nên phải chạy song song hoặc tối ưu batching.

Về phương án thì có thể làm **router**: BoW + MLP xử lý tin có xác suất chắc chắn (`p < 0.3` hoặc `p > 0.7`, hoặc tùy rule, có thể phân tích dataset trước khi set ngưỡng), chỉ chuyển tin không chắc chắnsang LLM. Như vậy giảm số lần gọi API, nhưng nhánh LLM vẫn có thể vượt 50 ms.

## 7. AI log và điều đã học

| Phần | Tôi hỏi AI gì | AI hỗ trợ gì | Tôi kiểm tra / chỉnh gì |
| --- | --- | --- | --- |
| 4a | Vì sao kết quả toàn invalid? | Gợi ý kiểm tra `max_tokens`, `reasoning_effort` và phản hồi | Thử tăng `max_tokens` từ 3 lên 64, chạy lại thấy `invalid=0` |

Sau bài này,  có thể thấy là F1 cao chưa chắc đã là lựa chọn tốt nhất vì còn có độ trễ và chi phí. Ngoài ra cũng có thể nhận thấy rằng tập test thông thường có thể làm mình quá tự tin, nên cần stress test và kiểm tra lỗi cụ thể.