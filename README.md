# Tối Ưu Hóa và Huấn Luyện Mô Hình SSDLite MobileNetV3 Cho Phát Hiện Phương Tiện Trên Hệ Thống Nhúng

Đây là dự án môn học **Hệ Thống Nhúng Nâng Cao**, tập trung vào việc huấn luyện, tối ưu hóa (Pruning và Quantization) mô hình $\text{SSDLite MobileNetV3}$ để đạt hiệu suất cao (tốc độ xử lý cao, kích thước nhỏ) khi triển khai trên các nền tảng nhúng.

---

## 🎯 Mục Tiêu Dự Án

Mục tiêu chính của dự án là tối ưu hóa mô hình $\text{SSDLite MobileNetV3}$ cho tác vụ **Phát hiện Phương tiện Giao thông** để đáp ứng các tiêu chí triển khai trên thiết bị nhúng:

1.  **Tăng tốc độ Inference (FPS)**: Giảm độ trễ xử lý.
2.  **Giảm kích thước mô hình**: Tối ưu hóa bộ nhớ.
3.  **Duy trì độ chính xác (mAP)**: Giảm thiểu sự suy giảm hiệu suất sau tối ưu hóa.

---

## 🛠️ Phương Pháp Luận và Công Cụ

Dự án áp dụng các kỹ thuật chính sau:

| Giai đoạn | Kỹ thuật | Công cụ chính | Mô tả |
| :--- | :--- | :--- | :--- |
| **1. Huấn luyện** | Fine-Tuning | PyTorch | Huấn luyện mô hình $\text{SSDLite MobileNetV3}$ trên tập dữ liệu phương tiện giao thông để tạo ra mô hình $\text{baseline}$ (FP32). |
| **2. Tối ưu hóa** | Pruning Kênh | $\text{torch-pruning}$ | Cắt bỏ $\text{20%}$ kênh ít quan trọng nhất trên toàn mạng theo chiến lược **Global Pruning** (Norm L1). |
| **3. Định dạng** | Quantization | ONNX | Chuyển đổi mô hình $\text{FP32}$ đã prune sang định dạng $\text{INT8}$ để giảm $\text{4x}$ kích thước và tăng tốc độ inference. |
| **4. Triển khai** | Inference & Benchmark | Python, OpenCV | Đánh giá tốc độ xử lý thực tế ($\text{FPS}$) của các phiên bản mô hình ($\text{FP32}$ và $\text{INT8}$). |

---

## 📂 Cấu Trúc Thư Mục và Mô Tả Tệp

| Tệp/Thư mục | Mô tả | Giai đoạn |
| :--- | :--- | :--- |
| **`Retrain_Prunning_Quantization.ipynb`** | **Notebook chính**. Chứa toàn bộ quy trình: Fine-tuning, Pruning, Finetuning sau Pruning và $\text{Export}$ mô hình sang $\text{ONNX}$ ($\text{FP32}$ và $\text{INT8}$). | Toàn bộ quy trình |
| **`Test_FPS.py`** | Script thực thi. Dùng để $\text{Load}$ mô hình ($\text{ONNX}$), chạy $\text{inference}$ trên video, vẽ $\text{bounding box}$ và tính toán **tốc độ khung hình thực tế** ($\text{FPS}$). | Đánh giá/Triển khai |
| **`ssdlite320_mbv3_finetuned_full.pth`** | Mô hình $\text{PyTorch}$ **Baseline** sau $\text{Fine-tune}$ ($\text{FP32}$, chưa $\text{prune}$). | Output Giai đoạn 1 |
| **`ssdlite320_mbv3_pruned_finetuned_full.pth`** | Mô hình $\text{PyTorch}$ đã được **Prune** và **Train lại** ($\text{Finetune}$ phục hồi). Đây là mô hình nguồn để $\text{Export}$ sang $\text{ONNX}$. | Output Giai đoạn 2 |
| **`ssd_pruned_fp32.onnx`** | Mô hình $\text{ONNX}$ sau $\text{Pruning}$, định dạng **$\text{FP32}$**. Dùng để $\text{benchmark}$ $\text{FP32}$ sau $\text{pruning}$. | Output Giai đoạn 3 |
| **`ssd_pruned_int8.onnx`** | Mô hình $\text{ONNX}$ sau $\text{Quantization}$ định dạng **$\text{INT8}$**. Kích thước nhỏ, tốc độ cao, sẵn sàng triển khai trên hệ thống nhúng. | Output Giai đoạn 3 |

---

## 👥 Thành Viên Thực Hiện

| Tên thành viên | Mã số sinh viên |
| :--- | :--- |
| Bùi Nguyễn Hoài Thương | (Điền MSV) |
| Trương Vũ Hoàng Anh | (Điền MSV) |
| Phạm Thanh Toàn | (Điền MSV) |

---

## 🏃 Hướng Dẫn Sử Dụng (Quick Start)

1.  **Môi trường:** Đảm bảo bạn đã cài đặt $\text{PyTorch}$, $\text{torch-pruning}$ và các thư viện hỗ trợ $\text{ONNX/OpenCV}$.
2.  **Huấn luyện/Tối ưu hóa:** Mở và chạy $\text{Notebook}$ **`Retrain_Prunning_Quantization.ipynb`** theo trình tự các bước để tái tạo các mô hình tối ưu.
3.  **Kiểm tra FPS:** Chỉnh sửa đường dẫn video trong $\text{Script}$ **`Test_FPS.py`** và chạy để đánh giá tốc độ xử lý thực tế của mô hình $\text{INT8}$ đã tối ưu.

```bash
# Ví dụ chạy script kiểm tra FPS (sau khi có file ssd_pruned_int8.onnx)
python Test_FPS.py --model_path ssd_pruned_int8.onnx --video_source my_video.mp4
