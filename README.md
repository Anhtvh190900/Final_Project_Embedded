# Báo cáo môn học: Hệ Thống Nhúng Nâng Cao

## Đề tài
**Training and Optimization of SSD Lite MobileNet V2 for Vehicle Detection on Embedded Systems**

Đề tài tập trung vào:
- Huấn luyện mô hình **SSD Lite MobileNet V2** trên dữ liệu phương tiện giao thông tại Việt Nam  
- Tối ưu mô hình bằng **Pruning** và **Quantization (INT8)**  
- Chuyển đổi mô hình sang **TF-Lite / ONNX / TensorRT**  
- Kiểm thử mô hình trên nền tảng **hệ thống nhúng**  
- Đánh giá hiệu năng thông qua mAP, FPS và kích thước mô hình  

---

## Học viên thực hiện
- **Bùi Nguyễn Hoài Thương**  
- **Trương Vũ Hoàng Anh**  
- **Phạm Thanh Toàn**

---

## 🎥 Video Test  
Video kiểm thử mô hình trên dữ liệu thực tế:

🔗 https://drive.google.com/file/d/1TqE110X4sbsguWQiXTFAC9JnlVz-g0zL/view?usp=sharing

---

## 📄 Nội dung chính của đồ án

### 1. Chuẩn bị và tiền xử lý dữ liệu
- Chuyển đổi dataset gốc sang **COCO-style**  
- Sinh file **TFRecord** để train trên TensorFlow  
- Sinh dataset COCO JSON để optimize bằng PyTorch  

### 2. Training & Fine-tuning
- Sử dụng mô hình **SSD Lite MobileNet V2 FPNLite 320×320 pretrained COCO**  
- Fine-tune trên bộ dữ liệu **Vehicle Vietnam – Cần Thơ**  
- Huấn luyện với optimizer Momentum + Cosine LR Decay  

### 3. Đánh giá Baseline
- Tính toán **mAP (COCO metrics)**  
- Đo **FPS** trên GPU/CPU  
- Lưu mô hình baseline FP32  

### 4. Pruning (PyTorch)
- Áp dụng **Structured Channel Pruning**  
- Giảm FLOPs và số tham số  
- Fine-tune lại để phục hồi accuracy  

### 5. Quantization (ONNX INT8)
- Export mô hình sang ONNX FP32  
- Áp dụng **Post-training Dynamic Quantization**  
- Tối ưu kích thước mô hình phục vụ nhúng  

### 6. Đánh giá mô hình tối ưu
  

---


