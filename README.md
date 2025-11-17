Báo cáo môn học: Hệ Thống Nhúng Nâng Cao
Đề tài: Training and Optimization of SSDLite MobileNetV3 for Vehicle Detection on Embedded Systems
Thành viên thực hiện

Bùi Nguyễn Hoài Thương

Trương Vũ Hoàng Anh

Phạm Thanh Toàn

Mô tả các file trong thư mục
Retrain_Prunning_Quantization.ipynb

Notebook chính của dự án.
Chứa toàn bộ quy trình fine-tuning, pruning bằng torch-pruning, và export ONNX FP32/INT8.

Test_FPS.py

Script đo FPS.
Load model, chạy inference trên video, vẽ bounding box và tính tốc độ khung hình thực tế.

ssdlite320_mbv3_finetuned_full.pth

Mô hình PyTorch sau khi fine-tune (baseline FP32, chưa prune).

ssdlite320_mbv3_pruned_finetuned_full.pth

Mô hình PyTorch đã được prune và train lại, dùng để xuất ONNX.

ssd_pruned_fp32.onnx

Mô hình ONNX sau pruning, định dạng FP32.
Dùng để benchmark hoặc làm bước trung gian trước INT8.

ssd_pruned_int8.onnx

Mô hình ONNX INT8 sau quantization.
Kích thước nhỏ, chạy nhanh, phù hợp triển khai nhúng.

README.md

Tài liệu mô tả dự án và hướng dẫn sử dụng.
