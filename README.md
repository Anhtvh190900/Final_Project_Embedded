# Training and Optimization of SSDLite MobileNetV3 for Vehicle Detection on Embedded Systems

This project, part of the **Advanced Embedded Systems** course, focuses on training and optimizing the $\text{SSDLite MobileNetV3}$ model using Channel Pruning and Quantization techniques. The goal is to achieve high performance (fast inference speed, small size) for deployment on resource-constrained embedded platforms.

---

## Project Goals

The primary objective is to optimize the $\text{SSDLite MobileNetV3}$ model for **Vehicle Detection** to meet embedded system deployment criteria:

1.  **Increase Inference Speed (FPS)**: Minimize processing latency.
2.  **Reduce Model Size**: Optimize memory footprint.
3.  **Maintain Detection Accuracy (mAP)**: Minimize performance degradation after optimization.

---

## Methodology and Key Tools

The project employs the following core techniques:

| Phase | Technique | Key Tool | Description |
| :--- | :--- | :--- | :--- |
| **1. Training** | Fine-Tuning | PyTorch | Fine-tuning the $\text{SSDLite MobileNetV3}$ model on a vehicle detection dataset to establish an $\text{FP32 baseline}$. |
| **2. Optimization** | Channel Pruning | $\text{torch-pruning}$ | Removing $\text{20%}$ of the least important channels across the network using the **Global Pruning** strategy ($\text{L1}$ Norm based). |
| **3. Format Conversion** | Quantization | ONNX | Converting the pruned $\text{FP32}$ model to the $\text{INT8}$ format to achieve up to $\text{4x}$ size reduction and inference acceleration. |
| **4. Deployment** | Inference & Benchmark | Python, OpenCV | Evaluating the real-world processing speed ($\text{FPS}$) of the optimized $\text{INT8}$ model. |

---

## Project Structure and File Descriptions

| File/Folder | Description | Phase Output |
| :--- | :--- | :--- |
| **`Retrain_Prunning_Quantization.ipynb`** | **Main Project Notebook**. Contains the complete pipeline: Fine-tuning, Pruning, Post-Pruning Finetuning, and $\text{Export}$ to $\text{ONNX}$ ($\text{FP32}$ and $\text{INT8}$). | Full Pipeline |
| **`Test_FPS.py`** | Execution Script. Used to $\text{Load}$ the $\text{ONNX}$ model, run $\text{inference}$ on a video stream, draw $\text{bounding boxes}$, and calculate the **real-time Frame Per Second ($\text{FPS}$)**. | Evaluation/Deployment |
| **`ssdlite320_mbv3_finetuned_full.pth`** | The $\text{PyTorch}$ **Baseline Model** after $\text{Fine-tune}$ ($\text{FP32}$, unpruned). | Phase 1 Output |
| **`ssdlite320_mbv3_pruned_finetuned_full.pth`** | The $\text{PyTorch}$ model that has been **Pruned** and subsequently **Finetuned** for performance recovery. This is the source model for $\text{ONNX}$ export. | Phase 2 Output |
| **`ssd_pruned_fp32.onnx`** | The $\text{ONNX}$ model after $\text{Pruning}$, in **$\text{FP32}$** format. Used for $\text{FP32}$ post-pruning $\text{benchmarking}$. | Phase 3 Output |
| **`ssd_pruned_int8.onnx`** | The $\text{ONNX}$ model after $\text{Quantization}$ in **$\text{INT8}$** format. Small size, high speed, and ready for embedded deployment. | Phase 3 Output |

---

## Project Team

| Member Name |
| :--- |
| Bùi Nguyễn Hoài Thương |
| Trương Vũ Hoàng Anh |
| Phạm Thanh Toàn |

---

