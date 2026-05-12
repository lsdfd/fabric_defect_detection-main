# Fabric 二分类蒸馏实验说明

第一版只做 AITEX fabric patch 的二分类蒸馏：

```text
原项目 BinaryClassifier teacher
        ->
一层 OpticalConvBank student + 小型 FC backend
        ->
导出 student optical kernels
        ->
后续接 PSF/超表面相位设计
```

## 文件结构

- `src/fdd/data.py`
  - 标准化 AITEX patch dataset。
  - 修正原 notebook 中 Windows 风格路径拆分问题。
- `src/fdd/models.py`
  - `BinaryClassifier`：原 fabric teacher 架构。
  - `OpticalStudentClassifier`：一层 optical CNN frontend + 小型 FC backend。
- `src/fdd/training.py`
  - 二分类 metrics。
  - classic KD loss。
- `scripts/evaluation/evaluate_teacher.py`
  - 评估原 teacher checkpoint。
- `scripts/training/train_teacher.py`
  - 当原 teacher 权重只有 Git LFS 指针文件时，重新训练原始 `BinaryClassifier`。
- `scripts/training/train_student_kd.py`
  - 训练一层 CNN student。
  - 支持 `baseline` 和 `kd` 两种模式。
- `scripts/evaluation/evaluate_student.py`
  - 单独评估 student checkpoint。
- `scripts/export/export_student_kernels.py`
  - 导出 student 第一层 CNN kernels，并拆成 positive/negative。

## 运行前环境

当前默认 Python 环境可能没有安装 `torch`。建议使用独立环境，不要直接硬装原始 `requirements.txt`，因为里面有平台相关 pin。

最小依赖大致是：

```bash
pip install torch torchvision opencv-python numpy scikit-learn
```

如需完全复现原 notebook，再视情况补 `torchmetrics`、`torchinfo`、`streamlit` 等。

## 评估 Teacher

注意：当前目录里的 `models/bigger_binary_F1_0.98.pth` 可能只是 Git LFS 指针文件，不是真实 104MB 权重。如果评估时报 `invalid load key, 'v'`，先重新训练 teacher：

```bash
cd fabric_defect_detection-main
conda run -n metamat python scripts/training/train_teacher.py --epochs 5 --batch-size 16
```

训练出的 teacher 默认保存到：

```text
outputs/teacher/binary_classifier_best.pt
```

```bash
cd fabric_defect_detection-main
python scripts/evaluation/evaluate_teacher.py
```

默认读取：

```text
models/bigger_binary_F1_0.98.pth
```

## 训练 Student Baseline

```bash
cd fabric_defect_detection-main
python scripts/training/train_student_kd.py \
  --mode baseline \
  --epochs 10 \
  --batch-size 16 \
  --optical-kernels 16 \
  --kernel-size 7 \
  --pooled-size 6 \
  --hidden-dim 256
```

## 训练 KD Student

```bash
cd fabric_defect_detection-main
python scripts/training/train_student_kd.py \
  --mode kd \
  --teacher-checkpoint outputs/teacher/binary_classifier_best.pt \
  --epochs 10 \
  --batch-size 16 \
  --optical-kernels 16 \
  --kernel-size 7 \
  --pooled-size 6 \
  --hidden-dim 256 \
  --alpha 0.5 \
  --temperature 2.0
```

输出默认会按实验配置自动写入，例如：

```text
outputs/student_baseline_k16_s7_p6_h256/
outputs/student_kd_k16_s7_p6_h256/
```

包括：

- `student_best.pt`
- `student_last.pt`
- `student_optical_kernels.pt`
- `history.json`

## 评估 Student

```bash
cd fabric_defect_detection-main
python scripts/evaluation/evaluate_student.py \
  --checkpoint outputs/student_kd_k16_s7_p6_h256/student_best.pt
```

## 导出 Kernels

```bash
cd fabric_defect_detection-main
python scripts/export/export_student_kernels.py \
  --checkpoint outputs/student_kd/student_best.pt \
  --output outputs/student_kd/student_kernels.npz
```

导出的 `.npz` 包含：

- `kernels`：原始 learned kernels；
- `positive`：正权重部分；
- `negative`：负权重取反后的正值部分。

这一步是后续适配 `卷积核->超表面相位设计代码/TF_for_PSF_Engineering_CIFAR.ipynb` 的入口。

## 第一版 Student 设计依据

参考 RGB/多色 optical encoder 论文和补充材料，第一版 student 采用：

- 单层 optical convolution frontend；
- 默认 16 个 `7 x 7` kernels；
- `6 x 6` pooled feature maps；
- 256 hidden dim 的小型 FC backend。

这些参数是参考起点，不是最终结论。后续应系统比较 kernel 数量、kernel size、pooled size 和 hidden dim。
