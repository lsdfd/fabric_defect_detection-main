# Fabric Defect Detection Research Workspace

这个仓库现在不再只是原作者的 notebook demo，而是我们后续做 fabric 缺陷检测蒸馏与超表面映射的主工作区。

当前目标分两步：

1. 严格复现 AITEX fabric 的二分类 patch classifier 和 U-Net 分割基线。
2. 在这个基线上做一层 CNN student 的知识蒸馏，并把 learned kernels 迁移到超表面相位设计流程。

## 当前主线

- 数据集：AITEX fabric
- 基线任务：
  - patch-level binary classification
  - defect patch segmentation with U-Net
- 后续研究任务：
  - one-layer optical CNN student
  - KD / 后续 NTKD
  - kernel to metasurface phase design

## 仓库结构

- `src/fdd/`
  我们自己的主线代码 package。
- `scripts/`
  标准训练、评估、导出脚本。
- `docs/`
  项目结构、复现状态、服务器说明。
- `notebooks/`
  notebook 分为 `reproduction/` 和 `archive/`。
- `train/`
  原作者 notebook 依赖的 Python 文件，保留做严格复现参考。
- `app/`
  原作者 Streamlit 推理 demo，后续会逐步收成我们自己的推理入口。
- `models/`
  原始权重与原作者日志。
- `outputs/`
  实验输出，默认不进版本控制。

结构说明见：
[docs/PROJECT_STRUCTURE.md](/Users/lishengxin/Desktop/毕设/科研/图像处理/CNN/课题：超表面+工业缺陷检测/fabric_defect_detection-main/docs/PROJECT_STRUCTURE.md)

## 当前复现进度

复现状态见：
[docs/REPRO_STATUS.md](/Users/lishengxin/Desktop/毕设/科研/图像处理/CNN/课题：超表面+工业缺陷检测/fabric_defect_detection-main/docs/REPRO_STATUS.md)

当前已知情况：

- binary classifier 已基本复现成功，4090 上可到 `F1 ~= 0.9752`
- U-Net 100 epoch 训练已跑通
- U-Net 最终是否“完整复现”还差同口径 IoU 评估盖章

## 常用脚本

二分类 notebook 复现：

```bash
python scripts/reproduction/reproduce_binary_patch_notebook.py \
  --epochs 50 \
  --batch-size 16 \
  --lr 0.001 \
  --num-workers 2 \
  --output-dir outputs/gpu_50ep
```

U-Net 训练：

```bash
python scripts/training/train_unet.py \
  --epochs 100 \
  --batch-size 4 \
  --lr 0.001 \
  --num-workers 2 \
  --output-dir outputs/unet_repro_100ep
```

teacher 评估：

```bash
python scripts/evaluation/evaluate_teacher.py
```

student KD：

```bash
python scripts/training/train_student_kd.py \
  --mode kd \
  --teacher-checkpoint models/bigger_binary_F1_0.98.pth
```

## 服务器

服务器使用方式见：
[docs/SERVER.md](/Users/lishengxin/Desktop/毕设/科研/图像处理/CNN/课题：超表面+工业缺陷检测/fabric_defect_detection-main/docs/SERVER.md)

## 说明

- 原作者 notebook 和历史尝试暂时保留，不做激进删除。
- 新功能优先落在 `src/fdd/` 和 `scripts/`。
- 长时间训练优先在 4090 服务器执行，不再默认用本地 CPU 慢跑。
