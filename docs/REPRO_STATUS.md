# Reproduction Status

## Binary Patch Classifier

状态：已基本复现成功。

- 原始权重：
  `models/bigger_binary_F1_0.98 (1).pth`
- 评估结果：
  `F1 = 0.9750692520775623`
- 4090 上 50 epoch 脚本化复现结果：
  `F1 = 0.9752066115702479`

结论：
二分类 teacher 基线已经打通，可作为后续 student/KD 的固定 teacher。

## U-Net Segmentation

状态：训练已跑通，最终“是否完整复现”仍在做评估口径对齐。

已完成：

- 按 notebook 结构脚本化 U-Net。
- 4090 上跑完 100 epoch。
- 最终 train loss：
  `0.3556753727`

当前问题：

- 原作者保存的日志主要是训练 loss。
- README 报告的是 mean IoU，但 notebook/deploy 的 IoU 计算口径包含：
  - raw output per-image min-max normalization
  - threshold = `0.75`
- 因此不能直接用常规 sigmoid + `0.5` 阈值指标去判断复现是否成功。

下一步：

- 用与 notebook/app 一致的后处理重新评估同一个 checkpoint。
- 再判断是否接近 README 所说的 train/test IoU。

## 当前主线

1. 固定 binary teacher 复现链路。
2. 补齐 U-Net 同口径评估。
3. 进入一层 CNN student 的 baseline 和 classic KD。
4. 再进入卷积核导出与超表面相位设计适配。
