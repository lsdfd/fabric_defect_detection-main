# Next Refactor Plan

这不是立即执行的大重构，而是后续分阶段演进路线。

## Phase 1

目标：保持现有功能不坏，先把主入口理顺。

- 完成 binary / U-Net 复现脚本固定化
- 补齐 U-Net 同口径评估
- 把服务器训练流程文档化

## Phase 2

目标：让二分类蒸馏成为仓库第一公民。

- 把 `train_student_kd.py` 变成主训练入口之一
- 明确 student 配置、输出目录、评估指标
- 补 kernel 导出和可视化

## Phase 3

目标：把“超表面映射”接进主线。

- 增加 `src/fdd/metasurface/`
- 增加 kernel normalization / positive-negative split
- 适配相位恢复 notebook 的输入输出接口

## Phase 4

目标：把旧 demo 代码边缘化，而不是一上来强删。

- 把非主线 notebook 归档
- 精简 `app/`，只保留需要的推理逻辑
- 统一模型定义来源，减少 `train/` 和 `app/` 双份结构

## 当前不建议立刻做的事

- 大规模改动数据目录
- 删除原 notebook
- 一次性重写全部 deploy 代码
- 在 U-Net 评价口径没对齐前大改 segmentation 训练逻辑
