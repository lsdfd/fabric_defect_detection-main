# Student KD Sweep Plan

这份文档用于指导 fabric 二分类 student baseline / classic KD 的系统实验。

目标不是盲目多跑，而是回答三个问题：

1. 一层 optical-style student 本身能到什么水平；
2. classic KD 对这个 fabric teacher 是否稳定有帮助；
3. 哪个 student 架构最适合后续做 kernel 到超表面相位设计。

## 总体原则

- 先定评测协议，再扫模型。
- 先扫 `student baseline`，再扫 `KD`。
- 参考文献提供结构原则，不直接照抄数值。
- student 设计既要看分数，也要考虑后续超表面映射的物理友好性。

## 评测协议

所有实验先统一遵循以下设置：

- 数据：AITEX fabric patch binary classification
- teacher：原始 `BinaryClassifier`
- student：一层卷积前端 + 小型 FC backend
- 主指标：`val F1`
- 同时记录：
  - `val precision`
  - `val recall`
  - `val loss`
  - `train task loss`
  - `train kd loss`
  - `epoch seconds`
- 默认 seed：
  - 粗扫阶段：`42`
  - 确认阶段：`42, 52, 62`

## 阶段 1：Student Baseline 架构粗扫

这一阶段完全不加蒸馏，只回答一个问题：

`哪种一层 optical student 架构本身最适合 fabric patch 二分类？`

### 候选配置

这一轮先严格对齐补充材料的 student 逻辑：

- 单卷积层；
- 两层全连接；
- 不额外引入复杂电子后端；
- 把 `student input size` 作为核心变量一起扫。

| id | input_size | optical_kernels | kernel_size | pooled_size | hidden_dim | 设计意图 |
|---|---:|---:|---:|---:|---:|---|
| R1 | 64 | 16 | 7 | 6 | 256 | 最接近低分辨率 optical student 假设 |
| R2 | 96 | 16 | 7 | 6 | 256 | 当前主推荐起点 |
| R3 | 128 | 16 | 7 | 6 | 256 | 保留更多纹理细节 |
| R4 | 96 | 8 | 7 | 6 | 256 | 测 kernel 数减少的影响 |
| R5 | 96 | 16 | 11 | 6 | 256 | 测更大感受野是否更适合 fabric |

### 选择标准

- 首先看 `val F1`
- 若分数接近，优先：
  - kernel 数更少；
  - kernel size 更适合后续 PSF 设计；
  - backend 更小；
  - 收敛更稳定

阶段 1 结束后，选前 2 到 3 个结构进入 KD。

## 阶段 2：Classic KD 参数扫

这一阶段只对阶段 1 里最好的 2 到 3 个 baseline student 做 KD，不再全量扫全部结构。

### KD 候选参数

| id | alpha | temperature | 含义 |
|---|---:|---:|---|
| K1 | 0.9 | 2 | 真实标签主导，teacher 轻辅助 |
| K2 | 0.8 | 2 | 当前推荐起点 |
| K3 | 0.8 | 4 | 更软 teacher |
| K4 | 0.7 | 2 | 略加强 distillation |
| K5 | 0.7 | 4 | 强 soft target 版本 |

### 说明

- 当前不建议默认使用 `alpha=0.5` 作为主起点。
- 对这个二分类 sigmoid teacher，更合理的第一轮倾向是让 `task loss` 主导，`KD loss` 作为辅助。
- teacher 输入保持原始训练习惯，student 保持我们当前 patch 视角。

## 阶段 3：训练策略微调

在最优 student + KD 配方附近，再调训练策略：

| 项目 | 候选 |
|---|---|
| epoch | `30 / 50 / 100` |
| batch_size | `16 / 32` |
| lr | `1e-3 / 5e-4` |
| sampler | `balanced sampler` 开 / 关 |

建议顺序：

1. 先固定 `lr=1e-3`
2. 看 `30 epoch` 是否已经接近收敛
3. 若仍上升，再跑 `50 epoch`
4. 只有有意义时再上 `100 epoch`

## 阶段 4：最佳配置确认

最终最优配置不要只看一次幸运结果。

确认实验要求：

- 至少跑 `3` 个 seed：`42, 52, 62`
- 汇总：
  - `mean +- std`
  - 最佳 checkpoint
  - 参数量
  - 卷积核 shape
  - 推理时间
- 保存 optical kernels，作为后续超表面设计输入

## 当前执行顺序建议

先跑这 5 个 baseline：

- `R1`
- `R2`
- `R3`
- `R4`
- `R5`

然后：

1. 按 `val F1` 选前 2 名
2. 对前 2 名跑 `K2, K3, K4`
3. 选当前最优配置
4. 把最优配置拉长到 `50 epoch`
5. 再决定是否需要 `100 epoch`

## 输出规范

建议所有输出目录包含结构和 KD 参数信息，例如：

```text
outputs/student_baseline_R2_seed42/
outputs/student_kd_R2_K2_seed42/
```

其中：

- `R2` 对应结构配置
- `K2` 对应 KD 配置
- `seed42` 对应随机种子

## 研究判断记录

当前主观判断最值得优先关注的是：

- `96 input + 16 kernels + 7x7`
- `128 input + 16 kernels + 7x7`
- `96 input + 16 kernels + 11x11`

原因：

- 补充材料中的 student 不仅压缩了网络，也建立在较低输入分辨率上；
- fabric 是灰度纹理任务，直接让一层 student 吃 `256 x 256` 可能过难；
- `16 x 7x7` 是参考工作最强配置，应该先作为主对照；
- 在此基础上再判断 larger receptive field 对 fabric 是否真有帮助。

## 当前补充验证：二分类 KD 修正版

在第一轮 classic KD 明显输给 baseline 后，当前优先做一个小而硬的补充验证，专门回答两个问题：

1. 是不是 `30 epoch` 对 KD 来说太短；
2. 是不是当前二分类 `probability KD` 形式本身不合适。

当前验证只围绕最优 baseline `R2` 展开：

| id | 说明 |
|---|---|
| `B50` | `R2` baseline，训练 `50 epoch`，开启 `auto_pos_weight` |
| `L90` | `R2` logit KD，`alpha=0.9`，`T=2`，训练 `50 epoch`，开启 `auto_pos_weight` |
| `L95` | `R2` logit KD，`alpha=0.95`，`T=2`，训练 `50 epoch`，开启 `auto_pos_weight` |

设计理由：

- `alpha` 更高，让 task loss 主导；
- `logit KD` 比直接蒸 sigmoid 概率更适合当前二分类 teacher；
- `auto_pos_weight` 用于缓解正负样本不平衡；
- `50 epoch` 用于排除“KD 只是收敛更慢”的可能。

对应 manifest 可直接生成：

  ```bash
python scripts/experiments/generate_student_sweep.py \
  --stage binary-kd-fix \
  --selected-baselines R2 \
  --output outputs/plans/student_binary_kd_fix_R2_seed42.json
```

## 下一轮消融：去掉 pos_weight 的高 alpha 二分类 KD

在发现 `auto_pos_weight` 可能把训练推坏之后，下一轮优先做一个更干净的消融：

| id | 说明 |
|---|---|
| `B50N` | `R2` baseline，`50 epoch`，不加 `pos_weight` |
| `P90N` | `R2` probability KD，`alpha=0.9`，`T=2`，`50 epoch`，不加 `pos_weight` |
| `P95N` | `R2` probability KD，`alpha=0.95`，`T=2`，`50 epoch`，不加 `pos_weight` |
| `L95N` | `R2` logit KD，`alpha=0.95`，`T=2`，`50 epoch`，不加 `pos_weight` |

目的：

1. 把 `pos_weight` 从变量里移除；
2. 保留 balanced sampler；
3. 只比较更高 `alpha` 下，`prob KD` 和 `logit KD` 哪个更稳。

对应 manifest：

```bash
python scripts/experiments/generate_student_sweep.py \
  --stage binary-kd-ablation \
  --selected-baselines R2 \
  --output outputs/plans/student_binary_kd_ablation_R2_seed42.json
```
