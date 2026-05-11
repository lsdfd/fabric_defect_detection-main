# Project Structure

这个仓库现在分成两层：

1. `train/` 和 `app/`
   原作者留下的 notebook/demo 体系，作为复现依据保留。
2. `src/fdd/` 和 `scripts/`
   我们自己的可复现研究主线，后续蒸馏、超表面映射都从这里继续长。

## 当前建议结构

- `README.md`
  仓库入口，优先说明我们自己的主线，而不是原作者历史尝试。
- `docs/`
  面向项目协作和复现的说明文档。
- `src/fdd/`
  主线 Python package。
- `scripts/`
  可直接执行的训练、评估、导出脚本。
- `notebooks/`
  原始 notebook，分为严格复现和历史归档两层。
- `train/`
  notebook 依赖的上游 Python 代码。
- `app/`
  原始 Streamlit/demo 推理代码。
- `models/`
  原始/参考权重与原作者日志。
- `outputs/`
  本地实验输出，默认不进版本控制。

## `src/fdd/` 角色划分

- `fdd/data.py`
  AITEX 数据读取、patch 切分、split/sampler 等。
- `fdd/models.py`
  teacher/student 二分类模型。
- `fdd/training.py`
  二分类指标、KD loss。
- `fdd/unet.py`
  U-Net 结构、loss、分割评估。

后续建议逐步扩展为：

- `fdd/classification.py`
- `fdd/segmentation.py`
- `fdd/kd.py`
- `fdd/inference.py`
- `fdd/metasurface/`

但现在先不大拆，避免影响已经跑通的复现链路。

## `scripts/` 命名约定

- `reproduce_*`
  严格复刻 notebook/原始流程。
- `train_*`
  我们自己的标准训练入口。
- `evaluate_*`
  标准评估入口。
- `export_*`
  导出中间产物，用于蒸馏或超表面设计。

## 清理原则

- 原 notebook 不删。
- 历史探索 notebook 先留在 `notebooks/archive/`，但不作为主入口。
- 未来可以把非主线 notebook 归档到 `docs/archive/` 或 `notebooks/archive/`，但要等主线彻底稳定后再动。
- 新功能优先加到 `src/fdd/` + `scripts/`，不要继续堆在 notebook 里。
