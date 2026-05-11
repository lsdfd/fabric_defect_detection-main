# Server Notes

本文件记录仓库相关的 GPU 服务器使用方式。

## Server

- Host: `ssh.bj8.bz1.paratera.com`
- Port: `2233`
- User: `root@ackcs-00gjgqpy`
- GPU: RTX 4090
- Remote repo root: `/root/fabric_run/fabric_defect_detection-main`

## Login

```bash
ssh -p 2233 -o StrictHostKeyChecking=no -l 'root@ackcs-00gjgqpy' ssh.bj8.bz1.paratera.com
```

密码不要写进会同步到远端仓库的文档。
如需本地长期保存，放在不进 git 的文件中，例如：

- `SERVER.local.md`
- `.env.local`

## Upload

远端没有 `rsync` 时，可先打 tar 包再 `scp`：

```bash
tar \
  --exclude='fabric_defect_detection-main/outputs' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='fabric_defect_detection-main/.git' \
  -czf /tmp/fabric_gpu_payload.tgz \
  fabric_defect_detection-main
```

```bash
scp -P 2233 \
  -o StrictHostKeyChecking=no \
  -o User='root@ackcs-00gjgqpy' \
  /tmp/fabric_gpu_payload.tgz \
  ssh.bj8.bz1.paratera.com:/root/fabric_run/fabric_gpu_payload.tgz
```

远端解压：

```bash
cd /root/fabric_run
rm -rf fabric_defect_detection-main
tar -xzf fabric_gpu_payload.tgz
cd fabric_defect_detection-main
```

## Typical Runs

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

## Sync Principle

- 本地负责代码整理与实验组织。
- 服务器负责正式训练和长时间评估。
- `outputs/` 默认不纳入 git。
- 真正需要共享的结果，导出为小型 `json/png/csv` 再同步回仓库。
