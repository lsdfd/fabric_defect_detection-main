from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs = torch.sigmoid(inputs)
    smooth = 1.0
    iflat = inputs.view(-1)
    tflat = targets.view(-1)
    intersection = (iflat * tflat).sum()
    return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.size() != inputs.size():
            raise ValueError(f"Target size ({targets.size()}) must match input size ({inputs.size()})")

        max_val = (-inputs).clamp(min=0)
        loss = inputs - inputs * targets + max_val + ((-max_val).exp() + (-inputs - max_val).exp()).log()
        invprobs = F.logsigmoid(-inputs * (targets * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha: float, gamma: float):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.alpha * self.focal(inputs, targets) - torch.log(dice_loss(inputs, targets))
        return loss.mean()


class UNetDownBlock(nn.Module):
    def __init__(self, input_channel: int, output_channel: int, down_size: bool):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.down_size = down_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_size:
            x = self.max_pool(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNetUpBlock(nn.Module):
    def __init__(self, prev_channel: int, input_channel: int, output_channel: int):
        super().__init__()
        self.up_sampling = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv1 = nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()

    def forward(self, prev_feature_map: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class NotebookUNet(nn.Module):
    """Original U-Net definition from train/unet_segmentation.ipynb."""

    def __init__(self):
        super().__init__()
        self.down_block1 = UNetDownBlock(1, 64, False)
        self.down_block2 = UNetDownBlock(64, 128, True)
        self.down_block3 = UNetDownBlock(128, 256, True)
        self.down_block4 = UNetDownBlock(256, 512, True)
        self.down_block5 = UNetDownBlock(512, 1024, True)

        self.up_block1 = UNetUpBlock(512, 1024, 512)
        self.up_block2 = UNetUpBlock(256, 512, 256)
        self.up_block3 = UNetUpBlock(128, 256, 128)
        self.up_block4 = UNetUpBlock(64, 128, 64)
        self.last_conv = nn.Conv2d(64, 1, 1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)
        x = self.up_block1(x4, x5)
        x = self.up_block2(x3, x)
        x = self.up_block3(x2, x)
        x = self.up_block4(x1, x)
        return self.last_conv(x)


@dataclass
class SegmentationDistillationBreakdown:
    total: torch.Tensor
    task: torch.Tensor
    kd: torch.Tensor
    ntkd: torch.Tensor


def segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    preds = (torch.sigmoid(logits).view(logits.size(0), -1) > threshold).float()
    true = (targets.view(targets.size(0), -1) > threshold).float()

    pred_sum = preds.sum(-1)
    true_sum = true.sum(-1)
    neg_index = torch.nonzero(true_sum == 0, as_tuple=True)[0]
    pos_index = torch.nonzero(true_sum >= 1, as_tuple=True)[0]

    dice_values = []
    if len(pos_index) > 0:
        pred_pos = preds[pos_index]
        true_pos = true[pos_index]
        dice_pos = 2 * (pred_pos * true_pos).sum(-1) / (pred_pos + true_pos).sum(-1).clamp_min(1e-12)
        dice_values.append(dice_pos)
    if len(neg_index) > 0:
        dice_neg = (pred_sum[neg_index] == 0).float()
        dice_values.append(dice_neg)

    if not dice_values:
        dice = torch.empty(0, device=logits.device)
    else:
        dice = torch.cat(dice_values)
    jaccard = dice / (2 - dice).clamp_min(1e-12)
    return dice, jaccard


def notebook_deploy_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.75) -> torch.Tensor:
    """Mean-IoU metric used in the original notebook/deploy path.

    The notebook does not threshold sigmoid probabilities directly. It min-max
    normalizes each raw output mask, thresholds at 0.75, then computes binary
    Jaccard/IoU against the target mask. This function preserves that evaluation
    convention so reproduction results are compared on the same scale.
    """

    masks = logits.detach().view(logits.size(0), -1)
    true = (targets.detach().view(targets.size(0), -1) > 0.5).float()
    mins = masks.min(dim=1, keepdim=True).values
    maxs = masks.max(dim=1, keepdim=True).values
    norm = (masks - mins) / (maxs - mins).clamp_min(1e-12)
    pred = (norm > threshold).float()
    intersection = (pred * true).sum(dim=1)
    union = ((pred + true) > 0).float().sum(dim=1)
    return intersection / union.clamp_min(1e-12)


def pooled_ntk_tokens(logits: torch.Tensor, pooled_size: int) -> torch.Tensor:
    pooled = F.adaptive_avg_pool2d(logits, (pooled_size, pooled_size))
    return pooled.flatten(1)


def token_gram_matrix(tokens: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(tokens, p=2, dim=1, eps=1e-12)
    return normalized @ normalized.transpose(0, 1)


def segmentation_baseline_loss(
    student_logits: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> SegmentationDistillationBreakdown:
    task_loss = criterion(student_logits, targets)
    zero = torch.zeros_like(task_loss)
    return SegmentationDistillationBreakdown(total=task_loss, task=task_loss, kd=zero, ntkd=zero)


def segmentation_distillation_loss(
    student_logits: torch.Tensor,
    targets: torch.Tensor,
    teacher_logits: torch.Tensor,
    criterion: nn.Module,
    alpha: float,
    beta: float,
    temperature: float = 1.0,
    kd_mode: str = "logit",
    ntkd_weight: float = 0.0,
    ntkd_pooled_size: int = 8,
) -> SegmentationDistillationBreakdown:
    task_loss = criterion(student_logits, targets)

    teacher_logits = teacher_logits.detach()
    if kd_mode == "logit":
        kd_loss = F.mse_loss(student_logits / temperature, teacher_logits / temperature) * (temperature**2)
    elif kd_mode == "prob":
        kd_loss = F.mse_loss(torch.sigmoid(student_logits / temperature), torch.sigmoid(teacher_logits / temperature))
    else:
        raise ValueError(f"Unsupported kd_mode: {kd_mode}")

    if ntkd_weight > 0:
        student_tokens = pooled_ntk_tokens(student_logits, ntkd_pooled_size)
        teacher_tokens = pooled_ntk_tokens(teacher_logits, ntkd_pooled_size)
        student_gram = token_gram_matrix(student_tokens)
        teacher_gram = token_gram_matrix(teacher_tokens)
        ntkd_loss = F.mse_loss(student_gram, teacher_gram)
    else:
        ntkd_loss = torch.zeros_like(task_loss)

    total = alpha * task_loss + beta * kd_loss + ntkd_weight * ntkd_loss
    return SegmentationDistillationBreakdown(total=total, task=task_loss, kd=kd_loss, ntkd=ntkd_loss)
