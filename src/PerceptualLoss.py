
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PerceptualLoss(nn.Module):
    """
    VGG-16 feature-space loss.  Compares relu2_2 and relu3_3 activations.
    Produces much sharper reconstructions than pixel-MSE alone, especially
    important for mode 0 where spatial information is most compressed.
    Runs on the server (same device as decoder).
    """
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        # relu2_2 = layers 0–9, relu3_3 = layers 0–16
        self.slice1 = nn.Sequential(*list(vgg.children())[:10]).to(device).eval()
        self.slice2 = nn.Sequential(*list(vgg.children())[:17]).to(device).eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        f1_p, f1_t = self.slice1(pred),  self.slice1(target)
        f2_p, f2_t = self.slice2(pred),  self.slice2(target)
        return F.mse_loss(f1_p, f1_t) + F.mse_loss(f2_p, f2_t)


def frequency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Penalises differences in the FFT magnitude spectrum.
    Encourages the model to reproduce high-frequency edges and textures —
    the first thing MSE sacrifices under compression.

    Runs on CPU because aten::_fft_r2c is not implemented for MPS.
    The .cpu() calls are no-ops on CUDA/CPU, so this is device-agnostic.
    Gradients still flow back through the CPU tensors to the MPS graph.
    """
    pred_cpu   = pred.cpu()
    target_cpu = target.cpu()

    def mag(x: torch.Tensor) -> torch.Tensor:
        return torch.abs(torch.fft.rfft2(x, norm="ortho"))

    return F.l1_loss(mag(pred_cpu), mag(target_cpu))
