
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def _conv_bn_relu(in_ch: int, out_ch: int, **kwargs) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, bias=False, **kwargs),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        _conv_bn_relu(in_ch, out_ch, kernel_size=3, padding=1),
        _conv_bn_relu(out_ch, out_ch, kernel_size=5, padding=2),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Transmission payload
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EncoderOutput:
    """
    Transmitted from encoding device → decoding server.

    base      — bottleneck latent (Bx1024xH/64xW/64); always present.
    residuals — 0-5 tensors of 32 channels each, ordered coarse→fine.
    skips     — optional raw encoder features for ultra-high-quality mode.
    """
    base: torch.Tensor
    residuals: List[torch.Tensor] = field(default_factory=list)
    skips: List[torch.Tensor] = field(default_factory=list)

    @property
    def mode(self) -> int:
        return len(self.residuals)


# ═══════════════════════════════════════════════════════════════════════════════
# Residual head — one per residual level, runs on device
# ═══════════════════════════════════════════════════════════════════════════════

class ResidualHead(nn.Module):
    """
    Depthwise-separable conv with ReLU between dw and pw, producing a 32-ch
    residual matrix.  Kept lightweight so mode-5 encoding stays fast.
    """
    def __init__(self, in_ch: int, out_ch: int = 32) -> None:
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn2(self.pw(F.relu(self.bn1(self.dw(x)), inplace=True)))


# ═══════════════════════════════════════════════════════════════════════════════
# Residual fusion — runs on server
# ═══════════════════════════════════════════════════════════════════════════════

class ResidualFusion(nn.Module):
    """
    Projects a 32-ch residual into decoder_ch and adds it via a learned
    sigmoid gate, so the decoder learns how much to trust each residual.
    """
    def __init__(self, residual_ch: int, decoder_ch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(residual_ch, decoder_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(decoder_ch)
        self.gate = nn.Conv2d(residual_ch, decoder_ch, 1, bias=True)   # learned scale

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor]) -> torch.Tensor:
        if residual is None:
            return x
        r = F.interpolate(residual, size=x.shape[2:], mode="bilinear", align_corners=False)
        correction = self.bn(self.proj(r))
        scale = torch.sigmoid(self.gate(r))   # gate: how much to add
        return x + scale * correction


# ═══════════════════════════════════════════════════════════════════════════════
# LIGHT ENCODER  (runs on the device)
# ═══════════════════════════════════════════════════════════════════════════════

class LeHdEncoder(nn.Module):
    """
    ResNet-18 backbone + widened bottleneck (1024 ch) for better mode-0 quality.

    Bottleneck size for a 256x256 input:
        v1: 512 x 4 x 4  =  8 192 values
        v2: 1024 x 4 x 4 = 16 384 values  (2x more capacity at same spatial size)

    The two-layer bottleneck (instead of one) gives the model more depth to
    compress useful information before transmission.
    """

    RESIDUAL_CH  = 32
    BOTTLENECK_CH = 1024   # widened from 512
    _ENC_CH = [64, 64, 128, 256, 512]   # e1…e5 channel counts (ResNet-18)

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.enc2 = resnet.layer1    # 64-ch
        self.enc3 = resnet.layer2    # 128-ch
        self.enc4 = resnet.layer3    # 256-ch
        self.enc5 = resnet.layer4    # 512-ch

        # Two-layer bottleneck: 512 → 1024 (stride-2 in first layer: 8→4)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, self.BOTTLENECK_CH, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.BOTTLENECK_CH),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.BOTTLENECK_CH, self.BOTTLENECK_CH, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.BOTTLENECK_CH),
            nn.ReLU(inplace=True),
        )

        # Residual heads: coarse (e5=512ch) → fine (e1=64ch)
        enc_ch = self._ENC_CH
        self.res_heads = nn.ModuleList([
            ResidualHead(enc_ch[4], self.RESIDUAL_CH),  # res0 ← e5
            ResidualHead(enc_ch[3], self.RESIDUAL_CH),  # res1 ← e4
            ResidualHead(enc_ch[2], self.RESIDUAL_CH),  # res2 ← e3
            ResidualHead(enc_ch[1], self.RESIDUAL_CH),  # res3 ← e2
            ResidualHead(enc_ch[0], self.RESIDUAL_CH),  # res4 ← e1
        ])

    def forward(
        self,
        x: torch.Tensor,
        mode: int = 0,
        transmit_skips: bool = False,
    ) -> EncoderOutput:
        if not (0 <= mode <= 5):
            raise ValueError(f"mode must be 0-5, got {mode}")

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        base = self.bottleneck(e5)   # Bx1024x4x4

        enc_features = [e5, e4, e3, e2, e1]
        residuals = [self.res_heads[i](enc_features[i]) for i in range(mode)]
        skips = [e1, e2, e3, e4, e5] if transmit_skips else []

        return EncoderOutput(base=base, residuals=residuals, skips=skips)


# ═══════════════════════════════════════════════════════════════════════════════
# HEAVY DECODER  (runs on the server)
# ═══════════════════════════════════════════════════════════════════════════════

class LeHdDecoder(nn.Module):
    """
    Server-side decoder.  Accepts the widened 1024-ch bottleneck.
    Each stage fuses the corresponding residual (if transmitted).
    """

    RESIDUAL_CH = 32
    BOTTLENECK_CH = 1024

    def __init__(self) -> None:
        super().__init__()

        BC = self.BOTTLENECK_CH  # 1024

        # d1: 1024x4 → 1024x8
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = _dec_block(BC, BC)
        self.fuse1 = ResidualFusion(self.RESIDUAL_CH, BC)     # fuses res0 (e5)

        # d2: 1024x8 → 512x16
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = _dec_block(BC, 512)
        self.fuse2 = ResidualFusion(self.RESIDUAL_CH, 512)

        # d3: 512x16 → 256x32
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = _dec_block(512, 256)    
        self.fuse3 = ResidualFusion(self.RESIDUAL_CH, 256)    # fuses res2 (e3)

        # d4: 256x32 → 128x64
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec4 = _dec_block(256, 128)
        self.fuse4 = ResidualFusion(self.RESIDUAL_CH, 128)    # fuses res3 (e2)

        # d5: 128x64 → 64x128
        self.up5 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec5 = _dec_block(128, 64)
        self.fuse5 = ResidualFusion(self.RESIDUAL_CH, 64)     # fuses res4 (e1)

        # d6: 64x128 → 3x256
        self.up6 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec6 = nn.Sequential(
            _conv_bn_relu(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 3, kernel_size=1),
        )

    @staticmethod
    def _res(residuals: List[torch.Tensor], idx: int) -> Optional[torch.Tensor]:
        return residuals[idx] if idx < len(residuals) else None

    def forward(self, payload: EncoderOutput) -> torch.Tensor:
        b, r = payload.base, payload.residuals

        d1 = self.fuse1(self.dec1(self.up1(b)),  self._res(r, 0))
        d2 = self.fuse2(self.dec2(self.up2(d1)), self._res(r, 1))
        d3 = self.fuse3(self.dec3(self.up3(d2)), self._res(r, 2))
        d4 = self.fuse4(self.dec4(self.up4(d3)), self._res(r, 3))
        d5 = self.fuse5(self.dec5(self.up5(d4)), self._res(r, 4))

        out = self.dec6(self.up6(d5))
        return torch.sigmoid(out)

class LeHd(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = LeHdEncoder(pretrained=pretrained)
        self.decoder = LeHdDecoder()

    def forward(self, x: torch.Tensor, mode: int = 0) -> torch.Tensor:
        return self.decoder(self.encoder(x, mode=mode))

    def encode(self, x: torch.Tensor, mode: int = 0) -> EncoderOutput:
        return self.encoder(x, mode=mode)

    def decode(self, payload: EncoderOutput) -> torch.Tensor:
        return self.decoder(payload)

