
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
        self.slice2 = nn.Sequential(*list(vgg.children())[:18]).to(device).eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        f1_p, f1_t = self.slice1(pred),  self.slice1(target)
        f2_p, f2_t = self.slice2(pred),  self.slice2(target)
        return F.mse_loss(f1_p, f1_t) + F.mse_loss(f2_p, f2_t)
    

"""
Vision Transformer Perceptual Loss
===================================
ViT-based alternatives to VGG perceptual loss, using self-supervised models
like DINO that capture semantic structure better than supervised CNNs.

Why ViT/DINO over VGG?
──────────────────────
• VGG features are texture-biased (trained for classification)
• DINO features are shape-biased (trained via self-supervised SSL)
• ViTs capture long-range dependencies (global attention vs local conv filters)
• DINO explicitly learns semantic correspondence between image regions
• For talking heads: DINO "understands" face structure better — eyes, nose,
  mouth as coherent objects rather than texture patches

Recommended for LeHd:
    DINOPerceptualLoss with dinov2_vits14 — best quality/speed trade-off
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DINO v1 / DINO v2 perceptual loss
# ═══════════════════════════════════════════════════════════════════════════════

class DINOPerceptualLoss(nn.Module):
    """
    DINO-based perceptual loss using intermediate ViT blocks.

    Available models (all from torch.hub):
    ────────────────────────────────────────────────────────────────────────
    DINO v1 (2021):
      'dino_vits16'   — ViT-Small/16,  21M params,  good speed
      'dino_vitb16'   — ViT-Base/16,   86M params,  better quality
      'dino_vits8'    — ViT-Small/8,   21M params,  slower, finer detail

    DINO v2 (2023, RECOMMENDED):
      'dinov2_vits14' — ViT-Small/14,  21M params,  best speed/quality
      'dinov2_vitb14' — ViT-Base/14,   86M params,  highest quality
      'dinov2_vitl14' — ViT-Large/14, 300M params,  overkill for 256px

    Layers extracted:
      • layer 3  (early)  — local edges, texture
      • layer 6  (middle) — part-level structure (eyes, nose, mouth)
      • layer 9  (late)   — global semantic layout
      • CLS token (optional) — global image representation

    Usage:
        loss_fn = DINOPerceptualLoss('dinov2_vits14', device)
        loss = loss_fn(pred, target)  # both B×3×H×W in [0,1]
    """

    _LAYER_WEIGHTS = {
        3: 1.0,   # early features (texture, edges)
        6: 1.0,   # mid features (object parts)
        9: 0.5,   # late features (global semantics)
    }

    def __init__(
        self,
        model_name: str = "dinov2_vits16",
        device: torch.device = None,
        use_cls: bool = False,   # include CLS token in loss
    ) -> None:
        super().__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading {model_name} for perceptual loss...")
        
        # Load from torch.hub
        if model_name.startswith("dinov2"):
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                pretrained=True,
                force_reload=True
            ).to(device).eval()
        else:  # dino v1
            self.model = torch.hub.load(
                "facebookresearch/dino:main",
                model_name,
                pretrained=True,
                force_reload=True
            ).to(device).eval()

        self.use_cls = use_cls
        self.device = device
        
        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad_(False)

        # ViT patch size (needed for reshaping)
        self.patch_size = self.model.patch_embed.patch_size[0]
        
        print(f"  Loaded successfully (patch_size={self.patch_size})")

    def _extract_features(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        """
        Extract features from specified transformer blocks.
        Returns dict: {layer_idx: features}
        
        Features are B×N×D where N = num_patches + 1 (CLS token).
        We typically drop the CLS token and reshape to spatial: B×D×H'×W'
        """
        B, C, H, W = x.shape
        
        # Normalize like ImageNet (DINO expects this)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_norm = (x - mean) / std

        # Prepare patches
        x_patches = self.model.prepare_tokens_with_masks(x_norm, masks=None)
        
        features = {}
        
        # Forward through transformer blocks, extracting at specified layers
        for i, blk in enumerate(self.model.blocks):
            x_patches = blk(x_patches)
            
            if i + 1 in self._LAYER_WEIGHTS:
                # x_patches is (B, N+1, D) where first token is CLS
                if self.use_cls:
                    features[i + 1] = x_patches  # keep CLS
                else:
                    # Drop CLS token, reshape to spatial
                    tokens = x_patches[:, 1:, :]   # B, N, D
                    D = tokens.shape[-1]
                    
                    # Compute spatial dims (tokens form a grid)
                    n_patches_h = H // self.patch_size
                    n_patches_w = W // self.patch_size
                    
                    # Reshape to B×D×H'×W'
                    tokens_spatial = tokens.transpose(1, 2).reshape(
                        B, D, n_patches_h, n_patches_w
                    )
                    features[i + 1] = tokens_spatial
        
        return features

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: B×3×H×W in [0, 1]
        
        Returns weighted MSE across transformer block features.
        """
        # Extract features
        pred_feats = self._extract_features(pred)
        targ_feats = self._extract_features(target)
        
        loss = 0.0
        for layer_idx, weight in self._LAYER_WEIGHTS.items():
            fp = pred_feats[layer_idx]
            ft = targ_feats[layer_idx]
            loss = loss + weight * F.mse_loss(fp, ft)
        
        return loss


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Combined VGG + DINO loss (best of both worlds)
# ═══════════════════════════════════════════════════════════════════════════════

class HybridPerceptualLoss(nn.Module):
    """
    Combines VGG (texture) + DINO (semantics).

    VGG is good at local texture/edges via conv filters.
    DINO is good at global structure via attention.
    
    For talking heads: VGG captures skin texture, DINO captures face geometry.
    
    Weights:
        0.5 * VGG + 0.5 * DINO   (default, balanced)
    """

    def __init__(
        self,
        device: torch.device,
        dino_model: str = "dinov2_vits16",
        w_vgg: float = 0.5,
        w_dino: float = 0.5,
    ) -> None:
        super().__init__()
        
        # VGG component (lightweight: just relu2_2 and relu3_3)
        import torchvision.models as models
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg_slice1 = nn.Sequential(*list(vgg.children())[:10]).to(device).eval()
        self.vgg_slice2 = nn.Sequential(*list(vgg.children())[:17]).to(device).eval()
        
        # DINO component
        self.dino = DINOPerceptualLoss(dino_model, device)
        
        self.w_vgg = w_vgg
        self.w_dino = w_dino
        
        for p in self.vgg_slice1.parameters():
            p.requires_grad_(False)
        for p in self.vgg_slice2.parameters():
            p.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # VGG loss
        f1_p = self.vgg_slice1(pred)
        f1_t = self.vgg_slice1(target)
        f2_p = self.vgg_slice2(pred)
        f2_t = self.vgg_slice2(target)
        vgg_loss = F.mse_loss(f1_p, f1_t) + F.mse_loss(f2_p, f2_t)
        
        # DINO loss
        dino_loss = self.dino(pred, target)
        
        return self.w_vgg * vgg_loss + self.w_dino * dino_loss


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Lightweight ViT loss (MAE-pretrained, faster than DINO)
# ═══════════════════════════════════════════════════════════════════════════════

class MAEPerceptualLoss(nn.Module):
    """
    Perceptual loss using MAE (Masked Autoencoder) pretrained ViT.
    
    MAE is faster than DINO (simpler training) but still captures good semantics.
    Good middle-ground if DINO is too slow.
    
    Available models:
      'vit_base_patch16_224.mae'  — 86M params, good quality
      'vit_large_patch16_224.mae' — 304M params, best quality
    
    Requires: pip install timm
    """

    _LAYER_INDICES = [3, 6, 9]  # early, mid, late blocks

    def __init__(self, device: torch.device, model_name: str = "vit_base_patch16_224.mae") -> None:
        super().__init__()
        
        try:
            import timm
            print(f"Loading {model_name} (MAE-pretrained ViT)...")
            self.model = timm.create_model(model_name, pretrained=True, force_reload=True).to(device).eval()
            self._available = True
        except ImportError:
            print("Warning: timm not installed. MAEPerceptualLoss disabled. Run: pip install timm")
            self._available = False
            return
        
        self.device = device
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self._available:
            return torch.tensor(0.0, device=pred.device)
        
        # MAE/timm models expect ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)
        
        pred_norm = (pred - mean) / std
        targ_norm = (target - mean) / std
        
        # Forward through blocks, extracting intermediate features
        # (This is timm-specific; adjust if using different ViT implementation)
        def extract(x):
            x = self.model.patch_embed(x)
            x = self.model._pos_embed(x)
            features = []
            for i, blk in enumerate(self.model.blocks):
                x = blk(x)
                if i in self._LAYER_INDICES:
                    features.append(x[:, 1:, :])  # drop CLS token
            return features
        
        pred_feats = extract(pred_norm)
        targ_feats = extract(targ_norm)
        
        loss = 0.0
        for fp, ft in zip(pred_feats, targ_feats):
            loss = loss + F.mse_loss(fp, ft)
        
        return loss / len(pred_feats)


# ═══════════════════════════════════════════════════════════════════════════════
# Usage examples
# ═══════════════════════════════════════════════════════════════════════════════

"""
# Drop-in replacement for VGG loss in lehd_v2.py:

# OLD:
from PerceptualLoss import PerceptualLoss
perc_fn = PerceptualLoss(device)

# NEW (pure DINO):
from vit_perceptual_loss import DINOPerceptualLoss
perc_fn = DINOPerceptualLoss('dinov2_vits14', device)

# OR (VGG + DINO hybrid):
from vit_perceptual_loss import HybridPerceptualLoss
perc_fn = HybridPerceptualLoss(device)

# Then use exactly as before:
loss = perc_fn(pred, target)


# ──────────────────────────────────────────────────────────────────────────────
# Model comparison:
# ──────────────────────────────────────────────────────────────────────────────

Model              Params  Speed       Quality         Best for
─────────────────  ──────  ──────────  ──────────────  ─────────────────────────
VGG-16             138M    fastest     good (texture)  baseline
dinov2_vits14      21M     fast        excellent       talking heads, faces
dinov2_vitb14      86M     medium      best            high-quality mode 3-5
dino_vits16        21M     fast        very good       DINOv1 alternative
MAE ViT-B          86M     medium      very good       if DINO unavailable
Hybrid VGG+DINO    159M    slow        excellent       maximum quality

Recommendation for LeHd mode 0:
    DINOPerceptualLoss('dinov2_vits14')  ← best speed/quality for 256px faces
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════



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


if __name__ == "__main__":
    device = torch.device("mps")
    print(f"Testing on {device}\n")
    
    # Create dummy images
    pred   = torch.randn(2, 3, 256, 256, device=device) * 0.5 + 0.5  # [0, 1]
    target = torch.randn(2, 3, 256, 256, device=device) * 0.5 + 0.5
    
    print("=" * 60)
    print("Testing DINOPerceptualLoss (dinov2_vits16)...")
    print("=" * 60)
    dino_loss = DINOPerceptualLoss("dinov2_vits16", device)
    loss = dino_loss(pred, target)
    print(f"Loss: {loss.item():.6f}\n")
    
    print("=" * 60)
    print("Testing HybridPerceptualLoss (VGG + DINO)...")
    print("=" * 60)
    hybrid_loss = HybridPerceptualLoss(device)
    loss = hybrid_loss(pred, target)
    print(f"Loss: {loss.item():.6f}\n")
    
    print("All tests passed!")