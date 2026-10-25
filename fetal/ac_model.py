"""
fetal/ac_model.py
=================
Dedicated 2-Class (Binary) Attention U-Net for Fetal Abdominal Circumference (AC) Segmentation.

Designed specifically for isolating the outer fetal abdominal boundary / skin contour
from 2D ultrasound transverse planes (per ISUOG guidelines and ACOUSLIC-AI challenge).

Architecture Highlights:
  - Contracting Encoder path with 4 DoubleConv downsampling blocks.
  - Expanding Decoder path with Attention Gates filtering skip connections.
  - Single-channel (or 2-channel) segmentation head for binary background vs. abdomen segmentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm2d => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate for Attention U-Net.
    Suppresses irrelevant background ultrasound noise and highlights fetal abdominal contour.
    Reference: Oktay et al., 'Attention U-Net: Learning Where to Look for the Pancreas.'
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1)
        )
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        net = self.relu(g1 + x1)
        attn = self.sigmoid(self.psi(net))
        return x * attn


class UpBlock(nn.Module):
    """Bilinear Upsampling followed by Attention Gate and Double Convolution."""
    def __init__(self, in_channels: int, out_channels: int):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.attn = AttentionGate(F_g=out_channels, F_l=out_channels, F_int=max(16, out_channels // 2))
        self.conv_after_up = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv_after_up(x)
        skip_attn = self.attn(g=x, x=skip)
        if x.shape[2:] != skip_attn.shape[2:]:
            x = F.interpolate(x, size=skip_attn.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip_attn, x], dim=1)
        return self.conv(x)


class FetalACAttentionUNet(nn.Module):
    """
    Dedicated Binary Attention U-Net for Fetal Abdominal Circumference (AC) segmentation.
    
    Inputs:
        x: Ultrasound image tensor (B, in_channels, H, W), typically (B, 1, H, W) or (B, 3, H, W).
    Outputs:
        Raw logits of shape (B, 1, H, W) for binary segmentation.
        Pass through torch.sigmoid(logits) for pixel probabilities.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_filters: int = 32
    ):
        super(FetalACAttentionUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        f = base_filters

        # Encoder (Contracting Path)
        self.enc1 = DoubleConv(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(f, f * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(f * 4, f * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(f * 8, f * 16)

        # Decoder (Expanding Path with Attention Gates)
        self.up4 = UpBlock(f * 16, f * 8)
        self.up3 = UpBlock(f * 8, f * 4)
        self.up2 = UpBlock(f * 4, f * 2)
        self.up1 = UpBlock(f * 2, f)

        # Final 1x1 classification head
        self.final_conv = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        out = self.final_conv(d1)
        return out

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Utility for inference returning boolean/binary mask (B, 1, H, W)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if self.num_classes == 1:
                probs = torch.sigmoid(logits)
                return (probs >= threshold).to(torch.uint8)
            else:
                probs = torch.softmax(logits, dim=1)
                return torch.argmax(probs, dim=1, keepdim=True).to(torch.uint8)


if __name__ == "__main__":
    model = FetalACAttentionUNet(in_channels=1, num_classes=1, base_filters=32)
    sample_input = torch.randn(2, 1, 256, 320)
    output = model(sample_input)
    print("FetalACAttentionUNet check passed!")
    print(f"Input shape: {sample_input.shape} -> Output logits shape: {output.shape}")
