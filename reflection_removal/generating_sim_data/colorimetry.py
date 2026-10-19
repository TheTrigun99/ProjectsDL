"""Colour matrices and conversions. No dependency on the rest of the package.

Replaces the notebook's "COLORS and DISPLAY" cell (and the old colometry.py).
The display functions live in display.py: this module does not import
matplotlib, so it is usable inside the workers.
"""
from __future__ import annotations

import numpy as np

D50 = np.array([0.9642, 1.0, 0.8249], np.float32)
LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)   # linear sRGB luminance

BRADFORD = np.array([[0.8951, 0.2664, -0.1614],
                     [-0.7502, 1.7135, 0.0367],
                     [0.0389, -0.0685, 1.0296]], np.float32)

# ICC / IEC 61966-2-1
XYZ_D50_TO_SRGB = np.array([[3.1338561, -1.6168667, -0.4906146],
                            [-0.9787684, 1.9161415, 0.0334540],
                            [0.0719453, -0.2289914, 1.4052427]], np.float32)
XYZ_D65_TO_SRGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                            [-0.9692660, 1.8760108, 0.0415560],
                            [0.0556434, -0.2040259, 1.0572252]], np.float32)


def cat_matrix(src_white, dst_white=D50):
    """Bradford chromatic adaptation: XYZ(illuminant src) -> XYZ(dst)."""
    s = BRADFORD @ src_white
    d = BRADFORD @ dst_white
    return (np.linalg.inv(BRADFORD) @ np.diag(d / s) @ BRADFORD).astype(np.float32)


def apply_matrix(img, M):
    """M applied to the last axis of img (H, W, 3)."""
    return img @ M.T


def luminance(lin_srgb):
    return lin_srgb @ LUM


def srgb_encode(x):
    """sRGB gamma: linear [0, 1] -> encoded [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def srgb_decode(x):
    """Inverse sRGB gamma. With no rendering curve, this really is linear sRGB."""
    return np.where(x <= 12.92 * 0.0031308, x / 12.92,
                    ((x + 0.055) / 1.055) ** 2.4)


def xy_to_xyz(xy, Y=1.0):
    """Chromaticity -> tristimulus, with luminance Y imposed."""
    x, y = float(xy[0]), float(xy[1])
    return np.array([Y * x / y, Y, Y * (1.0 - x - y) / y])
