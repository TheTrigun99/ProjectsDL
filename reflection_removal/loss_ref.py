"""
Losses for reflection removal: pixel + perceptual + gradient.

    crit = ReflectionLoss().to(device)
    total, parts = crit(pred, target)
    total.backward()

`parts` holds each weighted term, already detached. Log it: the three should
stay within about one order of magnitude of each other. A term that drifts to
1% of the total is doing nothing, one that dominates is all the model optimises.

The perceptual term needs torchvision (`pip install torchvision`); the rest
runs on torch alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------

def _grad(x):
    """Forward differences. Returns (d/dx, d/dy), each one pixel smaller."""
    return x[..., :, 1:] - x[..., :, :-1], x[..., 1:, :] - x[..., :-1, :]


class GradientLoss(nn.Module):
    """L1 on the image gradients: penalises blur that the pixel term tolerates.

    Two images can differ by a soft blur and still have a small L1, because the
    error is spread thinly everywhere. Their gradients differ a lot, so this
    term is what pushes the network to keep edges."""

    def forward(self, pred, target):
        px, py = _grad(pred)
        tx, ty = _grad(target)
        return F.l1_loss(px, tx) + F.l1_loss(py, ty)


# ---------------------------------------------------------------------------
# Perceptual
# ---------------------------------------------------------------------------

# relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 in torchvision's VGG19 .features
_VGG_RELU = (1, 6, 11, 20, 29)
# Zhang et al. CVPR 2018: deeper layers carry fewer, more abstract activations,
# so they are re-weighted up to contribute comparably.
_VGG_W = (1 / 2.6, 1 / 4.8, 1 / 3.7, 1 / 5.6, 10 / 1.5)
# _VGG_W = (1/2.6, 1/4.8, 1/3.7, 1/5.6, 1/1.5) 

class PerceptualLoss(nn.Module):
    """L1 between VGG19 activations of pred and target.

    Why it matters here: a residual reflection is a *structured* error. The
    pixel term sees a small average deviation and is happy; VGG sees the ghost
    of a whole object and is not. This is the term that actually removes
    reflections rather than dimming them."""

    def __init__(self, layers=_VGG_RELU, weights=_VGG_W):
        super().__init__()
        try:
            from torchvision.models import VGG19_Weights, vgg19
        except ImportError as e:
            raise ImportError("PerceptualLoss needs torchvision: "
                              "pip install torchvision") from e

        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.blocks = nn.ModuleList()          # one block per cut point
        prev = 0
        for i in layers:
            self.blocks.append(vgg[prev:i + 1])
            prev = i + 1
        self.weights = weights

        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

        # VGG was trained on ImageNet-normalised input, not on [0, 1]
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        # model.train() must not put the frozen VGG back into training mode
        return super().train(False)

    def forward(self, pred, target):
        x = (pred - self.mean) / self.std
        y = (target - self.mean) / self.std
        loss = 0.0
        for block, w in zip(self.blocks, self.weights):
            x, y = block(x), block(y)
            loss = loss + w * F.l1_loss(x, y.detach())
        return loss


# ---------------------------------------------------------------------------
# The whole thing
# ---------------------------------------------------------------------------

class ReflectionLoss(nn.Module):
    """pixel + perceptual + gradient, each with one weight.

    L1 rather than L2 on the pixel term: the real pairs (nature, real89) are
    slightly misaligned, and L2 would spend its budget on a residual the
    network cannot predict -- the L2-optimal answer to noise is blur.

    Set a weight to 0 to drop that term; perceptual=0 also skips loading VGG.
    """

    def __init__(self, pixel=1.0, perceptual=0.1, gradient=0.1):
        super().__init__()
        self.w = {"pixel": pixel, "perceptual": perceptual, "gradient": gradient}
        self.grad = GradientLoss() if gradient else None
        self.percep = PerceptualLoss() if perceptual else None

    def forward(self, pred, target):
        parts, total = {}, pred.new_zeros(())
        terms = [("pixel", lambda: F.l1_loss(pred, target)),
                 ("perceptual", lambda: self.percep(pred, target)),
                 ("gradient", lambda: self.grad(pred, target))]
        for name, fn in terms:
            w = self.w[name]
            if w:
                term = w * fn()
                total = total + term
                parts[name] = term.detach()
        return total, parts


@torch.no_grad()
def psnr(pred, target, data_range=1.0):
    """Per-image PSNR, averaged over the batch. Batch-wide MSE would let one
    good image hide a bad one."""
    mse = F.mse_loss(pred.clamp(0, 1), target, reduction="none").flatten(1).mean(1)
    return (10 * torch.log10(data_range ** 2 / mse.clamp_min(1e-10))).mean()
