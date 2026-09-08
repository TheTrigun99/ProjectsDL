import torch
import torch.nn.functional as F


def ssim_fast_t(m, t, data_range=1.0):
    """(B,3,H,W) float32 -> (moyenne, écart-type) par image, tenseurs (B,).

    Transposition de ssim_fast : boîte 7x7, covariance non biaisée, pondération
    par canal de Kee (Sec. D.1). Le crop [3:-3] est gratuit : avec un noyau 7x7,
    la sortie 'valid' EST exactement la région qu'on gardait. Différentiable.
    """
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    cn = 49 / 48
    f = lambda x: F.avg_pool2d(x, 7, stride=1)          # pas de padding
    ux, uy = f(m), f(t)
    vx  = cn * (f(m * m) - ux * ux)
    vy  = cn * (f(t * t) - uy * uy)
    vxy = cn * (f(m * t) - ux * uy)
    S = ((2*ux*uy + C1) * (2*vxy + C2)) / ((ux*ux + uy*uy + C1) * (vx + vy + C2))

    w = m.mean(dim=(2, 3))                              
    w = w / w.sum(1, keepdim=True).clamp_min(1e-8)
    S = (S * w[:, :, None, None]).sum(1)                # (B,H-6,W-6)
    return S.flatten(1).mean(1) #, S.flatten(1).std(1, correction=0)
