"""ACR steps 1-4: linearisation, demosaicing, black level, conversion to XYZ.

A RAW goes in, (xyz, illuminant, saturation mask) comes out. The mask is the
delicate part: it is computed in the sensor frame (before demosaicing) and then
brought back into the frame of the output image.

No white balance (user_wb = 1,1,1,1): we want the "as-shot" XYZ, where the
colour of the illuminant is preserved, and it is the simulation that decides
about chromatic adaptation later on.
"""
from __future__ import annotations

import numpy as np
import rawpy

from .dng_metadata import camera_to_xyz
from .colorimetry import xy_to_xyz

# rotations applied by LibRaw (raw.sizes.flip)
_FLIP = {
    0: lambda a: a,                     # no rotation
    3: lambda a: a[::-1, ::-1],         # 180 deg
    5: lambda a: np.rot90(a, 1),        # 90 deg counter-clockwise
    6: lambda a: np.rot90(a, -1),       # 90 deg clockwise
}

# linear postprocess, with neither white balance nor auto-brightness
_PP = dict(gamma=(1, 1), no_auto_bright=True, adjust_maximum_thr=0,
           output_bps=16, use_camera_wb=False, use_auto_wb=False,
           user_wb=[1.0, 1.0, 1.0, 1.0], output_color=rawpy.ColorSpace.raw)


def clip_level(raw, min_pixels=64):
    """The lower of the two ceilings: the sensor's real plateau, and the
    white_level above which LibRaw flattens everything in postprocess. No
    plateau detected -> no sensor clipping -> only LibRaw's ceiling counts.

    Kept for reference and for the equivalence test: like the notebook, the
    pipeline uses raw.white_level directly and never calls this.
    """
    wl = float(raw.white_level)
    vis = raw.raw_image_visible
    mx = int(vis.max())
    if int((vis >= mx - 1).sum()) >= min_pixels:
        return min(wl, float(mx))
    return wl


def align_mask(mask, shape, flip=0, path=None, half_size=True):
    """HxW boolean mask, from the sensor frame to the `shape` frame.

    The fast path reproduces LibRaw's 2x2 binning (half_size) then the
    rotation. If the geometry is not affine (fuji_rotate, pixel_aspect...), we
    push the mask through the SAME postprocess as the image, which dilates it a
    little: conservative, therefore acceptable.
    """
    h, w = mask.shape
    if (h, w) == tuple(shape):          # linear DNG: no binning
        return np.ascontiguousarray(mask)

    m = np.pad(mask, ((0, h % 2), (0, w % 2)))        # ceil, as LibRaw does
    m = m.reshape(-1, 2, m.shape[1] // 2, 2).any(axis=(1, 3))
    m = _FLIP.get(flip, lambda a: a)(m)
    if m.shape == tuple(shape):
        return np.ascontiguousarray(m)

    with rawpy.imread(str(path)) as raw:
        v = raw.raw_image_visible
        v[:] = np.where(mask, int(raw.white_level), 0)
        out = raw.postprocess(half_size=half_size, **_PP)
    return np.ascontiguousarray(out.max(-1) > 0)


def read_raw_to_xyz(path, half_size=True):
    """RAW -> (xyz, illuminant_xyz, saturated, ok).

    ok=False for sRAW / linear DNG: with no Bayer mosaic, the per-channel
    saturation mask makes no sense, so the image is dropped.
    """
    with rawpy.imread(str(path)) as raw:
        black = np.mean(raw.black_level_per_channel)
        wl = raw.white_level
        bayer = raw.raw_image_visible
        sat_b = bayer >= black + 0.98 * (wl - black)
        if sat_b.ndim == 3:                  # sRAW / linear DNG
            return None, None, None, False

        cam = raw.postprocess(half_size=half_size, **_PP)
        saturated = align_mask(sat_b, cam.shape[:2], raw.sizes.flip, path,
                               half_size=half_size)
        # the DNG ColorMatrix goes XYZ -> CAM; we invert it (ACR step 4)
        cam_to_xyz, xy = camera_to_xyz(path)

    illum = xy_to_xyz(xy)

    # output_bps=16 yields uint16: we normalise, so saturation is at 1
    cam = cam.astype(np.float32) / 65535.0

    # Func. S6: clamp each channel to the camera white before the matrix,
    # otherwise a clipped channel drags the chromaticity towards green.
    cam_white = np.linalg.solve(cam_to_xyz, illum)
    cam_white = np.clip(cam_white / cam_white.max(), 0.001, 1.0).astype(np.float32)
    cam = np.minimum(cam, cam_white)

    xyz = np.einsum("ij,hwj->hwi", cam_to_xyz, cam)
    return np.clip(xyz, 0, None), illum, saturated, True
