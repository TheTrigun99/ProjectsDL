"""Sampling of the glass and the camera, and the reflection's two optical effects.

A "scene" is the dict describing the glass as seen by the camera:
    R_map       (H, W, 1)  per-pixel Fresnel reflectance of both faces
    R1_map      (H, W)     reflectance of a single interface, natural light
    a_ghost     (H, W)     weight of the ghost copy relative to the first one
    defocus_px  scalar     diameter of the reflection's circle of confusion, px
    ghost_px    (H, W)     ghost displacement, in pixels
    ghost_dir   (H, W, 2)  direction of that displacement in the image plane
    f_px, theta0, direction, double_pane

The pixel-valued quantities depend on `size`: a scene sampled at 128 cannot be
used at 384 without rescaling (see sample_scene_lowres).

Sec. B of the supplementary material, simplified to a single flat pane.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import (DEFOCUS_MAX_PX, GLASS_IOR, K_MAP, N_FERME, N_OUVERT, P_OUVERT, P_PROCHE, S_PROCHE, SENSOR_W, SENSOR_W_c)

# --------------------------------------------------------------------------
# Fresnel
# --------------------------------------------------------------------------
def fresnel_single(cos_i, n=GLASS_IOR):
    """Unpolarised Fresnel, a single air -> glass interface."""
    cos_i = np.clip(cos_i, 1e-6, 1.0)
    sin_t = np.sqrt(np.clip(1 - cos_i ** 2, 0, 1)) / n
    cos_t = np.sqrt(np.clip(1 - sin_t ** 2, 0, 1))
    rs = ((cos_i - n * cos_t) / (cos_i + n * cos_t)) ** 2
    rp = ((n * cos_i - cos_t) / (n * cos_i + cos_t)) ** 2
    return 0.5 * (rs + rp)


def fresnel_angle(theta_i_deg, n=GLASS_IOR):
    """Same thing from the angle in degrees (the paper's formulation)."""
    theta_i = theta_i_deg * np.pi / 180
    theta_t = np.arcsin(np.sin(theta_i) / n)
    s, d = theta_i + theta_t, theta_i - theta_t
    rs = (np.sin(d) / np.sin(s)) ** 2
    rp = (np.tan(d) / np.tan(s)) ** 2
    return 0.5 * (rs + rp)


def fresnel_reflectance(R):
    """Two-interface glass, geometric sum of the round trips: 2R/(1+R)."""
    return 2 * R / (1 + R)


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
def tilt(rng, lo, hi, g=1.0, toward="lo"):
    """Same support [lo, hi], density tilted towards `toward`.

    g = 1.0 -> uniform, bit-identical to rng.uniform (RNG stream unchanged).
    g > 1.0 -> mass displaced: the median of u goes from 0.50 to 0.35 (g=1.5),
               0.25 (g=2), 0.17 (g=2.5), 0.12 (g=3).
    """
    u = rng.random()
    if g != 1.0:
        u = u ** g if toward == "lo" else 1.0 - (1.0 - u) ** g
    return lo + u * (hi - lo)


_GRID: dict[int, tuple] = {}


def _grid(size):
    """(xs, ys, r2), cached: they depend on size only."""
    g = _GRID.get(size)
    if g is None:
        ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) - (size - 1) / 2
        g = _GRID[size] = (np.ascontiguousarray(xs), np.ascontiguousarray(ys),
                           (xs * xs + ys * ys).astype(np.float32))
    return g


def _geometry(size, fov, phi, theta0, N, d_focus, d_refl, h_glass, gap,
              double_pane, ior=GLASS_IOR, sensor_w=SENSOR_W, polar=True, fov_det=False):
    """Per-pixel maps, once every scalar has been sampled.

    The part shared by sample_scene2 and deterministic_scene2: no sampling
    happens here, so calling this function does not touch the RNG stream.
    """
    f_px = (size / 2) / np.tan(np.radians(fov) / 2)

    # defocus: focus is on the transmitted subject, while the reflected scene
    # sits virtually at d_refl from the camera (mirror image behind the glass)
    f_m = f_px / size * sensor_w
    coc = f_m ** 2 / (N * max(d_focus - f_m, 1e-3)) * abs(d_refl - d_focus) / d_refl
    defocus_px = float(coc / sensor_w * size)

    xs, ys, r2 = _grid(size)
    nx = np.float32(np.sin(theta0) * np.cos(phi))
    ny = np.float32(np.sin(theta0) * np.sin(phi))
    nz = np.float32(np.cos(theta0))

    # p = v.n with v = (xs, ys, f_px) NOT normalised
    p = xs * nx
    p += ys * ny
    p += np.float32(f_px * nz)

    # inv = 1/|v|: avoids materialising d (H, W, 3) and the norm over axis=-1
    inv = r2 + np.float32(f_px * f_px)
    np.sqrt(inv, out=inv)
    np.reciprocal(inv, out=inv)

    cos_i = np.abs(p, dtype=np.float32) * inv
    np.clip(cos_i, 1e-6, 1.0, out=cos_i)

    # ghost direction: trace of the ray in the plane of the glass. The inv
    # factor cancels in the 2D normalisation, so we do not apply it.
    tx = xs - p * nx
    ty = ys - p * ny
    nrm2 = tx * tx
    nrm2 += ty * ty
    np.sqrt(nrm2, out=nrm2)
    np.maximum(nrm2, 1e-8, out=nrm2)              # ray parallel to the normal
    tx = tx / nrm2
    ty = ty / nrm2
    ghost_dir = np.stack([tx, ty], -1)

    # Fresnel and tan_t share sin_t and cos_t
    sin_t = cos_i * cos_i
    np.subtract(np.float32(1.0), sin_t, out=sin_t)
    np.sqrt(sin_t, out=sin_t)
    sin_t /= np.float32(ior)
    cos_t = sin_t * sin_t
    np.subtract(np.float32(1.0), cos_t, out=cos_t)
    np.sqrt(cos_t, out=cos_t)

    ncos_t = cos_t * np.float32(ior)
    rs = (cos_i - ncos_t) / (cos_i + ncos_t)
    rs *= rs
    ncos_i = cos_i * np.float32(ior)
    rp = (ncos_i - cos_t) / (ncos_i + cos_t)
    rp *= rp
    R1_map = rs + rp
    R1_map *= np.float32(0.5)

    # Incoherent multiple reflections between m interfaces of reflectance R:
    # m*R / (1 + (m-1)*R), m = 2 (single pane) or 4 (double glazing). Ghost
    # weight = second copy over first, (1-R)^2 (single) or (1-R)^4 (double).
    tot = ((lambda R: 4.0 * R / (1.0 + 3.0 * R)) if double_pane
           else (lambda R: 2.0 * R / (R + 1.0)))
    k = 4 if double_pane else 2
    if polar:
        # MISSION 7 (10/10): exact for natural light. Each polarization (s, p)
        # bounces with its OWN reflectance and the average comes last; averaging
        # R first (polar=False) overestimates R_map at oblique incidence (single
        # pane +3 % at 45 deg, +8 % at 60 deg; double glazing +9 % / +21 %).
        fs, fp = tot(rs), tot(rp)
        R_map = 0.5 * (fs + fp)
        a_s, a_p = (1.0 - rs) ** k, (1.0 - rp) ** k
        c1 = 0.5 * (fs / (1.0 + a_s) + fp / (1.0 + a_p))               # first copy
        c2 = 0.5 * (fs * a_s / (1.0 + a_s) + fp * a_p / (1.0 + a_p))   # ghost copy
        a_ghost = c2 / c1          # R_map * (r + a*g) / (1 + a) == c1*r + c2*g
    else:
        # behaviour of generator v1.1 (bit for bit): average the polarizations first
        R_map = tot(R1_map)
        a_ghost = (1.0 - R1_map) ** k

    # ghost: round-trip optical path inside the glass, plus the air gap of the
    # double glazing (where the ray is not refracted: full angle)
    # The two emergent beams are parallel, separated perpendicularly by
    # 2*path*cos_i (standard plane-parallel plate result), so a source at
    # d_refl is seen displaced by f_px * 2*path*cos_i / d_refl. ONE cos_i: it
    # is the projection of the in-plane walk direction onto the image plane.
    # "Project perpendicular to the ray" is the SAME cosine seen differently,
    # not a second one -- at the field centre the ray IS the optical axis.
    tan_t = sin_t / cos_t
    path = h_glass * tan_t
    if double_pane:
        tan_i = np.sqrt(np.maximum(1 - cos_i ** 2, 1e-8)) / np.maximum(cos_i, 1e-8)
        path = path + gap * tan_i
    ghost_px = (f_px * 2 * path * cos_i / d_refl).astype(np.float32)

    return {"f_px": f_px, "theta0": theta0,
            "direction": (float(np.cos(phi)), float(np.sin(phi))),
            "R_map": R_map[..., None], "R1_map": R1_map, "a_ghost": a_ghost,
            "defocus_px": defocus_px, "ghost_px": ghost_px,
            "ghost_dir": ghost_dir, "double_pane": bool(double_pane)}

def sample_scene2(rng, size, biais=False, biais_params=None,
                  ior=GLASS_IOR, sensor_w=SENSOR_W, polar=True):
    """Samples a camera, a pane and distances, then derives the maps.

    The order of the draws is frozen (it defines the RNG stream, hence the
    reproducibility of the datasets already generated): fov, phi, theta0, N,
    d_focus, d_glass, d_refl, h_glass, gap, double_pane.

    theta0 is bounded so that the angle of incidence stays <= 83 deg over the
    whole field, failing which Fresnel saturates at 1 in the corners.
    """
    B = biais_params if biais else None

    def T(lo, hi, key, toward="lo"):
        return tilt(rng, lo, hi, B[key] if B else 1.0, toward)

    fov = float(T(40, 90, "fov"))
    phi = float(rng.uniform(0, 2 * np.pi))       # BEFORE theta0: theta0_max depends on it
    half = np.arctan(np.tan(np.radians(fov) / 2)
                     * (abs(np.cos(phi)) + abs(np.sin(phi))))
    theta0_max = np.radians(83.0) - half
    theta0 = float(T(0, theta0_max, "theta0", "hi"))

    N = float(T(1.4 , 8.0, "N"))    # f-number
    d_focus = 10.0 ** float(T(-0.3, 0.9, "d_focus"))             
    d_hi = max(0.4, min(3.0, d_focus))               
    d_glass = float(np.exp(T(np.log(0.3), np.log(d_hi), "d_glass")))
    d_refl = d_glass + 10.0 ** float(T(-1.0, 1.7, "d_refl"))     # 0.1-50 m behind
    h_glass = float(T(0.003, 0.020, "h_glass", "hi"))            # glass thickness
    gap = float(T(0.006, 0.020, "gap", "hi"))                    # air gap
    double_pane = rng.random() < (B["p_double"] if B else 0.5)

    return _geometry(size, fov, phi, theta0, N, d_focus, d_refl, h_glass, gap,
                     double_pane, ior=ior, sensor_w=sensor_w, polar=polar)


# presets to isolate one effect (diagnostics, not generation)
DET_PARAM_PRESETS = {
    # maximal defocus: long focal length, wide open, most distant reflection
    "defocus": {"fov": 40, "phi": np.pi, "theta0": np.radians(45), "N": 1.8,
                "d_focus": 0.5012, "d_glass": 0.5, "d_refl": 50.1,
                "h_glass": 0.008, "double_pane": False, "gap": 0.0},
    # maximal ghosting: thick glass, double glazing, stopped down
    "ghost": {"fov": 40, "phi": np.pi, "theta0": np.radians(45), "N": 8.0,
              "d_focus": 0.5, "d_glass": 0.3, "d_refl": 1.0, "h_glass": 0.020,
              "double_pane": True, "gap": 0.020},
    # sharp reflection: neither defocus nor ghosting
    "net": {"fov": 40, "phi": np.pi, "theta0": np.radians(45), "N": 5,
            "d_focus": 0.5, "d_glass": 8, "d_refl": 1, "h_glass": 0.02,
            "double_pane": False, "gap": 0},
}

DRAW_ADOBE = {

             "fov": (40,90), "phi": (0,2*np.pi), "N" : (1.8,8), "d_focus": (-0.3,0.9), "d_glass": 0.3,
             "d_refl": (-1.0, 1.7), "h_glass": (0.08,0.2)

}

DRAW_CLOSE = {
             "fov": (40,90), "phi": (0,2*np.pi), "N" : (8, 11, 13, 16, 19, 22, 27, 32), "d_focus": (-0.3,0.9), "d_glass": 0.3,
             "d_refl": (-1.0, 1.7), "h_glass": (0.08,0.2)    
}
def deterministic_scene2(rng, size, params, ior=GLASS_IOR, sensor_w=SENSOR_W,
                         polar=True):
    """Like sample_scene2, but every parameter in `params` replaces its draw.

    A value that is absent, None or zero is sampled normally (which is why
    theta0=0 cannot be imposed: use an epsilon instead). `d_refl` is counted as
    an offset behind the pane, not as an absolute distance.

    Diagnostics only: unlike sample_scene2, the fallback draws here are
    uniform, and N and h_glass use narrower ranges (1.8-8.0 and 0.008-0.020
    instead of 1.4-8.0 and 0.003-0.020). The RNG stream also depends on WHICH
    parameters are supplied, since `or` short-circuits the draw.
    """
    get = lambda k: params.get(k) or None

    fov = get("fov") or float(rng.uniform(40, 90))
    phi = get("phi") or float(rng.uniform(0, 2 * np.pi))
    half = np.arctan(np.tan(np.radians(fov) / 2)
                     * (abs(np.cos(phi)) + abs(np.sin(phi))))
    theta0_max = np.radians(83.0) - half
    theta0 = get("theta0")
    if theta0 is None:
        theta0 = float(rng.uniform(0, theta0_max))
    else:
        assert theta0 <= theta0_max, f"theta0 {theta0:.3f} > theta0_max {theta0_max:.3f}"

    N = get("N") or float(rng.uniform(1.8, 8.0))
    d_focus = get("d_focus") or 10.0 ** float(rng.uniform(-0.3, 0.9))
    d_hi = max(0.4, min(3.0, d_focus))
    d_glass = get("d_glass") or float(np.exp(rng.uniform(np.log(0.3), np.log(d_hi))))
    d_refl = d_glass + (get("d_refl") or 10.0 ** float(rng.uniform(-1.0, 1.7)))
    h_glass = get("h_glass") or float(rng.uniform(0.008, 0.020))

    dp = params.get("double_pane")
    double_pane = (rng.random() < 0.5) if dp is None else bool(dp)
    gap = (get("gap") or float(rng.uniform(0.006, 0.020))) if double_pane else 0.0

    return _geometry(size, fov, phi, theta0, N, d_focus, d_refl, h_glass, gap,
                     double_pane, ior=ior, sensor_w=sensor_w, polar=polar)


def sample_scene_lowres(rng, size, coarse=128, **kw):
    """Scene sampled at low resolution then interpolated: ~9x less compute.

    The quantities expressed in pixels (f_px, defocus_px, ghost_px) are scaled
    by size/coarse, failing which the blur would be three times too small.
    """
    s = sample_scene2(rng, coarse, **kw)
    k = size / coarse
    n_int = 4 if s["double_pane"] else 2
    up = lambda a: cv2.resize(a, (size, size), interpolation=cv2.INTER_LINEAR)
    return {"f_px": s["f_px"] * k, "theta0": s["theta0"],
            "direction": s["direction"],
            "R_map": up(s["R_map"][..., 0])[..., None], "R1_map": up(s["R1_map"]),
            # polar=False: recomputed from the interpolated R1 map, as v1.1 did (bit for bit)
            "a_ghost": (up(s["a_ghost"]) if kw.get("polar", True)
                        else (1.0 - up(s["R1_map"])) ** n_int),
            "defocus_px": s["defocus_px"] * k,
            "ghost_px": up(s["ghost_px"]) * k,
            "ghost_dir": up(s["ghost_dir"]), "double_pane": s["double_pane"]}


def sample_scene3(rng, size, fov, biais_params=None, p_ouvert=P_OUVERT, sensor_w=SENSOR_W_c, polar=True):
    B = biais_params 

    def T(lo, hi, key, toward="lo"):
        return tilt(rng, lo, hi, B[key] if B else 1.0, toward)
    
    phi    = rng.uniform(0, 2*np.pi)

    half = np.arctan(np.tan(np.radians(fov) / 2) * (abs(np.cos(phi)) + abs(np.sin(phi))))
    theta0_max = np.radians(83.0) - half
    theta0 = float(T(0, theta0_max, "theta0", "hi"))

    lo, hi = N_OUVERT if rng.random() < p_ouvert else N_FERME
    N = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))  

    f_m = sensor_w / (2 * np.tan(np.radians(fov) / 2))
    mise_min = K_MAP * f_m
    proche = bool(rng.random() < P_PROCHE)
    if proche:                       
        d_focus = float(rng.uniform(max(0.3, mise_min), 1.0))               # t close, focus on it
        d_glass = float(rng.uniform(0.15, d_focus - 0.05))   # glass in front of t
        s = float(np.exp(rng.uniform(np.log(S_PROCHE[0]), np.log(S_PROCHE[1]))))   
    else:                                             # wild / real20 / nature20 : actual laws
        d_focus = d_focus = max(10.0 ** float(T(0, 0.9, "d_focus")), mise_min) # 1-8m           
        d_hi = max(0.4, min(3.0, d_focus))               
        d_glass = float(np.exp(T(np.log(0.3), np.log(d_hi), "d_glass")))
        s = 10.0 ** float(T(-1.0, 1.7, "d_refl"))
    d_refl = d_glass + s
    h_glass = float(T(0.003, 0.020, "h_glass", "hi"))            # glass thickness
    gap = float(T(0.006, 0.020, "gap", "hi"))                    # air gap
    double_pane = rng.random() < (B["p_double"] if B else 0.1)
    g = _geometry(size, fov, phi, theta0, N, d_focus, d_refl, h_glass, gap, double_pane,
                  sensor_w=sensor_w, polar=polar)      
    return {**g, "fov": float(fov), "N": N, "d_focus": d_focus, "d_refl": float(d_refl), "proche": proche}

# --------------------------------------------------------------------------
# Optical effects
# --------------------------------------------------------------------------
def disk_kernel_continuous(radius, ss=8):
    """Anti-aliased disk kernel (ss x ss subsamples per pixel)."""
    R = int(np.ceil(radius))
    o = (np.arange(ss) + 0.5) / ss - 0.5
    dy, dx = np.meshgrid(o, o, indexing="ij")
    y, x = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
    k = (((y[..., None, None] + dy) ** 2
          + (x[..., None, None] + dx) ** 2) <= radius ** 2).mean(axis=(2, 3))
    s = k.sum()
    return None if s <= 0 else (k / s).astype(np.float32)


def defocus_continuous(img, diameter):
    """Defocus blur: a disk kernel (bokeh), not a Gaussian."""
    radius = diameter / 2
    if radius < 0.5:
        return img
    return cv2.filter2D(img, -1, disk_kernel_continuous(radius))


def double_reflection(img, scene):
    """Ghost: the reflection off the 2nd face comes out displaced by an amount
    that varies PER PIXEL -- zero where the ray is normal to the pane, growing
    from there. The weight is (1-R1)^2 per traversal of the glass, ^4 for
    double glazing.

    SIGN -- the map is p + ghost_dir*shift, not minus. Inside the glass the ray
    keeps advancing along +t_hat, so the ghost path exits further along +t_hat
    and SAMPLES the scene further along +ghost_dir; cv2.remap wants exactly
    that sampling offset. Careful: the visible copy of a feature then appears
    displaced along -ghost_dir, toward the point of normal incidence. Those two
    statements are inverses of each other, and putting the second one in this
    map is what made the ghost land on the wrong side.
    """
    shift = scene["ghost_px"]
    if float(shift.max()) < 0.15:                 # negligible over the whole field
        return img
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dxy = scene["ghost_dir"]
    map_x = (xs + dxy[..., 0] * shift).astype(np.float32)
    map_y = (ys + dxy[..., 1] * shift).astype(np.float32)
    ghost = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT)
    a = scene["a_ghost"] if "a_ghost" in scene else \
        (1.0 - scene["R1_map"]) ** (4 if scene["double_pane"] else 2)   # older scene dicts
    a = a[..., None]
    return (img + a * ghost) / (1.0 + a)
