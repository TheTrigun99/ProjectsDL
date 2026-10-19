# genere par extract_notebook.py -- ne pas editer
import sys
sys.path.insert(0, '/home/damien/ProjectsDL/reflection_removal/main_files')
import numpy as np, cv2, random, rawpy, exifread
from pathlib import Path
PATCH, TAU, SEED = 384, 0.1329, 1009
# ===== CELL 8
DNG = {"ColorMatrix1": 0xC621, "ColorMatrix2": 0xC622,
    "CameraCalibration1": 0xC623, "CameraCalibration2": 0xC624,
    "AnalogBalance": 0xC627, "AsShotNeutral": 0xC628, "AsShotWhiteXY": 0xC629,
    "CalibrationIlluminant1": 0xC65A, "CalibrationIlluminant2": 0xC65B}


NOMS = {0xC621:"ColorMatrix1", 0xC622:"ColorMatrix2", 0xC623:"CameraCalibration1",
    0xC624:"CameraCalibration2", 0xC627:"AnalogBalance", 0xC628:"AsShotNeutral",
    0xC629:"AsShotWhiteXY", 0xC65A:"CalibrationIlluminant1",
    0xC65B:"CalibrationIlluminant2", 0xC65C:"BestQualityScale"}

_FLIP = {
    0: lambda a: a,                     # no rotation
    3: lambda a: a[::-1, ::-1],         # 180°
    5: lambda a: np.rot90(a, 1),        # 90° anticlockwise
    6: lambda a: np.rot90(a, -1),       # 90° clockwise
}


def dng_tag(tags, name):
    """Lit un tag DNG. exifread ne les nomme pas : on passe par l'ID hexa."""
    t = tags.get("Image Tag 0x%04X" % DNG[name])
    if t is None:
        return None
    out = []
    for v in t.values:
        out.append(float(v.num) / float(v.den) if hasattr(v, "num") else float(v))
    return np.array(out, np.float64)
def exif_tag(tags, name):
    """EXIF standard."""
    t = tags.get(name)
    if t is None or not t.values:
        return None
    v = t.values[0]
    return float(v.num) / float(v.den) if hasattr(v, "num") else float(v)

def reject_noms(dng_file):
        with open(dng_file[0], "rb") as f:
            tags = exifread.process_file(f, details=False)
        asshot = dng_tag(tags,"AsShotNeutral")
        whitexy = dng_tag(tags, "AsShotWhiteXY")
        exif = exif_tag(tags, "EXIF FNumber")
        if asshot is None and whitexy is None:
            return False
        if exif is None:
            return False
        return True

# ===== CELL 16


# temperatures of the normalized illuminants (EXIF LightSource, spec DNG ch. 6)
ILLUMINANT_TEMP = {1: 6504., 2: 4100., 3: 2856., 4: 5500., 9: 5500., 10: 6504.,
                   11: 7504., 12: 6430., 13: 5050., 14: 4150., 15: 3450.,
                   17: 2856., 18: 4874., 19: 6774., 20: 5503., 21: 6504.,
                   22: 7504., 23: 5003., 24: 3200.}
# ------------------------------------------------------- xy -> temperature
def xy_to_temp(xy):
    """CCT par l'approximation de McCamy (1992). L'ecart a la methode de
    Robertson de la spec DNG est de quelques kelvins pres du lieu planckien."""
    x, y = float(xy[0]), float(xy[1])
    n = (x - 0.3320) / (0.1858 - y + 1e-12)
    return 449.0 * n ** 3 + 3525.0 * n ** 2 + 6823.3 * n + 5520.33


# ------------------------------------------- Func. S7 : XYZ -> camera pour un xy
def find_xyz_to_camera(xy, cal):
    """Interpole entre les deux matrices de calibration selon la temperature
    du point blanc vise. L'interpolation se fait en 1/T (mireds), pas en T."""
    t1, t2 = cal["temp1"], cal["temp2"]
    cm1, cm2 = cal["cm1"], cal["cm2"]
    cc1, cc2 = cal["cc1"], cal["cc2"]
    if cm2 is None:                                  # une seule calibration
        cm, cc = cm1, cc1
    else:
        temp = xy_to_temp(xy)
        if temp <= min(t1, t2):
            g = 1.0 if t1 <= t2 else 0.0
        elif temp >= max(t1, t2):
            g = 0.0 if t1 <= t2 else 1.0
        else:
            g = (1.0 / temp - 1.0 / t2) / (1.0 / t1 - 1.0 / t2)
        cm = g * cm1 + (1.0 - g) * cm2
        cc = g * cc1 + (1.0 - g) * cc2
    return cal["ab"] @ cc @ cm                       # AB . CC . CM


# ------------------------------- Func. S3 : AsShotNeutral -> xy (point fixe)
def neutral_to_xy(neutral, cal, n_iter=30, tol=1e-9):
    """La matrice depend du point blanc, qui depend de la matrice : on itere.
    Depart a D50, comme la spec DNG."""
    xy = np.array([0.34567, 0.35850])                # D50
    for _ in range(n_iter):
        xyz = np.linalg.solve(find_xyz_to_camera(xy, cal), neutral)
        new = np.array([xyz[0], xyz[1]]) / max(xyz.sum(), 1e-12)
        if np.abs(new - xy).max() < tol:
            return new, True
        xy = new
    return xy, False

def read_calibration(path):
    """getting what Func. S7. needs"""
    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=False)
    ashot = dng_tag(tags, "AsShotNeutral")
    cm1 = dng_tag(tags, "ColorMatrix1")
    cm2 = dng_tag(tags, "ColorMatrix2")
    cc1 = dng_tag(tags, "CameraCalibration1")
    cc2 = dng_tag(tags, "CameraCalibration2")
    il1 = dng_tag(tags, "CalibrationIlluminant1")
    il2 = dng_tag(tags, "CalibrationIlluminant2")
    ab = dng_tag(tags, "AnalogBalance")
    eye = np.eye(3)
    cal = {"cm1": cm1.reshape(3, 3),
           "cm2": None if cm2 is None else cm2.reshape(3, 3),
           "cc1": eye if cc1 is None else cc1.reshape(3, 3),
           "cc2": eye if cc2 is None else cc2.reshape(3, 3),
           "ab": eye if ab is None else np.diag(ab),
           "temp1": ILLUMINANT_TEMP.get(int(il1[0]) if il1 is not None else 17, 2856.),
           "temp2": ILLUMINANT_TEMP.get(int(il2[0]) if il2 is not None else 21, 6504.), "ashot": ashot}
    return cal, tags

# ------------------------------------------------------------- Func. S3
def compute_white_xy(path):
    """WhitePoint in xy. Use AsShotWhiteXY if it exists, uses otherwise
    AsShotNeutral.Both are present in our current filtered DATASET."""
    cal, tags = read_calibration(path)
    xy = dng_tag(tags, "AsShotWhiteXY")

    if xy is not None:
        return np.asarray(xy[:2], np.float64), "AsShotWhiteXY", True

    neutral = dng_tag(tags, "AsShotNeutral")
    # if neutral is None:
    #     raise ValueError("ni AsShotWhiteXY ni AsShotNeutral dans %s" % path)
    
    xy, ok = neutral_to_xy(neutral[:3], cal)
    return xy, "AsShotNeutral", ok

def xy_to_xyz(xy, Y=1.0):
    """Chromaticity -> tristimulus, Luminance = Y = 1."""
    x, y = float(xy[0]), float(xy[1])
    return np.array([Y * x / y, Y, Y * (1.0 - x - y) / y])

def camera_to_xyz(path):
    """Matrice CAM -> XYZ (Func. S3 l.5-7) and WhitePoint as-shot."""
    cal, tags = read_calibration(path)
    xy = dng_tag(tags, "AsShotWhiteXY")
    if xy is not None:
        xy, ok = np.asarray(xy[:2], np.float64), True
    else:
        neutral = dng_tag(tags, "AsShotNeutral")
        # if neutral is None:
        #     raise ValueError(f"ni AsShotWhiteXY ni AsShotNeutral dans {path}")
        xy, ok = neutral_to_xy(neutral[:3], cal)
        if not ok:
            print(f"NeutralToXY n'a pas convergé: {path}")
    return np.linalg.inv(find_xyz_to_camera(xy, cal)).astype(np.float32), xy

# XYZ -> CAM :  M = find_xyz_to_camera(xy, cal)
# CAM -> XYZ:   inv(M)

# ===== CELL 19
def clip_level(raw, min_pixels=64):
    """Plus bas des deux plafonds : le plateau réel du capteur, et le white_level
    au-dessus duquel LibRaw aplatit tout dans postprocess. Pas de plateau
    détecté -> pas d'écrêtage capteur -> seul le plafond LibRaw compte."""
    wl  = float(raw.white_level)
    vis = raw.raw_image_visible
    mx  = int(vis.max())
    if int((vis >= mx - 1).sum()) >= min_pixels:
        return min(wl, float(mx))
    return wl


# ===== CELL 21
def read_exposure(path):
    """Exposition e = s * g / n**2 (Sec. 3.1 du papier)."""
    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=False)

    def tag(name, default):
        t = tags.get(name)
        if t is None or not t.values:
            print(f"Il manque la valeur {name}")
            return default, False
        v = t.values[0]
        try:
            return float(v.num) / float(v.den), True
        except AttributeError:
            return float(v), True

    s, bool_s = tag("EXIF ExposureTime", 1 / 60)         
    n, bool_n = tag("EXIF FNumber", 4.0)         
    g, bool_g = tag("EXIF ISOSpeedRatings", 100.0)       
    return s * g / max(n, 0.7) ** 2, bool_s * bool_n * bool_g

_PP = dict(gamma=(1, 1), no_auto_bright=True, adjust_maximum_thr=0, output_bps=16,
           use_camera_wb=False, use_auto_wb=False, user_wb=[1.0, 1.0, 1.0, 1.0],
           output_color=rawpy.ColorSpace.raw)


def align_mask(path, sat_b, **pp):
    """mask : bool HxW en repère capteur -> repère `shape` (= cam.shape[:2])."""
    with rawpy.imread(str(path)) as raw:
        v = raw.raw_image_visible
        v[:] = 0
        v[sat_b] = int(raw.white_level)
        out = raw.postprocess(**pp)
        print(out.shape, out.max(-1).shape, out.max())
    return np.ascontiguousarray(out.max(-1) > 0.25 * 65535)

def align_maskv2(mask, shape, flip=0, path=None, half_size=True):
    """mask : bool HxW en repère capteur -> repère `shape` (= cam.shape[:2]).
       optimized version
    """
    h, w = mask.shape
    if (h, w) == tuple(shape):            # DNG linéaire : pas de binning côté LibRaw
        return np.ascontiguousarray(mask)

    m = np.pad(mask, ((0, h % 2), (0, w % 2)))          # ceil, comme LibRaw
    m = m.reshape(-1, 2, m.shape[1] // 2, 2).any(axis=(1, 3))
    m = _FLIP.get(flip, lambda a: a)(m)
    if m.shape == tuple(shape):
        return np.ascontiguousarray(m)
    # géométrie non affine (fuji_rotate, pixel_aspect...) : on refait passer
    # le masque dans le MEME postprocess que l'image.
    with rawpy.imread(str(path)) as raw:
        v = raw.raw_image_visible
        v[:] = np.where(mask, int(raw.white_level),0)
        out = raw.postprocess(half_size=half_size, **_PP)
    return np.ascontiguousarray(out.max(-1) > 0) #really conservative, dilating the mask a bit

def read_raw_to_xyz(path, half_size=True):
    """ACR étapes 1-4 : linearization, dématriçage, black level, -> XYZ.

    user_wb=(1,1,1,1) no white balance, we get XYZ
    "as-shot" where illuminants color is preserved.
    
    Output (xyz, illuminant_xyz, saturated, sat_bool)

    sat_bool = False for sRAW images where the bayer mask to get saturation mask makes no sense
    """
    with rawpy.imread(str(path)) as raw:
        black = np.mean(raw.black_level_per_channel)
        wl    = raw.white_level #clip_level(raw, min_pixels=64) # 
        bayer = raw.raw_image_visible
        sat_b = bayer >= black + 0.98 * (wl - black)
        if sat_b.ndim == 3:              # sRAW / DNG linear : no mosaic, I have to take them off later
            return None, None, None, False
        
        cam = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            adjust_maximum_thr=0,
            output_bps=16,
            use_camera_wb=False, 
            use_auto_wb=False, 
            user_wb=[1.0, 1.0, 1.0, 1.0],
            output_color=rawpy.ColorSpace.raw,  
            half_size=half_size,)
        saturated = align_maskv2(sat_b, cam.shape[:2], raw.sizes.flip, path)

        # ColorMatrix of DNG : XYZ -> CAM (rgb camera) ; we inverse it (step 4 ACR)
        cam_to_xyz, xy = camera_to_xyz(path)

    illum = None
    white_xyz = xy_to_xyz(xy)
    illum = white_xyz

    # output_bps=16 gives uint16: (2^16 - 1 = 65535),
    # we normalize ([0,1])/ Saturation >= 1. 
    cam = cam.astype(np.float32) / 65535.0

    # func S6.
    cam_white = np.linalg.solve(cam_to_xyz, illum)   # = inv(cam_to_xyz) @ illum
    cam_white = np.clip(cam_white / cam_white.max(), 0.001, 1.0).astype(np.float32)
    cam = np.minimum(cam, cam_white) 
    
    xyz = np.einsum("ij,hwj->hwi", cam_to_xyz, cam)
    return np.clip(xyz, 0, None), illum, saturated, True
