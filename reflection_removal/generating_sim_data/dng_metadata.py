"""Reading DNG metadata: as-shot white point, matrices, exposure.

Follows chapter 6 of the DNG spec and functions S3 / S7 of the supplementary
material of Kee et al. (CVPR 2025). exifread does not name the DNG tags, so we
go through their hexadecimal identifier.
"""
from __future__ import annotations

import exifread
import numpy as np


DNG_TAGS = {"ColorMatrix1": 0xC621, "ColorMatrix2": 0xC622,
            "CameraCalibration1": 0xC623, "CameraCalibration2": 0xC624,
            "AnalogBalance": 0xC627, "AsShotNeutral": 0xC628,
            "AsShotWhiteXY": 0xC629, "CalibrationIlluminant1": 0xC65A,
            "CalibrationIlluminant2": 0xC65B, "BestQualityScale": 0xC65C}

# temperatures of the standard illuminants (EXIF LightSource, DNG spec ch. 6)
ILLUMINANT_TEMP = {1: 6504., 2: 4100., 3: 2856., 4: 5500., 9: 5500., 10: 6504.,
                   11: 7504., 12: 6430., 13: 5050., 14: 4150., 15: 3450.,
                   17: 2856., 18: 4874., 19: 6774., 20: 5503., 21: 6504.,
                   22: 7504., 23: 5003., 24: 3200.}


def read_tags(path):
    with open(path, "rb") as f:
        return exifread.process_file(f, details=False)


def dng_tag(tags, name):
    """DNG tag -> float vector, or None if absent."""
    t = tags.get("Image Tag 0x%04X" % DNG_TAGS[name])
    if t is None:
        return None
    out = [float(v.num) / float(v.den) if hasattr(v, "num") else float(v)
           for v in t.values]
    return np.array(out, np.float64)


def exif_tag(tags, name):
    """Standard EXIF tag -> float, or None."""
    t = tags.get(name)
    if t is None or not t.values:
        return None
    v = t.values[0]
    return float(v.num) / float(v.den) if hasattr(v, "num") else float(v)


def has_required_metadata(path):
    """Entry filter: with no white point and no aperture, the image is useless."""
    tags = read_tags(path)
    if dng_tag(tags, "AsShotNeutral") is None and dng_tag(tags, "AsShotWhiteXY") is None:
        return False
    return exif_tag(tags, "EXIF FNumber") is not None


# ------------------------------------------------------------------ xy <-> T
def xy_to_temp(xy):
    """CCT via McCamy's approximation (1992). The gap to the DNG spec's
    Robertson method is a few kelvin near the Planckian locus."""
    x, y = float(xy[0]), float(xy[1])
    n = (x - 0.3320) / (0.1858 - y + 1e-12)
    return 449.0 * n ** 3 + 3525.0 * n ** 2 + 6823.3 * n + 5520.33


# ---------------------------------------------- Func. S7 : XYZ -> camera en xy
def find_xyz_to_camera(xy, cal):
    """Interpolates the two calibration matrices from the temperature of the
    targeted white point. The interpolation is in 1/T (mireds), not in T."""
    t1, t2 = cal["temp1"], cal["temp2"]
    cm1, cm2 = cal["cm1"], cal["cm2"]
    cc1, cc2 = cal["cc1"], cal["cc2"]
    if cm2 is None:                                   # a single calibration
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
    return cal["ab"] @ cc @ cm                        # AB . CC . CM


# ------------------------------- Func. S3 : AsShotNeutral -> xy (point fixe)
def neutral_to_xy(neutral, cal, n_iter=30, tol=1e-9):
    """The matrix depends on the white point, which depends on the matrix, so
    we iterate. Starts at D50, as the DNG spec does."""
    xy = np.array([0.34567, 0.35850])                 # D50
    for _ in range(n_iter):
        xyz = np.linalg.solve(find_xyz_to_camera(xy, cal), neutral)
        new = np.array([xyz[0], xyz[1]]) / max(xyz.sum(), 1e-12)
        if np.abs(new - xy).max() < tol:
            return new, True
        xy = new
    return xy, False


def read_calibration(path):
    """Everything Func. S7 needs, plus the raw tags."""
    tags = read_tags(path)
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
           "temp2": ILLUMINANT_TEMP.get(int(il2[0]) if il2 is not None else 21, 6504.),
           "ashot": dng_tag(tags, "AsShotNeutral")}
    return cal, tags


def white_xy(path, cal=None, tags=None):
    """Func. S3: as-shot white point, in xy.

    AsShotWhiteXY if it exists, otherwise the fixed point on AsShotNeutral. At
    least one of the two is present in the dataset filtered by
    has_required_metadata. Returns (xy, source, converged).
    """
    if cal is None or tags is None:
        cal, tags = read_calibration(path)
    xy = dng_tag(tags, "AsShotWhiteXY")
    if xy is not None:
        return np.asarray(xy[:2], np.float64), "AsShotWhiteXY", True
    xy, ok = neutral_to_xy(cal["ashot"][:3], cal)
    return xy, "AsShotNeutral", ok


def camera_to_xyz(path):
    """CAM -> XYZ matrix (Func. S3 l.5-7) and as-shot white point, in xy.

    XYZ -> CAM = find_xyz_to_camera(xy, cal); we return its inverse.
    """
    cal, tags = read_calibration(path)
    xy, src, ok = white_xy(path, cal, tags)
    if not ok:
        print(f"NeutralToXY n'a pas converge: {path}")
    return np.linalg.inv(find_xyz_to_camera(xy, cal)).astype(np.float32), xy


def read_exposure(path):
    """Exposure e = s * g / n**2 (Sec. 3.1 of the paper).

    Returns (e, complete); complete=False if one of the three tags was missing
    and was replaced by a default -- those images are dropped from the cache.
    """
    tags = read_tags(path)

    def tag(name, default):
        v = exif_tag(tags, name)
        if v is None:
            print(f"Il manque la valeur {name}")
            return default, False
        return v, True

    s, ok_s = tag("EXIF ExposureTime", 1 / 60)
    n, ok_n = tag("EXIF FNumber", 4.0)
    g, ok_g = tag("EXIF ISOSpeedRatings", 100.0)
    return s * g / max(n, 0.7) ** 2, bool(ok_s and ok_n and ok_g)
