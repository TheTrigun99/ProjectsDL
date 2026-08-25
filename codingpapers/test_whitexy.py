# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import rawpy
from whitexy import (compute_white_xy, read_calibration, dng_tag,
                     find_xyz_to_camera, xy_to_temp, xy_to_xyz)

DNG_DIR = Path(__file__).parent / "smoke" / "fivek_data" / "dng"

print("=== quel tag chaque fichier fournit-il ? ===")
for p in sorted(DNG_DIR.glob("*.dng")):
    cal, tags = read_calibration(p)
    xyt = dng_tag(tags, "AsShotWhiteXY")
    nt = dng_tag(tags, "AsShotNeutral")
    print("  %-24s AsShotWhiteXY=%-3s AsShotNeutral=%-3s ColorMatrix2=%-3s  T1=%.0fK T2=%.0fK"
          % (p.stem, "oui" if xyt is not None else "NON",
             "oui" if nt is not None else "NON",
             "oui" if cal["cm2"] is not None else "NON",
             cal["temp1"], cal["temp2"]))

print("\n=== Func. S3 : resultat ===")
print("  %24s | %14s | %8s %8s | %8s | conv" % ("fichier", "source", "x", "y", "CCT (K)"))
print("  " + "-" * 78)
for p in sorted(DNG_DIR.glob("*.dng")):
    xy, src, ok = compute_white_xy(p)
    print("  %24s | %14s | %8.5f %8.5f | %8.0f | %s"
          % (p.stem, src, xy[0], xy[1], xy_to_temp(xy), "oui" if ok else "NON"))

print("\n=== convergence du point fixe ===")
p = sorted(DNG_DIR.glob("*.dng"))[0]
cal, tags = read_calibration(p)
neutral = dng_tag(tags, "AsShotNeutral")[:3]
xy = np.array([0.34567, 0.35850])
print("  %s, AsShotNeutral = %s" % (p.stem, np.round(neutral, 5)))
print("  %4s | %9s %9s | %7s | %10s" % ("iter", "x", "y", "CCT", "|delta|"))
for k in range(8):
    xyz = np.linalg.solve(find_xyz_to_camera(xy, cal), neutral)
    new = np.array([xyz[0], xyz[1]]) / xyz.sum()
    print("  %4d | %9.6f %9.6f | %7.0f | %10.2e"
          % (k, new[0], new[1], xy_to_temp(new), np.abs(new - xy).max()))
    xy = new

print("\n=== controle : le xy retrouve reproduit-il AsShotNeutral ? ===")
for p in sorted(DNG_DIR.glob("*.dng")):
    cal, tags = read_calibration(p)
    neutral = dng_tag(tags, "AsShotNeutral")[:3]
    xy, _, _ = compute_white_xy(p)
    pred = find_xyz_to_camera(xy, cal) @ xy_to_xyz(xy)
    pred = pred / pred[1] * neutral[1]
    print("  %-24s lu %s  reconstruit %s  ecart %.2e"
          % (p.stem, np.round(neutral, 5), np.round(pred, 5),
             np.abs(pred - neutral).max()))

print("\n=== le point de depart influence-t-il le resultat ? ===")
p = sorted(DNG_DIR.glob("*.dng"))[0]
cal, tags = read_calibration(p)
neutral = dng_tag(tags, "AsShotNeutral")[:3]
for depart, nom in [([0.34567, 0.3585], "D50"), ([0.3127, 0.3290], "D65"),
                    ([0.4476, 0.4074], "A"), ([0.20, 0.20], "aberrant")]:
    xy = np.array(depart, np.float64)
    for _ in range(80):
        xyz = np.linalg.solve(find_xyz_to_camera(xy, cal), neutral)
        xy = np.array([xyz[0], xyz[1]]) / xyz.sum()
    print("  depart %-9s -> x %.9f  y %.9f" % (nom, xy[0], xy[1]))

print("\n=== recoupement avec rawpy (camera_whitebalance) ===")
for p in sorted(DNG_DIR.glob("*.dng")):
    cal, tags = read_calibration(p)
    neutral = dng_tag(tags, "AsShotNeutral")[:3]
    with rawpy.imread(str(p)) as raw:
        wb = np.array(raw.camera_whitebalance[:3], np.float64)
    wb = wb / wb[1]
    print("  %-24s 1/AsShotNeutral %s   camera_whitebalance %s   ecart %.2e"
          % (p.stem, np.round(1.0 / neutral, 4), np.round(wb, 4),
             np.abs(1.0 / neutral - wb).max()))

print("\n=== sensibilite : McCamy contre une CCT decalee de +-100 K ===")
print("  (on force g avec une temperature biaisee, et on regarde le xy final)")
for p in sorted(DNG_DIR.glob("*.dng"))[:2]:
    cal, tags = read_calibration(p)
    neutral = dng_tag(tags, "AsShotNeutral")[:3]
    ref, _, _ = compute_white_xy(p)
    ligne = []
    for biais in (-100.0, 0.0, +100.0):
        import whitexy as W
        vrai = W.xy_to_temp
        W.xy_to_temp = lambda xy, b=biais, f=vrai: f(xy) + b
        xy = np.array([0.34567, 0.35850])
        for _ in range(60):
            xyz = np.linalg.solve(W.find_xyz_to_camera(xy, cal), neutral)
            xy = np.array([xyz[0], xyz[1]]) / xyz.sum()
        W.xy_to_temp = vrai
        ligne.append(np.abs(xy - ref).max())
    print("  %-24s ecart xy pour -100K : %.2e   +100K : %.2e"
          % (p.stem, ligne[0], ligne[2]))
