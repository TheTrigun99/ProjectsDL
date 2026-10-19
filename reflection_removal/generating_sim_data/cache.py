"""Runnable stage 1: DNG -> float16 XYZ cache + saturation mask.

Writes into cache_dir, for every image kept:
    <name>.xyz.npy   float16 (H, W, 3), as-shot XYZ, de-exposed when read
    <name>.sat.npy   bool    (H, W),    sensor saturation mask
    pool_index.json  the metadata (loc, cats, e, illum, shape)

Decoding runs in separate processes: rawpy/LibRaw does not release the GIL and
saturates one core per image.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from .config import CacheConfig
from .dng_metadata import has_required_metadata, read_exposure
from .raw_processing import read_raw_to_xyz


def _init_worker():
    cv2.setNumThreads(1)


def _save_atomic(path, arr):
    """tmp + rename: an interrupted run leaves no truncated .npy behind."""
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.rename(path)


def _convert_one(item, cache_dir, half_size=True, overwrite=False):
    """Child process: writes to disk, returns metadata only."""
    path, loc, cats = item
    path = Path(path)
    try:
        xyz_p = cache_dir / f"{path.stem}.xyz.npy"
        sat_p = cache_dir / f"{path.stem}.sat.npy"
        if not overwrite and xyz_p.exists() and sat_p.exists():
            e, ok_exp = read_exposure(path)
            if not ok_exp:
                return "skip", path.stem, "missing EXIF exposure"
            shape = list(np.load(xyz_p, mmap_mode="r").shape)
            sat = np.load(sat_p, mmap_mode="r")
            return "ok", {"name": path.stem, "loc": loc, "cats": cats, "e": e,
                          "illum": _illum_from_cache(path), "shape": shape,
                          "sat_sum": int(np.asarray(sat).sum())}

        xyz, illum, sat, ok_xyz = read_raw_to_xyz(path, half_size=half_size)
        if not ok_xyz:
            return "skip", path.stem, "sRAW / linear DNG"
        e, ok_exp = read_exposure(path)
        if not ok_exp:
            return "skip", path.stem, "missing EXIF exposure"
        _save_atomic(xyz_p, xyz.astype(np.float16))
        _save_atomic(sat_p, np.ascontiguousarray(sat))
        return "ok", {"name": path.stem, "loc": loc, "cats": cats, "e": e,
                      "illum": illum.tolist(), "shape": list(xyz.shape),
                      "sat_sum": int(sat.sum())}
    except Exception as exc:                        # one bad image must not
        return "fail", path.stem, f"{type(exc).__name__}: {exc}"   # stop the run


def _illum_from_cache(path):
    """Illuminant re-read from the DNG: the .npy cache does not store it."""
    from .dng_metadata import camera_to_xyz
    from .colorimetry import xy_to_xyz
    _, xy = camera_to_xyz(path)
    return xy_to_xyz(xy).tolist()


def list_dng(cfg: CacheConfig):
    """(path, loc, cats) for every DNG to convert.

    With index_json (FiveK's training.json, raise_outdoor.json), loc and cats
    come from the json; otherwise we glob the directory and loc = cfg.loc.
    """
    if cfg.index_json is not None:
        meta = json.loads(Path(cfg.index_json).read_text())
        items = []
        for name, item in meta.items():
            cats = item.get("categories", {})
            p = item.get("urls", {}).get("dng") or item.get("path")
            p = Path(p) if p and Path(p).is_absolute() else cfg.dng_dir / f"{name}.dng"
            if p.exists() and p.stat().st_size > 0:
                items.append((p, cats.get("location", cfg.loc), cats))
    else:
        items = [(p, cfg.loc, {}) for p in sorted(cfg.dng_dir.glob("*.dng"))
                 if p.stat().st_size > 0]
    return items[:cfg.limit] if cfg.limit else items


def build_cache(cfg: CacheConfig, filter_metadata=True):
    """Converts every DNG and writes pool_index.json. Returns the metadata."""
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    index_p = cfg.cache_dir / "pool_index.json"
    if index_p.exists() and not cfg.rebuild:
        meta = json.loads(index_p.read_text())
        print(f"{len(meta)} images deja dans {cfg.cache_dir} (--rebuild pour refaire)")
        return meta

    files = list_dng(cfg)
    print(f"{len(files)} DNG trouves")
    if filter_metadata:
        files = [f for f in files if has_required_metadata(f[0])]
        print(f"{len(files)} avec point blanc et ouverture")

    work = partial(_convert_one, cache_dir=cfg.cache_dir, half_size=cfg.half_size,
                   overwrite=cfg.rebuild)
    meta, dropped = [], []
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(cfg.n_workers, mp_context=ctx,
                             initializer=_init_worker) as ex:
        for res in tqdm(ex.map(work, files, chunksize=1), total=len(files),
                        desc="RAW -> XYZ"):
            if res[0] == "ok":
                meta.append(res[1])
            else:
                dropped.append(res[1:])

    index_p.write_text(json.dumps(meta))
    print(f"{len(meta)} gardees, {len(dropped)} ecartees -> {index_p}")
    for name, why in dropped[:10]:
        print("  -", name, why)
    return meta
