"""Loading the pools (XYZ cache -> memmap entries) and sampling pairs.

A pool entry is a dict:
    name, loc, cats, e (exposure), illum (XYZ), xyz (memmap), saturation (memmap)

POOL ORDER -- the radiance cache (`levels`) is indexed by POSITION in the list.
The canonical order is therefore frozen: fivek, raise, ibl. Inserting a source
in the middle of the list invalidates cache_tot/pool_levels.npy.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from . import config as C

# canonical order, do not reorder: see the warning in the module docstring
POOL_ORDER = ("fivek", "raise", "ibl")

_INDEX_NAMES = ("pool_index.json", "ibl_index.json", "exr_index.json")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def attach_entry(m, cache_dir, raise_outdoor=False):
    """Metadata + opening of the arrays as memmaps (nothing is read here)."""
    d = Path(cache_dir)
    sat = m.get("sat_file") or f"{m['name']}.sat.npy"
    return {**m,
            "raise_outdoor": bool(raise_outdoor),
            "illum": np.asarray(m["illum"], np.float64),
            "xyz": np.load(d / f"{m['name']}.xyz.npy", mmap_mode="r"),
            "saturation": np.load(d / sat, mmap_mode="r")}


def _index_path(cache_dir):
    for n in _INDEX_NAMES:
        p = Path(cache_dir) / n
        if p.exists():
            return p
    raise FileNotFoundError(f"aucun index json dans {cache_dir}")



def load_pool(cache_dir, raise_outdoor=False, require_files=True):
    """Reads a cache's json index and attaches the memmaps.

    require_files drops the entries whose .npy has disappeared: the caches have
    been regenerated several times and the index can run ahead of the disk.
    """
    cache_dir = Path(cache_dir)
    meta = json.loads(_index_path(cache_dir).read_text())
    if require_files:
        meta = [m for m in meta
                if (cache_dir / f"{m['name']}.xyz.npy").exists()]
        
    return [attach_entry(m, cache_dir, raise_outdoor) for m in meta]


def rescale_hdr_exposure(entries, src_med=None, target=None):
    """Re-exposes the HDR pools (IBL, EXR) towards a physical radiance.

    The cache was written with a common median `src_med`; we bring it back to
    the radiance expected for the location, so that `xyz / e` is in the same
    unit as the de-exposed photographs. Entries with no indoor/outdoor category
    are left as they are.
    """
    src_med = C.HDR_SRC_MED if src_med is None else src_med
    target = C.HDR_TARGET if target is None else target
    out = []
    for e in entries:
        cats = e.get("cats", {}).get("categories", [])
        loc = ("indoor" if "indoor" in cats
               else "outdoor" if "outdoor" in cats else None)
        out.append(e if loc is None else {**e, "e": src_med / target[loc]})
    return out


def load_hdr_pool(cache_dir, rescale=True):
    p = load_pool(cache_dir, raise_outdoor=False)
    return rescale_hdr_exposure(p) if rescale else p


# --------------------------------------------------------------------------
# Crops
# --------------------------------------------------------------------------
def random_crop(img, size, sat, rng):
    """Random square crop (40-100 % of the short side), then downscale to size.

    We never upsample. Returns (crop, saturated fraction of the crop): the
    fraction is measured on the mask BEFORE downscaling, as the mean survives
    the resize well (checked in image_raw_exp/Filter_saturation_mask_*.png).
    """
    h, w = img.shape[:2]
    C_ = int(min(h, w) * rng.uniform(0.4, 1.0))
    C_ = max(C_, size)                                   # never upscale
    y = int(rng.integers(0, h - C_ + 1))
    x = int(rng.integers(0, w - C_ + 1))
    crop = np.asarray(img[y:y + C_, x:x + C_], np.float32)
    out = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return out, float(sat[y:y + C_, x:x + C_].mean())


def make_t_crop(src, rng, patch=None):
    """The crop of a pool entry. This is what a group shares."""
    patch = patch or C.PATCH
    return random_crop(src["xyz"], patch, src["saturation"], rng)

def make_r_crop(src, rng, size, bande=None):
    """Crop of the REFLECTION whose content has a FOV inside `bande` (degrees, side of the square).

    The FOV of the simulated camera is the FOV of this content (scene.sample_scene3). Works on the focal length
    in pixels of the cache (src["f_src_px"], measured on the WHOLE image), so a half of a photo (mid.moitie)
    keeps the right FOV. Never upsamples. Returns (crop, saturated fraction, fov), or None when the source cannot
    give a FOV inside the band (telephoto source narrower than the band, or image smaller than size).
    """
    img, sat = src["xyz"], src["saturation"]
    h, w = img.shape[:2]
    bande = bande or C.BANDE_FOV
    img, sat, f = src["xyz"], src["saturation"], src["f_src_px"]
    lo = max(2 * f * np.tan(np.radians(bande[0]) / 2), size)       # side of the crop (cache px): FOV >= min
    hi = min(2 * f * np.tan(np.radians(bande[1]) / 2), min(h, w))  # FOV <= max, not larger than the image
    if lo > hi:
        return None
    C_ = int(rng.uniform(lo, hi))
    y = int(rng.integers(0, h - C_ + 1))
    x = int(rng.integers(0, w - C_ + 1))
    crop = np.asarray(img[y:y + C_, x:x + C_], np.float32)         # float16 memmap: cv2 refuses float16
    out = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    fov = float(np.degrees(2 * np.arctan(C_ / (2 * f))))           # REAL FOV of the content (after rounding)
    return out, float(sat[y:y + C_, x:x + C_].mean()), fov

# --------------------------------------------------------------------------
# Mean radiance per scene
# --------------------------------------------------------------------------
def _read_levels(path, n):
    """Reads a levels cache, or None if it is absent, unreadable or stale."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        L = np.load(path)
    except Exception as exc:        # an aborted try left a malformed .npy
        print(f"cache {path.name} illisible ({exc.__class__.__name__}), ignore")
        return None
    if L.ndim != 1 or not np.issubdtype(L.dtype, np.floating):
        print(f"cache {path.name} de forme inattendue {L.shape} {L.dtype}, ignore")
        return None
    if len(L) != n:
        print(f"cache {path.name} perime ({len(L)} != {n}), recalcul")
        return None
    return L

def levels(entries, cache=None):
    """Mean radiance of each scene, in the unit of the simulation.

    The result is indexed by POSITION in `entries`, so the cache file is named
    after the fingerprint of the names: changing the composition of the pool
    picks a different file instead of overwriting the previous one. The old
    pool_levels.npy (without a fingerprint) is still accepted if its length
    matches, so as not to recompute the 10012 levels already measured.

    NOTE -- as things stand that fallback is ALWAYS the one taken, so the
    fingerprinting of CHANGES.md item 5 currently buys nothing. The canonical
    pool (fivek + raise + ibl, 10012 entries) fingerprints to bb3987c8cc, and
    cache_tot/pool_levels_bb3987c8cc.npy is precisely the file left corrupted
    by an aborted try (it holds array([Ellipsis], dtype=object)). Every run
    therefore prints "cache ... illisible", falls back to pool_levels.npy on a
    length check alone, and -- because the fallback succeeds -- never reaches
    the np.save that would repair the fingerprinted file.

    Two consequences: the warning is permanent, and cache_tot also holds
    pool_levels_ordre_raise_ibl_normal.npy, which has the same 10012 length in
    a DIFFERENT order; it is kept out only by its name. Deleting the corrupt
    pool_levels_bb3987c8cc.npy makes the fingerprint real again.
    """
    n = len(entries)
    digest = hashlib.md5("|".join(e["name"] for e in entries).encode()).hexdigest()
    if cache is not None:
        candidates = [Path(cache)]
    else:
        candidates = [C.LEVELS_DIR / f"pool_levels_{digest[:10]}.npy",
                      C.LEVELS_DIR / "pool_levels.npy"]
    for p in candidates:
        L = _read_levels(p, n)
        if L is not None:
            return L

    out = candidates[0]
    print(f"calcul des radiances moyennes de {n} scenes -> {out.name}")
    L = np.array([float(np.nanmean(np.asarray(e["xyz"][::16, ::16], np.float32)))
                  / max(float(e["e"]), 1e-12) for e in entries])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, L)
    return L


# --------------------------------------------------------------------------
# Sampling indices
# --------------------------------------------------------------------------
def kind(m):
    """Source type: ibl, raise, indoor, outdoor."""
    if m["loc"] == "ibl":
        return "ibl"
    if m.get("raise_outdoor"):
        return "raise"
    return m["loc"]


class Pool:
    """The entries plus everything needed to sample pairs.

    Gathers what the notebook scattered across globals (KIND, KIND_IDX, LOG_L,
    _ORDER, _SORTED, _ORD_K, _SRT_K) and which had to be evaluated in the right
    cell order for pick_partner_weak to work.
    """

    def __init__(self, entries, pair_probs=None, levels_cache=None):
        self.entries = entries
        self.kinds = np.array([kind(m) for m in entries])
        self.kind_idx = {k: np.flatnonzero(self.kinds == k)
                         for k in np.unique(self.kinds)}

        self.log_l = np.log10(np.maximum(levels(entries, levels_cache), 1e-12))
        # increasing luminance order, per type: weak partner in O(log n)
        self._ord = {k: idx[np.argsort(self.log_l[idx])]
                     for k, idx in self.kind_idx.items()}
        self._srt = {k: self.log_l[o] for k, o in self._ord.items()}

        probs = dict(C.PAIR_PROBS if pair_probs is None else pair_probs)
        missing = [p for p in probs if p[0] not in self.kind_idx or p[1] not in self.kind_idx]
        if missing:
            raise ValueError(f"types absents du pool pour les paires {missing} "
                             f"(disponibles : {sorted(self.kind_idx)})")
        self._pairs = list(probs)
        cum = np.cumsum([probs[p] for p in self._pairs], dtype=float)
        self._cum = cum / cum[-1]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        return self.entries[i]

    def counts(self):
        return {k: len(v) for k, v in self.kind_idx.items()}

    # ---- standard sampling -----------------------------------------------
    def pick_pair_type(self, rnd):
        return self._pairs[int(np.searchsorted(self._cum, rnd.random()))]

    def sample_in(self, k, rnd):
        idx = self.kind_idx[k]
        return int(idx[rnd.randrange(len(idx))])

    def sample_reflection_in(self, i, r_kind, rnd, tries=8):
        """A reflection of the requested type, different from the transmission."""
        dst = self.kind_idx[r_kind]
        for _ in range(tries):
            j = int(dst[rnd.randrange(len(dst))])
            if j != i:
                return j
        return None

    def sample_pair_in(self, combo, rnd):
        t_kind, r_kind = combo
        i = self.sample_in(t_kind, rnd)
        j = self.sample_reflection_in(i, r_kind, rnd)
        return None if j is None else (i, j)

    # ---- weak-mode sampling ----------------------------------------------
    def pick_partner_weak(self, ia, rnd, r_kind, logl=(-0.45, 0.45), tries=4):
        """A partner whose luminance already puts the mixture in the band.

        We search the window [L_t + logl[0], L_t + logl[1]] of the requested
        type: it is the luminance ratio, not a gain, that makes the reflection
        weak.
        """
        o, srt = self._ord[r_kind], self._srt[r_kind]
        a = np.searchsorted(srt, self.log_l[ia] + logl[0])
        b = np.searchsorted(srt, self.log_l[ia] + logl[1])
        if b <= a:
            return None
        for _ in range(tries):
            ib = int(o[rnd.randrange(a, b)])
            if ib != ia:
                return ib
        return None

# fov = {k: e["fov"]["short_deg"] for k, e in json("fivek_camera.json").items() if e.get("fov")}

_FOV = None

def fov_table():
    """fov of the shirt side of each photos, degrees
    Read once, at the start build_pool
    """
    global _FOV
    if _FOV is None:
        with open(C.FOV_JSON) as fh:
            _FOV = {k: e["fov"]["short_deg"] for k, e in json.load(fh).items() if e.get("fov")}
        return _FOV

def attach_focal(entries):
    """adds fov_src (degrees, shirt_side) and f_src_px (focal length in pixel of the cache, on the WHOLE image) 
    """
    fov = fov_table()
    for e in entries:
        fov_src = C.FOV_IBL if kind(e)=="ibl" else fov[e["name"]]
        e["fov_src"] = float(fov_src)
        e["f_src_px"] =  float((min(e["xyz"].shape[:2]) / 2) / np.tan(np.radians(fov_src) / 2))
    return entries


def build_pool(pools=POOL_ORDER, with_exr=False, pair_probs=None,
               levels_cache=None):
    """Assembles the pool in canonical order and returns a ready-to-sample Pool."""
    want = set(pools)
    entries = []
    for src in POOL_ORDER:                     # frozen order, not `pools`' order
        if src not in want:
            continue

        if src == "fivek":
            entries += load_pool(C.POOL_CACHE, raise_outdoor=False)
        elif src == "raise":
            entries += load_pool(C.RAISE_CACHE, raise_outdoor=True)
        elif src == "ibl":
            entries += load_hdr_pool(C.IBL_CACHE)
            if with_exr:
                entries += load_hdr_pool(C.EXR_CACHE)


    return Pool(attach_focal(entries), pair_probs=pair_probs, levels_cache=levels_cache)
