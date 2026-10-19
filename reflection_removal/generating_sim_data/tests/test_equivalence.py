"""The package must compute exactly what data_clean.ipynb computed.

    python3 generating_sim_data/tests/extract_notebook.py
    python3 generating_sim_data/tests/test_equivalence.py [--n 120]

Compares bit for bit, and additionally checks that the RNG stream is consumed
identically: a stream divergence would shift every subsequent draw without
changing a single result taken in isolation.

The only expected difference: the notebook's "A_tdom " verdict has a trailing
space, which the package removed. Verdicts are therefore compared after
.strip().
"""
import argparse
import collections
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

ARW_FALLBACK = "/home/damien/ProjectsDL/rawtosrgb/rawboat.ARW"


def ecarts(a, b, tol=0.0):
    out = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            out.append(f"{k}: one side only")
            continue
        x, y = a[k], b[k]
        if isinstance(x, np.ndarray):
            if not np.array_equal(x, y):
                out.append(f"{k}: max gap {float(np.abs(x - y).max()):.3e}")
        elif isinstance(x, str):
            if x.strip() != y.strip():
                out.append(f"{k}: {x!r} vs {y!r}")
        elif isinstance(x, dict):
            continue
        elif isinstance(x, (tuple, list)):
            if tuple(x) != tuple(y):
                out.append(f"{k}: {x} vs {y}")
        elif isinstance(x, bool) or isinstance(y, bool):
            if bool(x) != bool(y):
                out.append(f"{k}: {x} vs {y}")
        elif abs(float(x) - float(y)) > tol:
            out.append(f"{k}: {x} vs {y}")
    return out


def dit(label, bad):
    print(("OK   " if not bad else "DIFF ") + label
          + ("" if not bad else " -> " + "; ".join(bad)))
    return not bad


def test_scene():
    import nb_ref
    from generating_sim_data import scene as S
    from generating_sim_data.config import BIAS_PRESETS
    assert nb_ref.BIAS == BIAS_PRESETS["defocus_low"], "the preset has changed"
    ok = True
    for size in (128, 384):
        for seed in (0, 1, 7, 1009):
            for biais in (False, True):
                r1, r2 = (np.random.default_rng(seed) for _ in range(2))
                a = nb_ref.sample_scene2(r1, size, biais=biais, biais_params=nb_ref.BIAS)
                b = S.sample_scene2(r2, size, biais=biais, biais_params=nb_ref.BIAS,
                                    polar=False)   # v1.1 behaviour (v2 adds polarization)
                ok &= dit(f"sample_scene2 size={size} seed={seed} biais={biais}",
                          ecarts(a, b))
                assert r1.random() == r2.random(), "RNG streams diverged"
    return ok


def test_sim(n=120):
    import nb_ref
    from generating_sim_data import culling as CU, pool as P, simulate as SIM
    from generating_sim_data.config import IBL_CACHE, POOL_CACHE, RAISE_CACHE

    pool = (P.load_pool(POOL_CACHE) + P.load_pool(RAISE_CACHE, raise_outdoor=True)
            + P.load_hdr_pool(IBL_CACHE))
    print(f"{len(pool)} scenes in the pool")
    ok, cnt = True, collections.Counter()
    rng0 = np.random.default_rng(7)
    for it in range(n):
        i, j = (int(rng0.integers(len(pool))) for _ in range(2))
        seed = int(rng0.integers(10 ** 6))
        r1, r2 = (np.random.default_rng(seed) for _ in range(2))
        a = nb_ref.simulate_example_enc(pool[i], pool[j], r1, bias_dict=nb_ref.BIAS)
        b = SIM.simulate_example(pool[i], pool[j], r2, bias_dict=nb_ref.BIAS,
                                 polar=False, wb="t")   # v1.1 behaviour
        bad = [k for k in ("m", "t", "r", "true_t") if not np.array_equal(a[k], b[k])]
        if bad:
            ok &= dit(f"layers it={it} i={i} j={j} seed={seed}", bad)
            continue
        ok &= not ecarts(a["params"], b["params"]) or dit(f"params it={it}",
                                                          ecarts(a["params"], b["params"]))
        assert r1.random() == r2.random(), "RNG streams diverged"
        for fa, fb, tag in ((nb_ref.measure, CU.measure, "measure"),
                            (nb_ref.measure_a_tdom, CU.measure_a_tdom, "a_tdom")):
            da, db = fa(a), fb(b)
            inter = ({k: v for k, v in da.items() if k in db},
                     {k: v for k, v in db.items() if k in da})
            bad = ecarts(*inter)
            if bad:
                ok &= dit(f"{tag} it={it}", bad)
            if tag == "a_tdom":
                cnt[da["verdict"].strip()] += 1
        bad = ecarts({"p": nb_ref.psnr_raw_to_jpeg(a["m"], a["true_t"])},
                     {"p": CU.psnr_raw_to_jpeg(b["m"], b["true_t"])})
        if bad:
            ok &= dit(f"psnr_raw_to_jpeg it={it}", bad)
        bad = ecarts(nb_ref.descripteurs(a["m"], a["t"], a["r"]),
                     CU.descripteurs(b["m"], b["t"], b["r"]))
        if bad:
            ok &= dit(f"descripteurs it={it}", bad)
    print(f"{n} examples compared; a_tdom verdicts: {dict(cnt)}")
    return ok


def test_raw(arw=ARW_FALLBACK):
    """On an arbitrary RAW: the DNG matrix is replaced by a constant, the same
    one on both sides, so that all the rest of the chain can be tested."""
    if not Path(arw).exists():
        print(f"(no RAW at {arw}, RAW test skipped)")
        return True
    import nb_raw
    import rawpy
    from generating_sim_data import dng_metadata as DM, raw_processing as RP

    M = np.array([[0.72, 0.15, 0.09], [0.26, 0.70, 0.04], [0.02, 0.09, 0.82]], np.float32)
    xy = np.array([0.3457, 0.3585])
    nb_raw.camera_to_xyz = RP.camera_to_xyz = lambda p: (M, xy)

    ok = dit("read_exposure", ecarts({"e": nb_raw.read_exposure(arw)[0]},
                                     {"e": DM.read_exposure(arw)[0]}))
    with rawpy.imread(arw) as raw:
        ok &= dit("clip_level", ecarts({"c": nb_raw.clip_level(raw)},
                                       {"c": RP.clip_level(raw)}))
    rng = np.random.default_rng(0)
    for h, w in ((400, 600), (401, 599)):
        for flip in (0, 3, 5, 6):
            m = rng.random((h, w)) > 0.97
            shape = (h // 2 + h % 2, w // 2 + w % 2)
            if flip in (5, 6):
                shape = shape[::-1]
            ok &= dit(f"align_mask {h}x{w} flip={flip}",
                      [] if np.array_equal(nb_raw.align_maskv2(m, shape, flip),
                                           RP.align_mask(m, shape, flip)) else ["differs"])
    for half in (True, False):
        A, B = nb_raw.read_raw_to_xyz(arw, half_size=half), RP.read_raw_to_xyz(arw, half_size=half)
        bad = [la for la, xa, xb in zip(("xyz", "illum", "saturated", "ok"), A, B)
               if not (np.array_equal(xa, xb) if isinstance(xa, np.ndarray) else xa == xb)]
        ok &= dit(f"read_raw_to_xyz half_size={half} {A[0].shape}", bad)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=120, help="simulation draws")
    ap.add_argument("--arw", default=ARW_FALLBACK)
    ap.add_argument("--skip", nargs="*", default=[], choices=["scene", "sim", "raw"])
    a = ap.parse_args()
    if not (HERE / "nb_ref.py").exists():
        sys.exit("run extract_notebook.py first")
    ok = True
    for name, fn in (("scene", test_scene), ("sim", lambda: test_sim(a.n)),
                     ("raw", lambda: test_raw(a.arw))):
        if name in a.skip:
            continue
        print(f"\n===== {name} =====")
        ok &= fn()
    print("\nRESULT:", "IDENTICAL" if ok else "DIVERGENCE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
