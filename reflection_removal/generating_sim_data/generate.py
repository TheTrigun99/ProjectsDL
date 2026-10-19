"""Runnable stage 2: produces the examples and writes them to disk.

A "group" is one transmission and the K reflections that share it. The members
of a group have the SAME true_t, bit for bit (same crop, same locked exposure
gain), which makes it possible to train on pairs that differ only by their
reflection.

Per example written: <out_dir>/ex_<group>_<member>.npz
    m, true_t            float16 (P, P, 3)  linear sRGB
    p_f_px, p_theta0, p_defocus_px, p_double_pane, p_exposure
    R_map, ghost_px      float16            pane maps (--no-maps to leave them
                                            out: ~0.5 MB per example)
    group, member, t_name, r_name

The float16 layers are stored in the "separated bytes" format of
../npz_octets.py (each one as k__lo / k__hi, lossless): 25% smaller for the
same CPU cost as deflating the float16 directly (measured over 8 examples:
size 0.75x, CPU 0.95x median), same format as the datasets converted on 06/10.
Read them with `ouvre_npz(p)`, NOT with np.load.

And alongside them the logs: trials.parquet (every trial, with its verdict),
trials_groups.parquet (the examples kept), scene_params.parquet, usage.npz,
meta.json.

Parallelism uses THREADS, not processes: the work sits in numpy and OpenCV,
which release the GIL. Beyond ~16 threads the GIL becomes the bottleneck again
(48 threads only deliver ~10 useful cores).
"""
from __future__ import annotations

import json
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm.auto import tqdm

from . import _paths  # noqa: F401  (inserts the repo root into sys.path)
from npz_octets import ecrit_octets  # noqa: E402

from .config import (BANDE_FOV, DEFOCUS_MAX_PX, K_MAP, N_FERME, N_OUVERT, P_OUVERT,  # noqa: E402
                     P_PROCHE, S_PROCHE, SENSOR_W_c, GenConfig)
from .culling import CULLERS  # noqa: E402
from .pool import Pool, make_t_crop, make_r_crop  # noqa: E402
from .simulate import WB_KEYS, simulate_example, simulate_weak  # noqa: E402

_MAP_KEYS = ("R_map", "ghost_px")


class Generator:
    """The generation loop. One object per run, driven by run()."""

    def __init__(self, cfg: GenConfig, pool: Pool):
        self.cfg = cfg
        self.pool = pool
        self.cull = CULLERS[cfg.cull]
        self.lock = threading.Lock()
        self.state = {"groups": 0, "saved": 0, "tried": 0, "stop": False}
        self.trials: list[dict] = []        # every trial, accepted or not
        self.rows: list[dict] = []          # the examples kept
        self.params: list[dict] = []        # pane parameters of those kept
        self.saved_by: Counter = Counter()
        self.use_t = np.zeros(len(pool))
        self.use_r = np.zeros(len(pool))
        self.pairs: list[tuple[int, int]] = []
        self.pbar = None

    # ---------------------------------------------------------------- trials
    def _trial(self, d):
        with self.lock:
            self.trials.append(d)

    def _draw_std(self, combo, rng, rnd):
        """A normal example: pair drawn from the combo, reflection taken as is."""
        pair = self.pool.sample_pair_in(combo, rnd)
        if pair is None:
            return None, None, -1, -1, None
        ia, ib = pair
        t_pre = make_t_crop(self.pool[ia], rng, self.cfg.patch)
        rc = make_r_crop(self.pool[ib], rng, self.cfg.patch)
        if rc is None:                       # this source cannot give a FOV inside the band
            d = {"verdict": "r_hors_bande", "mode": "std"}
            self._trial(d)
            return None, d, ia, ib, t_pre
        r_crop, frac_r, fov = rc        
        if rng.random() < 0.5:
            r_crop = r_crop[:, ::-1].copy()
        ex = simulate_example(self.pool[ia], self.pool[ib], rng,
                              patch=self.cfg.patch, t_pre=t_pre,
                              r_pre=(r_crop, frac_r, fov),
                              bias_dict=self.cfg.bias_dict,
                              polar=self.cfg.fresnel == "polar", wb=self.cfg.wb)
        wbp = {k: ex["params"][k] for k in WB_KEYS}
        if self.cfg.wb == "mix" and wbp["wb_dmired"] > self.cfg.wb_max_mired:
            d = {"verdict": "balance_blancs", "mode": "std", **wbp}
            self._trial(d)
            return None, d, ia, ib, t_pre
        d = self.cull(ex)
        d["mode"] = "std"
        d.update(wbp)
        self._trial(d)
        return (ex if d["verdict"] == "ok" else None), d, ia, ib, t_pre

    def _draw_weak(self, combo, rng, rnd):
        ex, d, ia, ib, t_pre = simulate_weak(
            combo, rng, rnd, self.pool, weak=self.cfg.weak,
            patch=self.cfg.patch, bias_dict=self.cfg.bias_dict, cull=self.cull,
            polar=self.cfg.fresnel == "polar", wb=self.cfg.wb,
            wb_max_mired=self.cfg.wb_max_mired if self.cfg.wb == "mix" else None)
        self._trial(d)
        return ex, d, ia, ib, t_pre

    # --------------------------------------------------------------- workers
    def _worker(self, w):
        cfg = self.cfg
        rng = np.random.default_rng([cfg.seed, 10_000 + w])
        rnd = random.Random(cfg.seed * 1_000_003 + 10_000 + w)

        while True:
            with self.lock:
                if (self.state["stop"]
                        or self.state["groups"] >= cfg.n_groups
                        or self.state["tried"] >= cfg.max_group_tries):
                    return
                combo = self.pool.pick_pair_type(rnd)
                self.state["tried"] += 1

            weak = rnd.random() < cfg.p_weak
            draw = self._draw_weak if weak else self._draw_std
            anchor, d, ia, ib, t_pre = draw(combo, rng, rnd)
            if anchor is None:
                continue

            # ---- the next members: same t, same exposure, different reflection
            members = [(anchor, d, ib)]
            e_lock = anchor["params"]["exposure"]
            wb_lock = anchor.get("wb_W")           # the anchor's white, as an XYZ array
            for _ in range(cfg.k - 1):
                for _try in range(cfg.max_r_tries):
                    if self.state["stop"]:
                        return
                    jb = self.pool.sample_reflection_in(ia, combo[1], rnd)
                    if jb is None:
                        break
                    ex = simulate_example(self.pool[ia], self.pool[jb], rng,
                                          patch=cfg.patch, t_pre=t_pre,
                                          r_pre=None, e_lock=e_lock,
                                          bias_dict=cfg.bias_dict,
                                          polar=cfg.fresnel == "polar",
                                          wb=cfg.wb, wb_lock=wb_lock)
                    if ex is None:           # this source cannot give a FOV inside the band
                        self._trial({"verdict": "r_hors_bande", "mode": "std"})
                        continue
                    dk = self.cull(ex)
                    dk["mode"] = "std"
                    dk.update({k: ex["params"][k] for k in WB_KEYS})
                    self._trial(dk)
                    if dk["verdict"] == "ok":
                        members.append((ex, dk, jb))
                        break            # plain reject: no bias on p(j|i)
            if len(members) < cfg.min_kept:
                continue
            if cfg.k > 1:
                ref = members[0][0]["true_t"]
                assert all(np.array_equal(ref, ex["true_t"])
                           for ex, _, _ in members), "true_t non partage"

            # ---- reserving the group identifier
            with self.lock:
                if self.state["groups"] >= cfg.n_groups:
                    return
                g = cfg.group_offset + self.state["groups"]
                self.state["groups"] += 1
                self.state["saved"] += len(members)
                self.use_t[ia] += len(members)
                self.saved_by[combo] += len(members)
                for ex, dk, jb in members:
                    self.use_r[jb] += 1
                    self.pairs.append((ia, jb))
                    self.rows.append({**dk, "group": g, "K": len(members),
                                      "t_name": self.pool[ia]["name"],
                                      "r_name": self.pool[jb]["name"],
                                      "t_loc": self.pool[ia]["loc"],
                                      "r_loc": self.pool[jb]["loc"],
                                      "t_kind": combo[0], "r_kind": combo[1]})
                    p = ex["params"]
                    self.params.append(
                        {k: v for k, v in p.items() if k not in _MAP_KEYS}
                        | {"ghost_px": float(p["ghost_px"].mean()), "group": g})
                self.pbar.update(1)

            for k, (ex, dk, jb) in enumerate(members):
                self._save(g, k, ia, jb, ex)

    # ------------------------------------------------------------------ disk
    def _save(self, g, k, ia, jb, ex):
        p = ex.pop("params")
        layers = {key: ex[key].astype(np.float16) for key in ("m", "true_t")}
        out = {f"p_{key}": np.asarray(v) for key, v in p.items()
               if key not in _MAP_KEYS}
        if self.cfg.save_params:
            for key in _MAP_KEYS:
                out[key] = p[key].astype(np.float16)
        with open(self.cfg.out_dir / f"ex_{g:05d}_{k:02d}.npz", "wb") as fh:
            ecrit_octets({**layers, **out, "group": g, "member": k,
                          "t_name": self.pool[ia]["name"],
                          "r_name": self.pool[jb]["name"]}, fh)

    def _suffix(self):
        """A continuation's logs must not overwrite those of the first run."""
        return f"_{self.cfg.group_offset:05d}" if self.cfg.group_offset else ""

    def _write_logs(self, elapsed):
        import pandas as pd
        cfg, s = self.cfg, self._suffix()
        out = cfg.out_dir
        if self.trials:
            pd.DataFrame(self.trials).to_parquet(out / f"trials{s}.parquet")
        if self.rows:
            pd.DataFrame(self.rows).to_parquet(out / f"trials_groups{s}.parquet")
        if self.params:
            pd.DataFrame(self.params).to_parquet(out / f"scene_params{s}.parquet")
        np.savez(out / f"usage{s}.npz", use_t=self.use_t, use_r=self.use_r,
                 pairs=np.asarray(self.pairs, np.int32),
                 names=np.asarray([e["name"] for e in self.pool.entries]))
        meta = {**cfg.to_json(), "elapsed_s": round(elapsed, 1),
                "camera_v3": dict(bande_fov=BANDE_FOV, n_ferme=N_FERME, n_ouvert=N_OUVERT,
                                  p_ouvert=P_OUVERT, p_proche=P_PROCHE, s_proche=S_PROCHE,
                                  k_map=K_MAP, sensor_w=SENSOR_W_c),
                "pool_counts": self.pool.counts(),
                "pool_size": len(self.pool),
                "saved_by": {f"{a}->{b}": n for (a, b), n in self.saved_by.items()},
                **{k: v for k, v in self.state.items() if k != "stop"}}
        (out / f"meta{s}.json").write_text(json.dumps(meta, indent=2))
        return meta

    # ------------------------------------------------------------------- run
    def run(self):
        cfg = self.cfg
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        cv2.setNumThreads(1)                  # parallelism lives one level up
        self.pbar = tqdm(total=cfg.n_groups, desc="groupes")
        t0 = time.time()

        # no `with`: an interruption would block inside shutdown(wait=True)
        # while the workers run. We raise the flag and they leave at the end of
        # the group in progress.
        tp = ThreadPoolExecutor(cfg.n_workers)
        futs = [tp.submit(self._worker, w) for w in range(cfg.n_workers)]
        try:
            for f in futs:
                f.result()
        except KeyboardInterrupt:
            self.state["stop"] = True
            print("\ninterruption : les workers terminent leur groupe en cours...")
        finally:
            tp.shutdown(wait=True)
            self.pbar.close()

        elapsed = time.time() - t0
        meta = self._write_logs(elapsed)
        st = self.state
        print(f"{st['groups']} groupes, {st['saved']} exemples "
              f"({st['saved'] / max(st['groups'], 1):.2f} par groupe) "
              f"en {elapsed / 60:.1f} min -> {cfg.out_dir}")
        taux = st["saved"] / max(len(self.trials), 1)
        print(f"essais {st['tried']}, taux d'acceptation {taux:.1%}")
        if st["groups"] < cfg.n_groups and not st["stop"]:
            besoin = int(cfg.n_groups / max(taux, 1e-9) / max(cfg.n_groups, 1)) + 1
            print(f"ARRET SUR LE BUDGET D'ESSAIS ({cfg.max_group_tries}), pas sur "
                  f"--n-groups : seulement {st['groups']}/{cfg.n_groups} groupes.\n"
                  f"  Au taux observe il faut --max-group-tries-factor >= {besoin} "
                  f"(actuel {cfg.max_group_tries_factor}).")
        if self.trials:
            import pandas as pd
            vc = pd.DataFrame(self.trials)["verdict"].value_counts(normalize=True)
            print("verdicts :", ", ".join(f"{k} {v:.1%}" for k, v in vc.items()))
        return meta

def generate(cfg: GenConfig, pool: Pool | None = None):
    """Entry point: builds the pool if needed and starts the generation."""
    if pool is None:
        from .pool import build_pool
        pool = build_pool(cfg.pools, with_exr=cfg.with_exr)
        print(f"pool : {len(pool)} scenes {pool.counts()}")
    return Generator(cfg, pool).run()
