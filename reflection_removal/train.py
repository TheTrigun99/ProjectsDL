"""Entraînement SIRR : un run = un process = un GPU.

Extrait des cellules 10-19 de training2.ipynb. Le notebook garde ce à quoi il
est bon (show_dataset, plot_parts, show_results) et relit runs/<name>/hist.json ;
ici on ne fait qu'entraîner et écrire sur disque.

    CUDA_VISIBLE_DEVICES=3 python3 train.py --name d32_lr1e3 --dim 32 --lr 1e-3
    python3 train.py --gpu 3 --cfg sweep/lr3e4.json

Sorties dans runs/<name>/ :
    config.json  les hyperparamètres résolus, pour savoir ce qui a produit quoi
    hist.json    loss / termes / PSNR, réécrit à chaque éval (lisible pendant le run)
    ckpt.pt      poids du modèle non compilé, rechargeables dans un Unet nu
                 (sauf avec --no-ckpt)
    log.txt      la même chose que stdout
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from utils.ssim_ref import ssim_fast_t
# Sans --gpu ni CUDA_VISIBLE_DEVICES, torch voit les 8 cartes et prend la 0,
# qui est presque toujours occupée par quelqu'un d'autre. On force donc un GPU
# par défaut, à changer ici quand il n'est plus libre (nvidia-smi).
DEFAULT_GPU = "5"

# --gpu doit être posé avant l'import de torch : CUDA_VISIBLE_DEVICES est lu à
# l'initialisation du contexte, le changer après n'a aucun effet.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", default=None)
_gpu = _pre.parse_known_args()[0].gpu
# Un CUDA_VISIBLE_DEVICES déjà posé dans l'environnement gagne sur le défaut :
# sweep.sh et les lancements manuels continuent de marcher tels quels.
if _gpu is None and "CUDA_VISIBLE_DEVICES" not in os.environ:
    _gpu = DEFAULT_GPU
if _gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu)

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

from acr_tone import finish                                     # noqa: E402
from dataset_sim import SIM, build_eval_loaders, build_train_loader  # noqa: E402
from loss_ref import ReflectionLoss, psnr                       # noqa: E402
from models.unet_simple import Unet                             # noqa: E402
from dataset_sim import SIM_big, SIM_biased, SIM_biased2, SIM_NM, SIM_consistent
DEVICE = "cuda" 


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    bool_flag = argparse.BooleanOptionalAction

    # Le défaut reflète le GPU réellement visible (posé plus haut), pour que
    # config.json dise sur quelle carte le run a tourné.
    p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES"),
                   help=f"index physique du GPU (défaut {DEFAULT_GPU} ; "
                        "pose CUDA_VISIBLE_DEVICES)")
    p.add_argument("--cfg", type=Path,
                   help="JSON d'hyperparamètres ; les options CLI le surchargent")
    p.add_argument("--name", help="nom du run (défaut : horodatage)")
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--force", action="store_true",
                   help="écraser un runs/<name>/ existant")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", action=bool_flag, default=True,
                   help="écrire ckpt.pt à chaque éval (--no-ckpt : hist.json seul)")

    g = p.add_argument_group("modèle")
    g.add_argument("--dim", type=int, default=32)
    g.add_argument("--dim-mults", default="1,2,4")
    g.add_argument("--channel", type=int, default=3)
    g.add_argument("--compile", action=bool_flag, default=True)

    g = p.add_argument_group("optimisation")
    g.add_argument("--lr", type=float, default=1e-3, help="max_lr de OneCycle")
    g.add_argument("--iters", type=int, default=5000)
    g.add_argument("--eval-every", type=int, default=500)
    g.add_argument("--eval-batches", type=int, default=100)

    g = p.add_argument_group("données")
    g.add_argument("--batch-size", type=int, default=16)
    # 7 runs x 8 workers = 56 process de décodage JPEG sur 64 coeurs, et l'éval
    # en rajoute 3 par run. Au-delà les GPUs attendent le CPU.
    g.add_argument("--workers", type=int, default=6)
    g.add_argument("--eval-workers", type=int, default=3)
    g.add_argument("--size", type=int, default=384)
    g.add_argument("--weights", default="0.96,0.02,0.02",
                   help="probabilités d'échantillonnage (simulé, nature, real89)")
    g.add_argument("--simulated", action=bool_flag, default=True)
    g.add_argument("--tone", action=bool_flag, default=True)
    g.add_argument("--npz-dir", type=Path, default=SIM)
    g.add_argument("--t-key", default="t")
    g.add_argument("--ev-jitter", default=(-1.9, 0.9))

    g = p.add_argument_group("loss")
    g.add_argument("--pixel", type=float, default=1.0)
    g.add_argument("--perceptual", type=float, default=0.1)
    g.add_argument("--gradient", type=float, default=0.1)

    args = p.parse_args(argv)

    # Le fichier de config ne surcharge que ce que la ligne de commande n'a pas
    # fixé explicitement : --cfg sweep/x.json --lr 3e-4 fait ce qu'on attend.
    if args.cfg:
        given = {a.lstrip("-").replace("-", "_")
                 for a in (argv if argv is not None else sys.argv[1:])
                 if a.startswith("--")}
        for k, v in json.loads(args.cfg.read_text()).items():
            k = k.replace("-", "_")
            if not hasattr(args, k):
                p.error(f"{args.cfg}: option inconnue '{k}'")
            if k not in given:
                setattr(args, k, type(getattr(args, k))(v)
                        if isinstance(getattr(args, k), Path) else v)

    args.dim_mults = tuple(int(x) for x in str(args.dim_mults).split(","))
    args.weights = tuple(float(x) for x in str(args.weights).split(","))
    args.name = args.name or time.strftime("run_%Y%m%d_%H%M%S")
    return args

# ---------------------------------------------------------------------------
# Un pas, une éval — identiques aux cellules 13
# ---------------------------------------------------------------------------

def run_step(batch, net, crit, opt=None, sc=None):
    """Un pas. opt=None -> pas de backward (évaluation)."""
    m = batch["mixed"].to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
    t = batch["transmission"].to(DEVICE, non_blocking=True)
    with torch.amp.autocast(DEVICE, enabled=(DEVICE == "cuda")):
        pred = net(m)
        total, parts = crit(pred, t)
    if opt is not None:
        opt.zero_grad(set_to_none=True)
        sc.scale(total).backward()
        sc.step(opt)
        sc.update()
    return total, parts, pred.float(), m, t


@torch.no_grad()
def evaluate(net, crit, loaders, max_batches=100):
    """PSNR mixture/transmission, pred/transmission, SSIM"""
    net.eval()
    out = {}
    for name, dl in loaders.items():
        got, base, ssim = [], [], []
        for i, b in enumerate(dl):
            if i >= max_batches:
                break
            _, _, pred, m, t = run_step(b, net, crit)
            # ssim_fast_t rend un SSIM par image, (B,) : .mean() avant .item().
            ssim.append(ssim_fast_t(pred, t, data_range=1).mean().item())
            got.append(psnr(pred, t).item())
            base.append(psnr(m, t).item())
        out[name] = [float(np.mean(got)), float(np.mean(base)), float(np.mean(ssim))]
    net.train()
    return out

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def build(args, log):
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model = Unet(dim=args.dim, dim_mults=args.dim_mults, channel=args.channel)

    torch.nn.init.zeros_(model.final_conv.weight)
    torch.nn.init.zeros_(model.final_conv.bias)
    n_params = sum(p.numel() for p in model.parameters())
    model = model.to(DEVICE).to(memory_format=torch.channels_last)
    if args.compile:
        model = torch.compile(model)
    try:
        crit = ReflectionLoss(pixel=args.pixel, perceptual=args.perceptual,
                              gradient=args.gradient).to(DEVICE)
    except ImportError as e:
        log(f"{e}\n-> terme perceptuel désactivé")
        crit = ReflectionLoss(pixel=args.pixel, perceptual=0.0,
                              gradient=args.gradient).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.iters + 10)
    scaler = torch.amp.GradScaler(DEVICE)

    data = build_train_loader(finish, batch_size=args.batch_size,
                              workers=args.workers, size=args.size,
                              weights=args.weights, simulated=args.simulated,
                              tone=args.tone, npz_dir=args.npz_dir,
                              t_key=args.t_key, ev_jitter=args.ev_jitter)
    eval_loaders = build_eval_loaders(size=args.size, workers=args.eval_workers)

    log(f"{n_params / 1e6:.1f} M paramètres  |  device {DEVICE}  "
        f"|  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    return model, crit, opt, sched, scaler, data, eval_loaders

# ---------------------------------------------------------------------------
# Boucle
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    run_dir = args.out / args.name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.force:
        sys.exit(f"{run_dir} existe déjà et n'est pas vide (--force pour écraser)")
    run_dir.mkdir(parents=True, exist_ok=True)

    logfile = (run_dir / "log.txt").open("a", buffering=1)

    def log(msg):
        print(msg, flush=True)
        logfile.write(msg + "\n")

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    log(f"=== {args.name} ===\n{json.dumps(cfg, indent=2)}")

    model, crit, opt, sched, scaler, data, eval_loaders = build(args, log)

    hist = {"iter": [], "loss": [], "parts": [], "eval": []}

    def flush():
        """Écrit à chaque éval : un run tué à mi-parcours reste exploitable, et
        on peut tracer les courbes pendant que les 7 tournent."""
        (run_dir / "hist.json").write_text(json.dumps(hist))
        if not args.ckpt:
            return
        # state_dict du module d'origine : les clés de la version compilée sont
        # préfixées _orig_mod. et ne se rechargent pas dans un Unet nu.
        net = getattr(model, "_orig_mod", model)
        torch.save({"model": net.state_dict(), "config": cfg, "iter": len(hist["iter"])},
                   run_dir / "ckpt.pt")

    # On compte en itérations, pas en epochs : le WeightedRandomSampler tire
    # avec remise, donc une « epoch » ne parcourt pas le dataset une fois.
    t0, it = time.time(), 0
    model.train()
    while it < args.iters:
        for batch in data:
            total, parts, pred, m, t = run_step(batch, model, crit, opt, scaler)
            sched.step()
            hist["iter"].append(it)
            hist["loss"].append(total.item())
            hist["parts"].append({k: float(v) for k, v in parts.items()})

            if it % args.eval_every == 0:
                res = evaluate(model, crit, eval_loaders, max_batches=args.eval_batches)
                hist["eval"].append((it, res))
                flush()
                line = "  ".join(f"{n} {b:5.2f}->{g:5.2f} ssim {s:.3f}"
                                for n, (g, b, s) in res.items())
                log(f"it {it:6d}  loss {total.item():.4f}  "
                    f"lr {sched.get_last_lr()[0]:.2e}  {(time.time() - t0) / 60:5.1f} min"
                    f"  |  PSNR mixture->modèle, SSIM : {line}")

            it += 1
            if it >= args.iters:
                break

    res = evaluate(model, crit, eval_loaders, max_batches=args.eval_batches)
    hist["eval"].append((it, res))
    flush()
    line = "  ".join(f"{n} {b:5.2f}->{g:5.2f} ssim {s:.3f}"
                    for n, (g, b, s) in res.items())
    log(f"it {it:6d}  FIN  {(time.time() - t0) / 60:.1f} min  |  {line}")
    logfile.close()


if __name__ == "__main__":
    main()
