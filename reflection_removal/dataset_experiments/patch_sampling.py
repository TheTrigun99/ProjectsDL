"""
Remplacements pour data_clean.ipynb.  Version SSIM seul, sans bande dlogY.
Trois cellules a changer : 66 (tirage des paires), 63 (measure), 79 (boucle).
Mesures justificatives : 2026-09-11_vis_verification.md.

Pourquoi c'est aussi court : une fois le tri fait sur SSIM, la composition ne
deplace presque plus la distribution. Mesure sur les 105 465 essais, fenetre
[0.65, 0.97] + std>=0.05 + expo :

  composition (I->I/I->O/O->I/O->O)     L1     essais/ex   SSIM p10/p50/p90
    15/25/20/40                        0.669       5.5     0.690/0.848/0.943
    10/20/20/50                        0.666       5.4     0.692/0.850/0.943
    30/30/10/30 (actuel)               0.669       4.9     0.689/0.843/0.941
    13/44/44/0  (papier)               0.691       7.4     0.684/0.842/0.942
  cible test sets                                          0.780/0.859/0.937

Toutes equivalentes. La composition se choisit donc pour la DIVERSITE des
sources, pas pour l'appariement de distribution. La forme finale se regle au
chargement avec un WeightedRandomSampler.
"""
import numpy as np

# =============================================================================
# CELLULE 66 — remplacer integralement
# =============================================================================

# Choisie pour la diversite : le pool outdoor est 3.4x plus divers (3489 vs
# 1021 images). Reutilisation moyenne a 100k exemples -> indoor 73x, outdoor 36x
# (contre 98x / 29x avec 30/30/10/30, et 110x / 25x avec le tirage du papier).
TARGET = {("indoor",  "indoor"):  0.15,
          ("indoor",  "outdoor"): 0.25,
          ("outdoor", "indoor"):  0.20,
          ("outdoor", "outdoor"): 0.40}

_C    = list(TARGET)
QUOTA = {c: int(round(v * N_EXAMPLES)) for c, v in TARGET.items()}
saved_by = {c: 0 for c in _C}                       # protege par g_lock

IDX = {}
for k, m in enumerate(pool):
    IDX.setdefault(m["loc"], []).append(k)
IDX = {L: np.asarray(v) for L, v in IDX.items()}


def sample_reflection_in(i, lr, rnd):
    """j de location `lr`, uniforme, j != i.
    Remplace sample_reflection_for : plus de ponderation, plus de CUM_WEIGHTS."""
    dst = IDX[lr]
    for _ in range(8):
        j = int(dst[rnd.randrange(len(dst))])
        if j != i:
            return j
    return None


def sample_pair_in(combo, rnd):
    """(i, j) uniforme dans la combinaison. C'est tout."""
    lt, lr = combo
    src = IDX[lt]
    i = int(src[rnd.randrange(len(src))])
    j = sample_reflection_in(i, lr, rnd)
    return None if j is None else (i, j)


def pick_combo(rnd, deficit):
    """Combinaison tiree proportionnellement au quota restant.
    Signature INCHANGEE : la cellule 68 l'appelle deja comme ca."""
    s = deficit.sum()
    if s <= 0:
        return None
    return _C[int(np.searchsorted(np.cumsum(deficit / s), rnd.random()))]


def deficit_now():
    """A appeler SOUS le lock."""
    return np.array([max(QUOTA[c] - saved_by[c], 0) for c in _C], float)


# =============================================================================
# CELLULE 63 — remplacer `measure`
# =============================================================================
CULL_SIZE = 384

def measure(exa):
    """Les deux criteres du papier (Sec. D.1) + l'exposition. Rien d'autre.

    Fenetre [0.65, 0.97] et pas le [0.40, 0.94] d'Adobe, pour deux raisons :
      - PLANCHER : leur modele de base recoit une photo contextuelle c du decor
        reflete (Sec. E.1). A SSIM 0.4 la transmission est quasi detruite ; ils
        ont c pour desambiguiser, pas nous -- dataset_sim.py ne charge que m et t.
      - PLAFOND : 0.94 -> 0.97 fait passer la couverture des 520 images de test
        de 87.3 % a 97.7 % (WildScene a beaucoup de reflets tres faibles
        au-dessus de 0.94). Ces exemples quasi-identite apprennent au modele a
        ne pas sur-effacer.

    Resultat : acceptation 16.0 %, SSIM p10/p50/p90 = 0.692/0.845/0.952
    contre 0.780/0.859/0.937 pour les test sets."""
    rs = lambda x: cv2.resize(x, (CULL_SIZE, CULL_SIZE), interpolation=cv2.INTER_AREA)
    mc, tc, rc = rs(exa["m"]), rs(exa["t"]), rs(exa["r"])
    Ym, Yt, Yr = luminance(mc), luminance(tc), luminance(rc)
    ssim_m_t, std_m_t = ssim_fast(mc, tc)

    d = {"ssim": ssim_m_t, "ssim_std": std_m_t, "Ym": float(Ym.mean()),
         # diagnostics uniquement, plus des criteres
         "vis": float(Yr.mean() / (Ym.mean() + 1e-8)),
         "m_mean": float(mc.mean()), "t_std": float(Yt.std()),
         "r_std": float(Yr.std()),
         "sat_frac": float((mc.max(axis=-1) > 0.99).mean())}

    if   not 0.65 < d["ssim"] < 0.97: d["verdict"] = "ssim"
    elif d["ssim_std"] < 0.05:        d["verdict"] = "ssim_std"
    elif not 0.04 < d["Ym"] < 0.4:    d["verdict"] = "exposition"
    else:                             d["verdict"] = "ok"
    return d

# Supprimes comme criteres, gardes comme colonnes de diagnostic :
#   vis      -> fonction fermee de R.rho, 95 % determinee par le couple source,
#               absente du papier ; ne distingue pas voile diffus / reflet localise
#   m_mean   -> 0.00 % de rejets residuels sous cette fenetre
#   t_std    -> 0.45 %
#   r_std    -> 7.1 %. Le remettre garantit un reflet structure ; le laisser
#               dehors donne des exemples quasi-identite (utiles contre le
#               sur-effacement). Choix ouvert, mesure connue.


# =============================================================================
# CELLULE 68 (pipeline SANS groupes) — AUCUN CHANGEMENT
# Elle appelle deja pick_combo(rnd_w, deficit) et sample_pair_in(combo, rnd_w).
# Seule contrainte : re-executer la cellule 66 juste avant (QUOTA + saved_by),
# avec N_EXAMPLES egal au nombre d'exemples vises.
#
# =============================================================================
# CELLULE 79 (pipeline AVEC groupes) — cinq retouches dans _group_worker
# =============================================================================
"""
0) N_EXAMPLES doit valoir le nombre d'EXEMPLES vises (= K * N_GROUPS), et la
   cellule 66 doit etre re-executee juste avant chaque generation : c'est elle
   qui calcule QUOTA et remet saved_by a zero.

1) Le bloc `with g_lock:` (l.29-32). Remplacer :

        with g_lock:
            if g_state["groups"] >= N_GROUPS or g_state["tried"] >= MAX_GROUP_TRIES:
                return
            g_state["tried"] += 1

   par :

        with g_lock:
            if g_state["groups"] >= N_GROUPS or g_state["tried"] >= MAX_GROUP_TRIES:
                return
            combo = pick_combo(rnd_w, deficit_now())
            if combo is None:
                return
            g_state["tried"] += 1

2) L'ancre (l.35). Remplacer :

        ia, ib = sample_realistic_pair(rnd_w)

   par :

        pair = sample_pair_in(combo, rnd_w)
        if pair is None:
            continue
        ia, ib = pair

3) Les freres du groupe (l.51). Remplacer :

        jb = sample_reflection_for(ia, rnd_w)

   par :

        jb = sample_reflection_in(ia, combo[1], rnd_w)
        if jb is None:
            break

4) Apres `g_state["saved"] += len(members)` (l.66), ajouter :

        saved_by[combo] += len(members)

5) MAX_R_TRIES = 30 -> 60.  O->I n'accepte qu'a 9.9 % : a 30 essais, 4.4 % des
   groupes O->I perdent leur frere et sont jetes en entier ; a 60, 0.2 %.

A SUPPRIMER : cellule 57 (pair_score, ALL_PAIRS, PAIR_WEIGHTS, CUM_WEIGHTS,
sample_realistic_pair) et cellule 59 (sample_reflection_for, _BLK). ~2.6 Go.
LOGY / _YB ne servent plus au tirage ; les garder comme colonne de diagnostic
dans trials.parquet reste utile (dlogY explique 94 % de SSIM).
Le biais geometrique BIAS peut rester : il explique 0.2 % de la variance du tri.

COUT : 5.5 essais par exemple retenu, soit ~550 000 simulations pour 100k.
A ton debit mesure (135 essais/s a 20 threads), ~1 h ; moins a 35 workers.
"""
