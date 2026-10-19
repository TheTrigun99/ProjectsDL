"""Physique de reference, ecrite independamment de generating_sim_data.

Trois briques, chacune derivee des principes et non de la formule du code :

  1. Fresnel  : coefficients d'AMPLITUDE a une interface n1 -> n2.
  2. Empilement : enumeration incoherente EXHAUSTIVE des copies emergentes
     d'un empilement de couches (DFS sur les rebonds). Donne pour chaque copie
     son poids et sa marche laterale le long de la vitre.
  3. Geometrie : tracage de Snell vectoriel 3D + parallaxe exacte, pour le
     deplacement image du fantome, pixel par pixel.

Aucun import de generating_sim_data ici.
"""
import numpy as np

# ---------------------------------------------------------------------------
# 1. Fresnel par amplitudes
# ---------------------------------------------------------------------------
def fresnel_R(n1, n2, th1):
    """(R_s, R_p) a l'interface n1 -> n2 pour un angle d'incidence th1 (rad)."""
    n1, n2 = np.asarray(n1, float), np.asarray(n2, float)
    s2 = (n1 / n2 * np.sin(th1)) ** 2
    c1 = np.cos(th1)
    c2 = np.sqrt(np.clip(1.0 - s2, 0.0, 1.0))          # cos(theta_2), Snell
    rs = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)
    rp = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)
    return rs ** 2, rp ** 2


# ---------------------------------------------------------------------------
# 2. Empilement : toutes les copies reflechies
# ---------------------------------------------------------------------------
def stack_copies(theta0, idx, thick, pol, wmin=1e-13, lvl_max=600):
    """Copies emergentes vers l'arriere, propagation incoherente par niveaux.

    idx   : indices des couches, idx[0] = milieu d'incidence, idx[-1] = sortie.
    thick : epaisseurs des couches internes, len(idx) - 2 valeurs (metres).
    pol   : 0 pour s, 1 pour p.

    Un etat est (interface, sens d'arrivee, nb de traversees de chaque couche
    interne) : deux chemins de meme etat sont indiscernables et se fusionnent,
    ce qui rend l'enumeration finie meme quand R -> 1 (incidence rasante).

    Retour : (poids, marches, fuite) tries par marche croissante. `fuite` est le
    poids abandonne sous wmin, a verifier negligeable.
    """
    idx = np.asarray(idx, float)
    M = len(idx) - 1                                     # interfaces 0..M-1
    th = np.arcsin(np.clip(np.sin(theta0) / idx, -1, 1))  # invariant de Snell
    R = np.array([fresnel_R(idx[i], idx[i + 1], th[i])[pol] for i in range(M)])
    nin = M - 1                                          # couches internes
    walk = np.zeros(len(idx))
    walk[1:-1] = np.asarray(thick, float) * np.tan(th[1:-1])

    z0 = (0,) * nin
    live = {(0, 0, z0): 1.0}        # sens 0 = descendant, 1 = montant
    out, fuite = {}, 0.0
    for _ in range(lvl_max):
        if not live:
            break
        nxt = {}
        def add(key, w):
            nonlocal fuite
            if w < wmin:
                fuite += w
            else:
                nxt[key] = nxt.get(key, 0.0) + w
        for (i, inc, cnt), w in live.items():
            if inc == 0:                      # dans la couche i, descendant
                wr = w * R[i]
                if i == 0:
                    out[cnt] = out.get(cnt, 0.0) + wr
                else:
                    c = list(cnt); c[i - 1] += 1
                    add((i - 1, 1, tuple(c)), wr)
                if i + 1 < M:
                    c = list(cnt); c[i] += 1
                    add((i + 1, 0, tuple(c)), w * (1 - R[i]))
            else:                             # dans la couche i+1, montant
                wt = w * (1 - R[i])
                if i == 0:
                    out[cnt] = out.get(cnt, 0.0) + wt
                else:
                    c = list(cnt); c[i - 1] += 1
                    add((i - 1, 1, tuple(c)), wt)
                if i + 1 < M:
                    c = list(cnt); c[i] += 1
                    add((i + 1, 0, tuple(c)), w * R[i])
        live = nxt
    fuite += sum(live.values())
    items = sorted((float(np.dot(c, walk[1:-1])), w) for c, w in out.items())
    # fusionne les marches egales (couches d'epaisseur identique)
    ws, ss = [], []
    for s, w in items:
        if ss and abs(s - ss[-1]) <= 1e-15 * max(1.0, abs(s)):
            ws[-1] += w
        else:
            ws.append(w); ss.append(s)
    return np.array(ws), np.array(ss), fuite


def layers(n_glass, h, gap, double):
    if double:
        return [1.0, n_glass, 1.0, n_glass, 1.0], [h, gap, h]
    return [1.0, n_glass, 1.0], [h]


def reflect_kernel(theta0, n_glass, h, gap, double, wmin=1e-13):
    """Noyau de reflexion non polarise : (poids, marches, fuite), moyenne s/p.

    Les marches sont les memes pour s et p (pure geometrie), seuls les poids
    changent ; on fusionne donc les deux supports.
    """
    acc, fuite = {}, 0.0
    for pol in (0, 1):
        w, s, f = stack_copies(theta0, *layers(n_glass, h, gap, double),
                               pol=pol, wmin=wmin)
        fuite += 0.5 * f
        for wi, si in zip(w, s):
            key = round(si, 15)
            acc[key] = acc.get(key, 0.0) + 0.5 * wi
    ks = sorted(acc)
    return np.array([acc[k] for k in ks]), np.array(ks), fuite


# ---------------------------------------------------------------------------
# 3. Geometrie : Snell vectoriel et parallaxe
# ---------------------------------------------------------------------------
def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _refract(d, n, eta):
    """d unitaire, n unitaire (orientation quelconque : auto-orientee), eta = n1/n2."""
    c = (d * n).sum(-1, keepdims=True)
    n = np.where(c > 0, -n, n)            # normale face au rayon
    c = -(d * n).sum(-1, keepdims=True)
    k = 1 - eta ** 2 * (1 - c ** 2)
    return eta * d + (eta * c - np.sqrt(np.clip(k, 0, None))) * n


def _reflect(d, n):
    return d - 2 * (d * n).sum(-1, keepdims=True) * n


def trace_walk(v, nrm, n_glass, h, gap, double):
    """Marche laterale par tracage explicite, SANS formule analytique.

    v   : (..., 3) direction du rayon partant de la camera (non normalise ok)
    nrm : (3,) normale de la vitre (orientation quelconque)
    Surfaces aux offsets 0, -h, -(h+gap), -(h+gap+h) le long de nrm, le rayon
    venant du cote +nrm. Reflexion speculaire sur la DERNIERE surface traversee
    avant retour : surface 1 (lame simple) ou surface 2 (1re face du 2e panneau
    d'un double vitrage, la copie que modelise le code).

    Retour : (walk_vec, dir_sortie). walk_vec est dans le plan de la vitre.
    """
    v = _unit(np.asarray(v, float))
    nrm = _unit(np.asarray(nrm, float))
    if (v * nrm).sum() > 0:               # nrm doit pointer vers la camera
        nrm = -nrm
    zs = [0.0, -h] if not double else [0.0, -h, -(h + gap), -(h + gap + h)]
    k_refl = 1 if not double else 2       # indice de la surface reflechissante
    ns = [1.0, n_glass, 1.0, n_glass, 1.0]   # indice de la couche sous la surface k

    P = np.zeros_like(v)
    d = v
    for k in range(k_refl):               # descente : traverse les surfaces 0..k_refl-1
        d = _refract(d, nrm, ns[k] / ns[k + 1])
        t = (zs[k + 1] - (P * nrm).sum(-1)) / (d * nrm).sum(-1)
        assert np.all(t > 0), "propagation vers l'arriere (descente)"
        P = P + t[..., None] * d
    d = _reflect(d, nrm)
    for k in range(k_refl - 1, -1, -1):   # remontee
        t = (zs[k] - (P * nrm).sum(-1)) / (d * nrm).sum(-1)
        assert np.all(t > 0), "propagation vers l'arriere (remontee)"
        P = P + t[..., None] * d
        d = _refract(d, nrm, ns[k + 1] / ns[k])
    return P, d


def ghost_offset_exact(x, y, f_px, nrm, n_glass, h, gap, double, d_refl,
                       scene_plane="depth"):
    """Offset d'echantillonnage (dx, dy) en pixels du fantome, exact.

    Le faisceau fantome est le faisceau direct translate de `walk_vec` ; le
    fantome est donc ce que verrait une camera deplacee de walk_vec (parallaxe).
    On projette le point de la scene virtuelle vu par la camera deplacee avec la
    camera d'origine : c'est la coordonnee source du remap.
    scene_plane : "depth" scene plane a la profondeur d_refl ;
                  "sphere" scene a la distance d_refl le long de chaque rayon.
    """
    v = np.stack(np.broadcast_arrays(x, y, np.full_like(x, f_px, dtype=float)), -1)
    walk_vec, d_out = trace_walk(v, nrm, n_glass, h, gap, double)
    vh = _unit(v)
    if scene_plane == "depth":
        Z = d_refl
    else:
        Z = d_refl * f_px / np.linalg.norm(v, axis=-1)
    # P' = walk_vec + s*vh avec (P')_z = Z
    s = (Z - walk_vec[..., 2]) / vh[..., 2]
    P = walk_vec + s[..., None] * vh
    xp = f_px * P[..., 0] / P[..., 2]
    yp = f_px * P[..., 1] / P[..., 2]
    return xp - x, yp - y, walk_vec


# ---------------------------------------------------------------------------
# 4. Noyau exact indexe par le nombre de traversees (pour les cartes par pixel)
# ---------------------------------------------------------------------------
def kernel_counts(theta0, n_glass, double, wmin=1e-9, lvl_max=400):
    """{(k_verre, k_lame_air): poids} non polarise, a une incidence donnee.

    La marche d'une copie vaut k_verre * h * tan(theta_t) + k_air * gap *
    tan(theta_i) : elle ne depend de h et gap que par ce facteur, donc la table
    ne depend que de theta0. Poids moyennes sur s et p.
    """
    idx = [1.0, n_glass, 1.0, n_glass, 1.0] if double else [1.0, n_glass, 1.0]
    M = len(idx) - 1
    th = np.arcsin(np.clip(np.sin(theta0) / np.asarray(idx, float), -1, 1))
    acc = {}
    for pol in (0, 1):
        R = np.array([fresnel_R(idx[i], idx[i + 1], th[i])[pol] for i in range(M)])
        live = {(0, 0, (0,) * (M - 1)): 1.0}
        out = {}
        for _ in range(lvl_max):
            if not live:
                break
            nxt = {}
            for (i, inc, c), w in live.items():
                def push(j, d, lay, ww):
                    if ww < wmin:
                        return
                    cc = list(c); cc[lay] += 1
                    k = (j, d, tuple(cc))
                    nxt[k] = nxt.get(k, 0.0) + ww
                if inc == 0:
                    if i == 0:
                        out[c] = out.get(c, 0.0) + w * R[i]
                    else:
                        push(i - 1, 1, i - 1, w * R[i])
                    if i + 1 < M:
                        push(i + 1, 0, i, w * (1 - R[i]))
                else:
                    if i == 0:
                        out[c] = out.get(c, 0.0) + w * (1 - R[i])
                    else:
                        push(i - 1, 1, i - 1, w * (1 - R[i]))
                    if i + 1 < M:
                        push(i + 1, 0, i, w * R[i])
            live = nxt
        for c, w in out.items():
            key = (c[0] + c[2], c[1]) if double else (c[0], 0)
            acc[key] = acc.get(key, 0.0) + 0.5 * w
    return acc


def kernel_table(n_glass, double, n_th=257, th_max_deg=84.0, wkeep=1e-4):
    """Tabule kernel_counts sur une grille d'incidence.

    Retour (cos_grid croissant, cles, W) avec W[i, j] le poids de la cle j a
    l'incidence i. Les cles retenues sont celles qui depassent wkeep du total
    a au moins une incidence.
    """
    ths = np.radians(np.linspace(0.0, th_max_deg, n_th))
    tabs = [kernel_counts(t, n_glass, double) for t in ths]
    keys = set()
    for d in tabs:
        tot = sum(d.values())
        keys |= {k for k, v in d.items() if v > wkeep * tot}
    keys = sorted(keys, key=lambda k: (k[1], k[0]))
    W = np.array([[d.get(k, 0.0) for k in keys] for d in tabs])
    cos_g = np.cos(ths)[::-1]              # croissant
    return cos_g, keys, W[::-1]
