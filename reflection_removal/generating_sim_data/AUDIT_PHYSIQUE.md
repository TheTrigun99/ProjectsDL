# Audit de la physique de reflexion (2026-10-12)

Contre-verification de `scene.py` / `simulate.py` par une physique de reference
ecrite independamment. Porte sur `R_map`, la polarisation, le fantome et le
double vitrage -- les trois chantiers de la mission 7.

Scripts : `audit_physique/`. Lancer depuis ce dossier (`python3 t0_selfcheck.py`,
etc.). `ref_physics.py` n'importe rien de `generating_sim_data`.

---

## Methode

Trois briques derivees des principes, pas de la formule du code :

1. **Fresnel par coefficients d'amplitude** a une interface n1 -> n2.
2. **Enumeration incoherente exhaustive** des copies emergentes d'un empilement
   de couches : propagation par niveaux avec fusion des etats indiscernables
   (etat = interface, sens, nombre de traversees par couche). Donne pour chaque
   copie son poids **et** sa marche laterale. Fini meme en incidence rasante,
   ou R -> 1 piege la lumiere.
3. **Tracage de Snell vectoriel 3D** + projection perspective exacte, sans
   aucune formule analytique.

Auto-controle de la reference (`t0_selfcheck.py`) : incidence normale =
((n-1)/(n+1))^2, R_p = 0 exactement a Brewster, -> 1 en rasance ; retrouve
`2R/(1+R)` et `4R/(1+3R)` a 1e-12 ; retrouve `2h.tan(theta_t)` et
`2(h.tan_t + gap.tan_i)` a 1e-13 par tracage.

Comparaisons ensuite faites sur la **vraie distribution** du sampler : l'ordre
des tirages de `sample_scene2` est recopie dans `render_ref.draw_scalars`, et
les rendus utilisent de vraies images (`real_photos_random`).

---

## Ce qui est exact

Verifie par calcul independant, pas relu.

| Quantite | Ecart a la reference |
|---|---|
| `fresnel_single`, `fresnel_angle` | 2.2e-16 |
| `R_map` avec `polar=True`, lame **et** double vitrage | 1e-13 (precision machine) |
| `a_ghost` : decomposition 2 copies par polarisation | c1 + c2 = R_map exactement |
| `defocus_px` (cercle de confusion) | 1e-15 |
| `T = 1 - R_map` | exact (verre sans absorption) |
| marche `2(h.tan_t + gap.tan_i)` | 7e-15 (tracage de Snell) |
| separation perpendiculaire `= marche . cos_i` | exacte |
| signe du fantome (`+ghost_dir`) | correct |

`4R/(1+3R)` pour 4 interfaces n'est pas evident et sort bien de la recurrence
(= `2Rs/(1+Rs)` applique a `Rs = 2R/(1+R)`).

Le biais de la v1.1 est reel, aux valeurs annoncees dans `CHANGES.md` pres :
`moyenne` surestime de **+3.36 %** a 45 deg et **+8.16 %** a 60 deg (lame),
**+8.86 %** / **+20.82 %** (double vitrage). C'est Jensen, `2R/(1+R)` est
concave. Corrections 13 et 14 confirmees.

---

## Erreur 1 -- projection du fantome dans le plan image

Touche toutes les scenes avec fantome (48 a 63 % selon le prereglage).

Le fantome n'est **pas** une deviation angulaire : les deux faisceaux emergents
sont paralleles, decales de `D = marche . t_hat`. Si la scene reflechie etait a
l'infini il n'y aurait aucun fantome. C'est de la **parallaxe** : le fantome est
ce que verrait une camera translatee de D.

Parallaxe d'une base D, scene a la profondeur Z, en composantes du repere camera :

```
offset = (f/Z) * ( Dx - x*Dz/f ,  Dy - y*Dz/f )
         \______/   \__________/
          lateral      radial
```

Le code ne garde que le lateral, avec le mauvais facteur :

| | code | physique |
|---|---|---|
| terme lateral | `\|D\|.cos_i` x dir(tx,ty) | `\|D\|.sqrt(tx^2+ty^2)/\|t\|` x dir(tx,ty) |
| terme radial | **absent** | `-(x,y).Dz/d_refl` |

`|D|.cos_i` est la separation des faisceaux mesuree **perpendiculairement aux
faisceaux** -- fait geometrique juste (c'est ce que validait la correction 11).
Mais un decalage en pixels se mesure dans le **plan image**, perpendiculaire a
l'**axe optique**. Les deux plans coincident uniquement sur l'axe. D'ou :

```
au centre du champ exact (theta0 = 5..75 deg) : ratio code/exact = 1.000000
```

C'est precisement pour cela que la verification au centre ne pouvait pas
attraper ce bug.

**Mesures hors axe** (`t3_geom.py`, grille 96^2, domaine du sampler) :

| | erreur de direction (med / max) | module code/exact (med, p10) |
|---|---|---|
| fov 40, th0 = 38 deg | 7.4 / 13.4 deg | 0.988 / 0.950 |
| fov 40, th0 = 60 deg | **16.7 / 31.5 deg** | 0.985 / 0.952 |
| fov 90, th0 = 23 deg | 3.5 / 4.7 deg | **0.816 / 0.696** |

**Attribution** (`t4_attrib.py`, erreur vectorielle relative mediane) : jeter Dz
seul donne 20 a 44 % ; le `cos_i` **compense partiellement**, le code finit a
13 a 29 % (max 54 %). Deux defauts qui s'annulent en partie -- c'est pourquoi le
module n'a l'air que legerement faux alors que la **direction** l'est beaucoup.

**Exemple sur un pixel** (`t11_pixel.py` ; fov 40, patch 384, th0 = 45 deg,
h = 6 mm, d_refl = 1 m ; coin haut-gauche, theta_i = 66.4 deg, Dz = +6.00 mm) :

```
facteur lateral  : sqrt(tx^2+ty^2)/|t| = 0.7497   cos_i = 0.4007   -> x1.87
terme lateral    = (-3.1644, -1.6856) px   |.| = 3.585
terme radial     = (+1.1487, +1.1487) px   |.| = 1.624   <- absent du code
exact (= trace)  = (-2.0156, -0.5368) px   |.| = 2.086   angle -165.1 deg
code             = (-1.6912, -0.9008) px   |.| = 1.916   angle -152.0 deg
```

**Effet sur la distribution reelle** (`t10_dist.py`, 3000 scenes) : `ghost_px`
systematiquement sous-estime.

```
defocus_high : rapport corrige/code  med 1.180  p90 1.492  max 1.726
uniform      :                       med 1.280  p90 1.572  max 1.727
hard         :                       med 1.182  p90 1.492  max 1.726
```

A noter : `ghost_px` est enregistre dans les npz quand `save_params=True`, donc
toute analyse qui s'en sert herite du biais de -18 a -28 %.

### Correctif

6 lignes dans `_geometry`, en gardant `tx`, `ty` **non normalises** :

```python
tz  = np.float32(f_px) - p * nz                      # composante hors-plan de t
nt  = np.sqrt(tx*tx + ty*ty + tz*tz)                 # |t|
qx  = tx - xs * tz / np.float32(f_px)                # lateral + radial
qy  = ty - ys * tz / np.float32(f_px)
amp = (np.float32(f_px) * 2 * path) / (d_refl * np.maximum(nt, 1e-8))
gx, gy    = amp * qx, amp * qy
ghost_px  = np.hypot(gx, gy)
ghost_dir = np.stack([gx, gy], -1) / np.maximum(ghost_px, 1e-8)[..., None]
```

`/nt` remplace `cos_i`, `- xs*tz/f_px` est le terme manquant. Le signe est
inchange, la correction 12 reste bonne. Verifie a **4.9e-10** d'ecart relatif
contre le tracage de Snell + projection exacte sur 72 configurations
(fov 40/60/90, th0 de 5 a 98 % de th0_max, 3 valeurs de phi, lame et IGU) --
`t9_fix.py`.

---

## Erreur 2 -- le double vitrage a 4 copies, pas 2

Touche 11 % (`defocus_high_v2`) a 50 % (`uniform`, `hard`) des exemples.

Structure exacte a theta = 45 deg, h = 14 mm, gap = 8 mm (`t2_ghost_kernel.py`,
`t5_kernel_px.py`) :

```
copie          0.00mm  14.71mm  30.71mm  45.43mm
exact          31.55%   26.20%   21.83%   18.61%
code           59.05%     --     40.95%     --
```

Les quatre reflexions (faces avant/arriere du panneau A, puis du panneau B) ont
des poids comparables : `(1-R)^2 ~ 0.90` entre copies successives.

**Ce qui reste juste** : l'energie totale est exacte, et -- demontre
analytiquement puis verifie -- la separation `2(h.tan_t + gap.tan_i)` est
**exactement** l'ecart barycentre-a-barycentre des deux paires {1,2} et {3,4},
et `(1-R)^4` **exactement** le rapport de leurs poids sommes. Le modele 2 copies
est le bon resume par paires, ce n'etait pas du hasard.

**Ce qui est faux** : chaque paire est ecrasee en un point, donc le doublet
interne de separation `2h.tan_t` disparait. Etalement (ecart-type du noyau)
sous-estime de 4.5 % (h=6, gap=12, 30 deg) a **29 %** (h=20, gap=6, 60 deg). Le
regime verre epais / lame d'air mince est le plus faux.

### Correctif

4 copies, poids proportionnels a `[1, A, A^2, A^3]` avec `A = (1-R)^2` **par
polarisation**, normalises a `4R/(1+3R)`, aux marches
`0, w_g, w_g+w_a, 2w_g+w_a` (`w_g = 2h.tan_t`, `w_a = 2.gap.tan_i`). Reproduit
98 a 100 % de la structure exacte (`t9_fix.py`). Meme forme pour la lame simple
avec 2 copies -- ce que le code fait deja.

Demande de rendre une liste de copies par `_geometry` au lieu du couple
`(R_map, a_ghost)`, et de sommer dans `double_reflection`.

---

## Impact mesure sur les couches rendues

Erreur L2 relative de la couche de reflet contre la reference complete, et PSNR
de la mixture m. 140 scenes par prereglage, vraies images (`t7_cond.py`).

| prereglage | sous-ensemble | A = code | B = geom corrigee | C = + noyau corrige | PSNR m (med/min) |
|---|---|---|---|---|---|
| defocus_high_v2 | tout | 0.51 % | 0.10 % | 0.05 % | 77.1 / 45.8 |
| | fantome >= 3 px (19 %) | 1.88 % | 0.43 % | 0.16 % | 62.8 / 45.8 |
| | **IGU, fantome >= 1 px (11 %)** | **2.79 %** (max 10.1 %) | 2.14 % | 0.10 % | 57.8 / 45.8 |
| uniform | IGU, fantome >= 1 px (36 %) | 4.07 % (max 10.4 %) | 3.17 % | 0.00 % | 54.8 / 45.3 |
| hard | IGU, fantome >= 1 px (39 %) | 4.44 % (max 15.7 %) | 3.25 % | 0.14 % | 54.8 / 42.9 |

Lecture : la geometrie domine en lame simple (A 1.00 % -> B 0.21 %), le noyau
domine en double vitrage (B reste a 3.2 %, C tombe a 0.1 %).

**Ordre des operateurs** : le code defocalise avant le fantome et avant `R_map`,
la physique defocalise apres. Residu C -> reference : mediane 0.00 a 0.17 %,
max 0.68 %. **Negligeable, ne pas y toucher.**

---

## Non modelise -- chiffre

Tous verifies negligeables ou benins (`t8_misc.py`) :

- **Fantome de la transmission** : poids relatif R^2, 0.2 % (0 deg) a 3 % (70 deg).
- **Decalage lateral de la transmission** par la lame : 0.19 px (10 deg) a
  2.65 px (80 deg) en global, sous-pixel en differentiel sur le champ. Et comme
  `m` et `true_t` viennent du meme crop, la simulation a **zero** desalignement --
  de vraies paires avec/sans vitre en auraient.
- **Defocus propre du fantome** (image virtuelle reculee de 2h/n, 8 a 26 mm) :
  0.2 a 5 % de d_refl.
- **Copies >= 3 d'une lame simple** : 0.1 % (30 deg) a 3.8 % (80 deg) de
  l'energie, et la normalisation du code les redistribue dans le bon rapport.
- **Interferences** : absentes, et c'est **correct** (longueur de coherence << h).

Deux manques de realisme qui ne sont pas des bugs :

- **Absorption / teinte verte du verre.** Verre suppose sans perte, donc
  `T = 1 - R` exactement. Du float 6 mm transmet ~0.87-0.90 avec un biais vert :
  la transmission simulee est optimiste et parfaitement neutre.
- **Couches bas-emissives.** Un double vitrage reel en a presque toujours une ;
  la reflectance visible passe de 0.15 a ~0.20-0.30 et n'est pas neutre
  spectralement. Probablement le plus gros ecart au reel de tout cet audit, pour
  les IGU.

Choix de modelisation sans reponse unique : `d_refl` est traite comme une
**profondeur** (plan fronto-parallele). L'interpretation « distance le long du
rayon » (sphere) change le fantome de -5 % (fov 40) a -20 % (fov 90). Le plan est
coherent avec l'usage de `r` comme texture et avec le `d_refl` du cercle de
confusion.

---

## Priorites

1. **Projection du fantome.** 48 a 63 % des scenes, +18 a +28 % d'amplitude en
   mediane et direction fausse de 7 a 17 deg (jusqu'a 31.5 deg). Diff de 6 lignes,
   forme fermee verifiee a 4.9e-10.
2. **4 copies du double vitrage.** 11 a 50 % des exemples selon `p_double`.
3. Rien d'autre cote justesse. L'absorption du verre et le bas-emissif relevent
   du realisme, pas de la correction.
