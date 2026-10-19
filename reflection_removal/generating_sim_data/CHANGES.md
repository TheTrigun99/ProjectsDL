# Ce qui change par rapport a data_clean.ipynb

Le calcul etait identique bit a bit jusqu'aux corrections 11 et 12 ci-dessous,
qui changent `ghost_px` et donc `m` et `r`. Tout le reste est inchange, flux
RNG inclus. Les differences ci-dessous sont des corrections volontaires.

## Bugs de physique corriges (changent les donnees generees)

11. **`ghost_px` avait un `cos(theta_i)` en trop.** Les deux faisceaux
    emergents d'une lame a faces parallele sont paralleles, separes
    perpendiculairement de `2*path*cos(theta_i)` ; une source a `d_refl` est
    donc vue decalee de `f_px * 2*path*cos(theta_i) / d_refl`. Un seul cosinus,
    celui de la projection de la marche laterale sur le plan image. Le code
    ecrivait `cos_i ** 2`. « Projeter perpendiculairement au rayon » est le
    MEME cosinus vu autrement, pas un second : au centre du champ le rayon est
    l'axe optique. Verifie par tracage de Snell explicite (separation
    perpendiculaire = `Delta*cos` a 1e-4 pres de 5 a 80 deg).
    Effet : facteur `1/cos(theta_i)`, mediane 1.06 au centre du champ sur la
    distribution reelle, mais rapport median 2.2 entre le centre et les coins
    d'un meme patch -- ce n'est donc pas une erreur d'echelle, la forme du
    champ de deplacement etait deformee.

12. **Le fantome etait decale du mauvais cote.** Dans le verre le rayon avance
    le long de `+t_hat`, donc le chemin fantome ressort plus loin sur `+t_hat`
    et ECHANTILLONNE la scene plus loin sur `+ghost_dir` ; `cv2.remap` veut
    exactement cet offset d'echantillonnage, donc `map = p + ghost_dir*shift`.
    Le code soustrayait. Le piege : la copie visible d'un detail apparait, elle,
    decalee sur `-ghost_dir`, vers le point d'incidence normale -- les deux
    enonces sont inverses l'un de l'autre.
    Verifie de bout en bout : blob gaussien sur le plan de la scene, les deux
    chemins traces par Snell pixel par pixel. Erreur max de la couche fantome
    0.041 avec `+` contre 0.331 avec `-`, et 0.196 en ne decalant pas du tout --
    le signe fautif etait donc pire que pas de fantome a 45 deg d'inclinaison.
    Le poids du fantome etant `(1-R1)^2 ~ 0.92`, la copie decalee represente
    ~48 % de la couche de reflet : ce n'etait pas un terme mineur.

Consequence sur les tests : `tests/test_equivalence.py` signale desormais
`DIFF ... ghost_px` sur `sample_scene2` et `DIFF layers` sur `test_sim`. C'est
attendu. Verifie : `ghost_px` nouveau x `cos_i` == ancien a 3.7e-07 (precision
du float32), `ghost_px` est la SEULE cle de `sample_scene2` qui bouge, et le
flux RNG est identique sur les 16 configurations testees.

Non repercute : `data/generating_sim.py` (copie plus ancienne, gelee pour les
controles publies) et `main_files/data_clean.ipynb` garde les deux bugs.

## Bugs corriges

1. **Mode faible inutilisable avec K > 1.** La boucle des membres
   supplementaires (cellule 88) utilisait `t_pre`, qui n'etait jamais defini
   dans la branche faible : `NameError` des que `P_WEAK > 0` et `K > 1`.
   `simulate_weak` renvoie maintenant le crop brut, que le generateur partage
   entre les membres comme dans la branche normale.

2. **`trials.parquet` du mode faible sans colonne `verdict`.** En cas de rejet,
   `simulate_weak` renvoyait la raison sous forme de chaine, et l'appelant
   l'ajoutait a `rej_weak` a cote de dicts : `pd.DataFrame(rej_weak)` melangeait
   lignes et scalaires, donc `df["verdict"]` echouait. Un rejet renvoie
   desormais toujours `{"verdict": raison, "mode": "weak"}`.

3. **`pick_partner_weak` dependait de l'ordre des cellules.** Elle etait definie
   deux fois (la seconde masquant la premiere) et lisait `KIND_IDX`, cree dans
   une cellule *posterieure*. Elle ne fonctionnait qu'apres avoir execute les
   cellules dans le bon ordre. C'est maintenant une methode de `Pool`, dont les
   index sont construits dans le constructeur.

4. **`deterministic_scene2` ignorait `params["gap"]`.** La valeur demandee etait
   ecrasee juste apres par un tirage aleatoire, donc le preset "ghost"
   (gap = 0.020 m) ne mesurait pas ce qu'il annoncait. `gap` est maintenant
   respecte, et n'est tire que si la vitre est un double vitrage -- ce qui
   decale le flux RNG de cette fonction (diagnostics seulement, la generation
   ne l'appelle pas).

5. **Cache des radiances invalide silencieusement.** `levels()` ne verifiait que
   la longueur : une permutation du pool passait inapercue et decalait toutes
   les luminances. Le fichier est maintenant nomme d'apres l'empreinte md5 des
   noms du pool, et un cache illisible ou de forme inattendue est ignore au lieu
   de faire echouer le chargement (`cache_tot/pool_levels_bb3987c8cc.npy`
   contenait un `Ellipsis` laisse par un essai).

6. **`attach_entry` posait `raise_outdoor` a l'envers** (`False if raise_out else
   True`), ce que la cellule 70 corrigeait ensuite a la main. Le drapeau est
   maintenant passe explicitement a `load_pool`.

## Changements de sortie

7. `measure_a_tdom` renvoyait `"A_tdom "` avec une espace finale. Corrige en
   `"A_tdom"` : a regrouper si vous comparez d'anciens `trials.parquet`.

8. `measure_a_tdom` renvoie en plus les mesures qu'il calculait deja et jetait
   (`ssim`, `ssim_std`, `Ym`, `m_mean`, `t_std`, `r_std`, `p90_part_loc`,
   `ratio_lum`, `sat_frac`). Les verdicts sont inchanges, les colonnes du
   parquet sont plus nombreuses.

9. `jpeg_psnr_384` devient `jpeg_psnr`, accompagne de `jpeg_psnr_size`. Le nom
   etait trompeur : `psnr_raw_to_jpeg` mesure apres reduction a `CULL_SIZE`
   (256), pas a 384. Il venait de `mission_claude/gen_weak.py`, ou la taille
   etait bien un parametre.

10. Les journaux d'une continuation prennent le decalage en suffixe
    (`trials_15000.parquet`) au lieu d'ecraser ceux du premier run, et
    `meta.json` enregistre la configuration complete.

## Code retire

Supprime parce que mort, remplace, ou casse :

- `sample_scene` (cellule 41) : remplacee par `sample_scene2`. Contenait un
  `print` de debogage par appel et l'ancienne formule du fantome (`/d_glass`
  au lieu de `/d_refl`).
- `simulate_example` (cellule 51) et `simulate_example_k` (cellule 52) :
  remplacees par `simulate_example_enc`. La premiere lisait `src_t["ibl"]`,
  cle qu'aucune entree de pool ne possede (`KeyError` a l'appel).
- `disk_kernel` et `defocus` : variantes sans anti-aliasing, non appelees.
- `align_mask` version "cellule 21" (avec `print`) : seule `align_maskv2` etait
  utilisee, elle reprend le nom `align_mask`.
- `det_params` : generateur de dict de parametres, remplace par
  `scene.DET_PARAM_PRESETS`.
- Les cellules 82-84 de quotas (`QUOTA`, `deficit_now`, `pick_combo`,
  `draw_in`...) : entierement commentees, remplacees par `PAIR_PROBS`.
- `import builtins; bool = builtins.bool` (cellule 72) : ne servait qu'a reparer
  le `bool` masque par la cellule 70.
- Les cellules d'installation `%pip` et de telechargement des DNG : le
  telechargement vit deja dans `build_raise_json.py`, et `cache.py` accepte son
  json via `--index-json`.

## Non repris

`data/generating_sim.py` (importe par `analysis/scripts/control_lowe.py` et
`control_hdri_ref.py`) est une copie plus ancienne : son `ghost_px` divise par
`d_glass` et son `d_refl` est tire sur `10**(0, 1.7)`. Je l'ai laisse tel quel
pour ne pas changer ces deux controles sous les pieds de leurs resultats
publies ; le paquet, lui, suit le notebook (`/d_refl`, `10**(-1, 1.7)`).

## Simulateur v2 (mission 7, 10/10, code ecrit a la demande de Damien)

Les deux corrections ci-dessous changent les donnees generees ; elles ne
consomment aucun alea (a graine egale, memes candidats qu'en v1.1). Le
comportement v1.1 reste disponible bit pour bit : `--fresnel moyenne --wb t`
(verifie contre une copie figee de v1.1, `mission7/sauvegardes/generating_sim_data_v1_1/`,
sur 6 graines dont mode faible et K = 2).

13. **Fresnel par polarisation** (`scene._geometry(polar=True)`). La lumiere
    naturelle est moitie s, moitie p, et chaque polarisation rebondit avec SA
    reflectance : R_map = 1/2 [f(R_s) + f(R_p)], f(R) = 2R/(1+R) (lame) ou
    4R/(1+3R) (double vitrage). Moyenner R_s et R_p AVANT f (v1.1) surestimait
    R_map en incidence oblique (lame +3 % a 45 deg, +8 % a 60 deg ; double
    vitrage +9 % / +21 %), f etant concave. Le poids du fantome suit :
    a_pol = (1-R_pol)^2 (ou ^4), ramene a un seul `scene["a_ghost"]` = c2/c1
    avec c1 = 1/2 sum f_pol/(1+a_pol), c2 = 1/2 sum f_pol a_pol/(1+a_pol),
    c1 + c2 = R_map. Verifie contre une propagation des faisceaux rebond par
    rebond (ecart max 2.2e-7).

14. **Balance des blancs de la mixture** (`simulate.white_point`, `wb="mix"`).
    v1.1 equilibrait m, t, r sur le blanc as-shot de la transmission. v2 imite
    l'AWB d'une camera sur la photo a travers la vitre (Kee et al. utilisent le
    balanceur C5, supp. Func. S2 ; ici une version idealisee) :
    w = moyenne Y(reflet) / moyenne Y(m) en XYZ (apres la vitre, avant
    exposition), W = (1-w) W_t + w W_r (Y = 1), M = XYZ_D50->sRGB . Bradford(W->D50),
    la MEME matrice pour m, t, r et true_t (entree et cible restent coherentes),
    exposition recalculee avec M. Rejet « balance_blancs » si
    |mired(W) - mired(W_t)| > 60 (McCamy ; ~4 % des exemples ; Kee rejette aussi
    les deplacements extremes). Le mode faible recalcule le blanc pour chaque
    alpha. Membres d'un groupe : blanc de l'ancre. Limite : `wb_dmired` ne voit
    que la temperature, pas la teinte (vert / magenta).

15. **Nouveaux generateurs**, ports fideles des pipelines qui avaient produit
    les datasets auxiliaires, sur la physique v2 :
    - `mid.py` (`generate-mid`) : mission4/gen_mid.py + crit4.py (reflets nets
      « durs mais lisibles », dataset_30knet_mid ; `--meme-scene` : dataset_meme_scene).
      Memes buckets, quotas, fenetres de luminance (`fenetres_mid.json` =
      mission4/fenetres.json), rejet precoce par l'intervalle d'alpha, criteres
      exacts. Verifie : fonctions de crit4 identiques (0 difference), et
      composer(alpha = 1) == simulate_example bit pour bit.
    - `weak2.py` (`generate-weak2`) : mission_claude/gen_weak2.py (reflets
      faibles 27.5-30 dB, alpha borne [0.25, 4] par bissection, marge 2.5 dB).
      Verifie contre les fonctions originales (0 difference a wb = "t"). Seul
      changement voulu : plus de vues Laval (EXR) dans les reflets.
