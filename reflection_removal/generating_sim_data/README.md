# generating_sim_data

Generation du dataset simule de suppression de reflets (Kee et al., CVPR 2025,
Func. 1 et S1). Remplace `main_files/data_clean.ipynb` par un paquet
executable : meme calcul, bit a bit, mais decoupe en modules et pilotable en
ligne de commande.

## Les deux etapes

```bash
# 1. RAW -> cache XYZ float16 + masque de saturation (une fois par source)
python -m generating_sim_data cache \
    --dng-dir   /home/damien/data_fivek_dng/raise_outdoor_dng \
    --cache-dir /home/damien/data_fivek_dng/pool_raise_cache \
    --index-json /home/damien/data_fivek_dng/raise_outdoor.json

# 2. cache XYZ -> exemples simules et tries
python -m generating_sim_data generate --out /tmp/essai --n-groups 50

# coup d'oeil sur quelques exemples (m | true_t | r en PNG), sans journal
python -m generating_sim_data preview --out /tmp/apercu -n 8
```

A lancer depuis `/home/damien/ProjectsDL/reflection_removal`.

## Modules

| module | role |
|---|---|
| `config.py` | chemins, constantes, `GenConfig` / `CacheConfig` / `WeakConfig`, presets de biais |
| `colorimetry.py` | matrices XYZ/sRGB, Bradford, gamma sRGB |
| `dng_metadata.py` | tags DNG, point blanc as-shot (Func. S3/S7), exposition EXIF |
| `raw_processing.py` | RAW -> XYZ + masque de saturation (etapes 1-4 d'ACR) |
| `cache.py` | etape 1 : tous les DNG -> `.npy` + `pool_index.json` |
| `pool.py` | cache -> entrees memmap, decoupes, radiances, tirage des paires |
| `scene.py` | tirage de la vitre, Fresnel, defocus, fantome |
| `simulate.py` | Func. 1 et S1 : composition de `m`, `t`, `r`, `true_t` |
| `culling.py` | mesures et verdicts de tri |
| `generate.py` | etape 2 : boucle multi-thread, ecriture des `.npz` et des journaux |
| `display.py` | visualisation (matplotlib, hors chemin de generation) |

Dependance exterieure au paquet : `acr_tone_fast.finish`, importe depuis
`main_files/` par `_paths.py`. C'est la meme courbe de rendu que le
dataloader d'entrainement ; la partager est volontaire, sans quoi le PSNR du
culling ne serait plus comparable a celui de l'evaluation.

## Ce que la generation ecrit

```
<out_dir>/ex_00000_00.npz     m, true_t (float16), p_* (parametres scalaires),
                              R_map, ghost_px (float16), group, member,
                              t_name, r_name
<out_dir>/trials.parquet        tous les essais, avec leur verdict
<out_dir>/trials_groups.parquet les exemples gardes (+ group, K, noms, types)
<out_dir>/scene_params.parquet  parametres de vitre des exemples gardes
<out_dir>/usage.npz             combien de fois chaque image a ete tiree
<out_dir>/meta.json             la configuration complete du run
```

`--no-maps` omet `R_map` et `ghost_px` : ~0.5 Mo par exemple, soit 7 Go sur
15 000 exemples.

## Groupes

Un groupe est une transmission et les `K` reflets qui la partagent. Les
membres d'un groupe ont le meme `true_t` **au bit pres** : meme crop (`t_pre`)
et meme gain d'exposition (`e_lock`). Avec `-k 1` (defaut) chaque groupe n'a
qu'un membre.

## Culling

Deux grilles, `--cull` :

- `std` (`measure`) : SSIM(m, t) dans une fenetre, reflet non plat, exposition
  correcte, transmission structuree.
- `a_tdom` (`measure_a_tdom`, defaut) : raisonne par cases 32x32 sur la
  dominance locale. La transmission doit dominer 40-65 % des cases, le reflet
  doit en dominer >= 20 % d'un facteur 2, et aucune case ne doit etre noyee a
  plus de 80 %. Beaucoup plus selectif : compter ~1-2 % d'acceptation.

Les deux renvoient toutes leurs mesures, pas seulement le verdict : c'est ce
qui part dans `trials.parquet`, d'ou on relit la cause des rejets.

```python
import pandas as pd
df = pd.read_parquet(".../trials.parquet")
df["verdict"].value_counts(normalize=True).mul(100).round(1)
```

## Mode faible

`--p-weak 1.0` tire les exemples par `simulate_weak` : le partenaire reflet est
choisi pour que sa luminance place le melange dans `--weak-band` (defaut
24-27.5 dB de PSNR JPEG). Le gain `alpha` reste a 1 : c'est le **choix du
partenaire** qui rend le reflet faible, pas un facteur libre, qui serait
irrealiste.

```bash
python -m generating_sim_data generate --out .../dataset_weak \
    --n-groups 15000 --p-weak 1.0 --cull std --weak-band 24 27.5 --seed 1009
```

Avec `-k` > 1, seule l'**ancre** du groupe est garantie dans la bande : les
membres suivants sont tires par `simulate_example` et ne passent que le
culling. C'est le comportement du notebook ; pour un dataset entierement dans
la bande, garder `-k 1`.

## Completer un dataset

Les quotas ne sont pas persistes : une continuation est un nouveau run avec une
autre graine et un decalage des numeros de groupe, pour ne pas ecraser les
`.npz` existants. Les journaux prennent le decalage en suffixe
(`trials_15000.parquet`), puis se concatenent.

```bash
python -m generating_sim_data generate --out .../dataset_weak \
    --n-groups 5000 --seed 2009 --group-offset 15000 --p-weak 1.0
```

## Reproductibilite

`sample_scene2` fige l'ordre de ses tirages (fov, phi, theta0, N, d_focus,
d_glass, d_refl, h_glass, gap, double_pane) : le modifier change tous les
datasets a graine egale. `tilt(..., g=1.0)` est identique bit a bit a
`rng.uniform`, donc passer un preset de biais neutre ne decale pas le flux.

Les workers sont semes par `[seed, 10_000 + w]` : a nombre de workers egal le
run est reproductible, mais changer `--workers` change l'assemblage.

L'ordre du pool est fige (`POOL_ORDER = fivek, raise, ibl`) parce que le cache
des radiances `cache_tot/pool_levels_*.npy` est indexe par position. Il est
nomme d'apres l'empreinte des noms du pool, donc une composition differente
prend un autre fichier au lieu d'ecraser le precedent.

## Verifie contre le notebook

`sample_scene2`, `random_crop`, `simulate_example`, `measure`,
`measure_a_tdom`, `descripteurs` et `psnr_raw_to_jpeg` ont ete compares aux
cellules de `data_clean.ipynb` sur 120 tirages (pool complet, 5 branches de
verdict) : sorties identiques bit a bit et flux RNG inchange. L'ordre du pool a
ete revalide contre `cache_tot/pool_levels.npy` (10012 entrees).

Les differences voulues sont listees dans `CHANGES.md`.

La verification est rejouable :

```bash
python3 generating_sim_data/tests/extract_notebook.py    # cellules -> nb_ref.py
python3 generating_sim_data/tests/test_equivalence.py --n 120
```

Elle compare bit a bit et verifie que le flux RNG est consomme a l'identique :
une divergence de flux decalerait tous les tirages suivants sans changer un
seul resultat pris isolement.

**Ce qui n'a PAS pu etre verifie sur de vraies donnees** : plus aucun DNG n'est
sur le disque (`dng/` et `raise_outdoor_dng/` sont vides), donc la
sous-commande `cache` n'a pas pu etre lancee de bout en bout sur sa vraie
entree. `read_raw_to_xyz`, `align_mask`, `clip_level` et `read_exposure` ont
ete compares au notebook sur le seul RAW restant de la machine
(`rawtosrgb/rawboat.ARW`, un ARW Sony), avec la matrice couleur DNG remplacee
par une constante identique des deux cotes. La lecture des tags DNG
proprement dite (`camera_to_xyz`, `neutral_to_xy`, `find_xyz_to_camera`) est un
portage fidele mais non execute.
