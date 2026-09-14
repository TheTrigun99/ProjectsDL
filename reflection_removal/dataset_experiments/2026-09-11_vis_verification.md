# `vis` mis à l'épreuve — et correction de l'avis du 11/09

**Date :** 2026-09-11
**Statut :** vérification par la mesure de [2026-09-10_distribution_tri_simulation.md](2026-09-10_distribution_tri_simulation.md)
et de [2026-09-11_avis.md](2026-09-11_avis.md). Plusieurs conclusions de ces deux notes tombent.

---

## 0. Résumé

1. **`vis` n'est pas un critère.** C'est une fonction fermée du rapport de luminance des
   deux images sources. 95 % de sa variance est fixée avant que la simulation ne commence.
   Il n'est pas dans le papier, et le papier explique pourquoi il ne peut pas y être.
2. **Le « 10-20 % de couverture » est en grande partie un artefact.** Les calques de SIR²
   ne sont pas sur la même échelle d'exposition que leurs mélanges. Après correction, le
   `vis` médian des trois sous-ensembles tient dans 0,10-0,19 — et ton dataset actuel est
   à 0,174.
3. **Sur les axes du papier, ton dataset actuel se superpose déjà à SIR².**
4. Le défaut réel, mesuré, est plus petit qu'annoncé : à SSIM égal ton reflet est
   ~1,4× moins concentré que les jeux *in the wild* — et pas moins concentré que les jeux
   de labo.
5. **Le tri n'est pas le problème, le tirage de paires l'est.** Appliquer le culling du
   papier tel quel à ton pool donne un SSIM médian de 0,72 contre 0,90 pour les test sets.

---

## 1. `vis` est déterminé avant la simulation

`vis = mean(Y_r) / mean(Y_m)` avec `r = R⊙j` et `t' = (1−R)⊙i` donne exactement

```
vis = R̄ρ / ((1 − R̄) + R̄ρ)        avec ρ = Y_j / Y_i
```

Vérifié à 6·10⁻¹⁴ près sur les 94 860 points de `rho_calibration.npz`.

Sur les **105 465 essais réels** de `trials.parquet`, variance expliquée (η², 20 quantiles,
donc capture aussi le non-monotone) :

| variable explicative | `vis` | `ssim` | `ssim_std` |
|---|---|---|---|
| **`dlogY` = log10(Y_reflet) − log10(Y_transmission)** *(connu avant simulation)* | **0,953** | **0,942** | **0,713** |
| theta0 | 0,002 | 0,002 | 0,002 |
| f_px | 0,000 | 0,000 | 0,000 |
| defocus_px | 0,000 | 0,000 | 0,001 |
| ghost_px | 0,000 | 0,000 | 0,001 |
| exposure | 0,157 | 0,132 | 0,200 |

**Tout le tri est décidé à 94-95 % par le choix du couple d'images.** Fresnel, defocus,
fantôme, focale, double vitrage : 0,0 à 0,2 %. Le rejection sampler à 21 essais par
exemple retenu recalcule une quantité que `LOGY[j] − LOGY[i]` donne en nanosecondes.

C'est aussi la réponse à *« ma physique est juste et je choisis les paramètres, pourquoi
ma distribution ne bouge pas ? »* : les paramètres que tu choisis ne pilotent pas la
distribution. Le pool la pilote.

## 2. Ce que le papier utilise réellement (Sec. D.1, D.3)

> *Simulated mixtures are culled if the mean SSIM between m and t falls outside of
> [0.4, 0.94], or if the standard deviation of this SSIM is below 0.05.*

Trois critères, pas sept : moyenne du pixel dans la normale du pool, `SSIM ∈ [0,4 ; 0,94]`,
`std(SSIM) ≥ 0,05`. Ni `vis`, ni `m_mean`, ni `t_std`, ni `r_std`.

Et la justification de `std(SSIM)` est mot pour mot le problème que `vis` prétendait
traiter :

> *the standard deviation of this SSIM image is checked to remove reflections that are
> imperceptible but nonetheless produce a low mean SSIM by **spreading their power
> broadly** (they have low spatial variance).*

Le papier répond à « voile diffus vs reflet localisé » par une statistique **spatiale**.
`vis` est une statistique **d'énergie globale** : par construction il ne peut pas distinguer
les deux. C'est le seul critère du tri dont la variance n'est pas dominée par `dlogY`
(0,713 contre 0,94-0,95) — le seul qui apporte une information propre.

Coût du critère inventé : **4,7 % d'acceptation contre 28,7 %**, soit 6,1× de calcul jeté.

## 3. Les calques de SIR² ne sont pas calibrés en exposition

Régression par image `m = a·t + b·r + c` en linéaire, R² médian **0,99** :

| sous-ensemble | a | **b** | `vis` brut | **`vis` corrigé (×b)** |
|---|---|---|---|---|
| SolidObject | 0,93 | **1,27 – 1,68** | 0,059 | **0,075 – 0,10** |
| Postcard | 0,87 | **0,79** | 0,236 | **0,19** |
| WildScene | 1,03 | **1,05** | 0,085 | **0,10** |

Le calque reflet de SolidObject est capturé 1,3 à 1,7× trop sombre par rapport au mélange.

**Conséquence directe sur la note du 10/09 :** « SIR² n'est pas une distribution mais
trois » (0,059 / 0,085 / 0,236) devient 0,10 / 0,10 / 0,19. L'écart entre sous-ensembles
est divisé par ~3. Et le `vis` médian de `dataset_soomin_final` est de **0,174** — dedans.

> Un soupçon que j'avais et qui est **faux** : le biais de l'estimateur `r̂ = clip(m − t)`
> sous courbe de tons. Testé sur 300 simulés où le vrai `r` est connu : ratio médian
> **1,06** (p10-p90 : 0,95-1,18). Les chiffres Real20/Nature20 du 10/09 tiennent.

## 4. Sur les axes du papier, la couverture est déjà là

384², domaine display, `ssim_fast` identique des deux côtés :

| jeu | n | SSIM p10/p50/p90 | std(SSIM) p10/p50/p90 |
|---|---|---|---|
| **`dataset_soomin_final`** | 700 | **0,836 / 0,912 / 0,947** | **0,083 / 0,136 / 0,210** |
| SIR² SolidObject | 200 | 0,835 / 0,896 / 0,945 | 0,073 / 0,122 / 0,188 |
| SIR² Postcard | 179 | 0,828 / 0,887 / 0,929 | 0,076 / 0,120 / 0,178 |
| SIR² WildScene | 101 | 0,812 / 0,927 / 0,972 | 0,052 / 0,115 / 0,236 |
| Nature20 | 20 | 0,708 / 0,814 / 0,901 | 0,091 / 0,183 / 0,234 |
| Real20 | 20 | 0,593 / 0,764 / 0,880 | 0,151 / 0,213 / 0,274 |

Superposition quasi exacte avec SIR². Les deux jeux *in the wild* sont un cran plus bas
(reflets plus forts), pas plus haut.

Couverture des **520 images de test** par une fenêtre sur SSIM :

| fenêtre (display) | global | Postcard | SolidObject | WildScene | nature20 | real20 |
|---|---|---|---|---|---|---|
| [0,40 ; 0,94] *(papier)* | 87,3 % | 0,99 | 0,90 | 0,57 | 1,00 | 1,00 |
| [0,40 ; 0,97] | **97,7 %** | 1,00 | 1,00 | 0,88 | 1,00 | 1,00 |

`std(SSIM) ≥ 0,05` : vérifié par **98,1 %** des images de test. Filtre valide, non destructif.

C'est le **plafond** qui coûte la couverture, pas le plancher. Le plancher 0,40 ne coupe
rien du tout.

> En domaine **linéaire** (ce que calcule ton `measure`) et hors Postcard, n=341 :
> [0,40 ; 0,94] → 86,5 % ; [0,40 ; 0,97] → 96,5 % ; `std ≥ 0,05` → 96,2 %.
> Les seuils dépendent du domaine : il faut choisir le même des deux côtés.

## 5. Le défaut réel : la concentration, et son amplitude

Part de l'énergie du reflet portée par ses 10 % de pixels les plus brillants (médiane) :

| jeu | top10 |
|---|---|
| SIR² Postcard | 0,241 |
| **SIM** | **0,308 – 0,313** |
| SIR² SolidObject | 0,335 |
| SIR² WildScene | 0,422 |
| Real20 | 0,444 |
| Nature20 | 0,501 |

**Nuance qui manquait au 11/09 :** le déficit de concentration existe contre les jeux
*in the wild* (0,31 contre 0,42-0,50, facteur ~1,4) et **n'existe pas** contre les jeux de
labo (Postcard est moins concentré que la simulation). Le diagnostic « pool trop plat »
est juste, mais il vise Real20/Nature20/WildScene, pas SIR².

Source du déficit, mesurée sur le pool (220 images/groupe, XYZ scene-referred) :

| | sat_frac méd. | % images écrêtées | p99/p50 | **p99,9/p50** |
|---|---|---|---|---|
| indoor | 0,0006 | 65 % | 8,1 | **11,3** |
| outdoor | 0,0001 | 49 % | 6,8 | **9,0** |

Une image FiveK indoor plafonne à ~1 décade au-dessus de sa médiane. Une pièce réelle avec
une lampe : 2 à 4 décades. Les sources sont là (65 % des images ont des pixels écrêtés) mais
coupées au white level, sur 0,06 % de la surface.

Le papier nomme lui-même l'alternative aux IBL (Sec. B.4) :

> *Artificial light sources in HDR images are typically not saturated […] **(or,
> under-exposed RAW images could be used)**.*

## 6. Le vrai problème est le tirage de paires, pas le tri

Écart de luminance de scène du pool : outdoor − indoor = **1,59 décade = 5,3 EV**.
Physiquement correct. Conséquence, par combinaison (105 465 essais) :

| t_loc → r_loc | essais | `vis` méd. | SSIM méd. | std(SSIM) méd. | accept. Adobe |
|---|---|---|---|---|---|
| indoor → outdoor | 57 407 | **0,812** | **0,354** | 0,179 | 29,5 % |
| outdoor → indoor | 33 877 | **0,015** | **1,000** | **0,001** | 10,9 % |
| indoor → indoor | 776 | 0,159 | 0,920 | 0,117 | 34,8 % |
| outdoor → outdoor | 13 405 | 0,165 | 0,930 | 0,095 | 28,8 % |

Les deux combinaisons inter-locations échouent aux deux bouts, **et la physique a raison
dans les deux cas** : à midi on ne voit pas dans une vitrine (I→O), et depuis l'intérieur
on voit très bien dehors (O→I). Ce qui manque à O→I n'est pas de l'énergie mais de la
structure : `std(SSIM) = 0,001`. C'est exactement le maillon HDR.

Effet sur la distribution retenue :

| configuration | SSIM p10/p50/p90 | `vis` p50 |
|---|---|---|
| actuelle (7 critères, `vis ∈ [0,02 ; 0,25]`) | 0,817 / **0,896** / 0,932 | 0,184 |
| culling Adobe + `pair_score` | 0,457 / **0,694** / 0,906 | 0,449 |
| culling Adobe + tirage uniforme `D` du papier | 0,466 / **0,724** / 0,912 | 0,418 |
| **cible : test sets** | 0,82 / **0,90** / 0,95 | 0,10 – 0,24 |

**La fenêtre [0,4 ; 0,94] du papier n'est pas une distribution cible, c'est une boîte de
sécurité.** Adobe la remplit uniformément parce que la moitié de leur pool est intérieure ;
ton pool ne la remplit que par le bas. Passer au tirage uniforme ne corrige presque rien
(0,724 contre 0,694) : O→I ne survit pas au tri.

Et cela explique pourquoi la configuration actuelle marche : `vis ∈ [0,02 ; 0,25]` est,
par accident, un bon proxy de « SSIM autour de 0,90 » — puisque `vis` et `ssim` sont à 94 %
la même variable.

## 7. Composition : ce que fait le papier

> *D = (O × I) ∪ (I × O) ∪ (I × I) − P […] The set O × O is uncommon, and should be
> included sparingly following empirical priors. **We omit them.***

`TARGET` actuel : I→I 30 %, I→O 30 %, **O→I 10 %**, **O→O 30 %**.
Donc 30 % sur la combinaison que le papier exclut, 10 % sur celle qu'il décrit comme
centrale. Avec ton pool (3 489 outdoor / 1 021 indoor), le tirage uniforme du papier donne
**O→I 43,6 % / I→O 43,6 % / I→I 12,8 %** — c'est déjà codé dans `sample_pair` (cellule 56),
non utilisé.

Réserve : 1 021 images indoor pour 100k exemples, c'est ~86 réutilisations par image
indoor. À 850 M paramètres, le risque de mémorisation du calque reflet est réel.

## 8. Ce que je ferais

1. **Supprimer `vis`, `m_mean`, `t_std`, `r_plat`, `pair_score`, `ALL_PAIRS` / `PAIR_WEIGHTS`
   / `CUM_WEIGHTS` (2,6 Go) et le biais `BIAS` sur la géométrie.** Garder
   `SSIM ∈ [0,40 ; 0,97]` + `std(SSIM) ≥ 0,05` + exposition. 97,7 % de couverture des test
   sets, ~29 % d'acceptation.
2. **Piloter la distribution là où elle se décide : le tirage de `(i, j)` par
   `LOGY[j] − LOGY[i]`.** `sample_pair_for_vis` (cellule 33) fait déjà exactement ça.
   Viser la fenêtre en SSIM, pas en `vis`.
3. **Ne pas lancer 100k avec le culling Adobe nu** : SSIM médian 0,72 contre 0,90 pour les
   test sets. Ce n'est pas un élargissement, c'est un déplacement — l'avertissement du 11/09
   était juste, dans la mauvaise unité.
4. **Le PSNR ventilé par sous-ensemble SIR² reste le chiffre qui décide.** Mais le régresser
   sur `std(SSIM)` et la concentration, pas sur `vis`.
5. **Sources indoor à sources ponctuelles** : avant Laval Indoor, tester la piste que le
   papier nomme — sélectionner comme source de reflet les images indoor à `sat_frac` faible
   et `p99,9/p50` élevé (prises sous-exposées, nuit/crépuscule).

## 9. Ce qui tombe des deux notes précédentes

| affirmation | verdict |
|---|---|
| « SIR² n'est pas une distribution mais trois » (0,059 / 0,085 / 0,236) | **faux après correction d'exposition** : 0,10 / 0,10 / 0,19 |
| « seul 10-20 % de ta distribution correspond aux test sets » | **artefact de `vis`** : sur (SSIM, std SSIM) la superposition est quasi exacte |
| « SolidObject à 4,4 % de couverture » | idem |
| « le matériau source plafonne » | **juste**, mais facteur ~1,4 et seulement contre les jeux *in the wild* |
| « garde `ssim_std` » | **juste**, et c'est le seul critère à information propre (η² 0,71 contre 0,94-0,95) |
| « lance les 100k avec la config actuelle (30/30/10/30) » | **à ne pas faire** : 30 % sur O×O que le papier omet, 10 % sur O×I qu'il juge central |
| « `vis_max` 0,25 → 0,5, acceptation attendue ~40 % » | **à ne pas faire** : déplace le SSIM médian de 0,90 à ~0,72 |
| biais de `r̂ = clip(m − t)` sous courbe de tons | **négligeable** (ratio médian 1,06) — soupçon à moi, infirmé |

---

*Mesures reproductibles : `trials.parquet` (105 465 essais), `pool_luminance.npz`,
`rho_calibration.npz`, `dataset_soomin_final` (700 tirés), les 520 images de
`robustsirr_test_dataset`.*

---

# Addendum — `std(SSIM)` sur reflet flou, et le cas de `O × O`

## A. Le reflet flou pénalisé par `std(SSIM)` : mécanisme réel, amplitude négligeable

Objection : une scène sombre avec de grosses lampes très défocalisées produit un voile
large, parfaitement visible, mais à faible variance spatiale de SSIM.

Mesure, à couple source comparable (12 quantiles de `dlogY`, quartiles de `defocus_px`) :

| quartile de defocus | `ssim_std` médian |
|---|---|
| q1 (net) | 0,1232 |
| q2 | 0,1213 |
| q3 | 0,1192 |
| q4 (flou) | 0,1147 |

Écart net → flou : **−0,0085**, contre un seuil à 0,05. Le mécanisme existe, il est six fois
trop petit pour mordre.

Et le coût réel du critère : parmi les essais qui passent déjà `ssim ∈ [0,40 ; 0,94]`,
`std(SSIM) < 0,05` en rejette **0,04 %**. Ceux-là sont effectivement plus flous
(`defocus_px` médian 3,64 contre 2,66). Le critère est gratuit.

> Réserve : ce constat vaut **pour ce pool-ci**. Le scénario décrit exige des sources
> ponctuelles non écrêtées, que FiveK n'a pas. Avec des IBL HDR, la combinaison
> « source ponctuelle + gros defocus » deviendrait fréquente et `std(SSIM)` pourrait se
> mettre à rejeter de vrais cas. À re-mesurer à ce moment-là, pas avant.

## B. `O × O` : le bon argument n'est pas celui du réalisme

### Ce que produit chaque combinaison, après culling `[0,40 ; 0,97]` + `std ≥ 0,05`

| combinaison | acceptation | n | SSIM p10/p50/p90 | `vis` p50 | `std` p50 |
|---|---|---|---|---|---|
| indoor → indoor | 40,3 % | 313 | 0,514 / 0,794 / 0,947 | 0,322 | 0,211 |
| indoor → outdoor | 31,5 % | 18 109 | 0,451 / **0,676** / 0,923 | 0,466 | 0,237 |
| outdoor → indoor | 13,2 % | 4 481 | 0,506 / **0,829** / 0,955 | 0,290 | 0,192 |
| **outdoor → outdoor** | 34,2 % | 4 581 | 0,530 / **0,820** / 0,950 | 0,311 | 0,186 |
| **cible : 520 images de test** | | 520 | 0,780 / **0,859** / 0,937 | | 0,146 |

`O × O` et `O → I` sont les deux combinaisons dont la médiane tombe le plus près de la
cible. À fenêtre large, l'ajustement de composition choisit **`O × O` à 100 %** — mais avec
une distance L1 résiduelle de 0,82, c'est-à-dire : la moins mauvaise de quatre mauvaises.

### La fenêtre est le levier, la composition est du second ordre

Recherche jointe fenêtre × composition contre l'histogramme SSIM des 520 images de test :

| L1 | fenêtre SSIM | I→I | I→O | O→I | O→O | acceptation |
|---|---|---|---|---|---|---|
| **0,358** | **[0,78 ; 0,96]** | 0 | 100 | 0 | 0 | 9,9 % |
| 0,367 | [0,78 ; 0,94] | 0 | 80 | 20 | 0 | 7,9 % |
| 0,391 | [0,75 ; 0,94] | 0 | 90 | 0 | 10 | 10,7 % |
| 0,431 | [0,75 ; 0,96] | 0 | 100 | 0 | 0 | 11,4 % |
| 0,792 | [0,40 ; 0,94] | 0 | 0 | 0 | 100 | 28,8 % |
| 0,824 | [0,40 ; 0,97] | 0 | 0 | 0 | 100 | 34,2 % |

**Resserrer la fenêtre fait passer la distance de 0,82 à 0,36. Changer la composition à
fenêtre fixe la déplace de 0,03 à 0,11.** Le choix `O × O` vs le reste pèse un ordre de
grandeur de moins que le choix des bornes.

À noter aussi : à fenêtre resserrée le vainqueur devient `I → O`, pas `O × O`. Le classement
des combinaisons dépend entièrement de la fenêtre — ce n'est donc pas une propriété stable
sur laquelle fonder la composition.

Résidu de la meilleure config : 38,7 % de la masse dans [0,90 ; 1,00] contre 27,7 % visé.
La simulation garde trop de reflets très faibles à l'intérieur de la fenêtre.

### Le vrai argument pour `O × O` : la diversité de contenu

Réutilisation moyenne d'une image source, pour 100 000 exemples
(pool : 1 021 indoor, 3 489 outdoor) :

| composition (I→I / I→O / O→I / O→O) | indoor | outdoor |
|---|---|---|
| tirage papier 12,8 / 43,6 / 43,6 / 0 | **110×** | 25× |
| `TARGET` actuel 30 / 30 / 10 / 30 | 98× | 29× |
| 15 / 25 / 20 / 40 | **73×** | 36× |
| 10 / 20 / 20 / 50 | **59×** | 40× |

Le pool outdoor est 3,4× plus divers. Chaque point de `O × O` retire de la pression sur les
1 021 images indoor. Pour un DiT à 850 M paramètres, c'est un argument autrement plus solide
que l'appariement de distribution — et il ne dépend d'aucune hypothèse sur les benchmarks.

### Sur le « pas accurate dans la vraie vie »

Le papier écrit *« The set O × O is uncommon »* — c'est un prior déclaré, pas une mesure. Le
verre en extérieur (abribus, garde-corps, vitrines photographiées depuis la rue, pare-brise,
serres) est courant, et l'inspection visuelle des calques de WildScene / Real20 / Nature20
montre du contenu extérieur des deux côtés dans plusieurs images. La pénalité de réalisme
de `O × O` est plus faible qu'annoncée.

## C. Conclusion de l'addendum

1. Garder `std(SSIM) ≥ 0,05` : coût 0,04 %. Le re-mesurer si des sources HDR entrent dans
   le pool.
2. Monter `O × O` est raisonnable — mais pour la diversité de contenu et la réutilisation,
   pas pour l'appariement de distribution, où il n'est vainqueur qu'à fenêtre large.
3. **Le paramètre à régler en priorité est la fenêtre SSIM, pas la composition.**
   `[0,78 ; 0,96]` réduit la distance aux test sets de moitié, à ~10 % d'acceptation.
4. Appariement de distribution ≠ maximisation de score. La chose qui nuit de façon fiable
   est un **déplacement** (n'entraîner que sur des reflets forts fait sur-effacer le
   modèle) ; couvrir le support et pondérer au chargement reste plus sûr que reproduire la
   forme exacte de la cible.

---

# Addendum 2 — pourquoi pas la fenêtre d'Adobe telle quelle

La recommandation `[0,65 ; 0,97]` de l'addendum 1 était mal justifiée. Correction.

## La fenêtre et la bande `dlogY` sont des substituts

Recherche jointe (cible = histogramme SSIM des 520 images de test) : **toutes les fenêtres
atteignent L1 ≈ 0,29-0,39 une fois leur bande `dlogY` optimisée.** La fenêtre n'est pas ce
qui décide ; elle échange contre la bande.

| fenêtre SSIM | bande `dlogY` optimale | L1 | rendement | **% du pool de paires** |
|---|---|---|---|---|
| [0,78 ; 0,96] | [+0,20 ; +0,90] | **0,289** | 46,5 % | **12,8 %** |
| [0,55 ; 0,96] | [+0,20 ; +0,50] | 0,357 | 74,0 % | 5,3 % |
| [0,65 ; 0,97] | [+0,30 ; +0,60] | 0,361 | 76,2 % | 5,4 % |
| **[0,40 ; 0,94]** *(Adobe)* | [+0,20 ; +0,50] | 0,369 | 66,3 % | 5,3 % |

Donc **la fenêtre d'Adobe convient parfaitement** — il suffit de resserrer la bande. Ce qui
la disqualifie n'est pas la qualité de l'appariement, c'est le prix payé en diversité.

## Le prix : la diversité de partenaires

Nombre d'images `j` admissibles pour un `i` donné :

| bande `dlogY` | % paires | partenaires p10 / méd. / p90 | **images sans partenaire** |
|---|---|---|---|
| [+0,20 ; +0,50] | 8,1 % | 2 / 93 / 540 | **2,3 %** |
| [+0,20 ; +0,70] | 13,0 % | 3 / 152 / 888 | 0,3 % |
| **[+0,20 ; +0,90]** | 17,4 % | **4 / 201 / 1192** | **0,1 %** |
| [−0,26 ; +1,17] | 35,9 % | 31 / 524 / 2159 | 0,0 % |

Resserrer la bande **affame les extrêmes du pool** : à [+0,20 ; +0,50], 2,3 % des images
n'ont aucun partenaire et le décile inférieur en a 2. Resserrer la fenêtre ne coûte que du
calcul. À 850 M paramètres, on paie en calcul.

## Les deux vraies raisons de différer d'Adobe

1. **Le plancher.** Leur modèle de base reçoit une photo contextuelle `c` du décor reflété,
   qui module les features (Sec. E.1). À SSIM 0,4 la transmission est quasi détruite : ils
   ont `c` pour désambiguïser. `dataset_sim.py` ne charge que `m` et `t` — pas de `c`. Leur
   plancher n'est pas transférable.
2. **Le plafond.** 0,94 → 0,97 fait passer la couverture des 520 images de test de 87,3 % à
   97,7 % (WildScene a beaucoup de reflets très faibles au-dessus de 0,94), pour un coût
   nul. Ces exemples quasi-identité apprennent au modèle à ne pas sur-effacer.

Ce qui reste vrai de l'addendum 1 : la fenêtre d'Adobe **n'est pas une distribution cible**,
c'est une boîte de sécurité. Ce qui la remplit, c'est le pool. Avec les paires non filtrées,
`[0,40 ; 0,94]` donne un SSIM médian de 0,694 contre 0,859 pour les test sets — d'où la
bande.

## Réglage retenu

Bande `dlogY ∈ [+0,20 ; +0,90]`, fenêtre `[0,65 ; 0,97]` :
SSIM p10/p50/p90 = **0,701 / 0,830 / 0,932** contre **0,780 / 0,859 / 0,937** visé,
rendement 70 %, dont 66 % dans [0,78 ; 0,96] — assez pour resserrer au chargement.
`[0,78 ; 0,96]` donne le meilleur appariement (L1 0,289) à 47 % de rendement, si tu préfères
trancher à la génération.

---

# Addendum 3 — `LOGY` n'est pas `vis`, et la bande n'est pas un critère

## `LOGY` ne dérive pas de `vis`

`LOGY[k] = log10(Y_scene)` de l'image `k` du pool : la luminance moyenne scene-referred,
`pixel / e`, calculée une fois depuis le RAW (`pool_luminance.npz`). Elle ne dépend d'aucune
simulation, d'aucun mélange, d'aucun critère.

Le sens de la dépendance est l'inverse de celui supposé : **c'est `vis` qui s'est révélé
être une fonction de `LOGY`** (η² = 0,95). C'est pour ça que `LOGY` apparaît dans le
diagnostic — pas parce qu'il en dérive. Supprimer `vis` ne rend pas `LOGY` caduc.

## Le critère, c'est SSIM. La bande est un accélérateur.

Deux objets distincts, appliqués à deux moments :

| | quand | rôle |
|---|---|---|
| `measure` : SSIM, std(SSIM), exposition | **après** la simulation | garde ou jette |
| bande `dlogY` | **avant** la simulation | décide quelles paires valent la peine d'être simulées |

La bande ne décide rien sur la qualité. Elle évite de simuler des couples condamnés
d'avance — et comme `dlogY` explique 94 % de SSIM, elle le fait bien.

## Combien elle coûte, mesuré

Critère fixé à `SSIM ∈ [0,65 ; 0,97]` + `std ≥ 0,05` + exposition :

| bande `dlogY` | L1 | **exemples valides perdus** | rendement | rendement `O→I` |
|---|---|---|---|---|
| aucune | 0,698 | 0,0 % | 16,0 % | 9,9 % |
| [−1,00 ; +1,60] | 0,695 | 0,5 % | 34,9 % | 29,9 % |
| **[−0,50 ; +1,30]** | 0,677 | **3,9 %** | **47,8 %** | **46,6 %** |
| [0,00 ; +1,10] | 0,608 | 21,5 % | 62,2 % | 66,7 % |
| [+0,20 ; +0,90] | 0,530 | **44,0 %** | 70,1 % | 74,4 % |

`[−0,50 ; +1,30]` est un accélérateur quasi pur : **3,9 % d'exemples valides perdus pour
×3,0 de calcul**, et le déblocage de `O→I` (9,9 % → 46,6 %), qui sinon fait caler le
mécanisme de quotas.

Au-delà, la bande cesse d'accélérer et se met à filtrer. `[+0,20 ; +0,90]` améliore bien le
L1 (0,530 contre 0,698) — mais en **jetant 44 % des exemples valides**. Or la même forme
s'obtient gratuitement au chargement, en gardant les exemples sur disque.

## Jeter à la génération coûte plus que pondérer au chargement

Taille d'échantillon effective (Kish) après pondération vers l'histogramme cible :

| bande | n retenus | **ESS après pondération** |
|---|---|---|
| aucune | 16 886 | 10 803 |
| **[−0,50 ; +1,30]** | 16 235 | **10 638** |
| [+0,20 ; +0,90] | 9 450 | 7 111 |

À couverture de cible identique (82,1 % des bins, 96,3 % de la masse dans les trois cas),
la bande large laisse **1,5× plus d'exemples effectifs**. Resserrer à la génération est
strictement perdant.

## Correction de l'addendum 2

`DLOG_LO, DLOG_HI = +0,20, +0,90` était un mauvais réglage : il utilisait la bande pour
façonner la distribution, travail qui revient au `WeightedRandomSampler`. Valeur retenue :
**`−0,50, +1,30`**, et la forme finale se règle au chargement.
