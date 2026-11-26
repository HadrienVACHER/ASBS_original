# Complément « correction de forme » : décalage δ = 1,0 σ_tar, deux départs, à (128, 8, 6) × 0,75

Écrit pour : Hadrien. Écrit et commité **avant** la location et avant tout résultat de ce lot ; aucune règle ne sera modifiée après coup (amendements datés seulement). Suite de `rules_shape_correction.md` (891328b ; résultats 1074394). **Étiquette de l'utilisateur, 2026-11-24 : l'ancien verdict de P2 (faux, 1074394) est conservé tel quel et étiqueté I** (témoin DSM non convergé : ρ_M(DSM) = 0,298, soit environ 43 % du décalage au-dessus du plancher). Métrique de masse principale : TV par bassin.

## Question
Le décalage de 0,3 σ_tar n'a été recopié qu'à moitié par le DSM : le test sur la position était non concluant. Avec un décalage plus grand (1,0) et, en plus, un départ sans mémoire d'AS, le DSM recopie-t-il la position du buffer (témoin), et AS la corrige-t-il (régime |A_m| = 0,54 < 1) ?

## Montage
- Cellule, architecture, optimiseur, 500 époques, nfe 200, graine 0, évaluations, matchers : **identiques à `rules_shape_correction.md`** (`AdjointVEShapeASMatcher` et `AdjointVEShapeDSMMatcher`, tests réussis).
- **Buffer (M')** : masses exactes, covariance exacte (I), moyenne de chaque mode décalée de **δ = 1,0 σ_tar** (σ_tar = 1) le long de la **même direction unitaire par mode** (graine 2027) ; 10⁴ tirages au départ puis 512 tirages neufs i.i.d. par époque.
- **Deux départs** : **(i)** le contrôleur commun actuel (`results/wf_start/d128/controller_start.pt`, entraîné par AS) ; **(ii)** contrôleur **réinitialisé** (poids aléatoires de l'initialisation par défaut, graine 0, aucun `init_controller`). Même buffer de départ pour (i) et (ii).
- **4 bras** : AS et DSM, pour chaque départ. Une graine.
- **Mesures** (comme avant, moyennes des évaluations des époques 350, 400, 450, écart entre évaluations, fin de run à côté) : ρ_M = moyenne sur les 8 modes de la projection du décalage de la moyenne des échantillons du mode sur u_k, **divisée par δ = 1,0** ; ρ_V ; TV par bassin (principale), Voronoï, Q1. Planchers de ρ_M sur buffer exact, ramenés à δ = 1 : o-AS −0,049, o-DSM −0,039 (bruit de mesure de l'ordre de 0,02), à la calibration des runs oracle (`calibration.json`).

## Règles (de l'utilisateur)
- **T (témoin)** : ρ_M(DSM) ≥ 0,8 sur **au moins un départ**. **Si T est faux, le test est déclaré non concluant, sans lecture sur AS.**
- **P2' (conditionnelle à T)** : sur **le même départ** que celui où T est vrai, ρ_M(AS) ≤ 0,3. (Si T est vrai sur les deux départs, P2' est rapportée pour chacun, et lue sur chacun.)
- **Descriptif** : ρ_M(AS) comparé à A_m = −0,54 ; ρ_V et TV par bassin pour les 4 bras.
- **Lecture fixée d'avance** : **T vraie et P2' vraie** = AS corrige aussi la position (découplage masses/forme complet dans le régime |A_m| < 1) ; **T vraie et P2' fausse** = AS ne corrige que la largeur. Précision déclarée : un apprenant qui **ignorerait** le buffer donnerait ρ_M(AS) ≈ −0,05 et passerait P2', comme à la fois précédente ; la lecture « corrige » demande donc, en descriptif, que ρ_M(AS) soit **négatif au-delà du plancher** (vers −0,54), sinon « ne recopie pas » est la formulation retenue ; cette précision n'ajoute aucun seuil.
- **Si le budget ne tient pas** : seulement le départ (ii) (2 bras). Il tient (voir ci-dessous).
- **Limites annoncées** : une graine ; 500 époques ; un seul décalage ; masses exactes ; le départ (ii) peut ne pas converger en 500 époques (les TV par bassin le diront) ; ρ_M sans pondération des modes.
- **Mes prédictions (sans effet)** : T vraie 55 % (sur (ii) 45 %, sur (i) 40 % : le DSM n'a recopié que 43 % à 0,3 ; un décalage plus grand est plus facile à lire mais pas forcément à suivre) ; si T vraie, P2' vraie 70 % (AS ne recopie pas ; ρ_M(AS) ≈ −0,54 : 25 %) ; test non concluant 45 %.

## Coût, plafond et arrêts (accord explicite de l'utilisateur : RTX 3070 ≈ 0,134 $/h, plafond dur **0,08 $**)
- Offre RTX 3070 ≤ 0,135 $/h, fiabilité > 0,99, ≥ 8 cœurs CPU ; sinon pas de location. **Même montage que le lot précédent** (4 runs en parallèle) : mesuré 9,5 min d'entraînement et 10,9 min de création à destruction pour 0,018 $.
- **Central ≈ 14 min ≈ 0,03 $ ; pessimiste ≈ 35 min ≈ 0,07 $** à 0,10–0,12 $/h, sous 0,08 $. Si l'offre trouvée rend le pessimiste supérieur à 0,08 $, version réduite (départ (ii) seul).
- **Arrêt d'urgence à la minute 33 depuis la création** : tout est arrêté, les résultats terminés et les contrôleurs sont rapatriés, l'instance est détruite, l'utilisateur est prévenu (≤ 0,075 $ à 0,135 $/h). Jamais de question pendant que l'instance tourne ; destruction dès les résultats rapatriés.

## Exécution
`scripts/vast_shape2_remote.sh` (`sd_Mp_{as,dsm}_{i,ii}_d128`) ; analyse locale `scripts/shape_analyse.py runs2`. Résultats dans `figures/shape_correction/analysis_runs2.json`. Section dans RESULTS.md.

## Amendement daté 2026-11-24 (décision de l'utilisateur, avant la location ; texte original ci-dessus conservé)
- **Version réduite : départ (ii) seulement** (contrôleur réinitialisé), **2 bras (AS et DSM) sur (M')**. Les runs du départ (i) ne sont pas lancés (`scripts/vast_shape2_remote.sh` : 2 runs ; l'analyse `runs2` lit les départs présents).
- **La règle T porte sur le départ (ii) seul** : ρ_M(DSM, départ ii) ≥ 0,8 ; faux = test non concluant, sans lecture sur AS. **P2' (ρ_M(AS) ≤ 0,3, même départ) et la lecture fixée d'avance sont inchangées.** La précision déclarée sur l'apprenant qui ignore le buffer reste valable.
- **Location** : la première location (offre 62266879) a échoué (`success: false`, instance bloquée en « loading », jamais utilisable, détruite après 12 min) : **comptée à l'estimation haute, 12 min × 0,1017 $/h ≈ 0,020 $**, dans le plafond dur de **0,08 $ total**. Accord de l'utilisateur pour une offre RTX 3070 de fiabilité ≥ 0,975 et ≤ 0,095 $/h ; une offre conforme aux critères d'origine (fiabilité > 0,99) est prise de préférence. **Si l'instance n'est pas utilisable dans les 10 min, elle est détruite et une seule nouvelle tentative est faite** (dont le coût est compté de même).
- **Estimation** : 2 runs en parallèle, environ 7 à 10 min d'entraînement ; **central ≈ 0,02 $ de location réussie, pessimiste ≈ 30 min × 0,095 $/h ≈ 0,048 $** ; avec la location ratée, total pessimiste ≈ 0,068 $ < 0,08 $. Avec une seconde tentative (10 min perdues ≈ 0,016 $) : ≈ 0,084 $ en pessimiste, donc arrêt d'urgence avancé (voir ci-dessous).
- **Arrêt d'urgence** : tout est arrêté, les résultats finis et les contrôleurs rapatriés, l'instance détruite, l'utilisateur prévenu, au plus tard à la minute **30** après la création de l'instance utilisée (minute **22** si c'est la seconde tentative), de sorte que le total estimé (ratée comprise) reste ≤ 0,08 $. Détruire l'instance dès les résultats rapatriés ; jamais de question pendant la facturation.
