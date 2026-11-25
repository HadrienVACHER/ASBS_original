# Dernier lot : confirmation de la correction de forme à (128, 8, 6) × 0,75

Écrit pour : Hadrien. Écrit et commité **avant tout calcul de ce lot** ; aucune règle ne sera modifiée après coup (amendements datés seulement). Suite de `rules_shape_correction.md` (891328b, 1074394) et `rules_shape_correction_2.md` (0ab9cb0, f7334b1). **Métrique de masse principale : TV par bassin.**

## Montage commun
Cellule GMM-dk (128, 8, 6), σ_max = 1,602001, FourierMLP 4 × 128, lr 1e-4, buffer 10⁴ (tirages exacts ou déformés, 512 neufs par époque), nfe 200, **départ commun** `results/wf_start/d128/controller_start.pt` ; matchers et mesures inchangés (`gmm_dk.shape_measures`, directions de la graine 2027, indépendantes de la graine d'entraînement) ; `tests/test_shape_matcher.py` et `tests/test_oracle_dsm_matcher.py` réussis.

## Volet 1 : largeur, 3 graines (0, 1, 2)
- Buffer **(V)** (covariance × 1,44, masses et moyennes exactes) : AS et DSM ; **oracle** (buffer exact) AS et DSM sur les mêmes graines, pour les planchers ; 500 époques, évaluation toutes les 50 époques.
- **Graine 0 réutilisée** (`sd_V_as_d128`, `sd_V_dsm_d128`, `sd_o_d128`, `sd_od_d128`, RTX 3070) : même configuration, mêmes matchers (seule différence de code depuis : une ligne d'affichage et une factorisation du tirage sans changement des tirages). Les graines 1 et 2 (8 runs) sont faites sur le cluster, donc **sur un autre matériel** ; pour une graine donnée, le bras et son plancher sont sur le même matériel.
- **ρ_V** : moyenne des évaluations des époques 350, 400, 450 (écart entre évaluations rapporté).
- **Règle W** : pour **chacune des 3 graines** : ρ_V(AS, V) − ρ_V(oracle AS) **≤ 0** ET ρ_V(DSM, V) − ρ_V(oracle DSM) **≥ +0,30**. W vraie si les 6 conditions sont vraies.
- **Descriptif** : moyenne et écart-type (3 graines) des deux écarts au plancher, contre la prédiction de population **−0,10** (AS) et **+0,44** (DSM) ; TV par bassin, Voronoï, Q1, ρ_M de tous les runs.

## Volet 2 : position, décalage 1,0, entraînement long, graine 0
- Buffer **(M')** (moyenne de chaque mode décalée de 1,0 σ_tar, covariance et masses exactes) : AS et DSM, départ commun, **2000 époques**, évaluation toutes les 250 époques (0 à 1750, plus 10⁴ échantillons de fin de run à 2000), contrôleurs sauvegardés aux époques 500, 1000, 1500, 2000.
- **T (témoin)** : ρ_M(DSM) **≥ 0,8 à la fin** (échantillons de fin de run, époque 2000). La moyenne des 3 derniers jeux (époques 1500, 1750, fin) est rapportée à côté ; si elle donne un autre verdict, c'est dit, le verdict restant celui de la fin de run.
- **Si T est faux : test non concluant, sans lecture sur AS.**
- **Si T est vrai : P2'** : ρ_M(AS) **≤ 0,3** (fin de run). **Descriptif** : ρ_M(AS) contre A_m = −0,54 ; trajectoires de ρ_M et ρ_V aux 9 évaluations ; TV par bassin.
- **Précision déclarée** (comme au lot précédent, aucun seuil ajouté) : un apprenant qui ignorerait le buffer donne ρ_M(AS) ≈ −0,05 et passe P2' ; « AS corrige » est dit si ρ_M(AS) est de plus négatif au-delà du plancher, sinon la formulation est « AS ne recopie pas ».

## Lecture fixée d'avance
- **W vraie** : la correction de largeur est confirmée sur 3 graines.
- **T et P2' vraies** : AS corrige aussi la position ; le découplage est complet dans le régime |A_m| < 1.
- W fausse, T fausse, ou P2' fausse : dit tel quel, avec les valeurs par graine, sans lecture ajoutée.

## Limites annoncées
Une graine pour la position ; un seul décalage et un seul facteur de largeur ; départ commun entraîné par AS (pour le DSM aussi) ; graine 0 et graines 1–2 sur des matériels différents ; masses exactes.

## Mes prédictions (sans effet)
W vraie 90 % ; T vraie 45 % (36 à 43 % du décalage en 500 époques ; à d = 8, 0,86 puis 0,89) ; si T vraie, P2' vraie 70 %.

## Calcul : cluster (pas de location)
- **Estimation sur RTX 3070 : environ 45 à 55 min de GPU** (8 runs de 500 époques en parallèle ≈ 18 min à environ 2 s par époque ; 2 runs de 2000 époques ≈ 28 min à 0,84 s par époque à 2 en parallèle ; plus 5 min d'installation), **au-delà de 30 min : pas de location** (consigne de l'utilisateur). Le job `scripts/pbs_shape_final.pbs` est soumis par l'utilisateur.
- Analyse locale après rapatriement : `python scripts/shape_analyse.py final` (CPU < 1 min). Résultats : `figures/shape_correction/analysis_final.json` ; section dans RESULTS.md.
