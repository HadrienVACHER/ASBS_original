# A3 : banque de minima (trempe depuis la loi de base) + darting post hoc sur GMM-dk (32, 8, 8)

Écrit et commité AVANT tout calcul (point A3 du message de l'utilisateur du 2026-11-23 ; ne conditionne pas l'étape B). CPU uniquement, aucune location ; chaque exécution < 1 min (chronométrée). Aucune règle n'est modifiée après coup.

## Contrôleurs (aucun entraînement)
- **Principal** : GMM-dk (32, 8, 8) à la règle σ_max (× 1 = 2,828427), graine 0, 1500 époques (`results/gmm_dk_sweep/gmmdk_d32_K8_D8_s0`, couverture 3/8 par AS).
- **Descriptifs**, un jeu chacun : × 0,5 (1/8), × 0,75 (3/8), × 1,5 (2/8), × 2 (1/8) (`results/gmm_dk_window/w1/gmmdk_d32_K8_D8_s0_sx*`).

## Banque de minima
- **Départs** : N_q = **1000** tirages de la loi de base p_base = N(0, ν₁ I), avec ν₁ = σ_max² − σ_min² du contrôleur (graine 0). Nombre fixé ici, avant tout calcul.
- **Trempe** convergée de `basin_lib.quench` (‖∇ log π‖ < 10⁻³, au plus 3000 itérations). La banque est l'ensemble des points convergés distincts, fusionnés à moins de 0,5. Une recherche aveugle ne connaît pas les vrais minima : ils ne servent qu'à vérifier la banque.
- **Rapporté** : le nombre de minima de la banque, le nombre de départs qu'il a fallu pour trouver les 8 minima (indice du départ, dans l'ordre des tirages, qui apporte le 8ᵉ minimum distinct), et le coût en évaluations de gradient.

## Darting post hoc avec la banque
- Protocole de `dart_lib.darting` inchangé (Hessiennes, 1 saut + 5 pas MALA par balayage, pas réglé une fois), mais avec **les minima de la banque à la place de la découverte sur les échantillons** (option `minima=`, ajoutée sans changer le reste : vérifié sur un résultat commité, d32 jeu 1, identique). S ∈ {1, 5, 10}.
- Sur 3 jeux indépendants de 10⁴ points terminaux d'AS (graines 11, 12, 13), pour le contrôleur principal.
- **Témoins** : MALA seul à coût égal ; darting avec les minima découverts sur les échantillons (protocole d'origine) ; IS de chemin (TV, ESS).
- **TV par Voronoï** contre la vérité exacte ; **couverture** = cellules avec p̂ ≥ 0,25 p ; plancher à n = 10⁴ (médiane de 100 répétitions).

## Règle A3 (contrôleur principal, darting avec la banque, S = 10, moyennes sur les 3 jeux)
**Couverture 8/8** (dans chacun des 3 jeux) **ET TV (Voronoï) ≤ 2 × plancher.** Les autres S et les contrôleurs descriptifs sont rapportés à côté.

## Mes prédictions (sans effet)
- La banque trouve les 8 minima avec 1000 départs : 90 %.
- A3 vraie : 55 %. La couverture 8/8 est probable (75 %), parce que les sauts proposent les modes absents avec un taux de Barker proche de w_j / (w_j + w_k) si les formes sont justes. Le plancher à S = 10 est moins sûr : il faut assez de balayages pour équilibrer 5 modes partis de zéro.
