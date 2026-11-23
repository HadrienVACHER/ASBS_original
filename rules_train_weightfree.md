# Étape B : correction des masses à l'entraînement, sans poids (darting) contre par poids de trajectoire (IS tempéré)

Écrit et commité AVANT la location et avant tout résultat de l'étape B. Aucune règle n'est modifiée après coup. Condition remplie : A1 vraie (`rules_darting_large_sigma_snapshots.md`, bd7743c ; G et H_saut_bassin vrais sur 3 instantanés sur 3 à d = 128 et à d = 256).

## Cellules et départ (graine 0, reprise de 500 époques, recette `gmm_dk_as`)
- **Famille 1 (IS qui marche)** : GMM-dk (8, 8, 8), σ_max × 0,5 = 2,236068, et (32, 8, 6), × 0,5 = 1,25. Contrôleurs de départ : ceux du test d'entraînement précédent (`results/gmm_dk_cont/*/controller_start.pt`, finals de la graine 0 à × 0,5).
- **Famille 2 (IS qui s'effondre)** : GMM-dk (128, 8, 6), σ_max × 0,75 = 1,602001, depuis l'**époque 1500** (`figures/gmmdk_step4/controllers/gmmdk4_d128_sx0.75_ep1500.pt`).
- **Buffer initial, le même pour les trois bras d'une cellule** : 10⁴ points terminaux frais du contrôleur de départ, simulés sur CPU avec leurs log W de chemin (`scripts/wf_make_start.py`, graine 7 ; `results/wf_start/`). ESS normalisé de départ des poids W / W^0,5 : 0,58 / 0,86 (d8), 0,62 / 0,90 (d32), 0,0030 / 0,089 (d128).
- Hyperparamètres d'origine (FourierMLP 4 × 128, lr 1e-4, buffer 10⁴, 100 itérations par époque, 512 nouveaux échantillons par époque, nfe 200), tous les t (t_focus = 0). Évaluation toutes les 50 époques sur 10⁴ échantillons, échantillons d'évaluation sauvegardés, **contrôleurs sauvegardés aux époques 100 à 500 et en fin de run**.

## Bras (9 runs)
- **(a)** AS seul (`gmm_dk_as`).
- **(t)** AS + IS tempéré, α = 0,5 (`gmm_dk_as_tempered_is`, `AdjointVETemperedISMatcher`). Le log W de chaque trajectoire est stocké à sa génération ; le poids de régression vaut W^α normalisé par la moyenne de W^α sur **tout le buffer courant**. Le minimiseur en population est la mise à jour d'AS sur q_α ∝ q^{1−α} π^α (`rules_tempered_is_quadrature.md`). Tests `tests/test_tempered_is_matcher.py` : T1–T4 réussis.
- **(d)** AS + darting, forme identité, **γ = 1,75**, la dose théorique 2(M − 1)/M pour M = 8 (mémo §2.2) (`gmm_dk_as_darting`, `AdjointVEGMMDartingMatcher` inchangé : minima découverts sur le buffer, rafraîchis toutes les 25 époques).
- Essais de 2 à 3 époques sur CPU pour (t) en d = 8 et pour (d) en d = 128 : sans erreur.

## Mesures (analyse locale sur CPU après rapatriement)
- **TV brut sans poids**, par Voronoï (règle) et par bassin (rapporté ; bassins ≈ cellules sous π pour Δ ≥ 6), **moyenné sur les 3 dernières évaluations** (époques 350, 400 et 450, 10⁴ échantillons chacune). Les échantillons de fin de run (10⁴) sont rapportés à côté.
- **Couverture** : nombre de cellules avec p̂ ≥ 0,25 p.
- **Q1** : rapport de la distance médiane au minimum le plus proche à celle de π, quench et minima de `basin_lib`, moyenné sur les 3 mêmes évaluations.
- **ESS et k̂ des poids d'entraînement de (t)** (statistiques du matcher, sur le buffer).
- **Énergies par mise à jour** (code et configurations, et statistiques des matchers).

## Règles
- **F1 (famille 1)** : TV brut(d) ≤ 0,5 × TV brut(a), **dans les deux cellules**.
- **F2 (famille 2, la question clé)** : TV brut(d) ≤ 0,5 × TV brut(a) ET TV brut(d) ≤ 0,5 × TV brut(t).
- **F3 (sûreté)** : le Q1 (rapport de distance) de (d) et de (t) est au plus 10 % plus mauvais que celui de (a), **dans chaque cellule**.
- **Descriptif, prédiction écrite d'avance** : (t) se dégrade en famille 2 (ESS minuscule). L'étape A a mesuré à × 0,75, d = 128 : Var(log W) de 14 à 19 nats², dont 0,2 % entre bassins, et un ESS de W^0,5 de 0,03 à 0,08.
- **Lecture fixée d'avance** : F2 vrai = la correction sans poids marche à l'entraînement là où la correction par poids de trajectoire échoue, et c'est la contribution à défendre.
- Variabilité (règle permanente) : moyennes temporelles sur 3 évaluations (écart-type rapporté). Une graine d'entraînement, ce qui est rapporté comme limite.
- **Mes prédictions** (sans effet) : F1 vraie à 40 % ; F2 vraie à 55 % ; F3 vraie à 70 %. (t) ≈ (a) en famille 2.

## Coût, plafond et arrêts (accord explicite : RTX 3070 ≈ 0,134 $/h, plafond dur 0,20 $)
- Offre RTX 3070 ≤ 0,135 $/h, fiabilité > 0,99, ≥ 12 cœurs CPU ; sinon pas de location.
- Estimation : 9 runs, au plus 6 en parallèle, donc 2 vagues (vague 1 : les 3 runs d128 et les 3 runs d8 ; vague 2 : les 3 runs d32). Temps de référence : test d'entraînement précédent, 6 runs de 500 époques en environ 12 min ; d = 128 à environ 0,7 s par époque à 4 en parallèle à l'étape 4 ; darting à d = 128 sur CPU 1,5 s par époque. 500 époques à environ 1,0 s, soit environ 9 min par vague, **pessimiste 2 s, environ 17 min** ; démarrage 4 min ; rapatriement 2 min. **Total ≈ 25 min ≈ 0,06 $ (central) ; ≈ 40 min ≈ 0,09 $ (pessimiste)**, sous 0,20 $.
- **Aucun nouveau run lancé après la minute 60** ; **arrêt d'urgence à la minute 80** : tout est arrêté, les résultats terminés et les contrôleurs sont rapatriés, l'instance est détruite, puis l'utilisateur est prévenu ; création → destruction ≤ 85 min (≤ 0,19 $). Jamais de question pendant que l'instance tourne ; destruction dès les résultats rapatriés.

## Exécution
`scripts/vast_train_weightfree_remote.sh` (instance) ; `scripts/wf_analyse.py` (local, CPU, après rapatriement). Résultats et contrôleurs dans `figures/train_weightfree/`. Section dans RESULTS.md.
