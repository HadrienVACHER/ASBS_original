# A4 : post hoc sur les 12 contrôleurs MoS-w de l'étape 3 (IS, darting, MALA seul, banque de minima)

Écrit et commité AVANT tout calcul (point A4 du message de l'utilisateur du 2026-11-23 ; ne conditionne pas l'étape B). Question, descriptive : **la perte de modes en cours d'entraînement se rattrape-t-elle après coup ?** CPU uniquement, aucune location, aucun entraînement ; chaque exécution < 1 min (chronométrée ; une commande par contrôleur et par bras). Aucune règle n'est modifiée après coup.

## Contrôleurs et données (locaux, aucune simulation nouvelle)
- MoS-w, d ∈ {10, 50}, σ_max = 23,2, graine 0, 1500 époques (`rules_mosw_step3.md`) ; instantanés aux époques 250, 500, 750, 1000, 1250 et 1500, soit **12 contrôleurs**.
- Pour chacun : les **2·10⁴ points terminaux** et leurs **log W de chemin**, sauvegardés à l'étape 3 (`results/mosw_step3/step3_results/d{10,50}/endpoints_*.npy`, `is_logw_*.npz`).
- Cible : produit de Student t 1D (ν = 2, s² = 1), 10 composantes, poids inégaux (0,012 à 0,331), moyennes et poids lus dans la configuration du run. Vérité p et plancher (n = 10⁴) : `figures/mog_mos_truth_mosw_d{10,50}.json`.

## Mesures (par bassin, la mesure principale, comme le Q2 de l'étape 3 ; Voronoï rapporté)
- Bassins : trempe et minima de `basin_lib`, identiques à l'étape 3.
- **TV par bassin** contre p ; **couverture** = bassins avec p̂ ≥ 0,25 p ; TV de Voronoï rapporté à côté.

## Bras (un jeu par contrôleur ; la variabilité entre évaluations est celle des 6 instantanés)
1. **Brut** : les 10⁴ premiers points terminaux, comme le Q de l'étape 3.
2. **IS de chemin** sur les 2·10⁴ points : simple, PSIS, tronqué à √n × moyenne ; ESS, k̂.
3. **Darting, protocole d'origine** (`dart_lib.darting`, minima découverts sur les 500 premiers points), sur les 10⁴ premiers points, **S ∈ {10, 40}** (valeurs de l'étape 3 pour le MoS-w).
4. **MALA seul** au même pas et à coût égal à chaque S du bras 3.
5. **Banque de minima + darting** : la banque ne dépend pas du contrôleur, une seule par dimension.
   - **N_q = 1000** tirages de la loi de base N(0, ν₁ I), ν₁ = σ_max² − σ_min², graine 0 ;
   - trempe convergée de `basin_lib.quench`, fusion à moins de 0,5 ;
   - rapportés : le nombre de minima, les vrais minima retrouvés sur 10, le nombre de départs nécessaire pour les trouver tous, les départs par minimum.
   - Puis darting avec `minima=` la banque, mêmes points, même graine, S ∈ {10, 40}.

## Lecture fixée d'avance (descriptive, pas de verdict de règle)
- Un contrôleur est **rattrapé** par un bras si, après ce bras, la couverture par bassin est de **10/10** ET le **TV par bassin ≤ 2 × plancher** (bras 3 à 5 à S = 40 ; bras 2 sur les 2·10⁴ points, meilleure des trois variantes).
- Il est **partiellement rattrapé** si le TV par bassin est au plus la moitié du TV brut.
- **Réponse à la question, par dimension** :
  - « oui » si au moins 4 des 6 instantanés sont rattrapés par la banque + darting ;
  - « en partie » si au moins 4 sont au moins partiellement rattrapés ;
  - « non » sinon.
  On dit aussi quel bras rattrape et à partir de quelle époque.
- Ce qui reste hors de portée est rapporté avec sa cause mesurée : distance aux minima (rapport Q1 de l'étape 3), acceptation des sauts, minima manquants.

## Mes prédictions (sans effet)
- Banque : 10/10 minima à d = 10, 85 % ; à d = 50, 60 % (le mode de poids 0,012).
- d = 10 : « oui » à 60 %. Les formes y sont apprises aux époques 1250 et 1500 (Q1 vrai), et les 10 bassins sont atteints.
- d = 50 : « non » à 80 %. Les points y sont 8 à 10 fois plus loin des minima que sous π. Les sauts sont des translations et gardent cette distance, et l'étape 2 avait mesuré 13 à 28 % d'acceptation.
- L'IS de chemin ne rattrape aucun contrôleur (k̂ 1,3 à 8 à l'étape 3), 85 %.

## Exécution
`scripts/mosw_posthoc.py` (commandes `prep`, `bank D`, `raw_is D E`, `dart D E`, `mala D E`, `dartbank D E`, `summary`). Résultats dans `figures/mosw_posthoc/`. Section dans RESULTS.md.
