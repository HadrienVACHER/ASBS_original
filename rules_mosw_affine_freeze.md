# MoS-w post hoc : forme du buffer corrigé (translation et affine, S ∈ {10, 50, 200}) ; pas MALA figé

Écrit et commité AVANT tout calcul (points 2 et 3 du message de l'utilisateur du 2026-11-23). CPU uniquement, aucune location ; chaque exécution < 1 min (sous-échantillonnage déclaré). Aucune règle n'est modifiée après coup. **Métrique de masse principale : TV par bassin** ; Voronoï en secondaire.

## Partie M : diagnostic MoS-w post hoc
- **Données** : points terminaux bruts de (a) à l'instantané de l'époque 250 (`results/mosw_step3/step3_results/d10/endpoints_d10_ep250.npy`), **sous-échantillonnés aux 5000 premiers** (déclaré, pour le temps). **Banque existante** : `figures/mosw_posthoc/bank_d10.npy` (10 minima, 107 départs).
- **Bras** : S ∈ {10, 50, 200} × {**translation** (`force_translation`), **affine** (Hessiennes de la banque, option par défaut de `dart_lib` quand elles diffèrent)}, soit 6 cellules, pas MALA réglé comme d'habitude.
- **Mesures** du buffer corrigé : **Q1** (rapport de la distance médiane au minimum le plus proche à celle de π, 5,23 ; quench et minima de `basin_lib`) et **TV par bassin** (principale), couverture, TV de Voronoï ; acceptation des sauts ; coût.
- **Règle M1** : **au moins une des 6 cellules atteint Q1 ≤ 1,5 ET TV par bassin ≤ 0,02.** On rapporte aussi le plus petit S qui y arrive, et l'arc (Hessiennes égales ou non à 10⁻⁶ près : si elles sont égales, les deux bras sont identiques, et on le dit).
- **Précédent déclaré avant le calcul** : l'étape A4 (`rules_mosw_posthoc.md`, c086793) a mesuré, sur ce même instantané avec la banque et les 10⁴ points, à **S = 40** : Q1 = 1,2 et TV par bassin 0,0078 ; à S = 10 : Q1 = 2,2 et TV 0,0185. M1 est donc très probablement vraie à S = 50 ou 200 ; l'information de ce diagnostic est le S requis et la différence entre bras.
- **Mes prédictions (sans effet)** : M1 vraie 90 % (à S = 50 ou 200, pas à 10) ; les deux bras identiques à plus de 90 % pour Q1 (Hessiennes presque égales pour des Student identiques) 60 %.
- **Lecture** : M1 vraie = le buffer peut avoir la bonne forme à S assez grand ; l'échec de forme de l'auto-entraînement (Q1 buffer 4,26 à S = 10) vient alors du S choisi, et se règle par S plus grand ou par un pas MALA adapté, à un coût proportionnel à S. M1 fausse = ni la translation ni l'affine ne suffisent à S ≤ 200.

## Partie F : pas MALA figé (réglé une fois au départ)
- **Mise en œuvre** : `dart_lib.darting(..., mala_step=)` (pas donné, aucun réglage, coût de réglage non compté) ; `AdjointVEDartBankMatcher(freeze_step=True)` règle le pas au premier lot puis le réutilise. Défaut de la classe : `False` (les runs de ed2617a ont été faits avec réglage à chaque lot) ; la configuration `adjoint_ve_dartbank` met `freeze_step: true` pour les lots futurs.
- **Vérification (règle F1)** : dans chacune des 3 cellules ((32, 8, 8) × 0,5, MoS-w d = 10, (128, 8, 6) × 0,75), le pas réglé sur le **premier lot** (500 premiers points du fichier de départ `results/wf_start/<cellule>/start_samples.pt`, banque de la cellule) est comparé au pas réglé sur **3 autres lots de 500 points** : les 500 premiers points des échantillons d'évaluation du run (s) aux époques 50, 250 et 450. **F1 vraie si les 9 rapports pas_figé / pas_réglé sont dans [0,9 ; 1,1].** La dichotomie du réglage a une résolution d'environ ±4,5 % (8 pas sur 23 unités de log).
- **Lecture** : F1 vraie = figer est sans effet notable sur le pas utilisé, donc sur ce qui est calculé, et divise le coût de (s) d'environ 40 % ; F1 fausse = le réglage varie, on garde le réglage à chaque lot (`freeze_step: false`) et on le dit.

## Exécution
`scripts/mosw_affine_diag.py` (commandes `M ARM S`, `F CELL`, `verdict`). Résultats dans `figures/mosw_affine/`. Section dans RESULTS.md.
