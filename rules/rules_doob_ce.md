# Correction de dynamique par poids de chemin (note NTU_25) : PICE / entropie croisée sur trajectoires

Écrit le 2026-11-27, **avant tout calcul**. Origine : note NTU_25 (correction de type h-transform de Doob par coefficient de martingale). C'est en substance **PICE / l'entropie croisée sur les trajectoires** (Kappen & Ruiz 2016 ; Nüsken & Richter 2021 ; références non revérifiées) : consigné comme tel. Méthode plug-in, réglages fixés ci-dessous, aucun réglage par cellule.

## Convention du schéma (vérifiée dans le code avant d'écrire)
Dépôt (`ControlledSDE`, `girsanov.log_path_weights`) : X_{k+1} = X_k + g_k² u(t_k, X_k) Δt_k + g_k ΔW_k, avec ΔW_k = √Δt_k ε_k et g_k = `sde.diff(t_k)`. La demande écrit la dérive σ_k u ; ici elle est g_k² u. **Correction : u_new = u_θ + v_φ / g_k**, soit la dérive g_k² u_θ + g_k v_φ = g_k (g_k u_θ + v_φ). N = 199 pas (grille de 200 points), t uniforme, X_0 = 0.

## Méthode (une itération)
- u_θ figé. M trajectoires sous Q_θ, états X_k et innovations ε_k conservés. ω_i = w̃_i / Σ w̃, log w̃ = −E(X_N) − log r_N(X_N) + log dP⁰/dQ_θ (rapport exact des transitions discrètes).
- v_φ : même architecture que u_θ (FourierMLP 4 × 128, dernière couche à 0). Perte Σ_i ω_i Σ_k Δt_k ‖ΔW_k/Δt_k − v_φ(t_k, X_k)‖². Minibatch sans biais de cette perte : i tiré selon ω (avec remise), k uniforme parmi les N pas, terme Δt_k ‖·‖², moyenne sur le lot.
- **Optimisation (identique pour toutes les cellules)** : Adam, lr 10⁻³ avec décroissance cosinus jusqu'à 0, **30 000 étapes, lot 4096**, pas d'EMA, pas d'écrêtage ; on garde le dernier itéré.
- Contrôleur corrigé sauvegardé (poids de v_φ). Échantillons bruts : 10⁴ trajectoires sous u_new, graine distincte de l'apprentissage.
- Itérations 2 et 3 (descriptives) : même procédé avec u_new figé à la place de u_θ et un nouveau v_φ à 0 ; M = 10⁴.
- Bras (h), descriptif, en dernier : y_{ik} = log ω̃_i − log (dP⁰/dQ_θ)_{0:k} (ω̃ = w̃ non normalisé) ; f_φ (FourierMLP 4 × 128, sortie scalaire, initialisée à 0) minimise Σ_i ω_i Σ_k Δt_k (f_φ(t_k, X_k) − y_{ik})² ; contrôleur u = ∇_x f_φ en convention du dépôt (dérive g² ∇ log ĥ, h relatif à la référence). **Biais déclaré** : sous la pondération ω, la moyenne conditionnelle de y vaut log h(t_k, x) + KL(P*‖Q_θ sur [t_k, 1] | x) ≥ log h ; l'estimateur littéral demandé n'est pas sans biais et son gradient non plus.

## Étape 0 (CPU, chaque exécution < 1 min)
- **Test unitaire** : u_θ = 0, d = 2, cible N((1 ; 0), 0,7² I), VE σ_max = 2, 199 pas, M = 10⁵ ; réseau et optimiseur ci-dessus sauf **5000 étapes, lot 1024** (limite CPU, déclarée). Solution exacte du schéma discret : v*(k, x) = g_k (μ_k(x) − x) / R_k, avec R_k = Σ_{j≥k} g_j² Δt_j, V = R_0, λ_k = 1/R_k + 1/s² − 1/V, μ_k(x) = (x/R_k + m/s²)/λ_k. Erreur relative ‖v_φ − v*‖ / ‖v*‖ en L² (Σ_k Δt_k) **sous Q_θ**, et sous P* (trajectoires repondérées) en descriptif ; moyenne et écart-type terminaux sous u_new. **Réussi si l'erreur sous Q_θ ≤ 0,25** ; sinon arrêt et diagnostic.
- **Plomberie** : (8, 8, 8) × 0,5, M petit, quelques étapes ; temps par trajectoire ; aucun résultat lu.
- Temps GPU estimé ensuite (amendement daté).

## Étape 1
- **Cellules** : (A) (128, 8, 6) × 0,5, époque 1500 (IS de chemin qui marche) ; (B) (128, 8, 6) × 0,75, époque 1500 ; (C) MoS-w d = 10, époque 250. **Un instantané et une graine par cellule** (déclaré : la variabilité entre instantanés n'est pas mesurée ; étiquette C au mieux).
- M ∈ {10⁴, 10⁵}. Les règles portent sur **M = 10⁴**, 1re itération ; M = 10⁵ est descriptif.
- Mesures, pour u_θ et u_new (10⁴ échantillons bruts chacun) : TV par bassin (principale), TV de Voronoï, Q1, erreur de position des modes ; ESS relatif et log Ẑ de l'IS de chemin sous le contrôleur (cible normalisée : log Z = 0). Pour les poids d'apprentissage : ESS, k̂, masse pondérée et effectif par bassin. Courbe de perte d'apprentissage et perte pondérée sur 2000 trajectoires fraîches indépendantes (diagnostic). Plancher : TV de 10⁴ tirages exacts. Rappel descriptif de l'IS tempéré (t) quand il existe sur la cellule.

## Règles (1re itération, M = 10⁴)
- **H1** (A) : TV par bassin brute de u_new ≤ 0,5 × celle de u_θ **ET** Q1(u_new) ≤ 1,1 × Q1(u_θ).
- **H2** (B) : idem. **H3** (C) : idem.
- **H4** (sûreté, A) : ESS de l'IS de chemin sous u_new ≥ celui sous u_θ.
- **Lecture fixée d'avance** : H1 et H4 vraies, H2 et H3 fausses = la correction de dynamique par poids de chemin (PICE) corrige les échantillons bruts là où l'IS de chemin marche, et échoue là où il s'effondre (conclusion de la note confirmée ; ligne de grille en R pour (B) et (C) si le mécanisme est confirmé par l'ESS des poids, sinon C). H2 ou H3 vraie = résultat inattendu, à examiner avant toute conclusion. H1 fausse = défaut de mise en œuvre ou de budget d'apprentissage, à diagnostiquer.
- « Mécanisme confirmé par l'ESS des poids » (précisé d'avance) : ESS relatif des poids d'apprentissage < 0,01 dans la cellule concernée.

## Prédictions (sans effet)
Test unitaire réussi 85 %. H1 vraie 30 % (la TV pourrait baisser sans que Q1 tienne, ou sans atteindre la moitié). H2 10 %. H3 5 %. H4 35 % (le bruit de régression élargit la dérive). Lecture « corrige là où l'IS marche » : 20 %.

## Calcul, coût et arrêts (accord explicite de l'utilisateur, 2026-11-27)
Vast.ai RTX 3070, fiabilité > 0,99 de préférence (≥ 0,975 accepté), ≤ 0,134 $/h, **plafond dur 0,15 $**. Priorité décroissante : fumée ; (A) M = 10⁴ ; (B) M = 10⁴ ; (C) M = 10⁴ ; M = 10⁵ (A, B, C) ; itérations 2 et 3 (A, B, C) ; bras (h) (A, B, C). **Une étape dont la durée projetée dépasse le temps restant n'est pas entamée, ni les suivantes.** Au plafond : sauvegarde, destruction, utilisateur prévenu ; destruction dès les résultats rapatriés. Rien d'autre n'est lancé.

## Exécution
`scripts/doob_ce.py`, `scripts/vast_doob_remote.sh`. Résultats `figures/doob_ce/` (résumés et poids de v_φ). RESULTS.md et ligne 18 de la grille.
