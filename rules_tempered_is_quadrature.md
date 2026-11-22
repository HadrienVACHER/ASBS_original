# Phase 0, tâche 2/4 : AS pondéré par IS tempéré (idée I1), quadrature 1D dans le cadre de *Failure Modes*

Écrit et commité AVANT tout calcul. Aucune règle n'est modifiée après coup. CPU uniquement ; chaque exécution < 1 min (chronométrée ; panneaux séparés, et α séparés si nécessaire).

## Méthode (population, exacte)
On régresse la cible d'AS avec des poids r(X₁)^α, r = π/q, auto-normalisés. Le minimiseur est u_α(x, t) = E_q̃[r^α Y | x_t] / E_q̃[r^α | x_t], c'est-à-dire la mise à jour d'AS évaluée sur la loi q_α ∝ q^{1−α} π^α. On la calcule directement, puis on la propage par Fokker–Planck et on mesure la loi terminale.
- Unités de l'article : ν₁ = 1, horloge ν, σ² = 0,05 ; dérive d'AS D_α(x) = E_{q̃_α}[s_π(X₁) + X₁ | X_ν = x].
- Postérieure réciproque q̃_α(y | x) ∝ q_α(y) N(x ; νy, ν(1 − ν)), avec log q_α = (1 − α) log q + α log π, évalué exactement (q et π sont des mélanges gaussiens) :
  - si τ = √((1 − ν)/ν) ≥ 4 dy, par quadrature sur une grille fine en y (dy = σ/50, y ∈ [−6, 6], restreinte à log q_α ≥ max − 60) ;
  - sinon, par Gauss–Hermite à 64 nœuds centrés en x/ν, d'échelle τ (vraisemblance étroite).
- Solveur : Fokker–Planck conservatif Crank–Nicolson de `scripts/cv_failure_modes_checks.py`, même grille (dx = σ/20, domaine [−6, 6], 400 pas, départ à ν₀ = (5 dx)²). Les runs témoins D_q (reproduit q) et D_* (reproduit π) sont intégrés sur la même grille, avec les postérieures en forme close du script d'origine.
- α ∈ {0 ; 0,25 ; 0,5 ; 0,9 ; 1}. α = 0 est AS : c'est le contrôle.

## Prédictions et règles (de l'utilisateur)
- **P_w** (mauvais poids, panneau (d) : π = (10/11) N(−1, σ²) + (1/11) N(1, σ²), q = ½ / ½, mêmes formes) : après une itération, l'erreur de logit du poids gauche vaut (1 − α) × l'erreur initiale. Poids gauche prédit : 0,640 / 0,760 / 0,888 / 0,909 pour α = 0,25 / 0,5 / 0,9 / 1. **Vraie si |logit(v₊) − logit prédit| ≤ 0,05 pour chaque α** (v₊ = masse terminale de x < 0).
- **P_g** (instabilité gaussienne : π = N(1, σ²), q = N(1,1 ; σ²)) : multiplicateur de l'erreur de moyenne A_α = (m₊ − m_*)/(m_q − m_*) = (1 − α) × A_m, A_m = −2,153. **Vraie si |A_α − (1 − α) A_m| ≤ 0,05 pour chaque α.**
- **P_m** (mode absent, panneau (c) : q = N(1, σ²), π = ½ N(−1, σ²) + ½ N(1, σ²)) : masse gauche après une itération ≈ q_α(x < 0) à un facteur 3 près pour α ∈ {0,5 ; 0,9 ; 1}, et dans [0,47 ; 0,53] pour α = 1. **Vraie si ces deux conditions tiennent.** Attendu : q_α(x < 0) varie comme e^{−α(1−α)·2/σ²}, donc très petit sauf pour α proche de 1. q_α(x < 0) est calculée par quadrature.
- **Contrôle** : α = 0 redonne les résultats d'AS du mémo (`figures/cv_failure_modes_checks.json`) à 10⁻³ près en valeur absolue : A_AS = −2,1513 (P_g) ; poids gauche 0,50025 (P_w) ; masse gauche 2,191·10⁻⁵ (P_m). Si le contrôle échoue, les verdicts du panneau sont « non évaluables ».
- **ESS (descriptif ; c'est la vraie limite de la méthode)** : ESS/N = (E_q r^α)² / E_q r^{2α}, par quadrature en espace log (grille y ∈ [−8, 8]), pour chaque panneau et chaque α.

## Mes attentes (écrites avant ; sans effet sur les verdicts)
- **P_w et P_g** sont exactes à l'ordre dominant. Pour P_w, à modes séparés et de même forme, q_α est un mélange de même forme de poids logit-interpolés, et AS est aveugle aux poids (Th. 3.8). Pour P_g, q_α = N(1 + 0,1(1 − α), σ²) exactement. Je les attends vraies, avec des écarts dus au solveur de l'ordre de 10⁻³.
- **P_m.** Pour α = 0,5, la masse gauche de q_α est une bosse centrée sur la barrière (x ≈ 0, masse ≈ ½ e^{−1/(2σ²)}). La prédiction « à un facteur 3 près » y est la plus fragile, car une itération d'AS déplace cette masse. À α = 0,9 et 1, je l'attends vraie.
- **ESS.** P_g : e^{−0,2 α²} (0,82 à α = 1). P_w : ≈ 0,60 à α = 1. P_m : ≈ 0,5 à α = 0,5, mais effondrement vers α → 1 (E_q r^{2α} est dominé par la région x ≈ −(1 + 2α)/(2α − 1) ; de l'ordre de e^{−57} à α = 0,9 et e^{−80} à α = 1). La méthode ne retrouve le mode absent qu'avec un IS de variance astronomique.

## Exécution et rapport
`scripts/tempered_is_quadrature.py PANEL` avec PANEL ∈ {g, w, m, ess}, chronométré. Si un panneau dépasse 60 s, il est séparé par α (déclaré). Résultats dans `figures/tempered_is_quadrature.json`, puis tableau panneau × α (résultat, prédiction, verdict), ESS/N et temps, dans la section « Phase 0 » de RESULTS.md.
