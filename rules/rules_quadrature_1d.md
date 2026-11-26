# Règles pré-enregistrées : quadrature 1D, vitesse de correction des poids (AS seul contre AS + darting)

Écrites et commitées AVANT d'écrire ou de lancer le script (`scripts/quadrature_1d_failure_modes.py`). Calcul CPU, déterministe, sans réseau ni job, un seul script, budget total < 1 min.

## Cadre (de l'énoncé, non modifié)
π = w N(+1, ε²) + (1−w) N(−1, ε²), w = 0,8 ; q = w' N(+1, ε²) + (1−w') N(−1, ε²), w' = 0,5 ; base AS : X₀ = 0, dX = σ_t u dt + σ_t dB, schedule VE du dépôt (σ_min = 0,001), σ_max = 3 ; ν_t = ∫₀ᵗ σ² = σ_max²(1 − r^(−2t)), r = σ_max/σ_min ; κ_t = ν_t/ν₁, v_t = ν_t(1−κ_t), ν_{1|t} = ν₁ − ν_t ; p_base = N(0, ν₁), φ = π/p_base. Postérieures de X₁ sachant X_t = x en forme close (mélange gaussien) ; toutes les espérances conditionnelles par Gauss–Hermite (48 nœuds par composante), sans Monte Carlo.
Dérives (la dérive de l'SDE est σ_t u ; en variable ν, ∂_ν p = −∂_x(D p) + ½ ∂_xx p avec D = σ_t u / σ_t² = E[...]) : D_AS = E_q[∂log φ(X₁)|x] ; D_CV(η) = D_AS + E_q[c(X₁, x)|x], c = ω (y − x₁)/ν_{1|t} (γ = 1), y = m_j + (1+η)(x₁ − m_k), valide si y tombe dans la cellule j (signe), sinon ω = 0, ω = π̃(y)|1+η| / (π̃(x₁) + π̃(y)|1+η|), π̃(z) = π(z) N(x; κ_t z, v_t), η ∈ {0 ; 0,1 ; 0,3} ; D_* = E_π[∂log φ(X₁)|x] ; D_q = E_q[∂log(q/p_base)(X₁)|x].
Masse terminale : Fokker–Planck conservatif (volumes finis, flux nuls aux bords), Crank–Nicolson (solve_banded), dx = ε/20, domaine [−L, L], L = max(1 + 8ε, 6√ν₁), départ à ν_{t₀} = (5dx)² avec p(t₀) = N(0, ν_{t₀}), ŵ = ∫_{x>0} p(1, x) dx (x = 0 est une face de maille).

## Définitions opérationnelles ajoutées (fixées ici, avant tout calcul)
- Variable de temps : ν (la dynamique est écrite en ν). Grille de ν : 120 pas géométriques en ν de ν_{t₀} à ν₁/2, puis 280 pas géométriques en w = ν₁ − ν de ν₁/2 à w_fin = (dx/10)² (total 400 pas) ; le temps final est donc t₁ − δ avec un reste de variance w_fin ≪ dx² (déclaré ; le p(1, ·) du texte est p à w_fin). Dérive évaluée au milieu de chaque pas ; les 4 premiers pas en Euler implicite (lissage de Rannacher), puis Crank–Nicolson.
- Fenêtre active pour l'évaluation des dérives : |x| ≤ κ + 9√(v + κ²ε²) (dérive prolongée par la valeur du bord au-delà ; la masse y est < 1e-15).
- Plancher de discrétisation par ε : plancher(ε) = max(|ŵ_{u_q} − w'|, |ŵ_{u_*} − w|, 1e-6).
- Fraction de correction : ρ_m = (ŵ_m − ŵ_{u_q}) / (w − w'), m ∈ {AS, CV η = 0, CV η = 0,1, CV η = 0,3}. Une valeur avec |ŵ_m − ŵ_{u_q}| < 10 × plancher est rapportée « < plancher » et n'est pas interprétée.
- E_q[ω] : à chaque pas, E_q[ω](t) = ∫ p_t^q(x) E_q[ω | x] dx (p_t^q = marginale du processus q, mélange gaussien de moyennes ±κ, variance v + κ²ε²), puis moyenne en temps t uniforme (poids dt = t(ν_{n+1}) − t(ν_n)) sur [t₀, t_fin].
- Itération : CV η = 0 à ε = 0,3, w'₀ = 0,5, puis w'_{n+1} = ŵ_n (n = 1…5), q reste dans la famille ; « atteint |ŵ − w| < 0,01 en au plus 3 itérations » = il existe n ≤ 3 avec |ŵ_n − w| < 0,01, ŵ_n étant le résultat de la n-ième application.
- Contrôles (avant tout résultat) : u = 0 → TV(p(1), N(0, ν₁ − w_fin)) < 1e-4 (masses de maille exactes) ; u = u_* → ŵ ≈ w ; u = u_q → ŵ ≈ w' ; écarts = plancher, reportés. Vérification de t₀ : à ε = 0,3, ŵ_{u_q} et ŵ_AS avec ν_{t₀} divisé par 2 ; critère : différence < plancher(0,3).
- Temps et réduction de grille : mesure par chronomètre. Un pilote (15 pas à ε = 0,25 avec toutes les dérives) projette le temps total ; si la projection dépasse 40 s, on descend l'échelle : (A) dx = ε/20, 400 pas ; (B) dx = ε/10, 300 pas ; (C) dx = ε/8, 240 pas, 32 nœuds. Le script rapporte l'échelon utilisé et le temps total réel. Si le temps réel dépasse 60 s, le rapport le dit.

## H_AS (de l'énoncé + quantification ajoutée ici)
Sur les ε où |ŵ_AS − ŵ_{u_q}| > 10 × plancher (ensemble Q), log ρ_AS (logarithme népérien) est affine en 1/(2ε²) de pente dans [−1,5 ; −0,5] ET ρ_AS < 0,05 pour ε ≤ 0,35.
Quantification : « affine » = moindres carrés sur Q avec |Q| ≥ 3 et R² ≥ 0,95. Si |Q| < 3 : première partie « non évaluable ». Deuxième partie : vraie si ρ_AS < 0,05 mesuré, ou si ρ_AS est « < plancher » avec 10 × plancher / (w − w') < 0,05 (borne supérieure). Verdict : « soutenue » si les deux parties sont vraies ; « non soutenue » si l'une est fausse ; « non concluante » si la première est non évaluable et la seconde vraie.

## H_CV
ρ_CV(η = 0) ≥ 0,5 pour tout ε ∈ {0,6 ; 0,5 ; 0,4 ; 0,35 ; 0,3 ; 0,25}, ET l'itération atteint |ŵ − w| < 0,01 en au plus 3 itérations. Verdict : « soutenue » si les deux sont vraies, « non soutenue » sinon.

## H_flux (descriptive, sans seuil)
ρ_CV(η) et E_q[ω] en fonction de η ∈ {0 ; 0,1 ; 0,3}, par ε.

## Livrable
Tableau ε × méthode (ρ, ŵ), plancher, temps total, figure log ρ en fonction de 1/(2ε²), verdict H_AS / H_CV. Rien d'autre.
