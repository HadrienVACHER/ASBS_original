# Règles pré-enregistrées : vérifications numériques du mémo « variables de contrôle et modes d'échec d'AS »

Écrites et commitées AVANT d'écrire ou de lancer le script `scripts/cv_failure_modes_checks.py`. Aucune règle n'est modifiée après les résultats ; tout écart sera déclaré comme tel. CPU seulement, une graine (0, partie 4), chaque exécution < 1 min (chronométrée ; si une exécution dépasse 60 s, elle est rapportée comme telle et reprise avec la grille réduite dx = σ/10, 300 pas, déclarée). Pas de GPU, pas de location, pas d'entraînement.

Ces vérifications testent des dérivations du mémo, pas des méthodes en conditions réelles. Les valeurs théoriques ci-dessous ont été calculées par évaluation des formules avant l'écriture du script.

## Cadre commun (parties 1, 2, 3, 5)
Unités de l'article *Failure Modes of Adjoint Sampling* après son changement d'échelle espace-temps (leur annexe A) : base = mouvement brownien standard en horloge ν ∈ [0, 1] (ν₁ = 1), X₀ = 0, p_base = N(0, 1) ; κ = ν, v = ν(1 − ν), ν_{1|t} = 1 − ν. Postérieures q̃ = loi de X₁ | X_ν = x sous la projection réciproque de q, π̃ idem pour π (mélanges gaussiens en forme close). Dérives en horloge ν (la dérive de l'EDS est σ_t u = σ_t² D) :
- D_q(x) = (E_q̃[X₁] − x)/(1 − ν) (reproduit q) ; D_*(x) = (E_π̃[X₁] − x)/(1 − ν) (contrôle optimal, produit π) ;
- D_AS(x) = E_q̃[s_π(X₁) + X₁] (mise à jour AS exacte, éq. (4) de l'article) ;
- VC de Stein c_S(x₁, x) = s_π(x₁) + (x − κx₁)/(1 − ν) ; Newton-Stein : D_NS(Ĥ) = D_AS + (Λ − 1) E_q̃[c_S], Λ = 1/((1 − ν)Ĥ + κ) ;
- darting en bain thermique à deux modes : image y = x₁ ∓ Δ, poids p = π̃(y)/(π̃(x₁) + π̃(y)) si l'image tombe dans l'autre cellule, 0 sinon ; cartes « adaptées à la postérieure » : Δ = a'_R − a'_L (moyennes des composantes de π̃), frontière au milieu de a'_L et a'_R ; cartes « des minima de la cible » : Δ = µ_R − µ_L, frontière (µ_L + µ_R)/2 ; D_{NS+HB} = D_NS + E_q̃[p (y − x₁)]/(1 − ν) ;
- interpolation à p constant : D_p = D_q + p (D_* − D_q) ;
- VC « de type iDEM » (partie 5) : une proposition indépendante y ~ N(x, 1 − ν) (postérieure du processus de base), poids de Barker φ(y)/(φ(y) + φ(x₁)), φ = π/p_base ; D = D_AS + E[α (y − x₁)]/(1 − ν).
Espérances sous q̃ par Gauss–Hermite (48 nœuds par composante ; 16 nœuds et grille fine en y pour la partie 5) ; moments exacts (D_q, D_*) en forme close. Fokker–Planck conservatif + Crank–Nicolson repris du script de quadrature existant (commit bed2cf3) : dx = σ/20 (σ = écart-type des composantes de la cible), domaine [−6, 6], départ à ν₀ = (5 dx)² avec la marginale de base, 400 pas (120 géométriques en ν jusqu'à ½, 280 géométriques en 1 − ν jusqu'à (dx/10)²), 4 premiers pas en Euler implicite. Chaque comparaison est faite contre des runs témoins D_q (reproduit q) et D_* (reproduit π) qui fixent le plancher numérique.

## Partie 1 — Théorème A (cible gaussienne, VC de Stein préconditionnée « Newton-Stein »)
π = N(1 ; 0,05) ; q₀ = N(1,1 ; 0,05). Multiplicateur A = (m₊ − m_*)/(m_q − m_*), m₊, m_q, m_* = moyennes terminales des runs (méthode, D_q, D_*).
- **P1a (validation du solveur contre le Th. 3.3 de l'article)** : A_AS ∈ [−2,203 ; −2,103] (théorie : 1 + ln 0,05/0,95 = −2,1534).
- **P1b (Th. A)** : Newton-Stein avec Ĥ = 1/0,05 : |A_NS| ≤ 0,02 et |s₊²/s_*² − 1| ≤ 0,02.
- **P1c (corollaire, courbure mal estimée Ĥ = 1/(ρ · 0,05))** : pour ρ ∈ {0,5 ; 2 ; 4 ; 6}, |A − (1 − F(ρ))| ≤ 0,05 avec F(ρ) = ρ ln ρ/(ρ − 1) ; prédictions 0,3069 ; −0,3863 ; −0,8484 ; −1,1501 (instable à ρ = 6 ; seuil ρ⋆ = 4,9216, soit s⋆ = 0,45076 de l'article).
- **P1d (Th. A « pour tout q »)** : q₀ = ½ N(0,8 ; 0,02) + ½ N(1,3 ; 0,02) : TV(p₊^NS, π) ≤ TV(p₊^{D_*}, π) + 0,005 (TV sur les mailles). Rapporté sans règle : TV(p₊^AS, π).
Verdict : « Théorème A confirmé » si P1a à P1d sont vraies ; si P1a est fausse, le solveur n'est pas validé et les autres verdicts sont « non évaluables ».

## Partie 2 — Mélanges (panneaux b, c, d de la figure 3 de l'article ; σ² = 0,05, modes ±1)
Validation préalable de chaque panneau : |ŵ_* − ŵ_π| ≤ 0,005 et |ŵ_q − ŵ_{q₀}| ≤ 0,005 (masses exactes), sinon panneau « non évaluable ».
- **(b) déplacement symétrique** : π = ½ N(±1 ; 0,05), q₀ = ½ N(±1,05 ; 0,05). Demi-séparation ν̂ = (E[X | X > 0] − E[X | X < 0])/2, rapport R = (ν̂₊ − ν̂_*)/(ν̂_q − ν̂_*). **P2b** : |R_{NS+HB}| ≤ 0,1 (cartes adaptées à la postérieure). Rapportés sans règle : R_AS, R_NS (la théorie prédit que Newton-Stein seul manque un terme d'étiquettes du premier ordre).
- **(c) mode manquant** : π = ½ N(±1 ; 0,05), q₀ = N(1 ; 0,05) ; masse gauche ŵ = P(X₁ < 0). **P2c1** : |ŵ_AS − ŵ_q| ≤ 1e−4 et |ŵ_NS − ŵ_q| ≤ 1e−4. **P2c2** : ŵ_{NS+HB} ∈ [0,47 ; 0,53]. Rapporté sans règle : HB avec cartes des minima de la cible.
- **(d) mauvais poids** : π = (10/11) N(−1 ; 0,05) + (1/11) N(1 ; 0,05), q₀ = ½ N(±1 ; 0,05). **P2d1** : |ŵ_AS − ŵ_q| ≤ 0,01 et |ŵ_NS − ŵ_q| ≤ 0,01. **P2d2** : ŵ_{NS+HB} ∈ [0,889 ; 0,929] (cible 0,9091). Rapporté : cartes de la cible.
Verdicts : « Théorème C (localisation), cas des VC de Stein, confirmé » si P2c1 et P2d1 sont vraies ; « Proposition B (bain thermique + Newton-Stein exact en une itération) confirmée » si P2b, P2c2 et P2d2 sont vraies.

## Partie 3 — Proposition C2 (interpoler deux dérives optimales = mélange géométrique des lois terminales, au facteur de Feynman–Kac près)
σ² = 0,16, π = ½ N(±1 ; 0,16), q₀ = N(1 ; 0,16), dérive D_p, p ∈ {0 ; 0,5 ; 0,75 ; 0,9 ; 1}. Prédiction : ŵ(p) ≈ ŵ_geo(p) = ∫_{x<0} q^{1−p} π^p / ∫ q^{1−p} π^p ; valeurs calculées avant le script : 0,00621 ; 0,02303 ; 0,08183 ; 0,24369 ; 0,50000.
**P3** : |ln(ŵ(p)/ŵ_geo(p))| ≤ 1,5 pour p ∈ {0,5 ; 0,75 ; 0,9}, avec validation |ŵ(1) − 0,5| ≤ 0,01 et |ŵ(0) − 0,00621| ≤ 0,002. Verdict : « Proposition C2 confirmée » si P3 est vraie.

## Partie 4 — Recherche par trempe sur le GMM excentrique de SSS (d = 50)
π = ½ N(0, I/16) + ½ N(¾·1, I/16) ; 10⁵ tirages x₀ ~ N(0, I₅₀) (graine 0) ; descente de gradient sur −log π (pas 0,05, 200 itérations) ; mode d'arrivée = centre le plus proche du point final.
**P4** : (i) fraction p̂ arrivant au mode lointain ∈ [0,0034 ; 0,0046] (théorie Φ(−¾√50/2) = 0,00400) ; (ii) le mode d'arrivée coïncide avec le demi-espace ⟨x₀, 1⟩/√50 > ¾√50/2 pour ≥ 99,9 % des tirages ; (iii) fraction des 100 blocs disjoints de 1000 tirages contenant au moins une arrivée au mode lointain ∈ [0,93 ; 1] (théorie 0,982). Verdict : « prédiction de couverture par recherche externe confirmée » si (i), (ii) et (iii) sont vraies.

## Partie 5 — descriptive, sans règle
AS + VC de type iDEM (une proposition) dans le panneau (c) pour σ² ∈ {0,05 ; 0,09} : ŵ rapporté, et p_eff défini par ŵ_geo(p_eff) = ŵ (inversion numérique de la loi géométrique de la partie 3 au σ² correspondant).

## Livrable
Sortie du script, JSON `figures/cv_failure_modes_checks.json`, verdicts dans le mémo `memo_cv_failure_modes.md` (section 6). Rien d'autre.
