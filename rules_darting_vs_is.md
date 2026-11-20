# Règles pré-enregistrées : darting post hoc contre IS pour corriger les proportions des modes, en dimension croissante

Écrites et commitées AVANT d'écrire ou de lancer les scripts (`scripts/darting_vs_is_stepA.py`, `scripts/darting_vs_is_stepB.py`). CPU uniquement, chaque étape < 1 min (temps mesuré ; au-delà, réduction de n ou d déclarée), aucun nouvel entraînement.

## Étape A : sampler synthétique, sans réseau (de l'énoncé, non modifié)
K = 8 modes, d ∈ {32, 64, 128, 256, 512}, moyennes m_k = (Δ/√2) e_k (axes orthogonaux, Δ = 8), poids w_k ∝ ρ^(k−1), ρ = 10^(−1/(K−1)) (rapport 10), k = 1…K.
A1 : π_k = N(m_k, I). A2 : π_k = produit de t de Student 1D (ν = 3, échelle² = 4/3, Hessienne I au minimum) centré en m_k pour k PAIR (k = 2, 4, 6, 8 en indexation à partir de 1), N(m_k, I) pour k impair.
Sampler q = Σ_k (1/K) q_k, q_k = π_k dilaté de (1+δ) par coordonnée autour de m_k, δ = 0,1 ; contrôle δ = 0.
IS : poids exacts π/q, proportions auto-normalisées, ESS (fraction n_eff/n), k̂ PSIS.
Darting post hoc : une chaîne MH par échantillon de q, partenaire j ≠ k tiré uniformément, carte y = m_j + H_j^(−1/2) H_k^(1/2)(x − m_k), règle de validité (y dans la cellule j), acceptation min(1, π(y)|J| / π(x)), S ∈ {1, 2, 5, 10} balayages ; minima et Hessiennes par une quench de 500 échantillons de q (pas d'oracle).
Budget n = 10⁴ ; coût en évaluations d'énergie : n pour l'IS, n(1+S) pour le darting ; IS à n = 10⁵ comme comparateur à coût égal de S = 10 (limité à d ≤ 128 si cela ne tient pas dans la minute).
Métriques : TV des proportions (cellules de Voronoï des moyennes), vérité par échantillonnage exact, plancher TV d'un sampler exact à n, ESS (IS), taux d'acceptation (darting).

## Définitions opérationnelles ajoutées (fixées ici, avant tout calcul)
- Voronoï = plus proche moyenne = argmax des K premières coordonnées (moyennes de même norme sur les K premiers axes). Vérité p_k : 2·10⁶ tirages exacts de π (seules les K premières coordonnées sont tirées, les autres sont du bruit identique pour tous les modes) ; elle ne dépend donc pas de d. Plancher = médiane de la TV d'un sampler exact à n tirages (200 répétitions) ; plancher(n = 10⁴) sert dans R1.
- ESS = (Σw)² / (n Σw²) (fraction). k̂ = forme c de l'ajustement de Pareto généralisé (scipy, loc = 0) aux min(0,2 n ; 3√n) plus grands poids (comme `gmm_dk_analysis.py`).
- Quench : descente de gradient (pas 0,7, 100 pas) de 500 échantillons de q sur −log π avec le gradient analytique, fusion gloutonne à 0,5 en norme, minima atteints par ≥ 5 échantillons retenus (même logique que `discover_minima` de `gmm40_darting.py`, avec la descente de gradient à la place d'Adam lr = 0,1 qui oscille en d élevé) ; Hessiennes par différences finies centrées du gradient analytique (h = 1e-4). Si toutes les Hessiennes sont égales à I à 1e-6 près (A1, et A2 par construction), la carte devient la translation m_j − m_k de jacobien 1 et le calcul est fait en O(1) par proposition (l'équivalence est vérifiée numériquement et rapportée) ; sinon le script s'arrête.
- Taux d'acceptation = propositions acceptées / propositions (toutes comprises, les invalides comptant comme rejetées), cumulé sur les S balayages. Les chaînes sont communes : TV(S = 1, 2, 5, 10) sont les instantanés d'une même chaîne.
- Tout ce qui n'est pas dans l'énoncé (contrôle δ = 0 en plus, k̂, variantes, etc.) est exploratoire.

## Règles (de l'énoncé ; quantifications ajoutées)
- **R1 (le darting gagne)** : en A1, il existe d ∈ {32, 64, 128, 256, 512} et S ∈ {1, 2, 5, 10} tels que ESS_IS(n = 10⁴) < 5 %, TV_darting(S) ≤ 0,5 × TV_IS(n = 10⁵) et TV_darting(S) ≤ 2 × plancher(n = 10⁴). (TV_IS(10⁵) non calculée pour un d : ce d ne compte pas pour R1.) Variante rapportée : S = 10 seulement.
- **R2 (le darting perd)** : en A2 : (a) acceptation(d = 512, S = 10) < 0,5 × acceptation(d = 32, S = 10) ; ET (b) il existe d avec TV_darting(S = 10) > TV_IS, où TV_IS = TV de l'IS à n = 10⁵ si calculée pour ce d, sinon à n = 10⁴ (les deux comparaisons sont rapportées). Si (a) ou (b) n'est pas observé, le signaler : A2 ne casse alors pas la carte.
- **R3 (sanity sur du réel)** : en B, TV_darting(S = 5) ≤ 2 × TV_IS pour chacune des cellules (8,8,8), (32,8,4), (32,8,6), TV_IS étant celle de l'IS de chemin sur les mêmes 10⁴ échantillons.

## Étape B : contrôleurs AS déjà entraînés, sigma_max × 0,5, cellules à 8/8
Cellules (8,8,8), (32,8,4), (32,8,6) ; (32,8,8) exclue de la comparaison (modes absents), seule la couverture après darting est rapportée. IS de chemin : `girsanov.log_path_weights` sur 10⁴ trajectoires simulées sur CPU (les log w sauvegardés de la graine 0 n'ont pas conservé les points terminaux, donc ne permettent pas la comparaison sur les mêmes échantillons) ; darting post hoc S ∈ {1, 5} sur les points terminaux de ces mêmes trajectoires (minima par la quench sur les 500 premiers). **Contrainte constatée avant d'écrire ces règles : les contrôleurs de la graine 1 n'ont pas été conservés (seuls les log w et les JSON l'ont été) ; le darting y est impossible sans réentraînement, exclu. B porte donc sur la graine 0 seule** ; les statistiques d'IS de chemin sauvegardées de la graine 1 sont rapportées à titre d'information (sans darting). Si la simulation des 10⁴ trajectoires dépasse 1 min CPU au total, un job GPU court serait proposé avec offre, coût et plafond avant toute location (rien n'est loué sans accord).

## Livrable
Tableau d × {IS n, IS 10n, darting S} (TV, ESS ou acceptation, coût), figure TV en fonction de d pour A1 et A2, verdicts R1–R3, temps total.
