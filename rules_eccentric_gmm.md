# Eccentric GMM (d = 50) : le darting marche là où le SMC et l'IS échouent ? (jouet CPU)

Écrit pour : Hadrien. Écrit et commité **avant** tout résultat de ce test ; aucune règle ne sera modifiée après coup (amendements datés seulement). Aucun GPU, aucune location, aucun entraînement ; chaque exécution < 1 min. **Métrique de masse principale : TV par bassin** ; ici les deux bassins sont ceux des deux minima et coïncident avec le Voronoï des deux moyennes.

## Question
Sur l'exemple de la littérature (Eccentric GMM, goulot d'étranglement), où le SMC perd le mode excentré, la banque de minima trouvée par trempe depuis ρ = N(0, I) et le darting transfèrent-ils la bonne masse ?

## Cible et méthodes
- **Cible** π = ½ N(0, (1/16) I₅₀) + ½ N((3/4)·1₅₀, (1/16) I₅₀) ; **ρ = N(0, I₅₀)**. Masse vraie de chaque mode 0,5 ; distance entre les moyennes 0,75 √50 = 5,30 ; écart-type 0,25 par coordonnée.
- **SMC** (`scripts/toy_eccentric.py`) : N = 1000 particules tirées de ρ, chemin π_β ∝ ρ^(1−β) π^β, Δβ par bissection à un ESS conditionnel de 0,5 N, **rééchantillonnage systématique à chaque pas** (c'est la correction par IS du SMC : après chaque pas les poids sont uniformes), 5 pas de MALA à pas adapté après chaque rééchantillonnage, β = 1 en fin. Je ne prétends pas reproduire le SMC de la source (autre SMC, autre noyau) : je mesure la fréquence de perte du mode pour **ce** SMC.
- **Banque** : N_q = 1000 départs tirés de ρ, trempe convergée de `basin_lib.quench`, fusion à 0,5. **Indépendante des particules du SMC** (graine 10000 + s).
- **Darting** : `dart_lib.darting` avec `minima=` la banque, S = 10 balayages (1 saut + 5 pas MALA par balayage), pas MALA réglé sur le lot, appliqué aux 1000 particules finales du SMC. **Deux écarts déclarés au code existant, sans effet sur les autres cellules (défauts identiques ; `tests/test_dartbank_matcher.py` T1–T6 réussis) :**
  - le pas de polissage de la banque (`M + 0,7 g`) est mis à l'échelle `0,7 std²` (égal pour std = 1, toutes les cibles précédentes) : le pas 0,7 divergeait pour std = 0,25 ;
  - **saut paresseux, `jump_prob` = 0,5** : avec 2 minima de poids égaux, le saut est accepté avec probabilité 1 et inverse le mode de tous les points à chaque balayage : le noyau est de période 2 et, après S = 10 balayages, la masse est ramenée au mode de départ (TV 0,5, constaté sur des graines de test). Le saut paresseux (chaque point tente son saut avec probabilité 1/2) préserve π et supprime la période. C'est un écart au protocole de `dart_lib` (S = 10 balayages de 1 saut), déclaré ici ; les 8 modes des autres expériences ne sont pas concernés. Application de saut : translation (Hessiennes égales).
- **Proxy effondré** : 1000 tirages de q = N(0, (1/16) I), c'est-à-dire du mode proche seul (un SMC qui a perdu le mode) ; l'IS (poids π/q auto-normalisés) et le darting avec la même banque sont appliqués.
- **Seeds** : 100 graines (1 à 100) pour le SMC, la banque (10000 + s) et le proxy. **Les graines 9001 à 9029 ont servi à des essais de mise au point (temps, bug du pas de polissage et de la période) ; leurs résultats ne sont pas dans les règles.** Observé sur elles avant d'écrire ces règles : SMC sans mode excentré dans 19 cas sur 20 (graines 9010 à 9029), banque trouvant les deux minima dans 10 cas sur 10, darting après correction à environ 0,5 de masse.

## Mesures
Masse p₂ du mode excentré = fraction des particules plus proches de la moyenne 2 ; **TV par bassin = |p₂ − 0,5|**.
- **SMC brut** (et IS du SMC = le SMC lui-même) ; **SMC + darting** ; **proxy : IS** (masse pondérée), **proxy + darting**.
- **Collapse** : un run du SMC est « effondré » si min(effectif du mode 1, effectif du mode 2) ≤ 10 sur 1000.
- **Banque** : les deux minima présents (à moins de 0,5 des vrais minima) ; nombre de départs tombant dans chaque bassin.
- **Coût** (évaluations d'énergie et de gradient, convention de `dart_lib`) : SMC, banque, darting.

## Règles
- **E1 (découverte)** : la banque (N_q = 1000, depuis ρ) contient les **deux** minima dans **au moins 95 des 100 graines**. (Probabilité par départ de tomber dans le bassin excentré : environ 0,4 % ; attendu 1 − e⁻⁴ ≈ 98 %.)
- **E2 (descriptif)** : fraction de runs de SMC effondrés ; distribution des effectifs du mode excentré.
- **E3 (principale : le darting marche où le SMC échoue)** : sur les runs effondrés, au nombre d'au moins 10 : la TV par bassin **médiane du SMC brut est ≥ 0,40** (il a échoué) **ET** la TV médiane de **SMC + darting est ≤ 0,05 ET au moins 90 % des runs effondrés ont une TV de SMC + darting ≤ 0,10**. (Une banque sans le mode excentré compte comme un échec du darting, sans exclusion.) Si moins de 10 runs sont effondrés, E3 est déclarée non évaluable et E4 porte la conclusion.
- **E4 (proxy effondré)** : sur 100 graines, la TV médiane de l'IS est ≥ 0,45 **ET** la TV médiane de proxy + darting est ≤ 0,05 **ET** au moins 90 % des graines ont une TV ≤ 0,10.
- **E5 (sûreté)** : sur les runs non effondrés (si au moins 5) : la médiane de [TV(SMC + darting) − TV(SMC brut)] ≤ 0,03.
- **E6 (descriptif)** : coûts médians et rapport au SMC ; acceptation des sauts.
- **Lecture fixée d'avance** :
  - **E1 et E3 (ou E4 si E3 non évaluable) vraies** : sur l'Eccentric GMM, la banque trouvée depuis ρ et le darting corrigent la masse là où le SMC (et son IS par rééchantillonnage) la perdent ; il y a donc un exemple de la littérature où le darting marche et pas l'IS. **Conditions de cette lecture** : modes de même forme (saut par translation, acceptation 1), minima accessibles par trempe depuis ρ avec probabilité d'environ 0,4 % par départ, 2 modes seulement, SMC particulier, coût non comparé à égalité.
  - **E1 fausse** : la banque ne trouve pas toujours le mode excentré ; le darting n'est pas utilisable sans connaissance préalable du mode.
  - **E1 vraie, E3 fausse** : la banque est bonne mais le darting ne corrige pas ; on dit si c'est la masse (TV) ou la dispersion.
  - E5 fausse : le darting dégrade les runs sains.
- **Limites annoncées** : 2 modes ; les modes ont la même forme et le saut est une translation d'acceptation 1 (les modes de formes différentes ne sont pas testés ici) ; d = 50 ; le SMC n'est pas celui de la source ; saut paresseux ; un seul N (1000).
- **Mes prédictions (sans effet)** : E1 vraie 85 % ; SMC effondré dans 90 % des runs ou plus ; E3 vraie 85 % ; E4 vraie 90 % ; E5 vraie 80 %.

## Exécution
`scripts/toy_eccentric.py` (`disc S0 S1`, `smc S0 S1`, `verdict`). Résultats : `figures/eccentric_gmm/`. Section dans RESULTS.md.
