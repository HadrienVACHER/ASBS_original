# Phase 0, tâche 3/4 : mécanisme de l'échec de la division sur DW-4 (idée I3). Des gabarits par bassin plus riches réduisent-ils la dispersion de log u ?

Écrit et commité AVANT tout calcul. C'est un diagnostic, pas un test de méthode : DW-4 est trivial pour l'IS. CPU uniquement ; chaque exécution < 1 min (chronométrée). L'EM est découpé en exécutions successives avec point de reprise (déclaré) ; les sous-échantillonnages sont déclarés ci-dessous. Aucune règle n'est modifiée après coup.

**Portabilité** (note de RESULTS.md) : tous les ajustements et toutes les évaluations sont refaits ici, localement, dans la base déterministe de Helmert (`dw4_division_step0.basis()`). Aucun paramètre ajusté sur le cluster n'est utilisé.

## Données
- **AS** : `results/dw4_e1500_ed8bb67_samples.pt` (`model_samples`, 10⁴). Bassin = état de la configuration quenchée (`quench`, `state_ids`). Les 5 types sont triangle + pendant, chemin de 2, deux paires disjointes, K4 moins une arête, étoile. Dans chaque bassin, les échantillons sont pris dans l'ordre du fichier : les indices pairs forment la **moitié d'ajustement**, les impairs la **moitié d'évaluation** (validation croisée).
- **PT** (`results/pt_samples_T1.pt`, b4e35e5, étiquettes de bassin quenchées stockées) : **moitié A** = chaînes 0–31 (sert seulement aux minima initiaux μ_a : état d'énergie quenchée la plus basse parmi 200 états de A tirés par type) ; **moitié B** = chaînes 32–63, **sous-échantillonnée à au plus 2000 états par bassin** (graine 0 ; déclaré, pour le temps). L'étoile a environ 320 états en B, tous gardés.
- **Pour mémoire** (rapporté, sans règle) : la moitié B de la référence DEM (indices impairs de `data/test_split_DW4.npy`), sur laquelle les valeurs de l'étape 1 ont été mesurées.

## Gabarits
Pour chaque bassin a, un mélange de J gaussiennes G-moyenné : N̄_a(x) = Σ_j c_j q̄(x ; μ_j, (1 − η_a) Σ_j), avec q̄ la moyenne sur O(2) × S4 × Z2 de `dw4_division_step0.log_qbar` (48 éléments × 256 angles, interpolation trigonométrique exacte). J ∈ {1, 3, 5, 10}.
- **EM** sur la moitié d'ajustement, latents (composante, élément du groupe, angle). Il généralise l'EM de l'étape 1 (`dw4_division_em.py`) :
  - poids c_j ;
  - moyennes et covariances pondérées des points alignés ;
  - covariance rétrécie vers la covariance poolée de toutes les composantes de tous les bassins, avec n₀ = 50 ;
  - plancher de variance 5·10⁻³ le long de la tangente de rotation de chaque μ_j ;
  - au plus 40 itérations, ou |Δ log-vraisemblance moyenne| < 10⁻⁴ pour chaque bassin.
- **Initialisation** :
  - J = 1 : celle de l'étape 1 (`fit_basin` aligné) ;
  - J > 1 : k-means (k-means++, graine 0, 25 itérations) sur les points alignés sur μ_a, puis covariances de grappe rétrécies (n₀ = 50) vers la covariance du bassin.
- **Rétrécissement η_a** : règle de la variante B, appliquée au gabarit ajusté (20 000 tirages du mélange de covariances (1 − η) Σ_j). On prend le plus petit η ∈ {0,2 ; … ; 0,8} tel que P(une distance de paire dans [3,5 ; 4,5]) < 1 % ; à défaut 0,8, signalé. η_a est choisi avant de regarder toute dispersion.
- **Limite déclarée** : l'étoile n'a que 81 échantillons d'ajustement (163 au total). Pour J ≥ 3, ses composantes sont surtout fixées par le rétrécissement vers la covariance poolée.

## Mesures (par bassin et par J)
log u_a = log N̄_a(x) + E(x)/τ (τ = 1). On mesure son **écart-type** sur les échantillons du bassin a (i) de la moitié AS d'évaluation et (ii) de PT B. On mesure aussi **k̂** de u_a dans le bassin (ajustement de Pareto généralisé sur les 20 % du haut, `dw4_division_b.khat`) sur les mêmes deux ensembles. Rapporté aussi : les écarts-types sans rétrécissement (η = 0).

## Règles (de l'utilisateur, avec précisions déclarées)
- **Cohérence** : J = 1 redonne à 15 % près (|SD/SD_ref − 1| ≤ 0,15 dans chaque bassin) les écarts-types de l'étape 1 (EM) sur PT B : 2,66 / 1,68 / 6,69 / 6,46 / 3,30 (triangle + pendant, chemin de 2, deux paires, K4 moins une arête, étoile). **Sinon, arrêt.** Précision : ces valeurs avaient été mesurées sur la moitié B de la *référence DEM* (indices impairs), avec un ajustement sur *tous* les échantillons AS. La règle porte, comme écrite, sur PT B ; la version sur la moitié B de la référence DEM est rapportée à côté.
- **V1** : pour J = 5, écart-type de log u_a ≤ 2 nats **sur PT B** dans les trois petits bassins (deux paires, K4 moins une arête, étoile).
- **V2** : pour J = 5, k̂ dans le bassin < 0,5 dans ces trois bassins. **Précision : sur la moitié AS d'évaluation** (les poids de la division sont évalués sur les échantillons du sampler) ; la valeur sur PT B est rapportée.
- **Contrôle négatif** : le gabarit J = 5 à covariances × 2 (même η_a) doit donner un écart-type plus grand, sur PT B, dans chacun des trois petits bassins. Sinon, la mesure ne discrimine pas, ce qui est signalé.
- **Lecture fixée d'avance** : si V1 et V2 sont vraies, l'échec de la division tenait à des gabarits trop pauvres, et la piste se rouvre sur d'autres cibles ; sinon, la division est fermée (R) quel que soit J.

## Exécution
`scripts/dw4_template_variance.py` : `prep`, `prepref` (moitié B de la référence DEM, rapportée), `fit J` (repris jusqu'à convergence, < 50 s par exécution), `eval J`, `verdict`. Résultats `figures/dw4_template_variance/` (JSON) ; données et points de reprise dans `results/` (hors dépôt). Tableau bassin × J (écarts-types, k̂), verdicts et temps dans la section « Phase 0 » de RESULTS.md.

## Amendement daté (2026-11-22, après l'arrêt ; demandé par l'utilisateur ; écrit et commité avant la reprise)
**Arrêt d'origine, conservé** : la cohérence telle qu'écrite (J = 1 à 15 % près des valeurs de l'étape 1, sur PT B) était fausse (deux paires +24 %, étoile +79 %), donc arrêt (RESULTS.md, 6129e15). **Cause : erreur de spécification de l'utilisateur.** Les valeurs de référence 2,66 / 1,68 / 6,69 / 6,46 / 3,30 venaient de la moitié B de la référence DEM, pas de PT B.
- **Vérification d'implémentation** (remplace la cohérence) : J = 1 reproduit l'étape 1 sur la moitié B DEM à 15 % près dans au moins 4 bassins sur 5. **Elle est déjà vraie** (4/5 : −2 %, +23 %, −2 %, −13 %, +9 %). L'écart du chemin de 2 s'explique par η = 0,4 au lieu de 0,3. Elle est rapportée comme telle.
- **Référence pour la suite** : J = 1 mesuré sur PT B dans cette même exécution (`eval_J1.json` : 2,51 / 1,56 / 8,27 / 5,79 / 5,92).
- **V1, V2 et le contrôle négatif : inchangés.**
  - V1 : J = 5, écart-type ≤ 2 nats sur PT B dans les trois petits bassins.
  - V2 : k̂ dans le bassin < 0,5, sur la moitié AS d'évaluation comme précisé plus haut, la valeur sur PT B étant rapportée.
  - Contrôle négatif : le gabarit J = 5 à covariances × 2 doit donner un écart-type plus grand sur PT B dans chacun des trois petits bassins.
- **Rétrécissement** : la règle de l'étape 1 (variante B : plus petit η ∈ {0,2 ; … ; 0,8} tel que P(une distance de paire dans [3,5 ; 4,5]) < 1 % sur 20 000 tirages, sinon 0,8, signalé) est **appliquée à chaque composante séparément** : η_j pour la composante j, avec covariance (1 − η_j) Σ_j. Pour J = 1, c'est la règle déjà appliquée. η est rapporté par bassin et par J (liste des η_j).
- **Lecture fixée d'avance, inchangée.**
- Puis : ajustement de J ∈ {3, 5, 10}, évaluation, et tableau bassin × J (écarts-types sur la moitié AS d'évaluation et sur PT B, k̂), avec verdicts et temps. Chaque exécution < 1 min.
