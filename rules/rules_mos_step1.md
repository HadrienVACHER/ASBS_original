# Règles pré-enregistrées : étape 1 CPU (darting sur le contrôleur MoS d = 50, σ_max = 23,2) et étape 2 (proposition MoS-w)

Écrites et commitées AVANT tout calcul de l'étape 1. Demande de l'utilisateur (2026-11-20) : pas de lot complet, pas d'extension de la grille de σ_max (l'IS échoue à cause des queues, quel que soit σ_max). CPU uniquement, chaque exécution < 1 min ; si nécessaire n = 5000 (plancher recalculé) ; temps de chaque exécution rapporté. Rien n'est loué.

## Correction d'un chiffre du rapport du pilote
Le plancher TV (n = 10⁴, sampler exact) du MoS d = 50 est **0,0123** (`figures/mog_mos_truth_mos_d50.json`), pas 0,0113 (valeur de d = 200) : 2 × plancher = 0,0246 ; le verdict G du pilote (faux, TV_darting ≥ 0,072) ne change pas.

## Étape 1 (sur le contrôleur sauvegardé `figures/mos_pilot_controllers/mospilot_s23.2.pt`)
Échantillons de départ pour (a) et (b) : les 10⁴ échantillons terminaux de ce contrôleur (`samples_final.pt` du pilote, local) ; n = 10⁴ si le temps le permet, sinon n = 5000 (plancher recalculé à ce n par la médiane de 100 sous-échantillons exacts). Protocole darting identique à la phase 0 et au pilote (`scripts/dart_lib.py` : quench sur 500 échantillons, 1 saut + 5 pas MALA par balayage, pas de MALA réglé UNE fois pour 0,6 d'acceptation, translation).
- **(a)** Darting post hoc, S ∈ {10, 20, 40} (instantanés d'une même chaîne) : TV, acceptation des sauts et de MALA, coût (E + ∇), couverture en fonction de S.
- **(b) Témoin MALA seul** : mêmes échantillons, MÊME pas MALA (celui réglé par le darting), sans aucun saut ; budget = le coût total mesuré du darting à S = 40 (donc 2n pour l'état initial, puis ⌊(C(40) − 2n)/(2n)⌋ pas MALA, chacun à 2n évaluations : E et ∇ au point proposé) ; TV rapporté aussi aux coûts de S = 10 et 20. Le témoin est donc généreux (il reçoit aussi le coût de la quench).
- **(c) Variantes robustes de l'IS de chemin.** Les log w du pilote n'ont PAS été sauvegardés : 20 000 trajectoires sont resimulées sur CPU (même protocole `girsanov.log_path_weights`, graines différentes car le GPU a un autre générateur : pas les mêmes log w). Variantes : IS auto-normalisé simple ; **PSIS** (lissage de Pareto : ajustement `scipy.stats.genpareto` (loc = 0) aux M = min(0,2 n ; 3√n) plus grands poids, remplacement de la queue par les quantiles (j − 0,5)/M de la loi ajustée, troncature au plus grand poids brut ; k̂ = la forme ajustée ; approximation de l'estimateur de Zhang & Stephens, déclarée) ; **IS tronqué** (Ionides 2008) w ← min(w, √n × moyenne des w). TV et ESS de chacune.
Si le temps CPU permet plus de trajectoires, une 2ᵉ taille (n = 10⁵) est ajoutée ; l'IS à COÛT ÉGAL du darting (≥ 10⁶ trajectoires) n'est pas atteignable sur CPU en une minute : le comparateur à coût égal est alors celui de GPU du pilote (IS simple, TV 0,698 à 0,760 à n = 1,07·10⁶ à 2,10·10⁶) et les variantes robustes sont évaluées au plus grand n disponible, marquées « comparateur tronqué ».

## Règles (pré-enregistrées)
- **G inchangé** : TV_darting(S) ≤ 0,5 × TV_IS à coût égal ET TV_darting(S) ≤ 2 × plancher, pour au moins un S ∈ {10, 20, 40}. Évalué contre (i) l'IS simple de coût égal (celui du pilote) et (ii) la MEILLEURE variante de l'IS (celle de plus petit TV parmi simple, PSIS, tronqué, au plus grand n disponible, « tronqué » signalé).
- **H_saut** : TV_darting(S = 40) ≤ 0,5 × TV_MALA seul (même coût). Sinon la correction ne vient pas des sauts.
- Tout ajout est exploratoire.

## Étape 2 : proposition seulement (rien ne se loue)
MoS-w (poids inégaux de la phase 0) d = 50 : un run d'AS à σ_max = 23,2, 300 époques, contrôleurs sauvegardés aux époques 100, 200, 300 (option `save_controller_epochs`, à ajouter à `train.py`) ; puis sur chaque contrôleur le protocole de l'étape 1 (IS de chemin, PSIS, IS tronqué, darting S ∈ {10, 40}, MALA seul). Offre, durée, coût et plafond soumis à l'utilisateur, qui décide au vu de l'étape 1.

---
# Amendement 1 : exécution de l'étape 2 (accord explicite de l'utilisateur, 2026-11-20 ; écrit AVANT la location)
- Offre RTX 3070 (≈ 0,134 $/h), **plafond DUR 0,06 $ (≈ 26,8 min), arrêt d'urgence à 18 min** : à l'approche sauvegarder les résultats terminés, détruire l'instance, puis prévenir ; détruire dès les résultats rapatriés ; jamais de dépassement sans accord explicite.
- MoS-w d = 50 (poids `configs/problem/mosw.yaml`), AS seul, σ_max = 23,2, 300 époques, graine 0, contrôleurs sauvegardés aux époques 100, 200, 300 (`controller_ep{100,200,300}.pt`) ; **les log w de l'IS de chemin sont SAUVEGARDÉS** (`is_logw_ep*.npz`, avec les cellules de Voronoï des points terminaux).
- Évaluation de chaque contrôleur (protocole de l'étape 1) : TV brut et couverture (20 000 points terminaux) ; IS de chemin (20 000 trajectoires sur GPU) simple, PSIS, tronqué (TV, ESS, k̂, log Ẑ) ; darting post hoc S ∈ {10, 40} sur les 10⁴ premiers points terminaux (quench 500, 1 saut + 5 MALA, pas de MALA réglé une fois) ; MALA seul au coût total de S = 10 et de S = 40 ; **G** (TV_darting ≤ 0,5 × TV_IS ET ≤ 2 × plancher ; comparateur = meilleure variante de l'IS à n = 20 000, « tronqué », plus l'IS simple à coût égal pour le contrôleur de l'époque 300 si le temps le permet) et **H_saut** (TV_darting(40) ≤ 0,5 × TV_MALA seul), par contrôleur. Plancher : celui de `figures/mog_mos_truth_mosw_d50.json` (n = 10⁴).

# Diagnostic CPU (descriptif, sans règle) sur le contrôleur MoS d = 50 de l'étape 1
1. Sauts refusés (S = 10 balayages) : invalides (l'image tombe hors de la cellule j, c.-à-d. du minimum j le plus proche) contre refusés par MH ; pour les refusés, quantiles de log π(y) − log π(x) (attendu sous π : ≈ 0 au cœur des modes).
2. Acceptation des sauts par décile de la distance de x à son minimum, sur les échantillons d'AS et sur des échantillons exacts de π (témoin), mêmes minima.
3. TV final du darting (S = 40) décomposé : (i) erreur sur les masses par bassin (chaque échantillon assigné à son minimum par quench : descente de gradient 150 pas puis minimum le plus proche) comparée à la vérité calculée de la même façon (échantillons exacts assignés par quench) ; (ii) le reste = écart entre TV de Voronoï et TV par bassin, dû à la masse des queues qui déborde d'une cellule de Voronoï à l'autre (TV entre proportions de Voronoï et proportions par bassin de la vérité et du darting). Échantillons : n = 5000 par assignation (coût du quench).
4. Contrôle de stationnarité : darting S = 40 (protocole complet) démarré depuis 10⁴ échantillons exacts de π ; le TV de proportions doit rester au plancher.
