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
