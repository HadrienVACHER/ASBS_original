# Règles pré-enregistrées : la référence DW-4 est-elle biaisée par rapport à exp(-E_repo)/Z ?

Écrites et commitées sur la branche `div` AVANT tout run de cette série. Les règles de l'utilisateur sont reprises telles quelles ; les précisions d'opérationnalisation (marquées « Précision ») sont déclarées ici, avant les runs.
Tout en local, CPU, sans cluster ni entraînement. Chaque exécution < 10 minutes, avec la progression affichée. Aucun verdict existant de RESULTS.md n'est réécrit ; une section « Amendement : référence DW-4 » est ajoutée à la fin.

## Conventions communes
- Énergie : `DoubleWellEnergy(dim=8, n_particles=4).eval` (l'énergie de l'entraînement, a=0.9, b=-4, c=0, offset 4, tau = 1 : la densité est e^{-E}). À température T du PT : e^{-E/T}.
- Sous-espace de centre de masse nul : base déterministe `dw4_division_step0.B` (Helmert).
- Bassins : même quench que partout (`adjoint_samplers.utils.dw4_quench.quench`, puis `dw4_states.state_ids`) ; les 5 types du quenché de la référence (triangle + pendant, chemin de 2, deux paires disjointes, K4 moins une arête, étoile) ; les états hors de ces 5 types sont comptés dans le dénominateur.
- Référence : `data/test_split_DW4.npy` entière (10 000), quenchée ; p_ref = fréquences, SE(log p_ref) = sqrt((1 - p)/(p n)) (échantillons supposés indépendants ; les données MCMC de la référence peuvent être corrélées, ce qui sous-estime le SE : à signaler).
- p_IS (IS de chemin du contrôleur 94252, `scripts/dw4_path_weights.py`, 10 000 chemins, graine 0) : valeurs enregistrées dans RESULTS.md : triangle + pendant 0.4118 (SE 0.0081), chemin de 2 0.4683 (SE 0.0087) ; SE(log p) = SE / p.

## TÂCHE 1 : parallel tempering indépendant
Protocole (précisions comprises) :
- 64 chaînes vectorisées × 8 températures T_k = 5^{k/7}, k = 0..7 (progression géométrique de 1 à 5). Graine unique 0.
- Initialisation surdispersée : 32 chaînes (groupe A) partent de points de la référence tirés au hasard, 32 chaînes (groupe B) d'échantillons AS ed8bb67 (`model_samples`, tirés au hasard) ; tous les niveaux de température d'une chaîne démarrent au même point, recentré.
- Marche aléatoire gaussienne dans le sous-espace de centre de masse nul : y = x + s_T B xi, xi ~ N(0, I_6). Pas initial s_T = 0.1 sqrt(T) ; pendant le burn-in, toutes les 50 pas, log s_T += 0.5 (acc_T - 0.30) / sqrt(1 + w/10) (w = indice de la fenêtre, acc_T = acceptation moyenne de la fenêtre sur les chaînes) ; figé à la fin du burn-in.
- Échanges entre températures voisines tous les 10 pas (paires paires / impaires en alternance, toutes chaînes), acceptation min(1, exp((1/T_i - 1/T_j)(E_i - E_j))).
- 2e5 pas, dont les 4e4 premiers (20 %) de burn-in jetés. Si le run dépasse 10 minutes, le nombre de pas est réduit et c'est dit.
- Sauvegarde des états à T = 1 tous les 100 pas après le burn-in (1600 par chaîne) avec bassin (quench) et énergie dans `results/pt_samples_T1.pt` (non commité).

Validité (pré-enregistrée) ; un critère échoue => run NON VALIDE, rapporté sans conclure, avec proposition d'un réglage plus long :
  (i) R-hat < 1.05 sur l'indicatrice de chacun des 5 bassins. Précision : R-hat de Gelman-Rubin « split » (chaque chaîne coupée en deux moitiés), calculé (a) sur les 64 chaînes (128 demi-chaînes), (b) séparément sur les 32 chaînes de chaque groupe d'initialisation ; et les moyennes des deux groupes (moyennes de chaînes) doivent concorder à 3 SE (SE_diff = sqrt(var_A/32 + var_B/32), variance entre moyennes de chaînes). Un indicateur de variance nulle (R-hat indéfini) compte comme un échec pour ce bassin.
 (ii) au moins 500 transitions triangle + pendant <-> chemin de 2 à T = 1, toutes chaînes confondues. Précision : changements de bassin entre états sauvegardés consécutifs (pas de 100) d'une même chaîne, de l'un à l'autre de ces deux bassins.
(iii) histogrammes d'énergie à T = 1 concordants entre les deux groupes : test de KS à deux échantillons (états de T = 1 sauvegardés de chaque groupe), p > 0.01.

Estimation : p_PT pour les 5 bassins ; SE par moyennes de lots, les chaînes servant de lots indépendants : p = moyenne des moyennes de chaînes, SE = écart-type entre moyennes de chaînes / sqrt(64) ; SE(log p_PT) = SE / p_PT.

Verdict (en log, SE_comb = sqrt(SE_1^2 + SE_2^2) avec les SE (log) des deux estimations comparées) :
- « Référence biaisée » si, pour LES DEUX bassins dominants (triangle + pendant, chemin de 2) : |log p_PT - log p_ref| > 3 SE_comb ET |log p_PT - log p_IS| < 3 SE_comb.
- « Référence correcte » si l'inverse est vrai pour les deux bassins : |log p_PT - log p_ref| < 3 SE_comb ET |log p_PT - log p_IS| > 3 SE_comb.
- Sinon « non conclusif ».

## TÂCHE 2 : origine de l'écart (seulement si la tâche 1 est valide)
a) Énergie au-dessus du minimum, par bassin : pour chacun des 5 bassins, moyenne de E - E_min(bassin) sur la référence et sur les états PT à T = 1 ; différence (référence - PT) et z. E_min(bassin) = énergie du minimum quenché de plus basse énergie du bassin dans la référence. SE de la référence : écart-type / sqrt(n) (indépendance supposée) ; SE du PT : moyennes de lots sur les chaînes. Règle :
   - même signe et |z| > 3 dans au moins 4 bassins sur 5 : « énergie ou température différente » ;
   - |z| < 3 dans tous les bassins : « énergies cohérentes, défaut de mélange de la référence » ;
   - sinon : « non conclusif ».
   Donné aussi, à titre indicatif et sans règle : le rapport des moyennes ref/PT par bassin.
b) Provenance : recherche dans le repo, et dans le README ou le code d'origine des données DW-4, de la façon dont la référence a été générée (énergie exacte, facteur 1/2 sur les paires, tau, type de MCMC et longueur), avec fichier et ligne ou URL. Si rien n'est trouvé, c'est dit sans supposition.

## TÂCHE 3 : T1' du darting sur échantillons neufs (seulement si la tâche 1 est valide)
- Le NO-GO de T1 reste le verdict écrit. L'erreur de conception est déclarée dans RESULTS.md : T1 traitait la référence comme des échantillons de pi.
- T1' : même statistique et même seuil que T1 de `scripts/dw4_darting_test.py` (bilan de flux lumpé w_a K_ab = w_b K_ba à kappa = 0 ; pour toute paire dont les deux flux lumpés dépassent 1e-3, |log(w_a K_ab / (w_b K_ba))| < 3 SE bootstrap), calculés sur les échantillons PT à T = 1 de la tâche 1. Précisions : 8000 états tirés au hasard sans remise parmi les états T = 1 sauvegardés (graine 0), le noyau étant construit comme dans le run de T1 (minima du quench de 3000 échantillons de référence tirés au hasard, graine 0) ; SE bootstrap sur les états (300 rééchantillonnages, comme T1) ; en plus, sans règle, le SE par bootstrap de chaînes entières.
- Les échantillons IS pondérés ont déjà été regardés et ne comptent pas. T1' est rapporté séparément de T1.

## TÂCHE 4 : recalcul des erreurs (seulement si la tâche 1 est valide, sans règle)
Tableau avec p_PT comme vérité. Lignes : AS ed8bb67 (`model_samples`, 10 000), AS 94252 (`model_samples` de `dw4_e1500_94252_samples.pt`), IS d'endpoint (les 2 réglages utilisables : gaussienne s = 1 et t5 s = 1, 20 000 tirages, graine 7, proposition ajustée par EM sur AS ed8bb67), IS de chemin, MH de chemin (contrôleur 94252, 10 000 chemins). Colonnes : log-ratio log(p_ligne / p_PT) de chaque bassin avec son SE (SE_comb des deux estimations ; SE binomial pour les échantillons AS bruts, bootstrap pour les IS et le MH), puis l'erreur sommée sur les 5 bassins ; dernière colonne : l'erreur sommée calculée contre la référence.

## RAPPORT
Pour chaque tâche : verdict selon la règle, chiffres clés avec SE, temps de calcul, écarts au protocole. Rien d'autre n'est relancé ; ni la division ni Barker ne sont touchés.
