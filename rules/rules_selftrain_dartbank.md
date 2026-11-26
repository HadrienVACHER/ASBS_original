# Auto-entraînement sur un buffer corrigé par banque de minima + darting (sans poids)

Écrit et commité AVANT la location et avant tout résultat de ce lot (point 3 du message de l'utilisateur du 2026-11-23). Aucune règle n'est modifiée après coup. **Question** : si l'on remplace l'IS et le rééchantillonnage à poids (SSS) par une correction sans poids du buffer, AS apprend-il un sampler aux bonnes masses là où les méthodes à poids échouent ?

## Méthode (bras s)
`AdjointVEDartBankMatcher` (`adjoint_samplers/components/matcher.py`, configuration `adjoint_ve_dartbank`, tests `tests/test_dartbank_matcher.py` T1–T5 réussis). La régression est celle d'AS, sans poids.
- **Chaque lot de nouveaux points terminaux** (512 par époque, plus les 10⁴ points initiaux, qui passent par le même chemin en 20 lots de 500) subit **S = 10 balayages** de darting, protocole de `dart_lib.darting` inchangé : 1 saut de Barker entre minima de la banque + 5 pas MALA par balayage, Hessiennes aux minima recalculées, pas MALA réglé sur le lot (acceptation visée 0,6).
- **Les points déplacés remplacent les points bruts** dans le buffer. Le gradient de l'énergie (`adjoint1`) est **recalculé aux points déplacés**.
- **Banque** :
  - au premier lot, N_q = **1000** départs tirés de N(0, ν₁ I), ν₁ = σ_max² − σ_min², graine 0, trempe convergée de `basin_lib.quench`, fusion à moins de 0,5 ;
  - **à chaque lot**, trempe des nouveaux points bruts et ajout des minima distincts nouveaux (fusion à 0,5, au plus 64 minima).
- **Coût compté** (sommes des évaluations d'énergie et de gradient, comme `dart_lib`) : initialisation de la banque (N_q × (itérations + 1)), trempes des lots, darting des lots. Rapporté par époque et par mise à jour de gradient (100 mises à jour par époque), à côté de (a) (512 gradients par époque) et (t) (512 gradients + 512 énergies).
- **Choix déclaré** : le buffer initial est lui aussi corrigé, pour que le buffer entier suive la même loi. L'entraînement part du contrôleur de départ, inchangé.

## Cellules et départ (graine 0, reprise de 500 époques, recette `gmm_dk_as`, fichiers `results/wf_start/<cellule>/`)
- **(32, 8, 8) × 0,5** (modes manquants) : σ_max = 1,414214 ; départ = `controller_final` du run de la fenêtre (`results/gmm_dk_window/w1/gmmdk_d32_K8_D8_s0_sx0.5`, 1500 époques, couverture 1/8 par AS) ; 10⁴ points frais du contrôleur de départ avec log W de chemin (`scripts/wf_make_start.py k8d32`, graine 7). ESS normalisé de W / W^0,5 : 0,028 / 0,48.
- **MoS-w d = 10**, σ_max = 23,2 (perte de modes) : départ = **instantané de l'époque 250** de l'étape 3 (`results/mosw_step3/asbs/results/mosw3_d10/controller_ep250.pt`) ; buffer initial = les 10⁴ premiers points terminaux de cet instantané et leur log W de chemin, déjà simulés à l'étape 3 (graine 0). ESS de W / W^0,5 : 0,0004 / 0,0038.
- **(128, 8, 6) × 0,75** (IS qui s'effondre) : σ_max = 1,602001, départ **identique à l'étape B** (`controller_ep1500` de `gmmdk4_d128_sx0.75`, mêmes 10⁴ points de départ, ESS 0,0030 / 0,089).

## Bras
- **(s)** : ci-dessus, dans les 3 cellules (3 runs).
- **(a)** AS seul et **(t)** IS tempéré α = 0,5 :
  - **(128, 8, 6) × 0,75 : réutilisés de l'étape B** (`figures/train_weightfree/`) : même configuration, même départ, même graine et mêmes hyperparamètres. Seule différence de code : le matcher (t) journalise maintenant l'ESS à chaque époque, sans effet sur l'entraînement. Reproductibilité exacte non garantie sur GPU ;
  - **(32, 8, 8) × 0,5 et MoS-w d = 10 : nouveaux runs** (4 runs), car les cellules n'ont pas été faites à l'étape B.
- Au total **7 runs**. Hyperparamètres d'origine (FourierMLP 4 × 128, lr 1e-4, buffer 10⁴, 100 itérations par époque, 512 nouveaux échantillons par époque, nfe 200), tous les t (t_focus = 0). Évaluation toutes les 50 époques sur 10⁴ échantillons du sampler appris (pas du buffer), échantillons sauvegardés ; contrôleurs sauvegardés aux époques 100 à 500 et en fin de run.

## Mesures (analyse locale sur CPU après rapatriement, `scripts/selftrain_analyse.py`)
- **TV brut SANS poids** des échantillons du sampler appris :
  - par Voronoï (règle) et par bassin (rapporté), **moyenné sur les 3 dernières évaluations** (époques 350, 400, 450 ; écart-type rapporté) ; échantillons de fin de run (10⁴) rapportés à côté ;
  - **couverture** (cellules ou bassins avec p̂ ≥ 0,25 p) **à chaque évaluation et en fin de run** ;
  - **Q1** : rapport de la distance médiane au minimum le plus proche à celle de π, quench et minima de `basin_lib`. Seuil de S2 : ratio ≤ 1,5 (valeur de l'étape 3, médiane exacte 5,23 à d = 10).
- **Énergies et gradients par mise à jour** pour (a), (t), (s), lus dans le code et dans les journaux (`sbank/cost_cum`).
- **ESS de (t) à chaque époque** (`tis/ess_alpha`, `tis/ess_w`).
- Rapportés aussi : taille de la banque et minima ajoutés en cours d'entraînement, acceptations des sauts, TV du buffer, temps par époque.

## Règles
- **S1 ((32, 8, 8) × 0,5)** : couverture brute **8/8** ET TV brut(s) ≤ 0,5 × TV brut(a) (Voronoï, moyennes sur les 3 évaluations).
- **S2 (MoS-w d = 10)** : **≥ 9 bassins sur 10** à la fin, avec Q1 vrai.
- **S3 ((128, 8, 6) × 0,75)** : TV brut(s) ≤ 0,5 × TV brut(a) ET TV brut(s) ≤ TV brut(t). Valeurs de l'étape B (Voronoï, 3 évaluations) : (a) 0,196, (t) 0,113 ; le seuil est donc 0,098.
- **S4 (sûreté)** : Q1 de (s) au plus 10 % plus mauvais que celui de (a), moyenné sur les 3 évaluations, dans chaque cellule.
- **Précisions déclarées (miennes, avant tout résultat)** :
  - « à la fin » (S2) et « couverture » (S1) sont lus sur **chacun des 4 jeux d'échantillons** (3 évaluations et fin de run), pas sur un seul instantané, au nom de la règle permanente de variabilité entre évaluations. La lecture sur le seul jeu final est rapportée à côté ;
  - Q1 de S2 est évalué sur les mêmes 4 jeux ;
  - la couverture par bassin de S2 utilise la même définition que le Q2 de l'étape 3 (p̂ ≥ 0,25 p).
- **Lecture fixée d'avance (de l'utilisateur)** : S1 et S3 vrais = l'auto-entraînement sans poids donne un sampler aux bonnes masses là où les méthodes à poids échouent (contribution à défendre). S2 est un bonus.
- **Limites annoncées** : une graine d'entraînement ; 500 époques de reprise ; une valeur de S (10) ; la comparaison n'est pas à coût égal (le bras (s) coûte environ 200 fois plus d'évaluations par époque que (a) : aucune affirmation à coût égal ne sera faite) ; (a) et (t) de la cellule d = 128 viennent d'un autre lot.

## Mes prédictions (sans effet)
- S1 vraie : 55 %. La couverture 8/8 est probable si le buffer est correct (le post hoc le fait, A3), mais 500 époques suffisent-elles à réapprendre 7 modes ?
- S2 vraie : 15 %. À 500 époques depuis l'époque 250, le Q1 d'AS était à 1,96 × π à l'époque équivalente de l'étape 3 (seuil 1,5), et la banque ne corrige pas la forme (A4).
- S3 vraie : 35 %. Le darting à l'entraînement n'a rien fait à d = 128 (étape B), mais ici le buffer lui-même est corrigé, et A1 montre que le post hoc y est au plancher.
- S4 vraie : 80 %.
- S1 et S3 vraies ensemble : 20 %.

## Coût, plafond et arrêts (accord explicite de l'utilisateur : RTX 3070 ≈ 0,134 $/h, plafond dur 0,15 $)
- Offre RTX 3070 ≤ 0,135 $/h, fiabilité > 0,99, ≥ 12 cœurs CPU ; sinon pas de location.
- **Estimation** : 7 runs, au plus 6 en parallèle (un emplacement libre démarre la suite) ; démarrage 4 min ; rapatriement 2 min.
  - Temps de référence de l'essai local de 2 époques (CPU du Mac) : (s) 0,5 s par époque en d = 32 et 10, 1,1 s en d = 128 ; étape B : (a) 0,8 s, (t) 1,2 s, darting 1,9 s par époque à 6 en parallèle.
  - Le run le plus long est (s) à d = 128 : central 1,5 s par époque, soit 13 min ; pessimiste 3 s, soit 25 min.
  - **Total central ≈ 21 min ≈ 0,05 $ ; pessimiste ≈ 35 min ≈ 0,08 $**, sous le plafond de 0,15 $.
- **Aucun nouveau run lancé après la minute 40 ; arrêt d'urgence à la minute 50** : tout est arrêté, les résultats terminés et les contrôleurs sont rapatriés, l'instance est détruite, puis l'utilisateur est prévenu ; création → destruction ≤ 55 min (≤ 0,123 $). Jamais de question pendant que l'instance tourne ; destruction dès les résultats rapatriés. Si l'estimation dépasse le plafond, pas de location.

## Exécution
`scripts/vast_selftrain_remote.sh` (instance) ; `scripts/selftrain_analyse.py` (local, CPU, après rapatriement). Résultats et contrôleurs dans `figures/selftrain_dartbank/`. Section dans RESULTS.md.
