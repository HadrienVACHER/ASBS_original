# Contrôle par buffer oracle à (128, 8, 6) × 0,75 : AS entraîné sur des tirages exacts de la cible

**PRÉPARÉ, NON LANCÉ, aucune location sans décision de l'utilisateur.** Écrit et commité avant toute exécution ; aucune règle ne sera modifiée après coup. **Métrique de masse principale : TV par bassin** ; Voronoï en secondaire.

## Question
À (128, 8, 6) × 0,75, le buffer corrigé de (s) est au plancher (Q1 1,001, TV de Voronoï 0,008) alors que le sampler appris est à 0,121 (par bassin 0,047). Le réseau, entraîné 500 époques sur un buffer **parfait**, atteint-il mieux ? Si non, le seuil de S3 (0,098) est hors de portée de cet entraînement, quel que soit le buffer.

## Montage
- **Oracle** : même configuration que (s) à (128, 8, 6) × 0,75 (`gmm_dk_as`, σ_max = 1,602001, même architecture FourierMLP 4 × 128, lr 1e-4, buffer 10⁴, 100 itérations par époque, 512 nouveaux points par époque, nfe 200, tous les t, graine 0, 500 époques), même contrôleur de départ (`results/wf_start/d128/controller_start.pt`).
- **Seule différence** : les points terminaux du buffer ne viennent pas du SDE mais de **tirages i.i.d. exacts de la cible GMM-dk** (composante puis N(μ_k, I) ; `gmm_dk.sample_exact`), 10⁴ au départ puis 512 par époque. Matcher `AdjointVEOracleMatcher` (`adjoint_ve_oracle`) ; le gradient adjoint est calculé aux points exacts, comme pour tout bras.
- Évaluation toutes les 50 époques sur 10⁴ échantillons du **sampler appris** ; contrôleurs sauvegardés.

## Mesures
TV par bassin (principale) et par Voronoï, couverture, Q1, **moyennés sur les 3 dernières évaluations** (époques 350, 400, 450 ; écart-type rapporté) ; fin de run à côté. Références (étape B et ed2617a, mêmes évaluations) : (a) 0,196 / bassin 0,106 ; (t) 0,113 / 0,080 ; (s) 0,119 / 0,037 ; Q1 1,046 / 1,057 / 1,050.

## Règles
- **O1 (de l'utilisateur)** : l'oracle atteint une **TV de Voronoï ≤ 0,098**.
- **O1b (principale selon la convention de métrique)** : l'oracle atteint une **TV par bassin ≤ 0,053** (= 0,5 × 0,106, la moitié de (a) par bassin).
- **O2 (descriptif)** : rapport TV(s) / TV(oracle) par bassin et par Voronoï ; Q1 de l'oracle.
- **Lecture fixée d'avance** :
  - **O1 vraie** : un buffer parfait donne le seuil de S3 ; l'échec de (s) vient alors du buffer pendant l'entraînement (la correction n'arrive pas à tenir le buffer au plancher à chaque lot, ou la dynamique de réapprentissage l'en éloigne) ;
  - **O1 fausse** : même un buffer parfait ne donne pas 0,098 en 500 époques ; le seuil de S3 mesure la capacité d'AS en 500 époques à cette σ_max, pas la qualité du buffer ; le verdict faux de S3 n'est pas un échec de la correction.
  - O1b se lit de la même façon par bassin.
- **Limites annoncées** : une graine ; la même durée ; l'oracle triche (buffer exact), il borne ce que la correction peut espérer, il n'est pas un bras de méthode.
- **Mes prédictions (sans effet)** : O1 vraie 40 % (les évaluations par Voronoï dépendent de la largeur : Q1 1,05 pour tous les bras) ; O1b vraie 65 %.

## Coût estimé (RTX 3070, pas de location sans l'accord de l'utilisateur)
- **1 run seul** (pas de parallélisme) : environ 0,5 à 0,8 s par époque à d = 128 sans darting, soit **4 à 7 min** pour 500 époques. Démarrage de l'instance : de 4 à 10 min (la dernière fois 9,8 min de chargement) ; installation et envoi 3 min ; rapatriement 2 min.
- **Central ≈ 15 à 20 min ≈ 0,03 $ ; pessimiste ≈ 30 min ≈ 0,06 $** au tarif réel de 0,10 à 0,14 $/h (sur la base de l'instance précédente : 0,1017 $/h).
- **Plafond proposé : 0,10 $.** Arrêt d'urgence à la minute 40 depuis la création : résultats rapatriés, instance détruite, utilisateur prévenu.
- Une location de 2 graines ajouterait environ 1 min de calcul (en parallèle) pour le même coût, à décider.

## Exécution (après décision)
`scripts/vast_oracle_remote.sh` (instance) ; analyse locale `scripts/selftrain_analyse.py` étendue (commande `oracle`). Résultats dans `figures/oracle_buffer/`.
