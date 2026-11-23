# Contrôle par buffer oracle à (128, 8, 6) × 0,75 : AS et DSM entraînés sur des tirages exacts de la cible

Écrit pour : Hadrien. Version du 2026-11-23 **après accord de l'utilisateur** (deux bras, O1b conservée, accord de location RTX 3070 plafond dur 0,15 $). Écrit et commité **avant** la location et avant tout résultat ; aucune règle ne sera modifiée après coup (amendements datés seulement). **Métrique de masse principale : TV par bassin** ; Voronoï en secondaire.

## Question
À (128, 8, 6) × 0,75, le buffer corrigé de (s) est au plancher (Q1 1,001, TV de Voronoï 0,008) alors que le sampler appris est à 0,121 (par bassin 0,047). Le réseau, entraîné 500 époques sur un buffer **parfait**, atteint-il mieux, et AS est-il alors le bon apprenant, ou le denoising score matching (DSM) fait-il aussi bien ou mieux ?

## Montage commun
- Configuration de (s) à (128, 8, 6) × 0,75 (`gmm_dk_as` : σ_max = 1,602001, FourierMLP 4 × 128, lr 1e-4, buffer 10⁴, 100 itérations par époque, lot 512, 512 nouveaux points par époque, nfe 200, tous les t, graine 0, 500 époques), **même contrôleur de départ** pour les deux bras (`results/wf_start/d128/controller_start.pt`, entraîné par AS : limite déclarée, voir plus bas).
- **Buffer exact** : 10⁴ tirages i.i.d. exacts de la cible GMM-dk au départ (20 lots de 500, le buffer de départ du fichier de l'étape B est ignoré par le matcher), puis **512 tirages neufs i.i.d. exacts à chaque époque** (le buffer est un FIFO de 10⁴ : tout est renouvelé en environ 20 époques). Même taille, même calendrier de rafraîchissement, **mêmes tirages en loi** pour les deux bras (même graine, même code de tirage).
- Évaluation toutes les 50 époques sur 10⁴ échantillons du **sampler appris** (SDE contrôlée, 200 pas), échantillons et contrôleurs sauvegardés. Même SDE inverse et même nombre de pas pour les deux bras.

## Bras
- **(o-AS)** : `AdjointVEOracleMatcher` (`adjoint_ve_oracle`) : régression d'AS ; la cible est −∇E(x₁) au point exact x₁, au pont gaussien x_t | x₀, x₁ du VE.
- **(o-DSM)** : `AdjointVEOracleDSMMatcher` (`adjoint_ve_oracle_dsm`) : même pont x_t | x₀, x₁, même réseau, même sortie u (dérive g² u) ; **aucun gradient d'énergie**. Avec V = σ_max² − σ_min², r = ∫₀ᵗ g² / V, la dérive conditionnelle du pont est u_cond = (x₁ − x_t) / (V (1 − r)) ; perte = Var(x_t | x₁) · ‖u − u_cond‖², avec Var(x_t | x₁) = V r (1 − r).
  - **C'est du denoising score matching** : u_cond diffère du score conditionnel de p(x_t | x₁) par un terme qui ne dépend que de (t, x_t), donc la perte pondérée par la variance est la perte DSM standard (pondération λ(t) = variance). Son minimiseur est le contrôleur optimal u* (la même cible que celle d'AS).
  - Testé (`tests/test_oracle_dsm_matcher.py`, CPU, réussi) : l'espérance conditionnelle de la cible redonne le contrôleur exact d'une cible gaussienne 1D (écarts ≤ 0,01), et la simulation de la SDE avec ce contrôleur redonne la moyenne et la variance de la cible (2,020 / 0,253 pour 2 / 0,25).
  - **Choix déclarés** : (i) la pondération est celle du DSM standard, sans réglage ; elle concentre le poids sur les petits t (avec σ_max/σ_min = 1602, r ≈ 1 dès t ≈ 0,2), comme le schéma géométrique ; une autre pondération n'est pas testée ; (ii) le temps t est tiré uniformément comme pour AS.
- **Contrôle de mémorisation** : requis seulement si le buffer exact n'est pas rafraîchi par des tirages neufs. Il l'est (voir plus haut), donc **pas requis** ; rapporté tout de même, à coût CPU nul, pour les deux bras : distance au plus proche voisin des 10⁴ échantillons du sampler dans le buffer final (10⁴ points sauvegardés) contre la distance au plus proche voisin dans 10⁴ tirages exacts neufs ; descriptif.

## Mesures
TV par bassin (principale), TV de Voronoï, couverture, Q1, **moyennés sur les 3 dernières évaluations** (époques 350, 400, 450 ; écart-type entre évaluations rapporté) ; fin de run (10⁴) à côté ; **pour les deux bras**. Références (étape B et ed2617a) : (a) Voronoï 0,196 / bassin 0,106 ; (t) 0,113 / 0,080 ; (s) 0,119 / 0,037 ; Q1 1,046 / 1,057 / 1,050.

## Règles
- **O1 (de l'utilisateur, o-AS)** : TV de Voronoï ≤ 0,098.
- **O1b (o-AS, principale)** : TV par bassin ≤ 0,053 (= 0,5 × 0,106, la moitié de (a) par bassin).
- **O2 (o-DSM)** : **O1 et O1b pour (o-DSM)**, mêmes seuils, mêmes mesures.
- **O3 (descriptif)** : Q1 des deux bras ; rapports TV(s) / TV(o-AS) et TV(s) / TV(o-DSM) par bassin et par Voronoï ; contrôle de mémorisation ; temps par époque.
- **Lecture fixée d'avance (de l'utilisateur)** :
  - **O1 / O1b faux et O2 vrai** : sur un buffer correct, DSM est le meilleur apprenant et AS n'apporte rien.
  - **O1 vrai** : l'échec de S3 vient de l'interaction avec le buffer évolutif de (s), pas d'AS.
  - **O1 et O2 faux** : limite du réseau ou du budget ; S3 était inatteignable à ce budget.
  - Les cas non listés (O1 vrai et O2 faux ; O1 faux avec O1b vrai) sont rapportés tels quels, sans lecture ajoutée ; une divergence entre Voronoï et bassin est dite.
- **Limites annoncées** : une graine d'entraînement ; 500 époques ; l'oracle triche (buffer exact), il borne ce que la correction peut espérer, il n'est pas un bras de méthode ; **le départ commun est un contrôleur entraîné par AS** (le DSM repart d'un contrôleur AS, ce qui peut le favoriser ou le gêner, non testé) ; pondération DSM standard seule ; deux bras seulement, pas de bras (a) ou (t) relancé.
- **Mes prédictions (sans effet)** : O1 vraie 40 % ; O1b vraie 65 % ; O2 vraie 35 % (le départ AS et la pondération concentrée sur les petits t pénalisent le DSM) ; DSM meilleur qu'AS en TV par bassin : 35 %.

## Coût, plafond et arrêts (accord explicite de l'utilisateur : RTX 3070 ≈ 0,134 $/h, plafond dur 0,15 $ pour les deux bras)
- Offre RTX 3070 ≤ 0,135 $/h, fiabilité > 0,99, ≥ 8 cœurs CPU ; sinon pas de location.
- **Deux runs en parallèle sur la même instance** : (o-AS) 0,5 à 0,8 s par époque seul, 1 à 1,6 s à deux en parallèle (le DSM ne calcule pas ∇E, un peu moins) : **500 époques en 8 à 13 min**. Démarrage 4 à 10 min (dernière fois 9,8 min), installation et envoi 3 min, rapatriement 2 min.
- **Central ≈ 25 min ≈ 0,056 $ ; pessimiste ≈ 40 min ≈ 0,09 $** (à 0,134 $/h), sous le plafond de 0,15 $. Si l'offre trouvée rend l'estimation pessimiste supérieure à 0,15 $, pas de location.
- **Arrêt d'urgence à la minute 50 depuis la création** : tout est arrêté, les résultats terminés et les contrôleurs sont rapatriés, l'instance est détruite, l'utilisateur est prévenu ; création → destruction ≤ 60 min (≤ 0,134 $). Jamais de question pendant que l'instance tourne ; destruction dès les résultats rapatriés.

## Exécution
`scripts/vast_oracle_remote.sh` (instance, deux runs `sd_o_d128` et `sd_od_d128`) ; analyse locale `scripts/selftrain_analyse.py oracle` (CPU). Résultats dans `figures/oracle_buffer/`, grands fichiers dans `results/selftrain_runs/`. Section dans RESULTS.md.
