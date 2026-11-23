# Grille exhaustive des pistes (méthodes × régimes), 2026-11-23

*Lecture seule : RESULTS.md (en entier), `audit_fermetures.md`, tous les `rules_*.md`, `memo_cv_failure_modes.md`. Aucun calcul, aucune location.*

**Notes de Hai-Dang : elles ne sont pas dans le dépôt.** Cette source est donc ignorée. Un fichier `~/Desktop/NTU/Recherche/Note.pdf` existe hors du dépôt ; il n'a pas été lu, conformément à la consigne. Les lignes 10 (division à gabarits tronqués sur ellipsoïde, normalisation χ²) et 11 (VC de rapport avec témoin de Rayleigh, étape M3) n'ont donc **pas de définition disponible ici**. Leurs cases sont « non testées, définition hors dépôt », sauf la variante voisine (boîte) de la ligne 10, testée sur DW-4.

## Règle de fermeture (en vigueur à partir d'aujourd'hui)
**Une méthode ne peut être déclarée fermée que si chacune de ses cases pertinentes est soit testée, soit exclue par un théorème** (cité, avec son emplacement). Une case « sans objet » (méthode inapplicable au régime, raison donnée) ne compte pas. Les fermetures antérieures restent écrites telles quelles. Elles valent pour les régimes testés, pas pour la méthode entière, tant que la grille contient des cases pertinentes non testées sur cette ligne.

## Légende
- **T** = testé, avec le verdict tel qu'écrit, l'étiquette (R réfuté / C une configuration / I inconclusif) et le commit des règles → résultats.
- **T₁D** = testé seulement en quadrature 1D (population exacte, pas de vrai AS).
- **N** = non testé.
- **X** = exclu par un théorème (lequel).
- **—** = sans objet (raison donnée).

Régimes :
- **A** : modes trouvés, formes justes, IS qui marche.
- **B** : IS qui s'effondre à grand σ_max.
- **C** : queues lourdes.
- **D** : modes manquants dans les échantillons d'AS.
- **E** : modes absents mais minima connus par la structure (ManyWell).
- **F** : symétrie, poids différents (simplexe).
- **G** : bassins anharmoniques (DW-4).
- **H** : instabilité gaussienne (quadrature 1D).
- **I** : perte de modes au fil de l'entraînement.

Théorèmes cités :
- **Th. C** : localisation ; mémo §2.3 (prouvé ; vérifié en P2c1 et P2d1, eea3ce1 → 2682f5d).
- **Th. 3.8 FM** : aveuglement aux poids de l'article *Failure Modes*, étendu par le mémo §2.2 à toute VC de Stein.
- **Supp.** : support. Un estimateur par repondération ne crée pas de masse là où q ≈ 0 (trivial ; constaté en (32, 8, 8), où l'IS fait passer le TV de 0,433 à 0,324 seulement).

## Tableau

| méthode | A | B | C | D | E | F | G | H | I |
|---|---|---|---|---|---|---|---|---|---|
| 1. AS seul | T (référence) | T (référence) | T (référence) | T (référence) | T (référence) | T (référence) | T (référence) | T (R) | T (référence) |
| 2. IS de chemin post hoc | T (R) | T (C) | T (C) | X (Supp. + Th. C) | N | N | T (C) | — | N |
| 3. AS + IS tempéré | T₁D (P_w vraie) | N | N | X (Th. C ; ESS 10⁻²⁵ mesuré) | N | N | N | T₁D (P_g vraie) | N |
| 4. AS + rééchantillonnage IS du buffer | N | N | N | X (Th. C ; limite déclarée de SSS) | N | N | N | N (prédit : α = 1 de la ligne 3) | N |
| 5. Barker réflexions / orbites | — | — | — | N | T (C) ✗ | T (succès, 3 graines × 2 d) | T (C) ✗ | — | N |
| 6. Darting post hoc | T (R) | T (C) ✓ | T (C) ✗ | X (Th. C) | N | N | N | — | N |
| 7. Darting à l'entraînement | T (C) ✗ | N | N | X (Th. C) | — | — | T (I) | — | N |
| 8. Bain thermique (dose 1) | T₁D (vraie) | N | N | T₁D (vraie, minima connus) | N | N | N | — | N |
| 9. Division, gabarits gaussiens | N (démo 1D ✓) | N | N | X (Th. C) | N | T (C) ≈ Barker | T (R) ✗ | — | N |
| 10. Division, gabarits tronqués (ellipsoïde, χ²) | N | N | N | X (Th. C) | N | N | T, variante boîte (C) ✗ | — | N |
| 11. VC de rapport, témoin de Rayleigh | N | N | N | N | N | N | N | N | N |
| 12. Newton-Stein | T (C) | T (C) | T (C) | T (C) + X (Th. C) | X (Th. C) | X (Th. 3.8 FM) | N | T (R) ✓ | T (C) |
| 13. Amortissement η fixé | T (C) | T (C) | T (C) | T (C) | N | X (Th. 3.8 FM) | N | T (R) | T (C) |
| 14. Banque de minima + darting | — | — | N | N | N | — | N | — | N |
| 15. MALA seul post hoc (témoin) | T (C) | T (C) | T (C) | X (Th. C) | N | N | N | — | N |

**Décompte (135 cases)** : **43 testées** (dont 5 en quadrature 1D seulement), **11 exclues par un théorème** (en comptant comme exclue la case D de Newton-Stein, aussi testée), **64 non testées**, **17 sans objet**.

## Détail par méthode (configuration ; ce qui n'a pas été exploré ; prédiction et actifs pour les cases N ; coût)

### 1. AS seul (référence de toutes les comparaisons)
- **A** : GMM-dk ×0,5 (d = 8, 32, 128, 256 ; 1500 époques ; graines 0 et 1 pour le choix de σ_max ; 788a0e8, f581ea6, 09cf984, 2527419).
- **B** : × 0,75 (d = 128, 256) et (32, 8, 4) × 1,5 et × 2.
- **C** : MoS d = 50 (300 époques, σ_max ∈ {5,8 ; 11,6 ; 23,2}) ; MoS-w d = 50 et 10 (1500 époques, 23,2) ; MoS-dk d = 32 et 128.
- **D** : (32, 8, 8), (8, 8, 8) × 1, toutes les cellules à × 0,25.
- **E** : ManyWell-32 (σ_max 1,41, 2000 époques, graine 0, c4345ea).
- **F** : simplexe d = 24 et 36 (3 graines, 2000 époques).
- **G** : DW-4 (ed8bb67, 94252 ; graines 1 à 4 en continuation).
- **H** : Th. 3.3 de FM vérifié (A_AS = −2,1513, P1a).
- **I** : MoS-w d = 10 (couverture 10 → 5), pilote MoS à 5,8 et 11,6.
- **Non exploré** : σ_max à 1500 époques sur le MoS-w (une seule valeur) ; lr, taille du buffer, réseau, arrêt anticipé ; une graine presque partout.

### 2. IS de chemin post hoc
- **A, T (R)** : ramène le TV au plancher dans les 3 cellules à 8/8 (graines 0 et 1), à d = 128 et 256 (ESS 0,16 à 0,19), et sur MoS-dk d = 32 (ESS 0,164).
- **B, T (C)** : effondrement : ESS 3,7·10⁻³ / 3·10⁻⁴ / 8,9·10⁻³, k̂ 1,0 / 3,7 / 0,67 (f8957c1 → 255f24d) ; un contrôleur par cible.
- **C, T (C)** : MoS d = 50 k̂ 7,4 (raté) ; MoS-w k̂ 1,3 à 8 ; MoS-dk d = 128 ESS 8·10⁻⁴.
- **G, T (C)** : ESS 34 %, k̂ 0,32, petits bassins corrigés contre p_PT (contrôleur 94252).
- **D, X** : Supp. + Th. C.
- **H, —** : l'IS post hoc ne change pas l'itération d'entraînement. Sur une loi courante gaussienne, il repondère exactement.
- **E, N** : les coordonnées jamais visitées sont exclues par Supp. ; les 8 coordonnées à mauvais poids ne sont pas testées. Pas d'actif local (contrôleurs ManyWell sur le cluster) : réentraîner, GPU ≈ 0,07 $.
- **F, N** : contrôleurs perdus (tâche 4/4 arrêtée) ; seuls des points terminaux restent (`~/Desktop/NTU/data`). Réentraîner (256×6, 2000 époques, 3 graines) : GPU ≈ 0,3 $.
- **I, N** : les log w et points terminaux des 12 contrôleurs MoS-w de l'étape 3 sont **locaux** (`figures/mosw_step3/logw`, `results/mosw_step3/step3_results`). **CPU < 1 min.**

### 3. AS + IS tempéré (poids r^α dans la régression)
- **A et H, T₁D** (356ec61 → 859f85e ; σ² = 0,05, α ∈ {0 ; 0,25 ; 0,5 ; 0,9 ; 1}, une itération exacte) :
  - P_w vraie : l'erreur de logit est multipliée par (1 − α) ;
  - P_g vraie : A_α = (1 − α) A_m ;
  - P_m fausse à α = 0,5 ; ESS 10⁻²⁵ à α = 0,9 sur le mode absent.
- **D, X** : Th. C ; l'ESS mesuré interdit la découverte.
- **Non exploré** : aucun vrai AS ; itérations répétées ; panneau (b).
- **B, C, E, F, G, I, N** :
  - **Prédiction** : amortissement en logit à la vitesse α par itération, limité par l'ESS du chemin à la puissance α. En B, l'ESS est de 10⁻⁴ à α = 1 mais de l'ordre de ESS^{α²} pour α petit (conjecture, non vérifiée).
  - **Actifs** : contrôleurs B, C et G locaux pour une continuation.
  - **Coût** : code à écrire (log w de chemin à l'insertion dans le buffer), puis GPU ≈ 0,03 à 0,05 $ par cible.

### 4. AS + rééchantillonnage du buffer par IS (version allégée de SSS)
- **Aucun résultat consigné.** Le code voisin existe (`AdjointVEIMHBufferMatcher`, `AdjointVECorrectedBufferMatcher` : chaînes MH sur le buffer, pas du rééchantillonnage IS) ; aucune section de RESULTS.md ne le teste.
- **D, X** : Th. C ; limite déclarée de SSS (« If D_k misses a mode, resampling cannot recover it »).
- **H, N (prédit)** : c'est la limite population α = 1 de la ligne 3 (exacte en une itération).
- **A, B, C, E, F, G, I, N** :
  - **Prédiction** : corrige les poids des modes visités (A, F, G) ; en B et C, dégénérescence (ESS 10⁻⁴, doublons).
  - **Coût** : code (rééchantillonnage des X₁ par leurs poids de chemin), GPU ≈ 0,05 $ par cible.

### 5. Barker par réflexions / orbites (VC à l'entraînement)
- **F, T (succès)** : γ = 3, centres à seuil, 256×6, 2000 époques, 3 graines, d = 24 et 36 (ec12405) : −86 % et −68 % sur |log R̄|. **Non exploré** : d > 36, coût réduit.
- **E, T (C) ✗** : γ ∈ {1 ; 3}, orbite produit s = 3 (d6aac91, ae74379), une graine. Aggrave les poids, mais **fait apparaître le puits léger sur 4 à 6 des 8 coordonnées jamais visitées**.
- **G, T (C) ✗** : 24 réflexions, γ = 3 (6a955dd) ; continuation γ ∈ {3 ; 10 ; 30} sans effet (aadf33b, réévalué contre p_PT, 4a1342f). Flux d'environ 1 % à κ > 0,7.
- **A, B, C, H, —** : pas de symétrie exacte dans ces cibles. Le même noyau, sans symétrie, est la ligne 7.
- **D, N** : avec une symétrie, les images atteignent les modes absents. Th. C ne s'applique pas, car les requêtes sont globales par symétrie ; c'est observé en E. Aucun test sur un simplexe où AS perd des modes (les géométries hétérogènes G1–G4 n'ont jamais reçu Barker). Pas d'actif ; GPU ≈ 0,1 $.
- **I, N** : prédiction : empêcher la perte en cours de route sur cible symétrique. Pas d'actif.

### 6. Darting post hoc (avec pas locaux)
- **A, T (R) : pas mieux que l'IS.**
  - étape B (f31f199 → bba6570) : R3 vraie, les deux au plancher ;
  - étape 4 (2527419) : G faux, l'IS de coût égal est meilleur ;
  - étape 5, GMM-dk et MoS-dk d = 32 (c47fe51) : G faux.
- **B, T (C) ✓ : G et H_saut_bassin vrais** à d = 128 et 256, × 0,75 (darting au plancher 0,008 à 0,012, MALA seul 0,08 à 0,10). À (32, 8, 4) × 2, la correction vient des pas locaux (f8957c1 → 255f24d). Un contrôleur par cible, 3 jeux de points ; comparateur de coût égal remplacé par une borne.
- **C, T (C) ✗** :
  - MoS-dk d = 128 : G vrai de justesse, H_saut_bassin faux (9d16c5a) ;
  - MoS et MoS-w réels : G faux, sampler raté ;
  - MoS synthétique : G vrai.
- **D, X** : Th. C, minima découverts sur les échantillons ; observé en (32, 8, 8) : 2/8.
- **E, N** : les minima sont connus par la structure, donc les sauts atteignent les puits jamais visités. C'est le cas où le darting peut faire ce que l'IS ne peut pas. Pas d'actif local ; GPU ≈ 0,07 $, puis CPU.
- **F, N** : pas de contrôleur.
- **G, N** : noyau validé (T1′, T2, T3), mais aucune mesure post hoc contre l'IS. Actifs locaux : échantillons ed8bb67, contrôleur 94252, pool PT, `dw4_darting.py`. CPU, quelques exécutions de moins d'une minute.
- **I, N** : contrôleurs et points terminaux MoS-w des étapes 3 et 4 locaux ; CPU < 1 min.

### 7. Darting à l'entraînement (forme identité, γ)
- **A, T (C) ✗** : (8, 8, 8) et (32, 8, 6), × 0,5, continuation de 500 époques, graine 0, γ = 30 (R_train faux) et γ = 1 (effet partiel) ; b3a8c18 → 81f8243.
  - **Erreur de dose** : la théorie (mémo §2.2) donne γ = 2(M − 1)/M = 1,75.
  - **Non exploré** : γ = 1,75 ; graines.
- **G, T (I)** : γ = 30, graines 2 à 4 (07b422f / 8ec415b → 9b4680c). (i) et (ii) vrais, (iii) faux → « non confirmé ». Réanalyse d943965 → 384ea55 : G3′ fausse sur la graine 3, géométrie étiquetée I ; bassins et energy_w2 meilleurs sur 3 graines sur 3. Aussi D3, D10, D100 et focus, graine 1.
- **D, X** : Th. C, minima tirés du buffer. Avec une banque externe, voir la ligne 14.
- **E, F, —** : c'est la ligne 5 (orbite produit et simplexe).
- **H, —** : un seul mode. Quadrature 1D à deux modes (A en 1D) : H_CV soutenue (c204938 → bed2cf3).
- **B, N** : **la plus prometteuse**, puisque le post hoc marche en B. Actifs : contrôleurs d128 et d256 × 0,75 et (32, 8, 4) × 2 locaux, matcher `AdjointVEGMMDartingMatcher` existant. Continuation de 500 époques, γ = 1,75 : GPU ≈ 0,03 à 0,05 $.
- **C, N** : MoS-dk d = 128 × 0,75 (contrôleurs locaux, `figures/dk_step5/controllers`), GPU ≈ 0,03 $.
- **I, N** : MoS-w d = 10 (contrôleurs locaux) : le rafraîchissement des minima peut-il empêcher la perte en cours de route ? GPU ≈ 0,03 $.

### 8. Bain thermique entre minima (dose 1, à l'entraînement)
- **A et D, T₁D** (mémo, eea3ce1 → 2682f5d) : panneau (d) 0,90870 et panneau (c) 0,50004, exacts en une itération (minima connus) ; panneau (b) P2b faux (R = −0,112, sauts queue à queue aux temps tardifs).
- **H, —** : un seul mode.
- **B, C, E, F, G, I, N** :
  - **Prédiction** : dose 1 sans double comptage dans le régime séparé ; correctif de validité aux temps tardifs à écrire (conjecture du mémo).
  - **Coût** : code (variante du matcher darting), puis GPU ≈ 0,05 $ par cible.

### 9. Division, gabarits gaussiens
- **F, T (C)** : d = K = 36, graine 0, 2000 époques, γ = 1 (a8e0c53 échec numérique → fec80d1 valide) : « matches Barker » (0,106 contre 0,117).
- **G, T (R) ✗** :
  - étapes 0, B, 1 et 1′ (k̂ > 0,5 dans les 3 petits bassins) ;
  - tâche 3/4 (18a9a28, 037ae34 → 5dce340) : J ∈ {1, 3, 5, 10} sans gain. Gabarits non rétrécis à 0,3 à 1,2 nat ; la dispersion vient du rétrécissement exigé par la domination.
- **A, N** : démo 1D ✓ (03bacb7 ; masses exactes).
- **D, X** : Th. C, gabarits ajustés sur le buffer.
- **H, —** : un seul mode.
- **B, C, E, I, N** :
  - **Prédiction** : sur des modes gaussiens, les gabarits sont exacts et le rétrécissement est sans fuite (la contrainte de paires est propre à DW-4), donc la division devrait marcher en A et en B. En C, la dispersion de log u sera grande (forme gaussienne contre Student).
  - **Actifs** : contrôleurs GMM-dk et MoS-dk locaux.
  - **Coût** : estimateur hors ligne de Gelfand–Dey sur les échantillons (comme l'étape 0 de DW-4) : CPU < 1 min. À l'entraînement : GPU ≈ 0,05 $.

### 10. Division, gabarits tronqués (postérieure tronquée sur ellipsoïde, normalisation χ²)
- **Définition hors dépôt** (note de Hai-Dang non disponible). Variante voisine testée en **G, T (C) ✗** : composantes tronquées sur une **boîte** (c = 1,5, b4f26a4) ; k̂ de 2,4 à 6,5 dans les 3 petits bassins ; FAIL.
- **D, X** : Th. C.
- **H, —** : un seul mode.
- **Autres, N** : prédiction et coût impossibles sans la définition. Coût de départ : lire la note.

### 11. VC de rapport avec témoin de Rayleigh β ∝ Σ⁻¹μ (note 1, étape M3)
- **Définition hors dépôt : les 9 cases sont N.** Ni prédiction ni pertinence ne sont jugeables ici ; même H (instabilité) n'est pas classable sans la définition. Coût de départ : lire la note.

### 12. Newton-Stein
- **H, T (R) ✓** : Th. A prouvé et vérifié (P1a à P1d ; eea3ce1 → 2682f5d).
- **A et B, T (C)** : (32, 8, 4) × 1 et × 2, Ĥ = 1, 1500 époques, graine 0 (656b270 → 3c3375a) : R1 faux. À × 2, ESS 5,6 fois celui d'AS et 3,4 fois celui de l'amortissement ; à × 1, 1,2 et 1,3 fois.
- **D, T (C) + X** : (32, 8, 8), R2 faux ; Th. C.
- **C et I, T (C)** : MoS-w d = 50 et 10, σ_max 23,2, Ĥ = 1,5 (e5c8b5d → a35b609, amendement b9fd20c). Gel à la loi de base, car Ĥ est la courbure du mode, hors du cadre du Th. A.
- **E, X** : Th. C. **F, X** : les VC de Stein sont aveugles aux poids (mémo §2.2 ; Th. 3.8 FM).
- **Non exploré** : Ĥ_t de la postérieure (régression de Stein sur le buffer) ; plus de graines à × 2. **Exclu par l'utilisateur** : aucune autre variante.
- **G, N** : stabilité sur bassins anharmoniques ; intérêt faible (AS DW-4 stable à σ_max 2). GPU ≈ 0,05 $.
- **Selon la règle de fermeture**, la case G empêche de déclarer Newton-Stein fermé au sens strict, à moins de la juger non pertinente (justification à écrire).

### 13. Amortissement η fixé (η = 0,5, éq. 176)
- **A, B, D, T (C)** : (32, 8, 4) et (32, 8, 8), × 1 et × 2 (656b270 → 3c3375a).
- **C et I, T (C)** : MoS-w d = 50 et 10 (e5c8b5d → a35b609) ; il perd des modes comme AS.
- **H, T (R)** : théorie (multiplicateur −0,577 à σ² = 0,05, mémo §2.1) et résultats réseau de FM (citations vérifiées).
- **F, X** : même point fixe qu'AS, donc aveugle aux poids (Th. 3.8 FM).
- **E et G, N** : intérêt faible ; GPU ≈ 0,05 $.

### 14. Banque de minima + darting (recherche externe par quench depuis la base)
- **Seule la recherche a été testée** : P4, GMM excentrique de SSS d = 50. La trempe depuis N(0, I) trouve le mode lointain dans 0,39 % des tirages, exactement le demi-espace (eea3ce1 → 2682f5d). Jamais combinée au darting.
- **A, B, —** : les minima sont déjà trouvés par la découverte sur les échantillons (8/8) ; c'est la ligne 6.
- **F, —** : la symétrie donne les images ; c'est la ligne 5.
- **H, —** : un seul mode.
- **D, N : le test clé.**
  - **Prédiction** : Th. C ne s'applique pas, car les requêtes sont globales depuis la base. Sur (32, 8, 8), la trempe depuis N(0, σ_max² I) devrait trouver les 8 minima avec environ 1000 tirages (bassins de mesures comparables sous la base). Le darting entre minima connus transfère ensuite la masse vers les modes absents, au rythme de π(y)/π(x).
  - **Actifs** : contrôleurs (32, 8, 8) × 0,5 à × 2 locaux (`results/gmm_dk_window/w1`, `results/gmm_dk_sweep`).
  - **Coût** : CPU < 1 min (petite modification : passer des minima externes à `dart_lib.darting`).
- **I, N** : banque pendant l'entraînement pour empêcher la perte de modes ; code (minima externes dans `AdjointVEGMMDartingMatcher`), GPU ≈ 0,05 $.
- **C, N** : MoS-w d = 10 ; actifs locaux ; CPU.
- **E, N** : les minima de ManyWell sont donnés par la structure ; pas de contrôleur local (GPU ≈ 0,07 $).
- **G, N** : les 5 minima de DW-4 sont connus ; CPU (ed8bb67).

### 15. MALA seul post hoc (témoin)
- **A, T (C)** : à coût égal, d = 128 et 256 × 0,5 : 0,35 et 0,26, ne corrige rien (2527419).
- **B, T (C)** : 0,08 à 0,10 à d = 128 et 256 ; égal au darting à (32, 8, 4) × 2 (255f24d).
- **C, T (C)** : MoS d = 50 : 0,26, ne corrige rien (257767a) ; MoS-dk d = 128 : égal au darting à S = 40 (9d16c5a).
- **D, X** : Th. C, mouvement local.
- **H, —** : un seul mode.
- **E, F, G, I, N** : témoin à joindre à chaque test des lignes 6 et 14 ; même coût.

## Cases non testées, de la moins chère à la plus chère (intérêt pour le critère A : corriger les masses des échantillons bruts *pendant l'entraînement*)

**CPU < 1 min par exécution, actifs locaux, 0 $**
1. **14-D** : banque de minima (trempe depuis la base) + darting post hoc sur (32, 8, 8). Contrôleurs locaux ; petite modification de `dart_lib`. **Intérêt A : élevé.** C'est le précurseur direct d'une banque à l'entraînement, et le seul moyen que la théorie laisse ouvert pour les modes absents.
2. **6-I, 2-I, 15-I, 14-C** : post hoc (IS, darting, MALA, banque) sur les 12 contrôleurs MoS-w de l'étape 3 (log w et points terminaux locaux). **Intérêt A : moyen.** Il montre si la perte de modes en cours de route est rattrapable après coup.
3. **9-A et 9-B** : division (Gelfand–Dey) hors ligne sur des modes gaussiens (GMM-dk × 0,5 et × 0,75, contrôleurs locaux). **Intérêt A : moyen.** Il dit si la division ne marche que sans contrainte de domination, ce qui en fait une candidate pour l'entraînement.
4. **6-G, 15-G, 14-G** : darting post hoc et MALA sur DW-4 (ed8bb67, minima connus) contre l'IS de chemin et p_PT. **Intérêt A : faible.** L'IS marche déjà sur DW-4.
5. **3 et 8 en quadrature 1D** : itérations répétées, panneau (b), α à l'itération ; **4-H** (α = 1). **Intérêt A : faible à moyen** (théorie).

**Code à écrire, puis GPU en centimes**
6. **7-B** : darting à l'entraînement à γ = 1,75 en continuation depuis d128 et d256 × 0,75 et (32, 8, 4) × 2 (contrôleurs locaux ; matcher existant). **≈ 0,03 à 0,05 $. Intérêt A : le plus élevé**, car le post hoc tient en B.
7. **7-A** à la dose théorique γ = 1,75 (réouverture n° 3 de l'audit) : ≈ 0,03 $. Intérêt A : moyen ; l'IS post hoc suffit dans ce régime.
8. **14-D et 14-I à l'entraînement** : banque externe dans `AdjointVEGMMDartingMatcher`, (32, 8, 8) et MoS-w d = 10. ≈ 0,05 $. **Intérêt A : élevé.**
9. **7-C et 7-I** : darting à l'entraînement sur MoS-dk d = 128 et MoS-w d = 10. ≈ 0,03 $ chacun. Intérêt A : moyen.
10. **8-B** : bain thermique à l'entraînement (nouveau matcher). ≈ 0,05 $. Intérêt A : élevé si le n° 6 marche (dose exacte).
11. **3-B, 3-G** : IS tempéré à l'entraînement (log w de chemin à l'insertion). ≈ 0,05 $. Intérêt A : élevé ; c'est un amortissement des poids en logit.
12. **4-A, 4-B, 4-G** : rééchantillonnage IS du buffer. ≈ 0,05 $. Intérêt A : élevé en A et G, faible en B (ESS).
13. **12-G, 13-E, 13-G** : intérêt faible. ≈ 0,05 $.

**Sans actif local : réentraîner d'abord**
14. **Colonne E (ManyWell)** : AS réentraîné (≈ 0,07 $), puis 2-E, 6-E, 14-E et 15-E sur CPU. **Intérêt A : élevé pour 6-E et 14-E** (minima connus par la structure).
15. **Colonne F (GMM simplexe)** : AS et Barker réentraînés (256×6, 2000 époques, 3 graines ; ≈ 0,3 $), puis 2-F, 6-F et 9-F post hoc, et la tâche 4/4 abandonnée. Intérêt A : moyen ; Barker y marche déjà.
16. **5-D** : Barker sur un simplexe où AS perd des modes (≈ 0,1 $). Intérêt A : moyen.

**Définition manquante (lire la note de Hai-Dang d'abord ; coût : lecture)**
17. Lignes **10** (hors variante boîte) et **11** : les 17 cases N.
