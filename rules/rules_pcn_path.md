# Phase 2 (PRÉPARÉE, NON LANCÉE) : MH local dans l'espace des bruits (pCN)

Écrit pour : Hadrien. **Rien n'est exécuté** ; décision après la phase 1 (`rules_imh_path.md`). Les règles ne seront figées (commit avant calcul) qu'après ta décision ; les valeurs ci-dessous sont des propositions.

## Idée
Les innovations browniennes ξ = (ξ_1, …, ξ_K) ∈ R^{K×d} (K = 200) déterminent la trajectoire : x_{k+1} = x_k + m_k(x_k) + √v_k ξ_k. Proposition pCN : ξ' = ρ ξ + √(1 − ρ²) η, η ~ N(0, I), trajectoire re-simulée, acceptation min(1, w̃(ξ') / w̃(ξ)) avec le w̃ de la phase 1. Le noyau pCN est réversible pour la loi N(0, I) des ξ : la chaîne vise P* (w̃ est la densité de P* par rapport à Q_θ exprimée en ξ). Grille ρ ∈ {0,9 ; 0,99 ; 0,999}.

## Cellule et montage proposés
- Cellule (128, 8, 6) × 0,75, où la dégénérescence est intra-bassin (étape A) ; instantanés des époques 1000, 1250, 1500 (variabilité).
- 512 chaînes par (instantané, ρ), départs = trajectoires iid sous Q_θ (graine fixe), 500 pas, chauffe 20 %. Témoin : IMH (ρ = 0) avec le même budget de trajectoires.
- Mesures : acceptation ; ESS par indicatrice de bassin et par trajectoire simulée (coût comptant chaque proposition) ; TV par bassin des états finaux et moyennés ; Q1 ; TV d'IS sur les mêmes trajectoires.
- Règles proposées : J1 : pour au moins un ρ, TV par bassin ≤ 0,5 × celle de l'IS de même budget (moyennes des 3 instantanés) ; J2 : le même ρ a une acceptation dans [0,1 ; 0,6]. J1 vraie = le MH local en bruit corrige ce que l'IS ne corrige pas ; fausse = rien à gagner dans ce régime.

## Coût
- Simulations par (instantané, ρ) : 512 × 500 = 2,56·10⁵ trajectoires ; 3 × 3 + 3 témoins = 12 configurations ≈ 3,1·10⁶ trajectoires (≈ 6·10⁸ passes de réseau).
- **CPU : hors de portée en < 1 min par exécution** (mesuré : 10⁴ trajectoires d = 128 en 2,6 à 6,9 s en lot de 10⁴, soit au mieux ≈ 13 min pour 3,1·10⁶ trajectoires). Un pilote CPU très réduit (64 chaînes, 100 pas, 1 instantané, ≈ 6·10³ trajectoires par ρ, < 10 s par ρ) ne serait qu'un contrôle de plomberie (ESS non mesurable).
- **GPU** : estimation non mesurée, de 2·10³ à 10⁴ trajectoires par seconde sur RTX 3070 : 5 à 25 min de calcul, plus 5 min d'installation : **≈ 0,02 à 0,05 $** à 0,1 $/h ; plafond proposé 0,10 $. Cluster : `scripts/pbs_pcn_path.pbs` (même dépôt, 1 GPU, 1 h).
- Fichiers nécessaires : `figures/gmmdk_step4/controllers/gmmdk4_d128_sx0.75_ep{1000,1250,1500}.pt` (versionnés), config `figures/gmmdk_step4/configs` ou le JSON de σ_max.
- **Code** : `scripts/pcn_path.py` écrit mais **jamais exécuté** ; il devra passer un test court avant tout lancement.
