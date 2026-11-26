# Contrôle CPU de H_saut par bassins sur le contrôleur MoS-dk d = 128 × 0,75, époque 1500 (suite de l'étape 5)

Écrit et commité AVANT toute exécution des 5 chaînes. Aucune règle n'est modifiée après coup. Aucune location, aucune nouvelle graine d'entraînement : calcul sur le CPU du Mac, chaque exécution < 1 min, jamais en arrière-plan.

## Contexte
La décision de `rules_dk_step5.md` a rouvert la piste (MoS-dk d = 128 : Q vrai, I vrai, G vrai à S = 1 et 5), mais H_saut (Voronoï) est faux (darting S = 40 : 0,0082 ; MALA seul de même coût : 0,0106) et les marges de G sont au niveau du bruit (un seul contrôleur, une seule réalisation).

## Contrôleur
`figures/dk_step5/controllers/mosdk5_d128_sx0.75_ep1500.pt` (config `results/dk_step5/asbs/results/mosdk5_d128_sx0.75/config.yaml`) : MoS-dk (128, 8, 6), σ_max = 1,602, 1500 époques, graine d'entraînement 0.

## Chaînes et mesures
- **5 chaînes indépendantes** (i = 0 … 4). La chaîne i = un jeu indépendant de n points terminaux d'AS (simulation CPU du contrôleur, `torch.manual_seed(1000 + i)`) ; sur ces points : **darting** post hoc (`dart_lib.darting`, protocole inchangé : quench de 500 échantillons, Hessiennes par différences finies, carte générale, 1 saut + 5 pas MALA locaux par balayage, pas MALA réglé une fois, graine de chaîne 100 + i), instantanés à **S ∈ {1, 5, 40}** d'une même chaîne ; **MALA seul** (même pas, mêmes points de départ, graine 200 + i) au coût mesuré de chaque S.
- **n = 10⁴** si une exécution « darting » (simulation + darting S ≤ 40 + quench des instantanés) d'une chaîne d'essai de graine 99, dont les résultats sont jetés et n'entrent dans aucune moyenne, dure moins de 50 s ; **sinon n = 5000**. Si n = 5000 : les planchers sont recalculés pour n = 5000.
- **Mesures** (moyenne ± écart-type sur les 5 chaînes, écart-type à n − 1) : TV **par bassin** (quench convergée de `rules_dk_step5.md` §3, assignés seulement) contre la **vérité par bassin** (`figures/dk_step5/truth_mos_d128.json`, 2·10⁵ tirages exacts), et TV **par cellules de Voronoï** contre la vérité de Voronoï (10⁶ tirages), pour le darting à S = 1, 5, 40 et le MALA seul à coût égal. Rapport aussi du TV brut d'AS des mêmes jeux. **Planchers par bassin et par Voronoï** à n : médiane de 2000 TV d'un tirage multinomial de taille n sous les proportions de la vérité (bassin, Voronoï).
- Rappel (non recalculé) de l'IS tronqué par bassin à 20 000 trajectoires, tiré de `figures/dk_step5/cmp_mos_d128.json` : TV contre la vérité par bassin (calculé à partir de ses proportions) et contre la vérité de Voronoï.

## Règle pré-enregistrée
**H_saut_bassin :** moyenne sur les 5 chaînes de TV_bassin(darting, S = 40) **≤ 0,5 ×** moyenne sur les 5 chaînes de TV_bassin(MALA seul, au coût de S = 40).
- **Si H_saut_bassin est faux** : la réouverture tient aux pas locaux, pas aux sauts ; **la piste darting est fermée, définitivement.**
- **Si H_saut_bassin est vrai** : rapport, et proposition (sans location) d'une réplication sur une nouvelle graine d'entraînement.
S = 1 et S = 5, les versions par Voronoï et les écarts-types sont descriptifs : aucun verdict. Rien d'autre n'est fait : pas de location, pas de nouvelle graine, pas d'étape 4b.
