# Variables de contrôle et modes d'échec d'Adjoint Sampling : ce qu'une VC peut corriger, et ce qu'elle ne peut pas

*Mémo de réflexion, 2026-11-21, branche `div`. Vérifications numériques : règles `rules_cv_failure_modes.md` (commit `eea3ce1`, écrites et commitées avant le script), script `scripts/cv_failure_modes_checks.py`, résultats `figures/cv_failure_modes_checks.json`. Aucun GPU, aucune location, aucun entraînement.*

**Statut des énoncés.** Chaque résultat est marqué : **[prouvé]** (preuve complète ici), **[esquisse]** (preuve esquissée, hypothèses explicites), **[vérifié]** (prédiction pré-enregistrée confirmée), **[réfuté]** (prédiction pré-enregistrée infirmée), **[conjecture]**, **[post hoc]** (constaté après coup, sans règle).

## Résumé

1. **Identité de départ [prouvé].** La cible de régression d'AS s'écrit $Y = Y_{\rm BM} + \sigma_t\,\nabla_{x_1}\log\tilde\pi_t(x_1\mid x_t)$. C'est un *bridge matching* sur la loi courante, qui laisse toute loi $q$ fixe, plus une variable de contrôle (VC) de Stein de gain 1. Le terme de Stein est la seule force qui pousse $q$ vers $\pi$ ; il équivaut à un pas de gradient de taille $\nu_{1|t}$ sur l'énergie de la postérieure. Les trois modes d'échec sont trois défauts de ce pas : il est mal calibré (mode 2) et il est local (modes 1 et 3).
2. **Mode (2), instabilité [prouvé, vérifié].** Ce n'est pas un problème de variance. AS fait un pas de Newton avec la courbure du processus de base ($1/\bar\sigma^2$) au lieu de celle de la cible. Pour des variances égales, le multiplicateur de l'erreur de moyenne vaut $1-F(\rho)$, avec $F(\rho)=\rho\ln\rho/(\rho-1)$. L'instabilité apparaît pour $\rho>\rho_\star=4{,}9216$, ce qui redonne exactement $s_\star=0{,}45076$ de l'article.
   - **Correctif.** La VC « Newton-Stein » change ce pas sans coût en énergie. Elle rend l'itération exacte en une étape pour toute cible gaussienne et toute loi courante.
   - **Mesure.** Le multiplicateur passe de $-2{,}1513$ (AS) à $0{,}0000$.
3. **Mode (3), découverte [prouvé].** Aucune VC dont les points d'évaluation de l'énergie restent locaux ne peut accélérer la découverte : c'est un théorème de localisation qui contient le Th. 3.5 de l'article.
   - Une VC ne « voit » un mode absent qu'au rythme où ses propositions l'atteignent.
   - Elle est donc dominée par une recherche externe qui utilise les mêmes propositions.
   - La piste « VC contre la découverte » est fermée.
4. **Mode (1), poids [esquisse, vérifié en partie].** En théorie, le darting en bain thermique combiné à Newton-Stein règle la dose ($\gamma=1$) et le double comptage.
   - Il est exact en une itération sur les panneaux « mode manquant » et « mauvais poids » de l'article : $0{,}50004$ et $0{,}90870$, contre $0{,}50000$ et $0{,}90869$ pour le contrôle exact.
   - Il surcorrige de 11 % quand $q$ a aussi des erreurs intra-mode (P2b fausse, $R=-0{,}112$), par des sauts « queue à queue » aux temps tardifs.
   - Il n'a pas de niche pratique face à l'IS de chemin post hoc.
5. **Prédiction réfutée.** J'avais prédit qu'une correction partielle de fraction $p$ produirait un mélange géométrique $q^{1-p}\pi^p$. C'est faux (P3 fausse) : le facteur de Feynman–Kac est du même ordre exponentiel que la pénalité géométrique. Au σ² de l'article, la masse transférée est fortement surlinéaire en $p$.
6. **GMM-50 excentrique de SSS.**
   - Aucune des VC proposées ne relève la couverture à budget de particules fixé.
   - Une recherche par trempe depuis $\mathcal N(0,I)$ trouve le mode lointain avec 0,40 % des tirages (mesuré 0,391 % ; 95 % des blocs de 1000 trempes le trouvent).
   - Elle coûte de l'ordre de $10^5$ évaluations de gradient. C'est le mécanisme de recherche qui compte, pas la VC.
7. **Verdict.**
   - Résultat négatif argumenté pour la découverte.
   - Newton-Stein est un stabilisateur gratuit et prouvé ; il mérite un test GPU ciblé sur la sensibilité à $\sigma_{\max}$ (≈ 0,08 à 0,09 $, proposition chiffrée en §7, non lancée).
   - Ce n'est pas la matière d'un article « VC contre les modes d'échec ».

---

## 1. Énoncés exacts des trois articles

### 1.1 Adjoint Sampling (Havens et al., arXiv 2504.11713v3, vérifié sur l'en-tête du PDF)

**Processus et coût terminal.** Éq. (2) : $dX_t=\sigma(t)u(X_t,t)\,dt+\sigma(t)\,dB_t$, $X_0=0$. Éq. (8) : $g(x)=\log p^{\rm base}_1(x)+\frac1\tau E(x)$.

**Cible de régression et sa dérivation.** Adjoint Matching résout le point fixe $u=-\sigma\nabla J(u)$ (éq. 34 de l'annexe). Avec une dérive de base nulle, l'état adjoint « lean » vérifie $\frac{d}{dt}\tilde a=0$ et $\tilde a(1;X)=\nabla g(X_1)$, d'où $\tilde a(t;X)=\nabla g(X_1)$ pour tout $t$ et la perte (11) :
$$\mathcal L_{\rm AM}(u)=\mathbb E_{X\sim p^{\bar u}}\Big[\int_0^1\tfrac12\|u(X_t,t)+\sigma(t)\nabla g(X_1)\|^2dt\Big].$$
La cible est donc $-\sigma(t)\nabla g(X_1)$, qui ne dépend que de l'extrémité.

**Projection réciproque.** Perte RAM (13) :
$$\mathcal L_{\rm RAM}(u)=\int_0^1\lambda(t)\,\mathbb E_{X_t\sim p^{\rm base}_{t|1},\,X_1\sim p^{\bar u}_1}\Big[\tfrac12\|u(X_t,t)+\sigma(t)\nabla g(X_1)\|^2\Big]dt,\qquad \lambda(t)=1/\sigma(t)^2 .$$
Projection (14) : $\mathcal P(u)=\arg\min_v D_{\rm KL}\big(p^v(X)\,\|\,p^{\rm base}(X\mid X_1)\,p^u_1(X_1)\big)$. Théorème 3.2, énoncé « informal » de l'article (cité) :

> « Starting with any control $u_i$, performing steps 1 and 2 to obtain $u_{i+1}$ equivalently satisfies $u_{i+1}=\mathcal P(u_i)-\frac{\delta\mathcal L_{\rm AM}}{\delta u}(\mathcal P(u_i))$ (17) [...] Moreover, the fixed point where $u=\mathcal P(u)$ and $u=u-\frac{\delta\mathcal L_{\rm AM}}{\delta u}(u)$ is the optimal control $u^*$ to (7). »

L'itéré explicite figure dans la preuve du théorème C.3 (annexe C) : « By minimizing $L(u;\tilde u_i)$ point-wise for a fixed $\tilde u_i=\mathcal P(u_i)$, we know the optimal iterates take the form $u_{i+1}(x,t)=-\sigma(t)\,\mathbb E_{p^{\tilde u_i}}[\nabla g(X_1)\mid X_t=x]$. »

**Buffer et coût.** Algorithme 1, lignes citées :

> « 5: $\{X^{(i)}_1\}_{i=1}^n\sim p^{\bar u}_1$ [...] 6: # gradient of energy is evaluated once per sample 7: $\nabla g^{(i)}\leftarrow\nabla g(X^{(i)}_1)$ 8: $\mathcal B\leftarrow\mathcal B\cup\{(X^{(i)}_1,\nabla g^{(i)})\}$ 9: while Inner-Loop do 10: $\{(X^{(j)}_1,\nabla g^{(j)})\}_{j=1}^m\sim\mathcal U(\mathcal B)$ 11: $t^{(j)}\sim\mathcal U([0,1])$, $X^{(j)}_t\sim p^{\rm base}_{t^{(j)}|1}(x\mid X^{(j)}_1)$ »

Table 1, colonne « # E(·) evals per gradient update† » : **0,002** pour AS, 1 pour PIS et DDS, 512 pour iDEM. La note précise : « †The values reported are per sample (i.e., divided by the batch size) and according to the LJ-55 experiment hyperparameters ». Chaque évaluation d'énergie sert donc à de l'ordre de 500 échantillons de régression.

**Processus de base** (annexe B.1). Les transitions sont gaussiennes en forme close. Avec le schedule géométrique (48) : $p^{\rm base}_1=\mathcal N\big(0,\sigma_{\max}^2(1-(\sigma_{\min}/\sigma_{\max})^2)I\big)$ et $p^{\rm base}_{t|1}(X_t\mid X_1)=\mathcal N(\alpha_{t|1}X_1,\alpha_{t|1}\nu_{1|t}I)$ (50).

**Pourquoi c'est scalable** (cité) : « Note that optimizing (13) in Step 2 above is extremely cheap due to the analytical form of the forward base process [...] This means that many gradient updates can be carried out without needing to simulate the controlled process or evaluation of the energy model. »

**Conséquence pour toute VC.** Le coût d'une VC se compte par échantillon de régression. Si elle exige des évaluations d'énergie qui dépendent de $(t,x_t)$, elle multiplie le budget d'énergie d'AS par environ 500 fois leur nombre : elle détruit l'avantage d'AS. Si ses évaluations ne dépendent que de $x_1$, elles peuvent être faites une fois à l'insertion dans le buffer, et le budget n'est multiplié que par (1 + leur nombre).

### 1.2 *Failure Modes of Adjoint Sampling: Instability and Slow Multimodal Adaptation* (soumission anonyme, « Under review as a conference paper at ICLR 2027 »)

**Cadre.** Toute l'analyse se fait à $\sigma(t)=1$ grâce à un changement d'échelle espace-temps (annexe A). Les variances y sont relatives à la variance totale de la référence, $\bar\sigma^2=\int_0^1\sigma^2$. L'itération exacte étudiée est (4) :
$$u_{\rm new}(x_t,t)=-\sigma(t)\,\mathbb E_{p^{\rm rec}_{1|t}(X_1|X_t=x_t)}[\nabla_xg(X_1)].$$
Je garde la numérotation de la consigne : (1) poids, (2) instabilité gaussienne, (3) découverte. L'article les présente dans l'ordre instabilité, découverte, poids.

**Mode (2), instabilité gaussienne.** Théorème 3.3 (cité) :

> « Given the control in Eq. 6, the resulting controlled process $dX_t=u_{G,G}(X_t,t)\,dt+dW_t$ will have the terminal distribution $p^{\rm next}(x_1)=\mathcal N(x_1;m_+,s_+^2)$ where $m_+=\mu+\big(1+\frac1{1-s^2}\log s^2\big)(m-\mu)$ if $\gamma=1$, $\mu+A_m(m-\mu)$ if $\gamma\neq1$ [...] with $\gamma:=\frac{s^2(1-\sigma^2)}{\sigma^2(1-s^2)}$ and $A_m:=\frac{(\gamma-1)(1-s^{2\gamma})-\gamma(1-s^{2(\gamma-1)})}{(1-s^2)(\gamma-1)}$. »

Suite du texte : « When the current variance is correct, $s^2=\sigma^2$, the mean-error multiplier becomes $A_m(\sigma^2;\sigma^2)=1+(1-\sigma^2)^{-1}\log\sigma^2$. If this multiplier is below $-1$, each update reverses the sign of the mean error and increases its magnitude. »

Légende de la figure 1 : instabilité « for $s<s_\star\approx0.45076$ ». Annexe A, éq. (27) : $A(r)=1+\frac{\log r}{1-r}$ avec $r=\sigma^2_{\rm tar}/\bar\sigma^2$, et « Changing $\bar\sigma^2$ while keeping the target fixed can therefore change stability. »

**Mode (3), découverte.** Théorème 3.5 (cité) :

> « Fix $\delta\in(0,1)$ independently of $\varepsilon$, with $\varepsilon^2\ll\delta$. There exists a time-dependent safe region $\mathcal S_t$ carrying all but exponentially small probability under the captured-mode dynamics such that $|u_{G,GMM}(x_t,t)-u_{G,G}(x_t,t)|=\mathcal O(\mathrm{poly}(\varepsilon^{-1})e^{-C\varepsilon^{-2}})$, $x_t\in\mathcal S_t$ and $\mathbb P_{G,GMM}\{X_t\notin\mathcal S_t\}=\mathcal O(\mathrm{poly}(\varepsilon^{-1})e^{-C\varepsilon^{-2}})$, for some $C>0$ independent of $\varepsilon$ and $t$. »

Corollaire 3.6 : « [...] the probability available for discovery of the missing basin is exponentially small in $1/\varepsilon^2$. » L'éq. (109) donne $\tau_{\rm disc}\sim\frac{\sqrt{2\pi}}{\varepsilon}\exp\big(\frac1{2\varepsilon^2}\big)$, avec la réserve des auteurs : « Eq. 109 is a characteristic asymptotic estimate, not a global mixing-time theorem for repeated nonlinear AM iterations. »

**Mode (1), poids.** La proposition 3.7 donne la décomposition exacte en responsabilités. Théorème 3.8 (cité) :

> « Suppose that the current mixture has the correct component locations and variances, $m=\nu$ and $s^2=\sigma^2$, but an inaccurate weight $w\neq\varpi$. Assume that $w$ and $\varpi$ remain bounded away from zero and one as $\varepsilon\downarrow0$. Then, on regions carrying all but exponentially small probability, $u_{\{L,R\}}(x_t,t)=u^G_{\{L,R\}}(x_t,t)+\mathcal O(\mathrm{poly}(\varepsilon^{-1})e^{-c\varepsilon^{-2}})$ [...] The leading-order control depends on the current weight $w$ through $r_L$ and $r_R$, but is independent of the target weight $\varpi$. »

Corollaire 3.9 : « In the infinitely separated limit, every mixture weight $w\in(0,1)$ is a leading-order fixed point of the AM iteration [...] Correction of inaccurate mode weights therefore occurs on an exponentially long characteristic iteration scale. » La proposition E.1 étend le résultat aux variances inégales, avec une borne en variation totale.

**Remèdes testés** (§4.2 et annexe G). Replay activé ou non, croisé avec la « direct velocity damping » de Blessing et al. (2026), éq. (176) : $Y^{(\eta)}_k=(1-\eta)u_{k-1}(X_t,t)+\eta Y_k$ avec $\eta=0{,}5$. Résultats cités :

- « Direct velocity damping substantially improves both the mean and the full terminal distribution without replay. Replay alone provides weaker moderation. »
- Mode manquant : « Damping suppresses these excursions but leaves the missing-mode probability near $10^{-4}$ ».
- Poids : « The final-20 mean left-basin probabilities range from approximately 0.53 to 0.61, far from the target 10/11. »

Ta lecture était donc juste : l'amortissement règle l'instabilité, mais ni lui ni le replay ne règlent les poids ou la découverte.

**Questions ouvertes** (Discussion, citées) :

- « A natural next step is to develop adaptive relaxation or path-space trust-region updates with stability guarantees for the complete AS iteration [...] »
- « Combining AS regression with importance weighting, sequential correction, or annealed exploration may provide this information while retaining some of its computational advantages. Importance weighting itself requires adequate proposal coverage and controlled weight variance; it is therefore a component of a possible remedy, rather than a complete solution. »

### 1.3 SSS (Nguyen, Dau, Thiéry, arXiv 2611.23072v1)

**Mécanisme.**

- **Cibles tempérées** (1) : $\bar\pi_\beta=\rho\,e^{-\beta E}$.
- **Un seul réseau** $r_\theta(z,t,\beta)$ pour le score relatif $r^*=\nabla\log\pi_{\beta,t}+z$ (4).
- **Cible de régression TSM** (5) : $\varphi_{\rm TSM}=\alpha_t(\nabla\log\pi_\beta(X)+X)$, qui vaut $-\alpha_t\beta\nabla E(X)$ pour une référence gaussienne standard.
- **Échantillonneur** : intégrateur exponentiel du processus d'Ornstein–Uhlenbeck inverse.
- **Poids de chemin** (8) :
$$w(y_{0:L})=e^{-\beta E(y_L)}\prod_{n=0}^{L-1}\frac{g_h(y_{n+1}\mid y_n)}{q_{\theta,n}(y_{n+1}\mid y_n,\beta)} .$$
- **Algorithme 1** : à chaque niveau $\beta_k$, propositions, rééchantillonnage par les poids, HMC $\pi_{\beta_k}$-invariant, repondération vers $\beta_{k+1}$ par $v=e^{-(\beta_{k+1}-\beta_k)E}$ (3), buffers FIFO, entraînement conjoint sur tous les niveaux.
- **Initialisation** par SMC adaptatif (annexe A.1).

**Ce que SSS corrige.** Les masses des modes présents dans les buffers. L'ablation de la figure 2 conclut : « the importance-correction, and not the MCMC updates, is what keeps the training data faithful to the relative masses of the target ».

**Ce que SSS ne corrige pas.** Les modes absents. Limitations (citées) :

> « The method depends on adequate coverage of the target distributions by the training buffers. If $\mathcal D_k$ misses a mode, resampling cannot recover it, and short MCMC runs may fail to discover it. The next buffer can then inherit the gap, which repeated training may reinforce. Even when all modes are represented, concentrated importance weights can lead to low effective sample size and many duplicates after resampling. Local MCMC moves can diversify these copies but may not restore the relative masses of distant modes. Self-normalized importance estimates also have finite-sample bias. »

Le « GMM-50 excentrique » est l'« Eccentric GMM » de l'annexe de SSS : $\pi=\frac12\mathcal N(0,\frac1{16}I_{50})+\frac12\mathcal N(\frac34\mathbf 1_{50},\frac1{16}I_{50})$, référence $\rho=\mathcal N(0,I_{50})$. Il ne faut pas le confondre avec le benchmark GMM50 factorisé de SSS, qui a $2^{50}$ modes. Sur ce mélange, à budget d'énergie de SMC fixé (« matching the energy budget of 80,000 particles with ten HMC moves per temperature »), la table 10 montre que les runs qui retiennent le mode lointain passent de 3 % à 1000 particules à 85 % à 80 000 particules. La figure 4 montre qu'avec 80 000 particules SSS fait passer la part de ce mode de 0,23 à 0,49, et qu'avec 1000 particules le mode est perdu et SSS ne le retrouve pas.

**Référence DW-4 (précision demandée).** L'annexe C.1 de SSS dit : « As reference samples, we use the MCMC samples of Klein et al. (2023), obtained from the DEM repository of Akhound-Sadegh et al. (2024). »

- **Ce que c'est.** Ce sont les fichiers DW-4 du dépôt DEM. Notre tempering parallèle (commit `b4e35e5`) a trouvé biaisés les trois splits de ce dépôt (test, train et val) : $p_{\rm PT}=0{,}4127/0{,}4792/0{,}0644/0{,}0374/0{,}0063$ contre $0{,}4762/0{,}4099/0{,}0592/0{,}0490/0{,}0057$ pour `test_split_DW4.npy`, $z\approx12$ sur les deux bassins dominants. Le passage cité ne dit pas quel split SSS utilise ; comme les trois partagent le biais, la conclusion n'en dépend pas.
- **Deux corrections à la consigne.**
  - SSS cite Klein et al. (2023) pour les échantillons, et Köhler et al. (2020) seulement pour la définition de la cible.
  - « Une seule chaîne de Metropolis » n'est pas établi : le nombre de chaînes et le type de MCMC de cette référence ne sont documentés dans aucune source que nous avons lue.
- **Ce qui n'a pas été mesuré.** L'effet sur leurs métriques DW-4 (W2 sur les distances, l'énergie et la géométrie). Le biais porte sur les proportions de bassins, donc ces métriques sont affectées dans une mesure inconnue.

**Équité.** SSS n'est pas une méthode « AS + VC ». Sa régression vise, à chaque rafraîchissement, une loi fixée (le score d'un buffer corrigé par IS), pas un point fixe mobile. Les modes (1) et (2) d'AS ne s'y appliquent donc pas tels quels. Sa limite déclarée est la couverture, l'analogue du mode (3), et elle se situe en amont, dans l'initialisation par SMC.

---

## 2. Ce qu'une VC peut faire, et ce qu'elle ne peut pas faire, mode par mode

### 2.0 Cadre commun

**Conventions VE.**

- Processus : $X_0=0$, $dX=\sigma_tu\,dt+\sigma_t\,dB$, $\nu_t=\int_0^t\sigma^2$, $\kappa_t=\nu_t/\nu_1$, $v_t=\nu_t(1-\kappa_t)$, $\nu_{1|t}=\nu_1-\nu_t$, $p_{\rm base}=\mathcal N(0,\nu_1I)$.
- Pont de base : $X_t\mid X_1\sim\mathcal N(\kappa_tX_1,v_tI)$.
- Postérieure réciproque d'une loi terminale $p$ : $\tilde p_t(x_1\mid x_t)\propto p(x_1)\,\mathcal N(x_t;\kappa_tx_1,v_tI)$.
- La dérive qui reproduit $p$ (transformée de Doob de la base) est
$$u_p(x,t)=\sigma_t\,\nabla_x\log h^p_t(x)=\sigma_t\,\frac{\mathbb E_{\tilde p_t}[X_1\mid x]-x}{\nu_{1|t}},\qquad h^p_t(x)=\mathbb E_{\rm base}\big[\tfrac{p}{p_{\rm base}}(X_1)\,\big|\,X_t=x\big].$$
- On note $u^*=u_\pi$ ; $u_q$ est la projection markovienne de la réciproque de $q$, de loi terminale $q$, c'est-à-dire $\mathcal P(u)$ dans le langage d'AS.

**Lemme 1 (décomposition de la cible d'AS) [prouvé].** On pose
$$Y=\sigma_t\big(s_\pi(X_1)+X_1/\nu_1\big)=-\sigma_t\nabla g(X_1),\qquad Y_{\rm BM}=\sigma_t\,\frac{X_1-X_t}{\nu_{1|t}},\qquad c_S(x_1,x_t)=\nabla_{x_1}\log\tilde\pi_t(x_1\mid x_t)=s_\pi(x_1)+\frac{x_t-\kappa_tx_1}{\nu_{1|t}} .$$
Alors :

- (i) $Y=Y_{\rm BM}+\sigma_t\,c_S$ identiquement.
- (ii) $\mathbb E_{\tilde q}[Y_{\rm BM}\mid x_t]=u_q$.
- (iii) $\mathbb E_{\tilde\pi}[c_S\mid x_t]=0$.
- (iv) La mise à jour exacte d'AS est
$$u_{\rm AS}[q]:=\mathbb E_{\tilde q}[Y\mid x_t]=u_q+\sigma_t\,\mathbb E_{\tilde q}[c_S\mid x_t]=u_q+\sigma_t\,\mathbb E_{\tilde q}[\nabla\log r(X_1)\mid x_t],\qquad r=\pi/q .$$
- (v) $u^*-u_q=\sigma_t\,(\mathbb E_{\tilde\pi}-\mathbb E_{\tilde q})[X_1\mid x_t]/\nu_{1|t}=\sigma_t\,\nabla_{x_t}\log\mathbb E_{\tilde q}[r(X_1)\mid x_t]$.

*Preuve.*

- (i) $Y-Y_{\rm BM}=\sigma_t\big[s_\pi+x_t/\nu_{1|t}+x_1(1/\nu_1-1/\nu_{1|t})\big]$ et $1/\nu_1-1/\nu_{1|t}=-\kappa_t/\nu_{1|t}$, car $\nu_t+\nu_{1|t}=\nu_1$.
- (ii) Formule de la dérive $u_p$ ci-dessus.
- (iii) $\int\nabla\tilde\pi=0$ (décroissance aux bords).
- (iv) $\tilde\pi/\tilde q\propto r$ et $\mathbb E_{\tilde q}[\nabla\log\tilde q]=0$.
- (v) Formule de $u_p$ pour $p=\pi$ et $p=q$, puis $h^\pi/h^q=\mathbb E_{\tilde q}[r]$. □

**Lecture.** Dans le langage du Th. 3.2 d'AS : $\mathcal P(u_i)=u_q$ et $-\frac{\delta\mathcal L_{\rm AM}}{\delta u}(\mathcal P(u_i))=\sigma_t\mathbb E_{\tilde q}[c_S]$. AS fait donc **un pas unité dans la direction de Stein** depuis la projection. Le *bridge matching* pur ($c\equiv0$) laisse toute loi $q$ fixe : toute la force qui ramène vers $\pi$ vient du terme de Stein.

On peut aussi écrire $\sigma_t\mathbb E_{\tilde q}[c_S]=\frac{\sigma_t}{\nu_{1|t}}\mathbb E_{\tilde q}[T(X_1)-X_1]$ avec $T(x)=x+\nu_{1|t}\nabla\log\tilde\pi_t(x)$, un pas de gradient de taille $\nu_{1|t}$ sur l'énergie de la postérieure. Par (v), la correction exacte est au contraire le déplacement moyen d'un noyau $\tilde\pi$-invariant qui mélange parfaitement. Les trois modes d'échec sont trois défauts de ce « noyau implicite » d'AS : pas mal calibré (mode 2), localité (modes 1 et 3).

**Deux classes de VC à moyenne conditionnelle nulle sous $\tilde\pi$.**

- **Stein.** $c=A\nabla_{x_1}\log\tilde\pi+\nabla_{x_1}\!\cdot A$. Alors $\mathbb E_{\tilde q}[c]=\mathbb E_{\tilde q}[A\nabla\log r]$, car $\mathbb E_{\tilde q}[A\nabla\log\tilde q+\nabla\!\cdot A]=0$.
- **Noyau.** $c_K=(K\,\mathrm{id}-\mathrm{id})(x_1)/\nu_{1|t}$ avec $K=K_{t,x_t}$ un noyau $\tilde\pi$-invariant. Alors $\mathbb E_{\tilde q}[c_K]=(\mathbb E_{\tilde qK}-\mathbb E_{\tilde q})[X_1]/\nu_{1|t}$.

Avec la cible $Y+\sum_i\gamma_i\sigma_tc_i$, la nouvelle dérive est $u_{\rm new}=u_q+\sigma_t\mathbb E_{\tilde q}[c_S]+\sigma_t\sum_i\gamma_i\mathbb E_{\tilde q}[c_i]$, et $\pi$ reste point fixe pour tous les $\gamma_i$.

### 2.1 Mode (2), instabilité

**Ce n'est pas une instabilité de variance.** L'article le dit : « These results hold with exact conditional expectations and density evolution ». Une VC qui ne réduit que la variance, c'est-à-dire de moyenne nulle aussi sous $\tilde q$, ne change pas $A_m$. Une VC de moyenne non nulle sous $\tilde q$ change l'application de point fixe, donc son jacobien en $\pi$, donc $A_m$.

**Proposition 2 (AS est un pas de Newton avec la courbure de la base) [prouvé].** Pour une courbure supposée $\hat H$, posons $\Lambda_t(\hat H)=(\nu_{1|t}\hat H+\kappa_tI)^{-1}$ et considérons la famille de mises à jour $u_q+\sigma_t\Lambda_t(\hat H)\,\mathbb E_{\tilde q}[c_S]$.

- $\hat H=I/\nu_1$, la courbure de $p_{\rm base}$, donne $\Lambda_t\equiv I$, car $\nu_{1|t}/\nu_1+\kappa_t=1$. C'est AS.
- Pour une cible gaussienne $\mathcal N(\mu,\Sigma)$, $\hat H=\Sigma^{-1}$ donne $u^*$ exactement (Théorème A, §4).

En dimension 1, en unités de l'article ($\nu_1=1$, $\kappa=\nu$), avec des variances courante et cible égales ($s^2=\sigma^2$) et $\hat H=1/\hat\sigma^2$, le multiplicateur de l'erreur de moyenne vaut
$$A=1-F(\rho),\qquad F(\rho)=\frac{\rho\ln\rho}{\rho-1},\qquad \rho=\hat\sigma^2/\sigma^2 .$$

*Preuve.*

- Ici $\mathbb E_{\tilde q}[c_S]=(\mu-m)/\sigma^2$ ne dépend ni de $x$ ni de $\nu$. L'incrément de dérive est donc déterministe et la pente de la dérive est inchangée : la variance terminale reste $\sigma^2$.
- La dérive est $-(1-\sigma^2)x/\psi(\nu)$ plus une constante, avec $\psi(\nu)=1-\nu(1-\sigma^2)$. Le propagateur vaut $\Phi(1,\nu)=\sigma^2/\psi(\nu)$.
- Le décalage terminal de la moyenne est donc $(\mu-m)\int_0^1\Lambda_\nu/\psi(\nu)\,d\nu$, avec $\Lambda_\nu=\hat\sigma^2/\hat\psi(\nu)$ et $\hat\psi(\nu)=1-\nu(1-\hat\sigma^2)$.
- La décomposition en éléments simples donne $\int_0^1\frac{\hat\sigma^2d\nu}{\hat\psi\psi}=\hat\sigma^2\frac{\ln\sigma^2-\ln\hat\sigma^2}{\sigma^2-\hat\sigma^2}=F(\rho)$. □

Pour AS, $\hat\sigma^2=\bar\sigma^2=1$, d'où $\rho=1/r$ et $A=1-F(1/r)=1+\frac{\ln r}{1-r}$ : c'est exactement l'éq. (27) de l'article. Le seuil $|A|=1$ correspond à $\rho\ln\rho=2(\rho-1)$, soit $\rho_\star=4{,}92155$ et $s_\star=\rho_\star^{-1/2}=0{,}450764$, la valeur $0{,}45076$ de l'article.

**Lecture.** AS est instable dès que la variance de la base dépasse 4,92 fois celle de la cible. Il ne s'agit pas de variance d'estimation, mais d'un pas trop long, calibré sur la mauvaise courbure.

**Lien avec l'amortissement.** Une VC de Stein avec un gain scalaire constant $\lambda-1$ donne la cible $\lambda Y+(1-\lambda)Y_{\rm BM}$, de moyenne $u_q+\lambda(u_{\rm AS}-u_q)$. L'amortissement de Blessing et al. (éq. 176) a pour moyenne $(1-\eta)u_{k-1}+\eta u_{\rm AS}$ : les deux coïncident quand $u_{k-1}=u_q$, c'est-à-dire quand le réseau est déjà réciproque.

Le multiplicateur vaut alors $1-\lambda F(1/r)$. À $\sigma^2=0{,}05$ et $\eta=\lambda=0{,}5$, on obtient $-0{,}577$ : stable, ce qui est cohérent avec le succès de l'amortissement dans l'article. La valeur constante optimale $\lambda=1/F=0{,}317$ annule l'erreur de moyenne. Newton-Stein (§4) annule moyenne et variance pour toute loi courante, sans paramètre.

### 2.2 Mode (1), poids

**Les VC de Stein sont aveugles aux poids [prouvé, dans le cadre du Th. 3.8].** $\mathbb E_{\tilde q}[c]=\mathbb E_{\tilde q}[A\nabla\log r]$, et $\nabla\log r$ est exponentiellement petit sur le support de $q$, sauf dans la couche de transition. L'argument de l'article (décomposition de domaine et queues gaussiennes) s'applique tel quel à toute matrice $A$ à croissance polynomiale. Aucune VC de Stein, Newton-Stein compris, ne corrige donc le mode (1). Vérifié : P2d1, Newton-Stein donne $0{,}50006$ pour une cible à $0{,}909$.

Il faut un noyau qui fasse des sauts entre bassins, c'est-à-dire le darting.

**Dose du darting de nos expériences [esquisse, lecture a posteriori].** Notre terme était $\frac1{M-1}\sum_j\omega_j(y_j-x_1)/\nu_{1|t}$, avec des poids de Barker $\omega_j=W_j/(W_j+W_k)$ si les formes sont appariées. Près de l'équilibre, avec $M$ modes de poids égaux, $\tilde q$ ayant des poids $W+\delta$, le calcul donne
$$\mathbb E_{\tilde q}[c]=\frac{M}{2(M-1)}\cdot\frac{(\mathbb E_{\tilde\pi}-\mathbb E_{\tilde q})[X_1]_{\rm étiquettes}}{\nu_{1|t}} .$$
- **Calcul.** $\sum_{k<j}(m_j-m_k)(\delta_k-\delta_j)/2=-\frac M2\sum_jm_j\delta_j$.
- **Exemples.** Exact pour $M=2$ ; $4/7$ pour $M=8$.
- **Conséquence.** La bonne dose était donc $\gamma\approx1{,}75$. Notre $\gamma=30$ sur GMM-dk était environ 17 fois trop fort, ce qui est cohérent avec le dépassement observé (terme 2 à 9 fois la cible). Le résultat partiel à $\gamma=1$ correspond à environ 57 % de la correction par itération.
- **Bain thermique.** Il choisit l'image $j$ avec une probabilité proportionnelle à $\tilde\pi(y_j)|J_{jk}|$, l'état courant compris. Il donne tout l'écart d'étiquettes : dose $\gamma=1$.

**Proposition B (bain thermique + Newton-Stein) [esquisse ; vérifiée en partie].**

*Hypothèses.* Cible $\pi=\sum_k\varpi_k\mathcal N(\mu_k,\Sigma)$ à forme commune et bien séparée. Loi courante $q$ quelconque dont chaque composante postérieure reste dans sa cellule.

*Construction.* Cartes adaptées à la postérieure : $y_j=x_1+C_t\Sigma^{-1}(\mu_j-\mu_k)$, la translation entre moyennes de composantes de $\tilde\pi_t$. Validité de l'orbite : l'image doit tomber dans la cellule $j$.

*Énoncé.* À termes de recouvrement près,
$$u_q+\sigma_t\Lambda_t\mathbb E_{\tilde q}[c_S]+\sigma_t\mathbb E_{\tilde q}[c_{\rm HB}]=u^* .$$

*Esquisse.*

- Dans le régime séparé, $\Lambda_t\mathbb E_{\tilde q}[c_S]=\sum_kw_k(a'_k-\mathbb E_{\tilde q_k}X_1)/\nu_{1|t}$ : c'est la partie intra-mode avec les étiquettes de $q$.
- Comme $\tilde\pi(a'_j+u)\propto W_j\varphi_C(u)$, le bain thermique donne $p_j=W_j$ quelle que soit la coordonnée intra-mode $u$. Sa contribution est donc $\big(\sum_jW_ja'_j-\sum_kw_ka'_k\big)/\nu_{1|t}$ : la partie étiquettes.
- La somme vaut $(\mathbb E_{\tilde\pi}-\mathbb E_{\tilde q})[X_1]/\nu_{1|t}$. □

Ce résultat règle sur le papier la dose (1 exactement) et le double comptage : chaque terme corrige une partie disjointe de l'écart.

*Vérification numérique.*

- **Décomposition des dérives.** Exacte point par point à 4 décimales pour $\nu\le0{,}7$ (diagnostic 2diag).
- **Panneaux (c) et (d).** Exacts en une itération : $0{,}50004$ et $0{,}90870$.
- **Panneau (b).** P2b est fausse : $R=-0{,}112$ pour une borne de $0{,}1$.
- **Cause, établie a posteriori.** La translation adaptée $\Delta_t=(\mu_R-\mu_L)/(\sigma^2P_t)$ se contracte quand $t\to1$. Le rapport $\Delta_t/\text{écart-type}_t=2/(\sigma^2\sqrt{P_t})$ passe de 7,4 à $\nu=0{,}9$ à 3,7 à $\nu=0{,}99$. Le bain thermique fait alors des sauts d'une queue à l'autre d'une même composante, qui corrigent l'erreur intra-mode déjà corrigée par Newton-Stein : terme +0,044 à $\nu=0{,}97$ et +0,25 à $\nu=0{,}99$, là où la vraie partie étiquettes est nulle.
- **Correctif possible [conjecture, non testé].** Une validité par dominance (l'image doit tomber où sa composante domine), ou couper le bain thermique quand $\Delta_t/\text{écart-type}_t<k$.

**Niche pratique : aucune.** Nos expériences (synthèse en tête de RESULTS.md) montrent que l'IS de chemin post hoc ramène les masses au plancher dès que les modes sont trouvés. Une VC qui ne corrige que les poids n'a donc pas d'usage pratique.

### 2.3 Mode (3), découverte

**Théorème C (localisation) [prouvé].**

*Hypothèses.*

- La VC est $c=\Psi\big(x_1,x_t,\xi,(\log\pi(z),\nabla\log\pi(z))_{z\in Z}\big)$, où $Z=Z(x_1,x_t,\xi)$ est un ensemble fini de points de requête et $\xi$ un aléa auxiliaire. Le facteur $\mathcal N(x_t;\kappa z,v)$ est analytique et peut entrer dans $\Psi$.
- $\Psi$ est invariante par ajout d'une constante à tous les $\log\pi(z)$ (c'est le cas des rapports de Metropolis ou de Barker et des gradients), $L$-lipschitzienne en ces valeurs pour la norme sup, et bornée par $B$.
- $\pi'$ est une autre cible telle que $\log\pi-\log\pi'=\mathrm{const}+e$ avec $|e|+|\nabla e|\le\delta$ sur un ensemble $R$.

*Énoncé.*
$$\big|\mathbb E_{\tilde q}[c_\pi\mid x_t]-\mathbb E_{\tilde q}[c_{\pi'}\mid x_t]\big|\le L\delta+2B\,\mathbb P_{\tilde q\otimes\xi}(Z\not\subset R).$$

*Preuve.* Sur l'événement $\{Z\subset R\}$, les arguments de $\Psi$ diffèrent de moins de $\delta$ à une constante près, d'où un écart d'au plus $L\delta$. Sur le complémentaire, l'écart est d'au plus $2B$. □

**Corollaire C (mode manquant, éq. 9 de l'article) [prouvé, avec la Prop. E.1 de l'article pour le passage à la variation totale].**

*Mise en place.*

- On prend $\pi'=\pi_R=\mathcal N(\nu,\sigma^2)=q$ et une règle de construction de VC valide pour toute cible ($c_{\pi'}$ a une moyenne nulle sous $\tilde\pi'$).
- Alors $\tilde q=\tilde\pi'$, donc $\mathbb E_{\tilde q}[c_{\pi'}]=0$ exactement.
- Pour la cible bimodale, $e(z)=\log(1+e^{-2\nu z/\sigma^2})$ : on peut prendre $R_\delta=\{z\ge z_\delta\}$ avec $z_\delta=\mathcal O(\nu\varepsilon^2\ln(1/\delta))$.

*Borne sur le signal.*

- Supposons les requêtes locales : $|z-x_1|\le r$ pour tout $z\in Z$.
- Moyenner en $x_t$ sous la réciproque de $q$ rend la loi marginale $q$ pour $X_1$. D'où
$$\mathbb E_{x_t}\big|\mathbb E_{\tilde q}[c_\pi\mid x_t]\big|\le L\delta+2B\,q(X_1\le z_\delta+r)\approx L\delta+2B\,e^{-(1-r/\nu)^2/(2\varepsilon^2)} .$$
- Le pas de Girsanov et Pinsker de la Prop. E.1 de l'article transporte cette borne à la masse terminale du bassin manquant, qui reste exponentiellement petite. Ce pas borne $\mathrm{KL}(\mathbb P^{\rm rec}\|\mathbb P^{\rm new})$ par $\frac12\int_0^1\mathbb E_{\rm rec}|u_{\rm new}-u_q|^2dt$, puis la variation totale des lois terminales par Pinsker.
- Ce corollaire contient le Th. 3.5 de l'article (cas $c=c_S$). Il s'applique à toute VC de Stein, Newton-Stein compris, et à toute VC de noyau MALA ou HMC à pas borné.

*Conséquence.* Tant que les itérés restent près de $q$, le temps de découverte est au moins $e^{c/\varepsilon^2}/\mathrm{poly}$. Le passage du gain par itération au nombre d'itérations est une [esquisse] : la stabilité locale est supposée, pas prouvée.

*Vérification.* P2c1 : après une itération, la masse gauche vaut $2{,}2\cdot10^{-5}$ pour AS et $1{,}7\cdot10^{-5}$ pour Newton-Stein, partant de $3{,}7\cdot10^{-6}$ pour $q$, alors que la cible est $0{,}5$.

**Proposition C1 (identité de flux) [prouvé].**

*Énoncé.* Soit $q\propto\pi\mathbf 1_R$ et une VC de noyau avec $K$ $\tilde\pi$-réversible. L'équilibre détaillé donne
$$\mathbb E_{\tilde q}[c_K\mid x_t]=\frac{\tilde\pi(L)}{\tilde\pi(R)}\cdot\frac{\mathbb E_{\tilde\pi|L}\big[\int_RK(Y,dx)\,(Y-x)\big]}{\nu_{1|t}} .$$
- **Preuve.** Les mouvements de $R$ vers $R$ s'annulent par antisymétrie ; il reste le flux de $R$ vers $L$, qu'on retourne par équilibre détaillé.
- **Direction.** Le terme pointe vers le mode absent si et seulement si le noyau sait *sortir* de ce mode vers le mode présent.
- **Norme.** Cote postérieure du mode absent × probabilité d'échappement × distance / $\nu_{1|t}$.
- **Cas du bain thermique avec cartes exactes.** La probabilité d'échappement vaut $W_R$, ce qui redonne tout l'écart d'étiquettes.
- **Cas des noyaux locaux.** La probabilité d'échappement est exponentiellement petite (Théorème C).
- **Composante gaussienne.** Si $q$ est la composante gaussienne plutôt que la restriction, l'écart est exponentiellement petit.

**Proposition C2 (représentation de Feynman–Kac) [prouvé comme identité ; approximation réfutée].**

*Énoncé.* Soit la dérive interpolée $D_p=D_q+p(D_*-D_q)$ (horloge $\nu$). Posons $H=h_q^{1-p}h_*^p$. Alors $\partial_\nu H+\frac12\Delta H=-VH$ avec $V=\frac12p(1-p)|D_q-D_*|^2\ge0$, d'où
$$q_{\rm new}(x)\propto q(x)^{1-p}\pi(x)^p\,\mathbb E_{\rm pont\,0\to x}\Big[e^{\int_0^1V(X_s,s)\,ds}\Big].$$

*Ce qui a été réfuté.*

- J'avais prédit que le facteur de pont restait d'ordre 1, c'est-à-dire une loi de mélange géométrique : P3 est fausse.
- Le long des ponts qui finissent dans le mode absent, $|D_q-D_*|\to|s_q-s_\pi|\approx2\nu/\sigma^2$ quand $t\to1$. Le facteur vaut donc $e^{\Theta(p(1-p)/\varepsilon^2)}$, du même ordre que la pénalité géométrique $e^{-2p(1-p)/\varepsilon^2}$.

*Mesures.*

- **À $\sigma^2=0{,}16$ (P3, pré-enregistrée).** Masse gauche $0{,}136$, $0{,}315$, $0{,}430$ pour $p=0{,}5$, $0{,}75$, $0{,}9$, contre $0{,}023$, $0{,}082$, $0{,}244$ prédits par la loi géométrique et $0{,}25$, $0{,}375$, $0{,}45$ pour le mélange arithmétique.
- **Au σ² de l'article, $0{,}05$ (3b, a posteriori).**

| $p$ | masse gauche après une itération | proportion de la cible (0,5) |
|---|---|---|
| 0,1 | $4{,}1\cdot10^{-5}$ | moins de 0,01 % |
| 0,25 | $1{,}5\cdot10^{-3}$ | 0,3 % |
| 0,5 | $0{,}067$ | 13 % |
| 0,75 | $0{,}294$ | 59 % |
| 0,9 | $0{,}426$ | 85 % |

La loi géométrique se trompe de plusieurs ordres de grandeur. Une VC qui fournit une fraction modeste de la correction d'étiquettes au mode absent transfère donc peu par itération, mais pas une quantité exponentiellement petite.

**Quantification dans le cadre de l'article ($\nu=1$, $\sigma^2=0{,}05$, $\varepsilon=0{,}2236$).**

| mécanisme | gain d'une itération ou durée de découverte | statut |
|---|---|---|
| AS | $\tau\approx\frac{\sqrt{2\pi}}\varepsilon e^{1/(2\varepsilon^2)}\approx2{,}5\cdot10^5$ itérations (éq. 109) ; incrément mesuré $1{,}8\cdot10^{-5}$ par itération | |
| toute VC locale | même ordre exponentiel (Th. C) ; Newton-Stein mesuré $1{,}4\cdot10^{-5}$ | |
| VC « de type iDEM », une proposition (C, §3) | $4{,}6\cdot10^{-3}$ par itération, environ 250 fois AS, ce qui équivaut sur la courbe 3b à une fraction uniforme $p\approx0{,}3$ [post hoc]. Nombre d'itérations jusqu'à $0{,}5$ non mesuré : de l'ordre de 10 si le gain reste proportionnel à l'écart, de l'ordre de 100 s'il reste constant. Taux d'atteinte du bassin d'ordre $\varepsilon$ par proposition [conjecture]. | post hoc et conjecture |
| darting en bain thermique, minimum connu | 1 itération ($0{,}50004$) | |

En 1D, trouver le minimum par trempe depuis des tirages de la base est trivial : la moitié des tirages tombe à gauche.

En dimension $d$, une proposition globale tirée de la base atteint le mode absent avec une probabilité de l'ordre de $e^{-\mathrm{KL}(\pi_L\|p_{\rm base})}$. Pour le mode lointain du GMM-50 excentrique de SSS, $\mathrm{KL}\approx60$ nats, soit environ $10^{-26}$ par requête.

**Proposition C3 (domination par la recherche) [esquisse].**

- Par le Théorème C, une VC ne porte d'information sur $L$ que lorsqu'une de ses requêtes tombe dans $L$.
- Les mêmes requêtes, utilisées comme recherche autonome, trouvent $L$ avec la même probabilité.
- Une fois $L$ trouvé, le bain thermique (ou l'IS) transfère toute la masse en une itération.
- Dans une régression, au contraire, une requête heureuse est moyennée avec les autres échantillons du même voisinage en $(t,x_t)$.
- À nombre de requêtes égal, aucune VC ne fait donc mieux que « recherche puis bain thermique » ; au mieux elle fait aussi bien.

---

## 3. Trois VC candidates

### A. Newton-Stein (VC de Stein préconditionnée)

- **Définition.** $c_A=(\Lambda_t-I)c_S$ avec $\Lambda_t=(\nu_{1|t}\hat H+\kappa_tI)^{-1}$. La cible devient
$$\hat Y=Y+\sigma_tc_A=\Lambda_tY+(I-\Lambda_t)Y_{\rm BM},$$
une interpolation matricielle, dépendant du temps, entre la cible d'AS et celle du *bridge matching*.
- **Moyenne nulle.** $\Lambda_t$ ne dépend que de $t$ (on peut aussi le faire dépendre de $x_t$), et $\mathbb E_{\tilde\pi}[c_S]=0$. Si $\Lambda$ dépendait de $x_1$, il faudrait ajouter $\nabla\!\cdot\Lambda$.
- **Effet sur l'application de point fixe.** $u_{\rm new}=u_q+\sigma_t\Lambda_t\mathbb E_{\tilde q}[c_S]=\mathcal P(u)-\Lambda_t\frac{\delta\mathcal L_{\rm AM}}{\delta u}(\mathcal P(u))$ : un pas de gradient fonctionnel préconditionné.
- **Information injectée.** Une courbure ; aucune information nouvelle sur l'énergie.
- **Coût par échantillon de régression.** Zéro évaluation d'énergie ou de gradient : $s_\pi(X_1)$ est déjà stocké dans le buffer. $\hat H$ s'estime sans évaluation par la régression de Stein $\hat H=\mathrm{Cov}(-s_\pi(X),X)\,\mathrm{Cov}(X)^{-1}$ sur le buffer, par grappe pour un mélange. Pour une cible gaussienne, cette régression donne $\Sigma^{-1}$ exactement, quelle que soit $q$, car $-s_\pi$ est affine.
- **Variance.** Le terme reste borné quand $t\to1$ : $\Lambda_t=I-\nu_{1|t}(\hat H-I/\nu_1)+\mathcal O(\nu_{1|t}^2)$, ce qui compense la singularité en $1/\nu_{1|t}$ de $Y_{\rm BM}$.
- **Cas d'échec.**
  - $\hat H$ mal estimée : stable si $\hat H\succeq H/\rho_\star$ (Corollaire A1).
  - Directions plates ou de courbure inférieure à celle de la base : $\Lambda>I$, donc sur-relaxation. Version sûre : $\hat H\leftarrow\max(\hat H,I/\nu_1)$, qui garantit $\Lambda\preceq I$.
  - Bassins anharmoniques ou à queues lourdes : $\Lambda$ de Laplace seulement approchée.
  - Toujours aveugle aux poids et à la découverte (Théorème C).

### B. Darting en bain thermique adapté à la postérieure (avec A)

- **Définition.** Minima $m_k$ et hessiennes $H_k$, découverts ou donnés. Images $y_j=a'_j+A_{jk}(x_1-a'_k)$ entre moyennes de composantes postérieures ; pour une hessienne commune, $y_j=x_1+C_tH(m_j-m_k)$. On ne garde que les orbites valides. Choix de l'image par bain thermique, $p_j\propto\tilde\pi(y_j)|J_{jk}|$, puis $c_B=\big(\sum_jp_jy_j-x_1\big)/\nu_{1|t}$.
- **Moyenne nulle.** C'est un pas de Gibbs sur l'étiquette de l'orbite, donc $\tilde\pi$-réversible.
- **$\mathbb E_{\tilde q}[c_B]$.** La partie étiquettes de l'écart (Proposition B). Avec A, la mise à jour vaut $u^*$ à termes de recouvrement près : dose 1, pas de double comptage tant que $\Delta_t/\text{écart-type}_t$ reste grand.
- **Information injectée.** $\tilde\pi(y_j)$ aux images, c'est-à-dire les masses relatives des modes connus.
- **Coût.**
  - Cartes adaptées à la postérieure : $M-1$ évaluations d'énergie par échantillon de régression, soit environ $500(M-1)$ fois le budget d'AS. On peut tabuler sur $n_t$ instants, soit $n_t(M-1)$ évaluations par échantillon de buffer.
  - Cartes des minima de la cible : $M-1$ évaluations par échantillon de buffer, précalculées, mais seulement approchées (panneau (c) : $0{,}424$ au lieu de $0{,}5$).
  - À ajouter : la découverte des minima par trempe et les hessiennes.
- **Variance.** Déplacements bornés divisés par $\nu_{1|t}$ ; aux temps tardifs, les poids dégénèrent.
- **Cas d'échec.** Sauts « queue à queue » aux temps tardifs (mesuré : $R=-0{,}112$). Minima inconnus. Bassins anharmoniques. $M^2$ paires.

### C. VC « de type iDEM » (proposition globale)

- **Définition.** $y\sim Q_t(\cdot\mid x_t)=\mathcal N(x_t,\nu_{1|t}I)$, la postérieure de la base, ou bien $y\sim p_{\rm base}$. Poids de Barker $\alpha=\phi(y)/(\phi(y)+\phi(x_1))$ avec $\phi=\pi/p_{\rm base}$ : pour $Q_t$, les facteurs de pont se simplifient. Puis $c_C=\alpha(y-x_1)/\nu_{1|t}$. Avec $N$ propositions, on obtient un i-SIR.
- **Moyenne nulle.** Noyau d'indépendance de Barker, $\tilde\pi$-réversible.
- **$\mathbb E_{\tilde q}[c_C]$.** Donné par l'identité de flux (C1). Quand $N\to\infty$, il tend vers $(u^*-u_q)/\sigma_t$ : c'est l'estimateur Monte Carlo de score d'**iDEM**.
- **Information injectée.** Des énergies globales en des points frais.
- **Coût.** Variante testée : une évaluation d'énergie par échantillon de régression et par proposition, soit environ 500 fois le budget d'AS, exactement le coût qu'AS a été conçu pour éviter (iDEM : 512 évaluations par mise à jour dans la table 1). Variante tirée de $p_{\rm base}$ : une évaluation par échantillon de buffer, non testée.
- **Variance.** Grande : atteintes rares multipliées par de grandes distances divisées par $\nu_{1|t}$.
- **Cas d'échec.** La grande dimension, avec un taux d'atteinte en $e^{-\mathrm{KL}}$, comme l'IS depuis la base.

---

## 4. Le candidat retenu : Newton-Stein

**Théorème A (exactitude en un pas pour toute cible gaussienne) [prouvé ; vérifié].**

*Hypothèses.* $\pi=\mathcal N(\mu,\Sigma)$ avec $\Sigma\succ0$. Loi courante $q$ quelconque, de second moment fini. Cible $\hat Y=Y+\sigma_t(\Lambda_t-I)c_S$ avec $\Lambda_t=(\nu_{1|t}\Sigma^{-1}+\kappa_tI)^{-1}$.

*Énoncé.*

- (a) $\mathbb E_{\tilde\pi}[\hat Y-Y\mid x_t]=0$ pour tout $x_t$ : $\pi$ reste point fixe.
- (b) $\mathbb E_{\tilde q}[\hat Y\mid x_t]=u^*(x_t,t)$ pour tout $(x_t,t)\in\mathbb R^d\times[0,1)$.

Une itération externe exacte envoie donc **toute** loi $q$ sur $\pi$.

*Preuve.*

- (a) $\Lambda_t$ ne dépend que de $t$, et $\mathbb E_{\tilde\pi}[c_S]=0$.
- (b) Le Lemme 1 donne $\mathbb E_{\tilde q}[\hat Y]=u_q+\sigma_t\Lambda_t\mathbb E_{\tilde q}[c_S]$.
  - $\tilde\pi_t=\mathcal N(a_t(x_t),C_t)$ avec $C_t=(\Sigma^{-1}+(\kappa_t/\nu_{1|t})I)^{-1}$, car $\kappa_t^2/v_t=\kappa_t/\nu_{1|t}$.
  - $c_S=-C_t^{-1}(x_1-a_t)$ est affine en $x_1$, donc $\mathbb E_{\tilde q}[c_S]=C_t^{-1}(a_t-\mathbb E_{\tilde q}X_1)$ quelle que soit $q$.
  - $\Lambda_t=C_t/\nu_{1|t}$, d'où $\sigma_t\Lambda_t\mathbb E_{\tilde q}[c_S]=\sigma_t(\mathbb E_{\tilde\pi}-\mathbb E_{\tilde q})[X_1]/\nu_{1|t}=u^*-u_q$ par le Lemme 1 (v). □

*Remarques.*

- **Hypothèse H1 inutile.** $u_q$ est la projection de la loi des données elle-même.
- **Avec le replay.** Le buffer est un mélange de lois passées ; le résultat s'applique à ce mélange.
- **Moyenne de la cible inutile.** $\Lambda_t$ ne fait intervenir que $\Sigma^{-1}$, pas $\mu$.

**Corollaire A1 (robustesse) [prouvé ; vérifié].** En 1D, avec des variances courante et cible égales et une courbure scalaire supposée $\hat H=1/(\rho\sigma^2)$, le multiplicateur vaut $1-F(\rho)$. Il est stable pour tout $\rho<\rho_\star=4{,}92$ : la courbure peut être sous-estimée jusqu'à un facteur 4,92. Pour $\rho<1$, la mise à jour est sous-relaxée : plus lente, mais stable.

Mesures (P1c) : $A=0{,}3069$ ; $-0{,}3863$ ; $-0{,}8482$ ; $-1{,}1497$ pour $\rho=0{,}5$ ; 2 ; 4 ; 6, contre $0{,}3069$ ; $-0{,}3863$ ; $-0{,}8484$ ; $-1{,}1501$ prédits. La variance terminale reste $\sigma^2$ à $10^{-4}$ près.

**Corollaire A2 (sécurité) [prouvé].** Si $\hat H\succeq I/\nu_1$, alors $\Lambda_t\preceq I$ : Newton-Stein ne fait jamais un pas plus long qu'AS. C'est un amortissement pur, fort aux petits $t$, où $\Lambda\approx(\nu_1\hat H)^{-1}$, et nul quand $t\to1$.

**Fait-il mieux que les alternatives ?**

- **Contre l'amortissement à $\eta$ fixé** (Blessing et al. 2026, testé par l'article).
  - Sur une gaussienne, Newton-Stein est exact en un pas pour la moyenne et la variance, là où l'amortissement contracte géométriquement (facteur $-0{,}577$ à $\sigma^2=0{,}05$, $\eta=0{,}5$).
  - Il n'a pas de paramètre, son gain dépend du temps, et il n'a pas besoin du réseau précédent.
  - Mais l'article montre que l'amortissement suffit déjà à stabiliser le cas gaussien en pratique (erreur finale 0,002). **Le gain pratique de Newton-Stein sur l'instabilité seule n'est pas démontré.**
- **Contre l'IS de chemin post hoc, SSS ou un MCMC local.** Ils ne stabilisent pas l'entraînement d'AS. Newton-Stein ne fait rien pour les poids ni pour la découverte (Théorème C), et il n'apporte rien à SSS, dont la régression n'est pas une itération de point fixe mobile.
- **Ce qui le maintient en vie.** C'est le seul des trois candidats qui ait une niche (stabilité d'AS seul, à coût nul) et une hypothèse pratique testable : la sensibilité à $\sigma_{\max}$.
  - En VE, $\bar\sigma^2=\nu_1\approx\sigma_{\max}^2$. Le pas d'AS est calibré sur $1/\sigma_{\max}^2$, ce qui borne $\sigma_{\max}\lesssim2{,}22\,\sigma_{\rm mode}$ pour la stabilité intra-mode, alors que l'exploration demande un grand $\sigma_{\max}$. Newton-Stein supprime la première contrainte.
  - Compatible a posteriori avec nos données : sur la cellule GMM-dk (32, 8, 4), à couverture 8/8, l'ESS de l'IS de chemin décroît de manière monotone quand $\sigma_{\max}$ augmente.

| $\sigma_{\max}$ (× règle) | $\rho=\sigma_{\max}^2/\sigma_{\rm mode}^2$ | $1-F(\rho)$ | ESS de l'IS de chemin |
|---|---|---|---|
| 1,12 (× 0,5) | 1,25 | −0,12 | 0,875 |
| 1,68 (× 0,75) | 2,8 | −0,60 | 0,50 |
| 2,24 (× 1) | 5,0 | −1,01 | 0,25 |
| 3,35 (× 1,5) | 11 | −1,66 | 0,056 |
| 4,47 (× 2) | 20 | −2,15 | 0,011 |

  - **[post hoc, non démontré]** : d'autres causes, comme la variance des poids de chemin qui croît avec $\sigma_{\max}$, sont possibles.

---

## 5. Comparaison avec les alternatives

| problème | AS | AS + A (Newton-Stein) | AS + B (bain thermique) | AS + C (type iDEM) | IS de chemin post hoc | SSS | MCMC local | recherche externe + IS de Laplace |
|---|---|---|---|---|---|---|---|---|
| instabilité (2) | instable au-delà de $\rho_\star=4{,}92$ | exact pour une cible gaussienne, coût nul | – | – | ne stabilise pas l'entraînement | non concerné | – | – |
| poids (1) | aveugle | aveugle (Th. C) | exact en une itération si les minima sont connus ; coût $(M-1)$ E par échantillon | partiel | corrige si l'ESS est suffisant | corrige (IS et rééchantillonnage) | seulement si les masses sont déjà presque justes | corrige |
| découverte (3) | $e^{1/2\varepsilon^2}$ | aucun gain (Th. C) | si et seulement si le minimum est dans la liste, donc problème de recherche | 1D : polynomiale ; $d$ grand : $e^{-\mathrm{KL}}$ | non | non (limite déclarée) | non | coût de la recherche |

**GMM-50 excentrique de SSS : prédictions quantitatives à budget de particules fixé.**

- **A.** Aucun gain de couverture. La couverture se décide dans l'initialisation SMC, que A ne touche pas, et A est de toute façon locale (Théorème C). Prédiction : 3 % à 1000 particules et 85 % à 80 000, comme sans A.
- **B.** La couverture est la probabilité que le minimum lointain figure dans la liste.
  - Si la liste vient de trempes faites sur le buffer d'AS, cette probabilité ne dépasse pas la couverture d'AS.
  - Si la liste vient de $N_q$ trempes depuis $\mathcal N(0,I_{50})$, elle vaut $1-(1-0{,}0040)^{N_q}$, soit **98 % pour $N_q=1000$**. Vérifié par P4 : $\hat p=0{,}391\,\%$, accord parfait avec le demi-espace $\langle x_0,\mathbf 1\rangle/\sqrt{50}>\tfrac34\sqrt{50}/2$, et 95 % des blocs de 1000.
  - Coût : environ $2\cdot10^5$ évaluations de gradient pour 1000 trempes de 200 itérations (P4), ou environ $2\cdot10^4$ avec 20 itérations, qui suffisent ici [post hoc]. C'est moins qu'une seule température de la SMC de la table 10 : 80 000 particules × 10 pas HMC, soit au moins $8\cdot10^5$ évaluations de gradient par température. Le nombre de températures n'est pas donné dans ce passage.
  - **Ce gain vient de la recherche, pas de la VC.** SSS pourrait l'exploiter de la même façon : ajouter les minima trouvés avec une proposition de Laplace et des poids d'IS exacts. Sur ce mélange gaussien, l'IS de Laplace est parfaite.
- **C.** Une requête tirée de la base atteint le mode lointain avec une probabilité de l'ordre de $e^{-60}$ : aucun gain.
- **Lecture juste de l'expérience de SSS.** Elle est construite pour isoler le goulot du chemin de tempérage (masse du mode lointain $1{,}2\cdot10^{-4}$ vers $\beta\approx0{,}2$). Elle ne prétend pas que ce mode est difficile à trouver. Une descente de gradient depuis la référence ne passe pas par ce goulot.

**Ce qu'une VC ajouterait à SSS.** Rien de substantiel. Les poids sont déjà corrigés par l'IS, la découverte se joue en amont, et l'instabilité de type AS ne s'applique pas.

---

## 6. Vérifications numériques (règles pré-enregistrées dans `rules_cv_failure_modes.md`, commit `eea3ce1`)

**Cadre.** Unités de l'article ($\nu_1=1$). Solveur de Fokker–Planck conservatif Crank–Nicolson repris du script `bed2cf3`, avec $dx=\sigma/20$ et 400 pas. Chaque exécution a duré moins de 20 s, sur CPU uniquement.

| prédiction (pré-enregistrée) | valeur attendue | mesure | verdict |
|---|---|---|---|
| P1a, $A_{\rm AS}$ (validation du solveur contre le Th. 3.3) | $[-2{,}203;-2{,}103]$ | $-2{,}1513$ | vraie |
| P1b, Newton-Stein exact : $\lvert A\rvert$ et variance | $\le0{,}02$ ; $\lvert s_+^2/s_*^2-1\rvert\le0{,}02$ | $0{,}0000$ ; $1{,}0000$ | vraie |
| P1c, $A(\rho)$ contre $1-F(\rho)$, $\rho\in\{0{,}5;2;4;6\}$ | écart $\le0{,}05$ | écart $\le4\cdot10^{-4}$ | vraie |
| P1d, Newton-Stein avec $q$ non gaussien | TV $\le$ plancher $+0{,}005$ | TV $=3{,}46\cdot10^{-4}$, égal au plancher (AS : $0{,}114$) | vraie |
| P2b, (b) $\lvert R_{\rm NS+HB}\rvert$ | $\le0{,}1$ | $0{,}112$ (AS : $-1{,}572$ ; NS seul : $0{,}039$) | **fausse** |
| P2c1, (c) localisation pour AS et NS | $\le10^{-4}$ | $1{,}8\cdot10^{-5}$ et $1{,}4\cdot10^{-5}$ | vraie |
| P2c2, (c) NS + bain thermique | $[0{,}47;0{,}53]$ | $0{,}50004$ | vraie |
| P2d1, (d) localisation pour AS et NS | $\le0{,}01$ | $2{,}5\cdot10^{-4}$ et $6\cdot10^{-5}$ | vraie |
| P2d2, (d) NS + bain thermique | $[0{,}889;0{,}929]$ | $0{,}90870$ | vraie |
| P3, loi géométrique, $\lvert\ln\text{rapport}\rvert$ à $p=0{,}5$ ; $0{,}75$ ; $0{,}9$ | $\le1{,}5$ | $1{,}77$ ; $1{,}35$ ; $0{,}57$ | **fausse** |
| P4, trempe GMM-50 : $\hat p$ ; accord ; blocs | $[0{,}0034;0{,}0046]$ ; $\ge99{,}9\,\%$ ; $[0{,}93;1]$ | $0{,}00391$ ; 100 % ; $0{,}95$ | vraie |

**Verdicts tels qu'écrits.**

- **Théorème A confirmé.**
- **Théorème C confirmé** dans le cas des VC de Stein.
- **Proposition B non confirmée.** Elle est exacte dans les panneaux (c) et (d), mais fausse dans le panneau (b), avec un écart de $0{,}012$ au seuil.
- **Proposition C2 non confirmée.** L'identité de Feynman–Kac est exacte, mais l'approximation géométrique que j'en tirais est réfutée.
- **Prédiction de couverture par recherche externe confirmée.**

**Diagnostics a posteriori (sans règle, dans le JSON).**

- **2diag.** La décomposition de l'écart de dérive est exacte point par point à 4 décimales pour $\nu\le0{,}7$. Aux temps tardifs, le terme du bain thermique s'écarte de la partie étiquettes ; c'est l'origine de l'échec de P2b (§2.2).
- **3b.** Courbe $p\mapsto$ masse transférée à $\sigma^2=0{,}05$ (§2.3).
- **Partie 5 (descriptive).** VC de type iDEM : masse gauche $4{,}6\cdot10^{-3}$ à $\sigma^2=0{,}05$ et $4{,}4\cdot10^{-2}$ à $\sigma^2=0{,}09$. Le $p_{\rm eff}$ de la règle, défini par inversion de la loi géométrique, vaut 0,84 et 0,83, mais n'a pas de sens puisque cette loi est réfutée. Par interpolation sur la courbe 3b mesurée, on obtient $p\approx0{,}3$.

---

## 7. Verdict franc

1. **Découverte (mode 3) : résultat négatif, à publier comme tel si besoin.** Une VC à moyenne nulle ne découvre un mode qu'au rythme où ses évaluations d'énergie l'atteignent.
   - Toutes les VC locales (Stein, MALA, HMC, et le terme d'AS lui-même) sont bloquées à $e^{-c/\varepsilon^2}$ (Théorème C).
   - Les VC à requêtes globales deviennent des estimateurs de type IS ou iDEM, avec leur coût (une évaluation d'énergie par échantillon de régression, soit environ 500 fois le budget d'AS) et leur effondrement en $e^{-\mathrm{KL}}$ avec la dimension.
   - Le darting ne fait que transmettre des minima trouvés ailleurs.
   - La bonne architecture est une recherche externe (trempe depuis la base, *basin hopping*, tempering parallèle) suivie d'une pondération correcte (IS de Laplace, SMC, bain thermique). **La piste « VC contre la découverte » est close.**
2. **Poids (mode 1) : résolu sur le papier, pas de niche.** Bain thermique + Newton-Stein donnent dose 1 et pas de double comptage dans le régime séparé : exact sur les panneaux (c) et (d). Il reste un double comptage aux temps tardifs quand $q$ a des erreurs intra-mode. Face à l'IS de chemin post hoc, efficace dès que les modes sont trouvés, cela ne vaut pas d'être poursuivi.
3. **Instabilité (mode 2) : un petit résultat positif et propre.** Newton-Stein est exact en un pas pour toute cible gaussienne, gratuit en énergie, robuste à un facteur 4,92 sur la courbure, et c'est un amortissement pur dans sa version sûre. Il formalise la « relaxation adaptative avec garanties » que l'article laisse ouverte, mais son avantage pratique sur l'amortissement à $\eta$ fixé n'est pas démontré. **À poursuivre seulement à travers le test suivant**, qui demanderait ton accord explicite :

   **Test GPU proposé (non lancé).**

   - *Plan.* AS contre AS + Newton-Stein, $\hat H=1$ (modes de variance unité), sur GMM-dk (32, 8, 4) et (32, 8, 6), $\sigma_{\max}\times\{0{,}5;1;2\}$, graine 0, 1500 époques : 12 runs.
   - *Prédictions, à pré-enregistrer avant toute location.*
     - (i) À × 1 et × 2 dans (32, 8, 4), l'ESS de l'IS de chemin avec Newton-Stein est au moins le double de celui sans.
     - (ii) La couverture est inchangée à un mode près (Théorème C).
     - (iii) Les statistiques intra-mode deviennent stationnaires.
   - *Coût estimé.* RTX 3070 à environ 0,134 $/h, 7 runs en parallèle (≈ 0,45 s par époque, deux vagues d'environ 13 min), plus installation, IS et rapatriement : environ 35 à 40 min, soit **≈ 0,08 à 0,09 $**.
   - *Code à écrire.* Une variante du matcher avec $\Lambda_t$ (quelques lignes).
   **Résultat (2026-11-22, `rules_ns_damping.md`, version modifiée par l'utilisateur avec un bras « amortissement » η = 0,5, section de RESULTS.md) : R1 faux, R2 faux, décision écrite : on arrête.** À (32, 8, 4) × 2, Newton-Stein donne un ESS 5,6 fois celui d'AS et 3,4 fois celui de l'amortissement ; à × 1, seulement 1,22 et 1,33 fois ; à (32, 8, 8), aucun bras ne retrouve les 8 modes. Coût 0,040 $.
4. **Une prédiction de ce mémo a été réfutée et deux règles sont fausses (P2b, P3)**, sans relâchement après coup. Les explications (§2.2, §2.3) sont a posteriori et devraient être retestées sur données neuves avant d'être affirmées.

---

## Annexe A. Travaux liés (vérifié / de mémoire)

**Vérifiés** (lus dans les PDF fournis, ou dans nos propres résultats).

- Havens et al. (2025), *Adjoint Sampling*, arXiv 2504.11713v3.
- Anonyme (soumission ICLR 2027), *Failure Modes of Adjoint Sampling: Instability and Slow Multimodal Adaptation*.
- Nguyen, Dau, Thiéry (2026), *Amortizing Tempered Sampling with Corrected Self-Training*, arXiv 2611.23072v1.
- `RESULTS.md` (commits cités).

**Cités dans ces PDF** (référence vérifiée par la citation ; contenu non lu par moi).

- Domingo-Enrich, Drozdzal, Karrer, Chen (2024), *Adjoint Matching*, arXiv 2409.08861.
- Blessing, Richter, Berner, Malitskiy, Neumann (2026), *Bridge matching sampler: Scalable sampling via generalized fixed-point diffusion matching*, arXiv 2603.00530 (amortissement de la vitesse, éq. 176 de l'article FM).
- Domingo-Enrich, Han (2026), *Adjoint matching through the lens of the stochastic maximum principle in optimal control*, arXiv 2604.08580.
- Blessing, Berner, Richter, Domingo-Enrich, Du, Vahdat, Neumann (2025), *Trust region constrained measure transport in path space for stochastic optimal control and inference*, NeurIPS 38.
- Li, Chen, Tai, E (2018), JMLR (MSA).
- Kerimkulov, Šiška, Szpruch (2021) (MSA modifiée).
- Wenliang, Kanagawa (2021), *Blindness of score-based methods to isolated components and mixing proportions*, arXiv 2008.10087 : le précurseur direct du constat « les VC de Stein sont aveugles ».
- Shi, De Bortoli, Campbell, Doucet (2023), *Diffusion Schrödinger bridge matching*, NeurIPS 36 (DSBM).
- De Bortoli, Hutchinson, Wirnsberger, Doucet (2024), *Target score matching*, arXiv 2402.08667.
- Phillips, Dau, Hutchinson, De Bortoli, Deligiannidis, Doucet (2024), *Particle denoising diffusion sampler*, ICML (PDDS).
- Akhound-Sadegh et al. (2024), iDEM, et (2025), PITA.
- Schopmans, Friederich (2025), TA-BG.
- Midgley et al. (2023), FAB.
- Wang et al. (2025), *Importance weighted score matching for diffusion samplers with enhanced mode coverage*, arXiv 2505.19431.
- Rissanen et al. (2025), *Progressive tempering sampler with diffusion*, ICML (PTSD).
- Chen, Richter, Berner, Blessing, Neumann, Anandkumar (2025), *Sequential controlled Langevin diffusions*, ICLR (SCLD).
- Choi et al. (2025), *Reinforced sequential Monte Carlo for amortised sampling*, arXiv 2510.11711.

**De mémoire, à vérifier avant toute citation.**

- VC de Stein et *control functionals* : Assaraf, Caffarel (1999, PRL) ; Mira, Solgi, Imparato (2013, Stat. Comput.) ; Oates, Girolami, Chopin (2017, JRSS-B) ; Gorham, Mackey (2015, NeurIPS).
- VC par noyau de Markov et équation de Poisson : Henderson (1997) ; Dellaportas, Kontoyiannis (2012, JRSS-B).
- *Darting* : Andricioaei, Straub, Voter (2001, J. Chem. Phys.) ; Sminchisescu, Welling (2007/2011) ; Ahn, Chen, Welling (2013).
- *Multiple-try* et i-SIR : Liu, Liang, Wong (2000, JASA) ; Tjelmeland (2004) ; Andrieu, Lee, Vihola (2018, Bernoulli).
- *Basin hopping* : Wales, Doye (1997).
- Barker (1965).
- Transformée de Doob et Feynman–Kac : manuels standard.

**Nouveauté (honnête).**

- L'identité $Y=Y_{\rm BM}+\sigma c_S$ est une réécriture élémentaire, probablement connue sous une forme ou une autre ; je n'en connais pas de référence.
- Le préconditionnement de Newton de l'itération d'AS et son exactitude gaussienne : je n'en connais pas de référence. Non vérifié, à rechercher avant toute revendication.
- Le Théorème C formalise un principe connu (Wenliang et Kanagawa 2021 pour les méthodes de score, Th. 3.5 de l'article FM pour AS), étendu aux VC.
- La VC de type iDEM n'est pas nouvelle : c'est l'estimateur d'iDEM.
