# ESO-GAD theory plan

Nov 14, 2026 · the theory is in the paper (Section 5 and Appendix A); this file tracks what is proved, what is checked, and what is left

## Where things stand

- **In the paper.** Section 5 of `icml_ESO_GAD` states seven theorems, two new propositions and five corollaries (`theory.tex`), and Appendix A proves them (`proof.tex`). The paper builds cleanly. The paper is the source of truth for statements and proofs, and this file uses its numbering.
- **Checked numerically.** `theory_check.py synthetic` ran 27 checks on Nov 14 and all passed. Every statement of Section 5 has one, with the exception of Proposition 3, which is a property of the procedure. The script was restored into the new `src/` clone on Nov 17 and all 27 checks pass again there.
- **Not yet backed by data in the paper.** No measured number is in Section 5. The oracle diagnostics on Amazon and YelpChi (see "Numerical verification") agree with the theory, but the tables and the figure that should stand next to the theorems wait for blocks B and C of `experiment.md`.
- **Theory first (author, Nov 17).** The next experiments are the ones that check or extend the theory: the "Theory-first track" of `experiment.md`. Its scouting part (C16–C19, C9, C10) decides which of the candidate results N1–N7 below are worth proving.
- **Open with the authors.** Six findings about the method (M1–M6) and the description of cross-fitting in Section 4.1 need a decision; see "Remaining theory work", task T1.
- **Code folder replaced on Nov 17.** `src-code/` is gone; `src/` is a fresh clone of the GitHub repository. `theory_check.py` and the Nov 14 code changes were restored from the session transcript and committed on branch `icml2027` as `85e68f9` (not pushed).
- **Workspace moved on Nov 19.** The source is now `wsdm-2026/BWGNN_codebase/` (read `src/` as that folder), the data is in `wsdm-2026/GADBench-master/datasets/`, the environment is `gcnenv`. **`theory_check.py` and the `runs/` folders did not come along**: the 27 synthetic checks and the real-graph diagnostics below were run on the earlier machine and cannot be re-run here until the script is copied over or rewritten. The code behind the WSDM tables was found in `wsdm-2026/GADBench-master/`; its cross-fitting rule is the one Section 4.1 describes. Details in `experiment.md`, "Workspace on this machine".

## What the theory says

The WSDM theory proved that the signed operator is well defined and stable. Reviewer 3 (and Reviewer 2, indirectly) asked the question it does not answer: *why should signing improve anomaly separability?*

1. **Why signing helps.** With correct signs, a heterophilic graph is spectrally identical to a perfectly homophilic one (Theorem 1). For any calibrator, the frequency of the label signal equals twice the weighted fraction of wrongly signed edges (Theorem 2). So signing lowers the label frequency exactly as far as the calibrator is right.
2. **Why sign and not prune.** At node level, signing and pruning both put a node on the correct side, but pruning throws away the evidence and signing keeps it. The signal-to-noise gain of signing over pruning is $1/\sqrt{1-h_u}$, largest for camouflaged anomalies (Theorem 3). Among all ways of turning edge scores into weights, the soft sign $1-2p$ is optimal and every non-negative scheme is strictly worse whenever some edge is more likely heterophilic than not (Theorem 4).
3. **What $\eta$ is.** $\eta$ is the slope that recalibrates a frozen, miscalibrated score. The family $1-2\eta\hat q$ contains the optimal weights exactly when the calibrator's reliability curve is linear (Corollary 3). This answers "the calibrator is frozen, so errors are only damped".
4. **What weak edge supervision identifies.** The "unlabeled endpoint is normal" target estimates whether an edge touches an anomaly, not heterophily, so anomaly–anomaly edges get the highest score; the rule helps exactly the anomalies that are mostly linked to normal nodes (Proposition 2, Corollary 4). The cross-fitting rule keeps a node's own label out of the scores of its edges, up to its edges to labeled nodes in other folds (Proposition 3).
5. **Stability with constants that mean something.** The WSDM constant $2d_{\max}(1+d_{\max})$ is $9.8\times10^{7}$ on Amazon and $5.1\times10^{5}$ on YelpChi, while $\|L-L'\|_2\le2$ always. Theorem 5 replaces it with a degree-free bound in a local, averaged calibration error, Theorem 6 bounds the operator error by the Brier score that Phase 1 minimises, and Theorem 7 with Corollary 5 carries both to the representation.

## Claims

The check ids are the stable handle: they are the ids printed by `theory_check.py`. Experiment ids refer to `experiment.md`.

| Check | Paper statement | Claim | Kind | Answers | Proof and check | Evidence on real data |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | Prop 1 `prop:valid` | $D\succeq I$ and $\operatorname{spec}(L_\eta)\subset[0,2]$ for every $\eta\ge0$ | exact | well-posedness | proved (WSDM), extended, checked | not needed |
| C3 | Thm 1 `thm:gauge` | Oracle signs make $L_\eta$ gauge-equivalent to a homophilic operator | exact identity | R3 "why separability" | proved, checked | oracle rows of D3 will show the headroom |
| C2 | Thm 2 `thm:freq` | Label frequency $=2\times$ frustration | exact identity | R3 | proved, checked to $10^{-16}$ | C1: not started; preview on Amazon and YelpChi |
| C5 | Cor 1 `cor:lift` | Unsigned: $\rho_0$ is twice the edge heterophily. For $\eta\le\frac12$ the label frequency falls iff the calibrator has positive lift | exact | R1 "errors only damped" | proved, checked | C1 |
| C4 | Cor 2 `cor:energy` | Label energy in the low-pass and high-pass Beta channels | bound | R3, gate | proved, checked | C3: not started |
| C6 | Thm 3(a,b) `thm:snr` | Node SNR factor $=m_u\sqrt{d_{\mathrm{eff},u}}$; unsigned, pruning and signing under oracle edges | exact | R3 | proved, checked, Monte-Carlo checked | B7, C2: not started; preview on Amazon and YelpChi |
| C7 | Thm 3(c) | Oracle signing dominates the unsigned operator at every node for every non-negative filter | inequality | R3 | proved, checked | not planned |
| C9 | Thm 3(d) | First-order effect of $\eta$ on the margin | exact | $\eta=0$ control | proved, checked | D1: not started |
| C8 | Thm 4 `thm:optimal`, Cor 3 `cor:slope` | Optimal edge weights are soft signs; non-negative schemes are strictly worse; $\eta$ is the reliability slope | optimality | R2/R3 sign against prune, R1 frozen calibrator | proved, checked, Monte-Carlo checked | D3, C4: not started; two data points for C4 |
| C13 | Prop 2 `prop:e1`, Cor 4 `cor:camouflage` | The E1 target identifies $\frac12q^*+P_{aa}$; the rule helps an anomaly iff $h_u>\frac12-\frac1{2\tilde d_u}$ | exact | R1, R2, R3 on the E1 assumption | proved, checked by simulation | B3, B4, B7, D4: not started; confirmed with a stand-in calibrator on Amazon and YelpChi |
| C14 | Prop 3 `prop:crossfit` | Which scores are free of a node's own label | exact | R2 "is cross-fitting grouped by node" | proved by inspection of the code | B5, D5: not started |
| C10 | Thm 5 `thm:stab` | Degree-free stability in $\varepsilon_{\mathrm{loc}}$, joint in $(\eta,q)$ | bound | replaces the WSDM stability theorem | proved, checked | C5: not started |
| C11 | Thm 6 `thm:brier` | Frobenius stability in the local Brier score | bound | links the Phase 1 loss to the operator | proved, checked | C7: not started |
| C12 | Thm 7 `thm:repr`, Lemma 2 `lem:frob`, Cor 5 `cor:misaligned` | Representation robustness; fraction of misaligned nodes | bound | replaces the WSDM representation theorem | proved, checked | C5, C8: not started |

## Setting and notation

Everything is stated for the operator the paper defines. Labels are written in $\{-1,+1\}$ for the algebra ($+1$ is an anomaly); the paper's $\{0,1\}$ labels are $(y+1)/2$.

- Graph $G=(V,E)$, $n$ nodes, undirected, no parallel edges. $\tilde A=A+I$, $\tilde d_u=\sum_v\tilde A_{uv}$. Sums over $(u,v)$ run over ordered pairs with $\tilde A_{uv}=1$, self-loops included.
- Labels $y\in\{-1,+1\}^n$, $Y=\operatorname{diag}(y)$. True heterophily $t_{uv}=\mathbb 1[y_u\ne y_v]$, true sign $s^*_{uv}=y_uy_v=1-2t_{uv}$, with $s^*_{uu}=1$.
- Scores $q\in[0,1]$ per edge, symmetric, $q_{uu}=0$. Strength $\eta\ge0$.
- $W=\tilde A\odot(1-2\eta q)$, entries $w_{uv}$, signs $\sigma_{uv}=\operatorname{sign}(w_{uv})$. $D=\operatorname{diag}(|W|\mathbf 1)$, $N=D^{-1/2}WD^{-1/2}$, $L_\eta(q)=I-N$.
- Local heterophily $h_u=\tilde d_u^{-1}\sum_v\tilde A_{uv}t_{uv}$. Class averages $\bar h_a$, $\bar h_n$.
- **Frustration** of the signed graph with respect to the labels:
  $$\mathrm{Fr}_\eta(q)=\frac{\sum_{u,v}|w_{uv}|\,\mathbb 1[\sigma_{uv}\ne s^*_{uv},\,w_{uv}\ne0]}{\sum_{u,v}|w_{uv}|}\in[0,1].$$
- **Label signal** $\tilde y=D^{1/2}y$ and its frequency $\rho_\eta(q)=\tilde y^\top L_\eta(q)\tilde y/\tilde y^\top\tilde y$.
- **Node margin** $m_u=\sum_vw_{uv}s^*_{uv}/\sum_v|w_{uv}|\in[-1,1]$ and **effective degree** $d_{\mathrm{eff},u}=(\sum_v|w_{uv}|)^2/\sum_vw_{uv}^2\in[1,\tilde d_u]$.
- **Retained mass** $\gamma=\min_uD_{uu}/\tilde d_u\in(0,1]$.
- For two score vectors $q,q'$: local mean absolute error $\bar e_u=\tilde d_u^{-1}\sum_v\tilde A_{uv}|q_{uv}-q'_{uv}|$, $\varepsilon_{\mathrm{loc}}=\max_u\bar e_u$; local Brier $b_u=\tilde d_u^{-1}\sum_v\tilde A_{uv}(q_{uv}-q'_{uv})^2$, $B=\sum_ub_u$.
- Beta channels in the paper's normalisation: $\mathcal W_k(L)=\kappa_k(L/2)^k(I-L/2)^{P-k}$, $\kappa_k=\frac{P+1}{2}\binom Pk$. The code's channels are $2\times$ these.
- $Q(x)=\Pr[\mathcal N(0,1)>x]$.

The derivations in this file keep the working notation above. The paper had to rename the symbols it already used:

| Quantity | This file | Paper |
| --- | --- | --- |
| Signed labels, their diagonal matrix | $y$, $Y$ | $\bar{\boldsymbol y}$, $\boldsymbol Y$ (the paper's $y$ stays in $\{0,1\}$) |
| True heterophily indicator | $t_{uv}$ | $t^\star_{uv}$ (the paper's $t_{uv}$ is the training target) |
| Local heterophily | $h_u$ | $\tilde h_u$ |
| Node margin | $m_u$ | $\mathfrak m_u$ |
| Node SNR factor | $\kappa_u$ | $\chi_u$ ($\kappa_k$ is the Beta prefactor) |
| Gaussian tail | $Q$ | $\bar\Phi$ |
| Aggregation weights, their variance term | $\omega_v$, $\tau_v$ | $\vartheta_v$, $\varsigma_v^2$ |
| Anomaly-incidence probability | $a(z)$ | $\iota(\boldsymbol z)$ |
| Labeling probabilities | $\rho_1,\rho_0$ | $r_1,r_0$ |
| Summed local Brier score | $B$ | $\mathcal B$ |

## Assumptions

Theorems 1, 2, 5, 6, 7, Propositions 1 and 3 and Corollaries 1, 2, 5 need none of these: they hold for every graph, every labeling and every score vector (Theorem 1 and Corollary 5 for two classes).

| ID | In the paper | Assumption | Used in | Comment |
| --- | --- | --- | --- | --- |
| A1 | Assumption 1 | Encoder output is class-centred with independent noise: $H^{(0)}=y\mu^\top+\Xi$, rows of $\Xi$ i.i.d. $\mathcal N(0,\sigma^2I)$, independent of $(A,q)$ | Thm 3 | Standard contextual-SBM feature model, but conditional on an arbitrary graph. Remark 2 covers the uncentred case. |
| A2 | Assumption 2 | Given the scores, the relative signs $y_uy_v$ of a node's neighbors are independent with $\Pr[y_u\ne y_v\mid q]=p(q_{uv})$ | Thm 4 | $p(\cdot)$ is the calibrator's reliability curve. Calibrated means $p(q)=q$. |
| A3 | Assumption 3 | Which nodes are labeled is independent of the edge descriptors given the node classes | Prop 2 | Holds for GADBench's random stratified draws. |
| A4 | in the text after Theorem 1 | Two classes | Thm 1 | With three or more classes the label-induced signs are only weakly balanced and Theorem 1 fails. |

## Results

### Proposition 1 (spectral validity, C1)

For every $q\in[0,1]^{|E|}$ and every $\eta\ge0$: $D\succeq I$ and every eigenvalue of $L_\eta(q)$ lies in $[0,2]$.

*Proof.* Unchanged from the WSDM appendix. The self-loop gives $D_{uu}\ge|w_{uu}|=1$ for any $\eta$ because $q_{uu}=0$, and the two quadratic-form identities use only the symmetry of $W$ and $D=\operatorname{diag}(|W|\mathbf 1)$. Nothing uses $\eta\le1$. $\square$

The extension to $\eta>1$ matters: Corollary 3 shows the recalibration slope can exceed 1.

### Theorem 1 (balance: oracle signing is a change of gauge, C3)

Let $q=t$ (true heterophily indicators) and $\eta\in[\frac12,1]$. Then

1. $L_\eta(t)=Y\,L_{1-\eta}(t)\,Y$, where $L_{1-\eta}(t)$ has only non-negative weights. At $\eta=1$ the right-hand side is the unsigned Laplacian.
2. $L_\eta(t)$ and $L_{1-\eta}(t)$ have the same spectrum, and $g(L_\eta(t))=Y\,g(L_{1-\eta}(t))\,Y$ for every polynomial $g$, in particular for every Beta channel.
3. $L_\eta(t)\,\tilde y=0$ for $\eta\in(\frac12,1]$: the label signal has frequency exactly zero.

*Proof.* With $q=t$, $w_{uv}=\tilde A_{uv}(1-2\eta t_{uv})$ equals $1$ on homophilic edges and $1-2\eta\le0$ on heterophilic ones. So $|w_{uv}|=\tilde A_{uv}(1-2(1-\eta)t_{uv})$, which is the weight matrix $W^{(1-\eta)}(t)$, and $\sigma_{uv}=s^*_{uv}=y_uy_v$ wherever $w_{uv}\ne0$. Hence $W^{(\eta)}(t)=Y\,W^{(1-\eta)}(t)\,Y$ entrywise. Both operators have the same $D$ because $D$ depends only on $|W|$, and $Y$ commutes with diagonal matrices, so $N_\eta=YN_{1-\eta}Y$ and (1) follows. $Y$ is orthogonal, which gives the spectrum and, since $Y^2=I$, $g(YMY)=Yg(M)Y$. For (3), $L_{1-\eta}(t)$ is a non-negatively weighted normalized Laplacian, so $L_{1-\eta}(t)D^{1/2}\mathbf 1=0$; then $L_\eta(t)D^{1/2}y=YL_{1-\eta}(t)YD^{1/2}y=YL_{1-\eta}(t)D^{1/2}\mathbf 1=0$. $\square$

**Reading.** Filtering with the oracle signed operator is the same as flipping the features by class, filtering on a graph with no heterophily, and flipping back. Signing does not change the spectrum; it moves the label signal to the bottom of it. This is the precise sense in which signing "encodes" heterophilic edges and pruning "removes" them: pruning also reaches frequency zero, but on a smaller graph (Theorem 3).

**Remark (binary labels; the paragraph after Theorem 1 in the paper).** The identity needs the product of true signs along any walk from $u$ to $v$ to equal $y_uy_v$, which holds for two classes only. This is why the multi-hop failure of signed message passing shown by Liang et al. (ICML 2024, "Sign is not a remedy") for multi-class node classification does not apply to binary anomaly detection.

**What is classical.** Balance and switching equivalence are Harary (1953) and Zaslavsky (1982); the zero eigenvalue of a balanced signed Laplacian is in Hou, Li and Pan (2003) and Kunegis et al. (2010). The new part is the statement for the paper's $\eta$-family ($L_\eta(t)=YL_{1-\eta}(t)Y$) and its use for GAD.

### Theorem 2 (label frequency equals frustration, C2)

For every graph, labeling, $q$ and $\eta\ge0$:
$$\rho_\eta(q)=2\,\mathrm{Fr}_\eta(q).$$

*Proof.* Put $h=\tilde y=D^{1/2}y$ in the quadratic form of the paper, $h^\top L_\eta h=\frac12\sum_{u,v}|w_{uv}|(f_u-\sigma_{uv}f_v)^2$ with $f=D^{-1/2}h=y$. Since $y_u,y_v\in\{\pm1\}$, $(y_u-\sigma_{uv}y_v)^2=4\cdot\mathbb 1[\sigma_{uv}\ne y_uy_v]$ when $w_{uv}\ne0$. So $\tilde y^\top L_\eta\tilde y=2\sum_{u,v}|w_{uv}|\mathbb 1[\sigma_{uv}\ne s^*_{uv}]$. The denominator is $\tilde y^\top\tilde y=\sum_uD_{uu}=\sum_{u,v}|w_{uv}|$. $\square$

**Corollary 1, first part (unsigned operator).** At $\eta=0$, $\rho_0=2\tilde h$ with $\tilde h$ the edge heterophily ratio (self-loops counted as homophilic). Heterophily *is* the frequency of the label signal. This recovers the heterophily–frequency link behind BWGNN's right-shift and GHRN in one line.

**Corollary 1, second part (the reweighting regime, C5).** For $\eta\in[0,\frac12]$ no edge is negative and
$$\mathrm{Fr}_\eta(q)=\frac{H-2\eta S_{\mathrm{het}}}{M-2\eta S_{\mathrm{all}}},$$
where $H$ is the number of heterophilic ordered edges, $M$ the number of all ordered edges including self-loops, and $S_{\mathrm{het}}$, $S_{\mathrm{all}}$ the sums of $q$ over those sets. It is strictly decreasing in $\eta$ iff $S_{\mathrm{het}}/H>S_{\mathrm{all}}/M$, that is, iff the mean score on heterophilic edges exceeds the mean score on all edges (positive *lift*).

*Proof.* $\frac{d}{d\eta}\mathrm{Fr}_\eta=2(S_{\mathrm{all}}H-S_{\mathrm{het}}M)/(M-2\eta S_{\mathrm{all}})^2$, whose sign does not depend on $\eta$. $\square$

So any calibrator better than chance lowers the label frequency for every $\eta$ up to $\frac12$. Beyond $\frac12$ edges flip and
$$\mathrm{Fr}_\eta(q)=\frac{\sum_{\mathrm{het}}(1-2\eta q)_++\sum_{\mathrm{hom}}(2\eta q-1)_+}{\sum_{\mathrm{all}}|1-2\eta q|},$$
which rewards correct flips and charges wrong ones. $\mathrm{Fr}_\eta(q)$ is piecewise rational in $\eta$ and its minimiser is computable from oracle labels in $O(m\log m)$; the experiments compare it with the learned $\eta$.

**Corollary 2 (where the label energy goes, C4).** Let $E_k=\|\mathcal W_k(L_\eta)\tilde y\|^2/(\kappa_k^2\|\tilde y\|^2)$ be the share of label energy passed by channel $k$. Then
$$(1-\mathrm{Fr})^{2P}\le E_0\le1-\mathrm{Fr},\qquad \mathrm{Fr}^{2P}\le E_P\le\mathrm{Fr}.$$

*Proof.* Let $\mu_y$ be the spectral measure of $\tilde y/\|\tilde y\|$; its mean is $\rho=2\mathrm{Fr}$. $E_0=\int(1-\lambda/2)^{2P}d\mu_y$. The integrand is convex, so Jensen gives the lower bound; it is at most $1-\lambda/2$ on $[0,2]$, which gives the upper bound. The same argument with $\lambda/2$ gives $E_P$. $\square$

So the signed operator moves label energy from the high-pass channel to the low-pass channel in proportion to the frustration it removes. Under oracle signs $E_0=1$ and $E_k=0$ for $k\ge1$. This is the testable version of "signing reallocates frequency", and it tells us what the gate should do: shift mass to low channels as calibration improves.

**What is classical.** Frustration bounds on the least signed-Laplacian eigenvalue are known (Belardo 2014; Atay and Liu 2020). The identity for the *label* partition, its $\eta$-dependence and Corollaries 1 and 2 are new, and they are what ties the calibrator's quality to the spectrum.

### Theorem 3 (node-level separability, C6, C7, C9)

Assume A1. Let $G$ be any $n\times n$ matrix that depends on $(A,q,\eta)$ only, for example a Beta channel $\mathcal W_k(L_\eta(q))$, and $Z=GH^{(0)}$.

**(a) Exact law.** Row $u$ of $Z$ is $z_u=[Gy]_u\,\mu+\zeta_u$ with $\zeta_u\sim\mathcal N(0,\sigma^2\|G_{u\cdot}\|_2^2I)$. The readout $\operatorname{sign}\langle z_u,\mu\rangle$ misclassifies $u$ with probability
$$Q\!\Big(\kappa_u(G)\,\frac{\|\mu\|}{\sigma}\Big),\qquad\kappa_u(G)=\frac{y_u[Gy]_u}{\|G_{u\cdot}\|_2}.$$
For one propagation step $G=D^{-1}W$: $\kappa_u=m_u\sqrt{d_{\mathrm{eff},u}}$.

**(b) Oracle comparison at one step.**

| Operator | Margin $m_u$ | Effective degree | $\kappa_u$ |
| --- | --- | --- | --- |
| Unsigned | $1-2h_u$ | $\tilde d_u$ | $(1-2h_u)\sqrt{\tilde d_u}$, negative when $h_u>\frac12$ |
| Oracle pruning, $w=1-t$ | $1$ | $(1-h_u)\tilde d_u$ | $\sqrt{(1-h_u)\tilde d_u}$ |
| Oracle signing, $w=1-2t$ | $1$ | $\tilde d_u$ | $\sqrt{\tilde d_u}$ |

Hence $\kappa^{\mathrm{sign}}_u/\kappa^{\mathrm{prune}}_u=(1-h_u)^{-1/2}$, and the class means after unsigned aggregation are separated by $2|1-\bar h_a-\bar h_n|\,\|\mu\|$ against $2\|\mu\|$ for oracle signing.

**(c) Domination for low-pass filters.** Let $p$ be a polynomial with non-negative coefficients, $G_0=p(N_0)$ on the unsigned operator and $G_s=p(N_s)$ on the oracle-signed one ($N_s$ is $N$ at $\eta=1$, $q=t$). Then for every node
$$\kappa_u(G_s)=\frac{\|G_{0,u\cdot}\|_1}{\|G_{0,u\cdot}\|_2}\ \ge\ |\kappa_u(G_0)|,$$
with strict inequality iff the support of row $u$ of $G_0$ contains both classes. The low-pass Beta channel $\mathcal W_0\propto((I+N)/2)^P$ is of this form.

**(d) First-order effect of $\eta$.** For $G=D^{-1}W^{(\eta)}$,
$$\frac{d\kappa_u}{d\eta}\Big|_{\eta=0}=4\sqrt{\tilde d_u}\;h_u(1-h_u)\,\big(\bar q^{\mathrm{het}}_u-\bar q^{\mathrm{hom}}_u\big),$$
where $\bar q^{\mathrm{het}}_u$ and $\bar q^{\mathrm{hom}}_u$ are the mean scores on $u$'s heterophilic and homophilic edges (the self-loop counts as homophilic with score 0).

*Proof.* (a) $Z=Gy\mu^\top+G\Xi$. Row $u$ of $G\Xi$ is a linear combination of independent Gaussians with coefficients $G_{uv}$, so it is $\mathcal N(0,\sigma^2\|G_{u\cdot}\|^2I)$. Then $\langle z_u,\mu\rangle/\|\mu\|\sim\mathcal N([Gy]_u\|\mu\|,\sigma^2\|G_{u\cdot}\|^2)$ and the error event is $y_u\langle z_u,\mu\rangle\le0$. For $G=D^{-1}W$: $y_u[Gy]_u=\sum_vw_{uv}y_uy_v/D_{uu}=m_u$ and $\|G_{u\cdot}\|_2=(\sum_vw_{uv}^2)^{1/2}/D_{uu}=d_{\mathrm{eff},u}^{-1/2}$.

(b) Substitute the three weightings; the self-loop is homophilic with weight 1 in all of them. The class-mean statement follows from $[Gy]_u=m_uy_u$.

(c) By Theorem 1, $G_s=YG_0Y$, so $y_u[G_sy]_u=[G_0\mathbf 1]_u=\|G_{0,u\cdot}\|_1$ because $G_0\ge0$ entrywise, and $\|G_{s,u\cdot}\|_2=\|G_{0,u\cdot}\|_2$. For the unsigned operator $|y_u[G_0y]_u|=|\sum_vG_{0,uv}y_uy_v|\le\sum_vG_{0,uv}$, with equality iff $y_v$ is constant on the support; the support contains $u$ itself because of the self-loop.

(d) Write $d=\tilde d_u$, $H_u=h_ud$, $S_{\mathrm{het}}$, $S_{\mathrm{hom}}$ the score sums over $u$'s edges. For small $\eta$, $m_u(\eta)=\frac{(d-2H_u)-2\eta(S_{\mathrm{hom}}-S_{\mathrm{het}})}{d-2\eta(S_{\mathrm{hom}}+S_{\mathrm{het}})}$, so $m_u'(0)=\frac{4}{d^2}\big[S_{\mathrm{het}}(d-H_u)-H_uS_{\mathrm{hom}}\big]=4h_u(1-h_u)(\bar q^{\mathrm{het}}_u-\bar q^{\mathrm{hom}}_u)$. Also, with $S=S_{\mathrm{hom}}+S_{\mathrm{het}}$ and $S_2$ the sum of squared scores, $d_{\mathrm{eff},u}(\eta)=\frac{(d-2\eta S)^2}{d-4\eta S+4\eta^2S_2}$ has zero derivative at $\eta=0$. $\square$

**Reading.**

- In the camouflage regime (anomalies mostly linked to normal nodes, $\bar h_a$ near 1, $\bar h_n$ near 0) the unsigned operator has $\bar h_a+\bar h_n\approx1$: the two class means collapse onto each other and camouflaged anomalies land on the normal side. Measured: 98.3% of Amazon anomalies and 94.1% of YelpChi anomalies have $h_u>\frac12$; the unsigned separation is 0.23 and 0.15 against 2.
- The unsigned high-pass channel does put every node on its own side (in random-walk form, $L_0=I-D^{-1}\tilde A$ gives $y_u[L_0y]_u=2h_u$) but does not average noise ($\|L_{0,u\cdot}\|_2^2=1-1/\tilde d_u$), so $\kappa_u\approx2h_u\le2$. The signed low-pass channel reaches $\sqrt{\tilde d_u}$. *Unsigned spectral GAD can see the anomaly signal in high frequencies but cannot denoise it; signing turns heterophilic neighbors into denoising evidence.*
- Part (d): any calibrator that ranks a node's heterophilic edges above its homophilic ones improves that node at first order. The base rate does not enter. This is the formal content of an $\eta=0$ control.

**Remark 1 (scope).** Parts (b) and (d) are one-step statements and (c) is for non-negative filters with a linear readout. A multi-channel unsigned bank followed by an MLP can combine low- and high-pass channels non-linearly, so the theorem does not say BWGNN *cannot* separate the classes; it says what one signed low-pass channel achieves that no unsigned channel does.

**Remark 2 (uncentred features and the "edge scores as features" pathway).** If $H^{(0)}=\mathbf 1c^\top+y\mu^\top+\Xi$, the offset is mapped to $\nu_uc$ with $\nu_u=\sum_vw_{uv}/\sum_v|w_{uv}|$, the net sign ratio of $u$'s neighborhood. For the unsigned operator $\nu_u\equiv1$. For the signed one $\nu_u$ is a scalar summary of how much of the neighborhood is flagged heterophilic, which is itself predictive of the class. So the signed operator has two pathways: the *contrast* pathway analysed above and a *bias* pathway that injects a neighborhood summary of the edge scores. Reviewer 2's control (edge scores as extra node features) isolates the second; experiment D2 in `experiment.md` separates them.

### Theorem 4 (optimal edge weights under uncertain signs, C8)

Assume A1 (Gaussianity not needed) and A2. Fix a node $u$ and aggregate $z_u=\sum_v\omega_vh^{(0)}_v$ over its neighbors (self included) with weights that depend on the scores only. Let $c_v=1-2p(q_{uv})$ be the expected relative sign ($c_u=1$), $\tau_v=\sigma^2+\|\mu\|^2(1-c_v^2)$, and define the deflection
$$J(\omega)=\frac{\mathbb E\big[y_u\langle z_u,\hat\mu\rangle\big]}{\operatorname{sd}\big[\langle z_u,\hat\mu\rangle\big]}=\frac{\|\mu\|\sum_v\omega_vc_v}{\big(\sum_v\omega_v^2\tau_v\big)^{1/2}},\qquad\hat\mu=\mu/\|\mu\|.$$

1. $J(\omega)\le J^*=\|\mu\|\big(\sum_vc_v^2/\tau_v\big)^{1/2}$, with equality iff $\omega_v\propto c_v/\tau_v$.
2. Over non-negative weights, $J(\omega)\le J^*_+=\|\mu\|\big(\sum_v(c_v)_+^2/\tau_v\big)^{1/2}$, with equality iff $\omega_v\propto(c_v)_+/\tau_v$.
3. $J^{*2}-J_+^{*2}=\|\mu\|^2\sum_{v:\,p_v>1/2}c_v^2/\tau_v$.
4. If $J(\omega)>0$, the readout $\operatorname{sign}\langle z_u,\mu\rangle$ errs with probability at most $1/(1+J(\omega)^2)$.

*Proof.* Conditional on $y_u$ and the scores, $\langle z_u,\hat\mu\rangle=y_u\|\mu\|\sum_v\omega_vs_{uv}+\sum_v\omega_v\langle\xi_v,\hat\mu\rangle$ with $s_{uv}=y_uy_v$. By A2 its mean is $y_u\|\mu\|\sum_v\omega_vc_v$ and its variance is $\sum_v\omega_v^2\big(\|\mu\|^2(1-c_v^2)+\sigma^2\big)$. (1) is Cauchy–Schwarz applied to $\sum_v(\omega_v\sqrt{\tau_v})(c_v/\sqrt{\tau_v})$. (2): for $\omega\ge0$, $\sum_v\omega_vc_v\le\sum_v\omega_v(c_v)_+$, then Cauchy–Schwarz. (3) is the difference of the two sums. (4) is Cantelli's inequality. $\square$

**Reading.**

- The optimal weight is an odd, increasing function of the soft sign $c=1-2p$. In the regime where aggregation matters ($\|\mu\|\ll\sigma$) it is exactly $\omega_v\propto1-2p(q_{uv})$, and for confident edges ($|c|\to1$) it equals $c$ at any signal level.
- Unsigned ($\omega\equiv1$), reweighting ($\omega=1-q$) and pruning ($\omega=\mathbb 1[q<\tau]$) are non-negative, so they are bounded by $J_+^*$. The loss is exactly the information carried by edges that are more likely heterophilic than not. For a fully camouflaged node $J_+^{*2}$ keeps only the self term while $J^{*2}$ grows with the degree.
- This is the theorem behind the Signed > Reweighting > Pruning table, and it explains the *calibrated* in the title: the weight that matters is the true posterior, so miscalibration costs performance even at fixed edge AUROC.

**Corollary 3 ($\eta$ is a recalibration slope).** If the frozen calibrator has a linear reliability curve $p(q)=\eta_0q$, the optimal weights in the low-signal regime are $1-2\eta_0\hat q_{uv}$, that is $W^{(\eta_0)}$. The ESO-GAD family contains the optimal aggregator for every such calibrator, and $\eta_0<1$ means the calibrator over-predicts heterophily.

Over-prediction is the expected case: the class-weighted loss in `edge_calibration.py` (`pos_weight`) and GADBench's labeled set (20% anomalies against 3–22% in the population) both inflate the scores. Under-prediction ($\eta_0>1$) happens with the E1 target, which halves the scores (Proposition 2). Proposition 1 holds for all $\eta\ge0$, so the sigmoid range $[0,1]$ is a design restriction, not a requirement of the theory.

**Limits.** $\eta$ repairs a linear miscalibration only. A calibrator that is non-monotone, or wrong on a specific edge type (Proposition 2), cannot be repaired by one scalar. A two-parameter Platt map $1-2\,\mathrm{sigmoid}(a\cdot\mathrm{logit}\,\hat q+b)$ is the natural next step and is listed as an ablation.

### Proposition 2 (what the "unlabeled endpoint is normal" target estimates, C13)

Let $q^*(z)=\Pr[y_u\ne y_v\mid z_{uv}=z]$ and $P_{aa}(z)=\Pr[y_u=y_v=\text{anomaly}\mid z]$. For an edge with exactly one labeled endpoint the paper's target is $\tilde t_{uv}=\mathbb 1[\text{the labeled endpoint is an anomaly}]$. Assume A3.

1. If every node is labeled with the same probability, $\mathbb E[\tilde t\mid z]=a(z):=\tfrac12q^*(z)+P_{aa}(z)$.
2. If anomalies are labeled with probability $\rho_1$ and normal nodes with $\rho_0$, the odds of $\tilde t=1$ given $z$ are
$$\frac{\rho_1\big[2(1-\rho_1)P_{aa}+(1-\rho_0)q^*\big]}{\rho_0\big[2(1-\rho_0)P_{nn}+(1-\rho_1)q^*\big]}\approx\beta\,\frac{a(z)}{1-a(z)},\qquad\beta=\frac{\rho_1}{\rho_0},$$
the approximation being exact up to factors $1-\rho\ge1-100/n$. Under GADBench, $\beta=(1-\pi)/(4\pi)$ for anomaly rate $\pi$.

*Proof.* (1) Given an edge with exactly one labeled endpoint, each endpoint is the labeled one with probability $\frac12$ independently of the labels, and $z$ is symmetric, so $\mathbb E[\tilde t\mid z]=\frac12\Pr[y_u=a\mid z]+\frac12\Pr[y_v=a\mid z]=\frac12(2P_{aa}+q^*)$. (2) The configuration "labeled endpoint of class $c$, other endpoint of class $c'$" has weight $\rho_c(1-\rho_{c'})$; sum over configurations with $\tilde t=1$ and $\tilde t=0$. $\square$

**Consequences.**

- The E1 target ranks edges by $a(z)$, the probability that the edge *touches an anomaly*, not by heterophily. An anomaly–anomaly edge is homophilic but has the largest possible $a$.
- For rare, camouflaged anomalies $P_{aa}\approx0$ and $a\approx q^*/2$: the scores are a halved heterophily probability, which $\eta$ can undo only if it may exceed 1.
- **Corollary 4 (when the E1 rule helps).** For the ideal E1 operator at $\eta=1$ (every anomaly-incident edge negative), a normal node has $m_u=1$ and an anomaly has $m_u=2h_u-1+2/\tilde d_u$, against $1-2h_u$ for the unsigned operator. The rule helps an anomaly iff $h_u>\frac12-\frac1{2\tilde d_u}$, and it reverses the benefit for anomalies in fraud rings ($h_u<\frac12$). The paper's sentence "consistent with the camouflage assumption" is exactly this condition, and it can now be checked per dataset through $\bar h_a$.
- **E2 alone is not an option at 100 labels.** The expected number of labeled–labeled edges is $|E|\cdot\frac{100\cdot99}{n(n-1)}$: about 305 on Amazon, 136 on T-Finance, 57 on Weibo, 37 on Tolokers, 18 on YelpChi, 14 on Reddit and 0.6 on Questions. This is the quantitative answer to Reviewer 3's "E2-only versus E1+E2".
- **A principled fix.** Replace the modal imputation by a soft one: $\tilde t_{uv}=y_u(1-\hat p_v)+(1-y_u)\hat p_v$, with $\hat p_v$ a cross-fitted node-level anomaly probability for the unlabeled endpoint. The paper's rule is the special case $\hat p_v\equiv0$. If $\hat p$ is calibrated, the target is unbiased for $q^*$.

### Proposition 3 (which scores are free of a node's own label, C14)

Folds are assigned per labeled node, stratified by class. Fold model $j$ is trained on labeled edges with both endpoints outside fold $j$. In `cross_fitted_edge_calibration` an edge is scored by the average of the models whose excluded fold contains one of its endpoints, or by all $K$ models if neither endpoint is labeled.

For a labeled node $u$ in fold $a$:

1. If $v$ is unlabeled or in fold $a$, $\hat q_{uv}$ comes from model $a$ alone and is a function of $(X,A)$ and the labels outside fold $a$. It does not depend on $y_u$.
2. If $v$ is labeled and in fold $b\ne a$, $\hat q_{uv}=\frac12[\phi^{(a)}+\phi^{(b)}](z_{uv})$, and $\phi^{(b)}$ was trained on edges incident to $u$ with $y_u$ in their targets. This score depends on $y_u$.

*Proof.* Read off the `eligible` and `touch` masks: model $b$ excludes only edges with an endpoint in fold $b$, and $u\notin$ fold $b$. $\square$

**Consequences.** The share of a labeled node's edges affected by case 2 is the share of its neighbors that are labeled and in another fold. Under GADBench (100 labels) that is about $100/n$, under 1%, and the leak is negligible. Under the repository's 40% protocol with $K=5$ it is about 32% of every training node's edges, which can make the training-node operator look better than the test-node operator and push $\eta$ up.

**Exact fix.** Score a cross-fold labeled pair with a model that excludes *both* folds: $K(K-1)/2$ pair models instead of $K$ (10 instead of 5, a few seconds). With this rule every edge incident to a node is scored by a model that never saw that node's label. This is the property Reviewer 2 asked about, and the paper should state it in one sentence.

**Why it matters for $\eta$.** $\eta$ is learned from the training nodes' loss. If their edges carry in-sample scores, the scores look sharper than they are on test nodes and $\eta$ is tuned to the wrong reliability curve (Corollary 3). Cross-fitting is what makes $\eta$ an honest recalibration, the same role sample splitting plays in double machine learning (Chernozhukov et al. 2018).

### Theorem 5 (degree-free stability, C10)

Let $L=L_\eta(q)$, $L'=L_{\eta'}(q')$ and $\gamma=\min_uD^{(\eta)}_{uu}(q)/\tilde d_u$. Put
$$\delta=\max_u\frac{\sum_v|w_{uv}-w'_{uv}|}{D_{uu}}\ \le\ \frac2\gamma\Big(\eta\,\varepsilon_{\mathrm{loc}}(q,q')+|\eta-\eta'|\,\bar q'_{\max}\Big),$$
where $\bar q'_{\max}=\max_u\tilde d_u^{-1}\sum_v\tilde A_{uv}q'_{uv}\le1$. Then
$$\|L-L'\|_2\le3\delta+\delta^2.$$

*Proof.* $L'-L=N-N'=T_1+T_2$ with $T_1=D^{-1/2}(W-W')D^{-1/2}$ and $T_2=D^{-1/2}W'D^{-1/2}-N'$.

*$T_1$.* The spectral norm of a symmetric matrix is at most that of its entrywise absolute value $M=D^{-1/2}|W-W'|D^{-1/2}$. $M$ is non-negative and symmetric, and for the positive vector $x=D^{1/2}\mathbf 1$, $(Mx)_u=D_{uu}^{-1/2}\sum_v|w_{uv}-w'_{uv}|\le\delta x_u$. By the Collatz–Wielandt bound $\|M\|_2=\rho(M)\le\delta$.

*$T_2$.* Let $R=D^{-1/2}D'^{1/2}$, diagonal with $r_u=\sqrt{D'_{uu}/D_{uu}}$. Then $D^{-1/2}W'D^{-1/2}=RN'R$ and $T_2=(R-I)N'R+N'(R-I)$. Since $|\sqrt x-1|\le|x-1|$ and $\big||w'_{uv}|-|w_{uv}|\big|\le|w'_{uv}-w_{uv}|$, $\|R-I\|_2\le\max_u|D'_{uu}-D_{uu}|/D_{uu}\le\delta$ and $\|R\|_2\le1+\delta$. Proposition 1 gives $\|N'\|_2\le1$. So $\|T_2\|_2\le\delta(1+\delta)+\delta$.

*The bound on $\delta$.* $|w_{uv}-w'_{uv}|=2\tilde A_{uv}|\eta q_{uv}-\eta'q'_{uv}|\le2\tilde A_{uv}\big(\eta|q_{uv}-q'_{uv}|+|\eta-\eta'|q'_{uv}\big)$; sum over $v$ and divide by $D_{uu}\ge\gamma\tilde d_u$. $\square$

**Comparison with the WSDM Theorem 1.**

| | WSDM | Theorem 5 |
| --- | --- | --- |
| Error measure | $\eta\lVert q-q'\rVert_\infty$ (one bad edge gives 1) | $\eta\,\varepsilon_{\mathrm{loc}}$ (worst node's *mean* error) |
| Constant | $2d_{\max}(1+d_{\max})$: $9.8\times10^{7}$ on Amazon, $5.1\times10^{5}$ on YelpChi | $3/\gamma$ at first order, no degree |
| Non-vacuous when | never on real graphs ($\lVert L-L'\rVert_2\le2$) | $\eta\,\varepsilon_{\mathrm{loc}}\lesssim\gamma/4$ |
| Lipschitz in $\eta$ | constant $2d_{\max}(1+d_{\max})$ | constant $6\bar q'_{\max}/\gamma\le6/\gamma$ |

$\gamma$ is the price: it is the smallest fraction of a node's edge mass that survives calibration. $\gamma=1$ at $\eta=0$ and for oracle scores at $\eta=1$; $\gamma\ge1-2\eta\max q$ for $\eta<\frac12$; in the worst case $\gamma\ge1/\tilde d_{\max}$ (only the self-loop survives), which recovers a degree-dependent bound. $\gamma$ must be reported for the learned operator.

### Theorem 6 (the Brier score controls the operator, C11)

For the same $\eta$, with $B=\sum_ub_u$ the sum of local Brier scores between $q$ and $q'$ and $\delta$ as in Theorem 5,
$$\|L_\eta(q)-L_\eta(q')\|_F\le\frac{2\eta}{\gamma}\Big(1+\sqrt2\,(2+\delta)\Big)\sqrt B.$$

*Proof.* Same decomposition. $\|T_1\|_F^2=\sum_{u,v}(w_{uv}-w'_{uv})^2/(D_{uu}D_{vv})\le\frac{4\eta^2}{\gamma}\sum_u\tilde d_u^{-1}\sum_v\tilde A_{uv}(q_{uv}-q'_{uv})^2=\frac{4\eta^2}{\gamma}B$, using $D_{uu}\ge\gamma\tilde d_u$ and $D_{vv}\ge1$. For $T_2$, its entries are $N'_{uv}(r_ur_v-1)$ and $|r_ur_v-1|\le\delta_u(1+\delta_v)+\delta_v$ with $\delta_u=|D'_{uu}-D_{uu}|/D_{uu}$. Rows of $N'$ have squared norm at most 1, because $\sum_vw'^2_{uv}/(D'_{uu}D'_{vv})\le\sum_v|w'_{uv}|/D'_{uu}=1$. So $\|T_2\|_F^2\le2[(1+\delta)^2+1]\sum_u\delta_u^2\le2(2+\delta)^2\sum_u\delta_u^2$. Finally $\delta_u\le\frac{2\eta}{\gamma}\bar e_u$ and $\bar e_u^2\le b_u$ by Jensen, so $\sum_u\delta_u^2\le\frac{4\eta^2}{\gamma^2}B$. $\square$

**Reading.** Dividing by $n$: the mean squared row perturbation of the operator is at most a constant times $\eta^2\gamma^{-2}$ times the node-averaged Brier score. With $q'=t$ this is the held-out Brier score of the calibrator. The Brier term in the Phase 1 loss is therefore not only a calibration heuristic: it is an upper bound on the distance to the oracle operator of Theorem 1.

### Theorem 7 and Corollary 5 (representation robustness and misaligned nodes, C12)

**Theorem 7.** For fixed $\eta$ and any gate on the simplex, $\|Z(q)-Z(q')\|_F\le\max_k\Lambda_k\,(3\delta+\delta^2)\,\|H^{(0)}\|_F$ with $\Lambda_k=\frac12P\kappa_k$ and $\delta\le2\eta\varepsilon_{\mathrm{loc}}/\gamma$.

**Lemma 2.** $\|\mathcal W_k(L)-\mathcal W_k(L')\|_F\le\sup_{[0,2]}|\mathcal W_k'|\cdot\|L-L'\|_F$.

**Corollary 5.** Let $\eta\in(\frac12,1]$, $L^*=L_\eta(t)$ the oracle operator, for which $\gamma^*=1-2(1-\eta)h_{\max}\ge2\eta-1$, and $\hat L=L_\eta(\hat q)$. The $D^*$-weighted fraction of nodes whose low-pass label response $[\mathcal W_0(\hat L)\tilde y]_u$ has the wrong sign is at most $\big(\tfrac P2(3\delta+\delta^2)\big)^2$ with $\delta=2\eta\,\varepsilon_{\mathrm{loc}}(\hat q,t)/\gamma^*$.

*Proof.* Theorem 7: the WSDM proof of its representation theorem with the new stability theorem in place of the old one; the gate drops out for the same reason. Lemma 2: write both matrices in their eigenbases; the entries of $U^\top(f(L)-f(L'))V$ are $(f(\lambda_i)-f(\lambda'_j))(U^\top V)_{ij}$, and $|f(\lambda_i)-f(\lambda'_j)|\le\mathrm{Lip}(f)\,|\lambda_i-\lambda'_j|$. Corollary 5: by Theorem 1, $\mathcal W_0(L^*)\tilde y=\kappa_0\tilde y$. A node with the wrong sign has $|[\mathcal W_0(\hat L)\tilde y]_u-\kappa_0\tilde y_u|\ge\kappa_0\sqrt{D^*_{uu}}$. Summing squares over such nodes: $\kappa_0^2\sum_{\mathrm{wrong}}D^*_{uu}\le\|(\mathcal W_0(\hat L)-\mathcal W_0(L^*))\tilde y\|^2\le\Lambda_0^2(3\delta+\delta^2)^2\sum_uD^*_{uu}$, and $\Lambda_0/\kappa_0=P/2$. $\square$

Corollary 5 is the end-to-end statement: the calibrated operator inherits the separability of the oracle operator up to a term controlled by the local calibration error. It is a worst-case bound and is only informative for small $\varepsilon_{\mathrm{loc}}$; the exact identities (Theorems 2 and 3) are what the experiments should report.

## Numerical verification

### Synthetic (`python theory_check.py synthetic`, about a minute on CPU)

27 checks over 40 random two-class graphs with random scores and $\eta\in[0,1.6]$; all pass. Every statement of Section 5 of the paper has at least one, with the exception of Proposition 3, which is a property of the procedure.

- Identities C1, C2, C3, C6 hold to $10^{-15}$; C9 to $10^{-6}$ (finite differences); C8's gap formula to $10^{-15}$.
- C13 holds to $10^{-3}$ on 3M simulated edges, for uniform and stratified labeling.
- Corollary 5 holds on every instance, but at the sampled noise levels no node was misaligned, so the check does not show how tight it is.
- Bounds are never violated. Worst observed ratio of left to right side: 0.16 for Theorem 5, 0.06 for Theorem 6, 0.06 for Theorem 7, 0.42 for the Frobenius Lipschitz statement. On the same 60-node instances the WSDM bound is loose by a factor of at least 732.

### Real graphs (`python theory_check.py real --dataset {amazon,yelp}`)

These use all ground-truth labels and are analysis only. Graphs are deduplicated (YelpChi matches GADBench's 3,846,979 edges). Output is in `runs/theory_real_{amazon,yelp}.json`.

| Quantity (theorem) | Amazon | YelpChi |
| --- | --- | --- |
| Anomaly rate | 9.5% | 14.5% |
| Label frequency of the unsigned operator, $\rho_0=2\tilde h$ (Cor 1) | 0.102 | 0.451 |
| $\bar h_a$ / $\bar h_n$ | 0.849 / 0.035 | 0.795 / 0.131 |
| Anomalies with $h_u>\frac12$ | 98.3% | 94.1% |
| Class-mean separation, unsigned (oracle signing is 2) (Thm 3b) | 0.23 | 0.15 |
| Mean $\kappa_u$ of anomalies: unsigned / oracle prune / oracle sign (Thm 3b) | −9.9 / 4.0 / 12.8 | −7.4 / 5.0 / 12.0 |
| Mean $\kappa_u$ of normal nodes: same order | 23.6 / 24.7 / 25.1 | 9.2 / 11.6 / 12.4 |
| Anomaly–anomaly share of anomaly-incident edges (Prop 2) | 4.5% | 10.1% |

The camouflage collapse and the sign-over-prune gain (3.2× and 2.4× for anomalies) are both large on real data.

**What a calibrator learns under GADBench's label budget** (100 labels, 20 anomalies; 5 random labeled sets; evaluated on edges with both endpoints unlabeled). This is a quick stand-in calibrator built from the repository's descriptor and loss, not the paper's experiment code, so treat the numbers as indicative.

| | Amazon: E2 only | Amazon: E1+E2 (paper rule) | Amazon: E1+E2 with true labels | YelpChi: E2 only | YelpChi: E1+E2 (paper rule) | YelpChi: E1+E2 with true labels |
| --- | --- | --- | --- | --- | --- | --- |
| Training edges (undirected) | 408 | 85,700 | 85,700 | 18 | 17,300 | 17,300 |
| Edge AUROC against true heterophily | 0.831 | 0.850 | 0.943 | 0.563 | 0.602 | 0.667 |
| Brier | 0.054 | 0.039 | 0.038 | 0.340 | 0.251 | 0.243 |
| Mean score: normal–normal / anomaly–normal / anomaly–anomaly | 0.03 / 0.48 / 0.33 | 0.01 / 0.30 / **0.79** | 0.06 / 0.73 / 0.54 | 0.32 / 0.40 / 0.44 | 0.09 / 0.17 / **0.24** | 0.27 / 0.48 / 0.50 |
| Label frequency at $\eta=1$ (unsigned: 0.102 / 0.451) | 0.104 | 0.075 | 0.057 | 0.737 | 0.506 | 0.543 |
| Lowest label frequency over $\eta$, and where | 0.056 at 0.50 | 0.070 at 0.60 | 0.028 at 0.57 | 0.407 at 0.50 | 0.420 at 0.50 | 0.345 at 0.54 |

Three things follow.

- **Proposition 2 is visible.** With the paper's rule, anomaly–anomaly edges get the highest score on both graphs, and 80% of them would be flipped on Amazon at $\eta=1$. The modal target is wrong on 3.0% of Amazon's E1 edges and 14.0% of YelpChi's.
- **The calibrator is the bottleneck, not the operator.** Oracle signing brings the label frequency to 0; the stand-in calibrators reach 0.03–0.07 on Amazon and 0.35–0.42 on YelpChi.
- **On YelpChi at 100 labels, flipping edges raises the label frequency for all three stand-in calibrators**; only reweighting ($\eta\le\frac12$) lowers it. The paper reports its largest gains on YelpChi, so the actual calibrator must be measured the same way before the mechanism is claimed there.

**Does the learned $\eta$ behave as Corollary 3 predicts?** Two data points from today's 500-epoch runs under the repository's 40% protocol:

| | Learned $\eta$ | Reliability slope $\eta_0$ of the E2 calibrator | $\eta$ minimising the label frequency |
| --- | --- | --- | --- |
| Amazon | 0.541 | 0.496 | 0.55 |
| YelpChi | 0.606 | 0.623 | 0.71 |

The learned strength is within 0.05 of the reliability slope on both, **but this is not evidence yet** (checked on Nov 18, runs in `runs/plan_check/`). $\eta$ starts at 0.5 in the code. Started at 0.2 and 0.8, it ends at 0.20 and 0.84 on Amazon after 500 epochs, so the Amazon agreement comes from the start value; test AUPRC is the same in all three runs. On YelpChi the starts 0.2 and 0.5 both end at 0.60 and the start 0.8 is at 0.78 and falling, which does support the corollary, but with a larger learning rate on $\eta$ all runs end near 0.9. Experiment C4 in `experiment.md` now uses three starts and counts a split only when they agree, with the best fixed $\eta$ as the fallback evidence. The sentence after Proposition 3 in the paper ("$\eta$ estimates the reliability slope of the scores") depends on that outcome.

**Outcome (DP0, Nov 18, `runs/dp0/`).** Under 100 labels, in 10 draws each on Amazon and YelpChi, runs started at 0.2, 0.5 and 0.8 never end within 0.05 of each other; $\eta$ moves by 0.02 to 0.03 on average because the training loss is near zero by epoch 100. With $\eta$ fixed on a grid, test AUPRC is flat within the spread over draws and the best value does not track the slope. Corollary 3 therefore has no empirical support yet: it stands as a statement about the optimal aggregator, not about what training finds. The same run shows no significant gain of any fixed $\eta$ over $\eta=0$ on these two graphs (details in `experiment.md`, "DP0 result").

## Consequences for the method

These follow from the analysis and need a decision from the authors. None changes the operator or the main-table protocol. Experiment ids refer to `experiment.md`.

| # | Finding | Proposed action | Cost | State |
| --- | --- | --- | --- | --- |
| M1 | The E1 target scores anomaly–anomaly edges as most heterophilic (Prop 2) | Keep the paper's rule as the default, add the soft-imputation target as an ablation (D4), and adopt it if it wins on calibration and AUPRC | one extra cross-fitted node MLP | Stated in the paper (Prop 2, Cor 4, one sentence on soft imputation). Code unchanged. Decision open |
| M2 | Cross-fold labeled pairs leak the node's own label (Prop 3) | Switch to pair-excluded models (D5) | 10 fold models instead of 5 | Stated in the paper (Prop 3). Code unchanged. Decision open |
| M3 | `pos_weight` in the calibrator loss deliberately miscalibrates the scores, against the "calibrated" claim (Thm 4) | Ablate with and without (D6); report the reliability diagram either way (B2) | none | Open |
| M4 | The optimal slope can exceed 1 (Cor 3), but $\eta=\mathrm{sigmoid}(\cdot)\le1$ | Ablate $\eta\in[0,2]$ (D6) | none | Paper side done: Prop 1 covers $\eta\ge0$ and Section 5.2 says the slope need not stay below one. Code unchanged |
| M5 | The gate has no theorem behind its benefit; the KL bound on $\min_kg_k$ is vacuous at $\alpha\le0.1$ | Present the gate as a design choice. Keep the square-root parameterisation, whose role in Theorem 7 is real. Stop claiming the KL term "stabilises training" unless the ablation shows it (D7) | none | Open. Section 4.2 of the paper still makes the claim (task T3) |
| M6 | The repository's driver skips the Salton and Jaccard statistics above 5M edges, which includes both graphs used so far | Compute overlaps in chunks before any reported run (E0.8) | minutes | Open |

## Paper status

Done on Nov 14:

- **Section 5** (`theory.tex`): 5.1 why signing helps (Theorems 1–2, Corollaries 1–2); 5.2 sign, reweight or prune (Theorems 3–4, Corollary 3, Remarks 1–2); 5.3 what weak edge supervision identifies (Propositions 2–3, Corollary 4); 5.4 robustness (Theorems 5–7, Corollary 5).
- **Appendix A** (`proof.tex`): all proofs, Lemmas 1–2, the table of expected E2 and E1 sizes at 100 labels (Table 7), and a paragraph on the numerical checks.
- **Outside Section 5:** one sentence in the abstract, three in the introduction, the cross-fitting rule and two pointers in Section 4.1, Proposition 1 stated for $\eta\ge0$, the robustness paragraph of Section 6 pointed at the new theorems, six verified references.

Three things in the paper go beyond a literal port and need the authors' confirmation:

- Section 4.1 now *describes* the cross-fitting rule (node-grouped folds, an edge scored by the models whose held-out fold contains an endpoint). This follows `src/edge_calibration.py`. **Confirmed on Nov 19:** the WSDM detector (`wsdm-2026/GADBench-master/models/gnn.py`, `echogad_cross_fitted_edge_calibration`) uses the same rule.
- Proposition 3 states the residual label dependence of that rule openly and names the pair-excluded variant as the exact fix, without claiming the method uses it.
- Section 5.3 mentions soft imputation of the unlabeled endpoint as the unbiased alternative to the paper's rule, again without claiming it is used.

## Remaining theory work

### Paper side, no new mathematics

| ID | Task | Depends on | Priority |
| --- | --- | --- | --- |
| T1 | Confirm or correct the cross-fitting description in Section 4.1, and decide M1 and M2, so that Section 4.1 and Propositions 2–3 describe what is actually run | authors | P0 |
| T2 | Put measured quantities next to the theorems: (a) label frequency against $\eta$ with the learned $\eta$ marked, for Theorem 2 (main-text figure); (b) $\tilde h_a$, $\tilde h_n$, the share of anomalies with $\tilde h_u>\frac12$ and the node factor under the unsigned, pruned and signed operators per dataset, for Theorem 3 (table); (c) learned $\eta$ against the reliability slope, for Corollary 3 (figure); (d) measured E1 and E2 counts under the GADBench splits in place of the expected counts of Table 7 | experiments B7, C1, C2, C4 | P0 |
| T3 | Remove or soften the two claims the theory and the baselines do not support: "the KL loss stabilizes training" in Section 4.2 and "best on six of seven" in Section 6.2 | M5 and experiment D7; experiment A2 | P0 |
| T4 | Cut to 8 pages. Proposal for Section 5 (about 1.5 pages): keep Theorems 1 and 2, Theorem 3 parts (a) and (b), Theorem 4 with Corollary 3, Proposition 2 and Theorem 5 in the main text; move Corollaries 1, 2, 4, 5, Remarks 1–2, Proposition 3, Theorem 3 parts (c) and (d) and Theorems 6–7 to the appendix with one sentence each. Elsewhere: related work to half a page; gate ablation, runtime table and algorithm box to the appendix | authors choose what stays | P0 before submission |
| T5 | Regenerate the robustness figure with $\eta\,\varepsilon_{\mathrm{loc}}$ on the horizontal axis and report $\gamma$. The text currently bridges with $\varepsilon_{\mathrm{loc}}\le\lVert q-q'\rVert _\infty$ | experiment C5 | P1 |
| T6 | Contribution list at the end of the introduction (draft below) | none | P1 |
| T7 | Related work: one paragraph on the theory of GNNs under heterophily (contextual block-model analyses) and on spectral theory of signed graphs | verify each reference first | P2 |

Draft contribution list:

1. A signed spectral operator whose only learned operator parameter is a recalibration slope (unchanged method).
2. A theory of *why* it works: gauge equivalence, the label-frequency identity, and optimality of soft signs over pruning and reweighting.
3. Degree-free stability in the calibrator's local and Brier error, with an end-to-end bound.
4. An analysis of weak edge supervision that shows what singly-labeled edges can and cannot identify.

### New mathematics, optional, ranked by value for the effort

Each candidate has a scouting experiment in `experiment.md` that says whether it is worth proving. Run the experiment first.

| ID | Result | Why it is worth having | Scouted by | Effort | Priority |
| --- | --- | --- | --- | --- | --- |
| N1 | Average-case stability: for independent zero-mean calibration noise of size $\varsigma$, $\lVert L_\eta(\hat q)-L_\eta(\bar q)\rVert _2=O\big(\eta\varsigma\gamma^{-1}\sqrt{\log n/d_{\min}}\big)$ with high probability, by matrix Bernstein | Theorem 5 is worst case. This would show that the calibrator's variance averages out over a node's edges and only its bias survives, which is the argument for cross-fitting and ensembling and against a biased target. It would also explain the small operator changes in the paper's robustness figure, which perturbs the scores with independent noise | C16 | 1–2 days. The absolute value in the degree needs a margin condition. Check numerically first | P1 |
| N2 | Multi-hop behaviour under noisy signs. Free part: the $k$-hop label coherence $\tilde y^\top N^k\tilde y/\lVert \tilde y\rVert ^2$ is at least $(1-2\mathrm{Fr})^k$ for even $k$, by Jensen on the spectral measure of Theorem 2. Open part: the decay rate under independent sign errors, about $(1-2\epsilon)^k$ on tree-like neighborhoods | Explains why $P\in\{2,3\}$ is enough and predicts the outcome of experiment C6 | C17, C6 | half a day for the free part, about 2 days for the rest | P1 for the free part, P2 for the rest |
| N3 | Plug-in rate: if the calibrator's excess Brier risk is $r$, then $\frac1n\lVert L_\eta(\hat q)-L_\eta(\bar q)\rVert _F^2\le C\eta^2\gamma^{-2}r$ by Theorem 6, with $\bar q$ the population target of Proposition 2 | Closes the old gap G1 cheaply and makes explicit that the calibrator converges to the anomaly-incidence probability, not to $q^*$ | C18 | half a day | P1 |
| N4 | Unbiasedness of the soft-imputation target under a calibrated node posterior | Needed only if M1 is adopted | B4, D4 | half a day | P2, conditional |
| N5 | Optimality of a two-parameter Platt map for logistic reliability curves | Needed only if the reliability curves are clearly not straight (C10) or experiment D6 shows a gain | C10, D6 | half a day | P2, conditional |
| N6 | What the single $\eta$ optimises: the strength that maximises the average node deflection of Theorem 4, and how far it is from the per-node optimum $\eta^\star_u$ of anomalies and of normal nodes | Would turn "$\eta$ is learned" into a statement of what it converges to, and would give a principled case for a class- or degree-dependent strength | C19 | 1 day | P1 if C19 shows a clear split between the classes |
| N7 | Theorems 3 and 4 with correlated noise between neighbors and dependent neighbor signs: the effective degree and the variance term get a dependence factor | Needed only if the assumption audits find correlation (C9) or over-dispersion (C10) large enough to break the predicted error rates | C9, C10 | 1–2 days | P2, conditional |
| N8 | Node-wise lift. In the reweighting regime the margin $m_u(\eta)$ is monotone in $\eta$, increasing iff $\bar q^{\mathrm{het}}_u>\bar q^{\mathrm{hom}}_u$ (the node-level form of Corollary 1; the $\eta$ terms of $m_u'$ cancel, as in the proof of Theorem 3(d)). With the E1 target, an anomaly with anomalous neighbors has negative lift in expectation (Proposition 2) | Explains the flat region found on Nov 18 as a prediction: the lift of anomalies is −0.18 on Amazon and −0.07 on YelpChi with the paper's target. Gives a score-only quantity that says, before any training, whether signing can help | lift by class per dataset, target and label budget (scores only); gain over $\eta=0$ against the mean lift of anomalies | half a day | P0 |
| N9 | $\eta$ is not identified by the node loss. The Beta channels sum to $\frac{P+1}{2}I$ for every $\eta$, so the model always contains the feature-only network and can fit the labeled nodes whatever the operator; at interpolation the gradient in $\eta$ is zero. Remedy from the theory: set $\eta$ in Phase 1, as the minimiser of the cross-fitted label frequency on labeled edges (Theorem 2) or as the reliability slope (Corollary 3) | Turns the DP0 failure into a statement and replaces "learned $\eta$" by a computed, start-independent value | computed $\eta$ against the best fixed $\eta$ on the grid of D1, per dataset | half a day | P0 |
| N10 | Soft imputation restores positive lift when the node posterior is informative (extends N4): with a calibrated $\hat\pi$ the E1 target is unbiased for $q^\star$, and the lift of an anomaly is positive once $\hat\pi$ separates its anomalous from its normal neighbors | Preview: lift of anomalies goes from −0.18 to +0.26 on Amazon and stays at −0.06 on YelpChi | B4, D4 with the lift and $\chi_u$ reported next to AUPRC | half a day | P1 |
| none | A theorem for the KL gate | Not pursued: the bound on $\min_kg_k$ is vacuous at $\alpha\le0.1$ (M5) | | | dropped |

Working rule: when a statement changes, change `theory.tex`, `proof.tex`, the matching check in `theory_check.py` and the claims table here in the same edit.

## Open items

- **Assumption 2** (conditional independence of neighbor signs) is a model. Theorem 4 should be presented as a design principle with its assumption visible, and Theorems 1, 2, 5, 6 as the assumption-free core.
- **The check of Corollary 5 is not informative yet.** It passes, but no node was misaligned at the sampled noise levels. An instance with larger, structured errors would show how loose the bound is.
- **The stand-in calibrator** in `theory_check.py real` skips neighborhood-overlap statistics on YelpChi (memory). The YelpChi edge AUROC may improve once they are computed.
- **References.** The six new entries in `references.bib` were verified online. Lemma 2 is proved in the appendix, so no reference is needed for it.
- **`theory_check.py` is committed** (`85e68f9` on `icml2027`, not pushed) **on the earlier machine only.** It is not in this workspace or in the `phase1` branch. The paper's appendix reports its results, so the script has to be in the repository that is released.
- **Directed graphs.** Everything here assumes an undirected graph. GADBench stores Weibo, Tolokers and Questions directed, and the WSDM runs used them as stored. Runs that are meant to match the theory have to symmetrize these three first.
