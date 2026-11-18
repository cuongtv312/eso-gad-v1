# ESO-GAD experiment plan for ICML 2027

Nov 17, 2026 · theory-first track verified Nov 18 · workspace moved Nov 19 (see "Workspace on this machine") · companion to `theory.md` · theory first · budget: under 100 GPU-hours in total, local GPU (RTX 3090, 24 GB, since Nov 19; the Nov 14–18 runs used an RTX 4060 Ti, 16 GB) plus Vast.ai rentals of \$0.5–1.5 each, in small batches · theorem numbers are the paper's

## Where things stand

| Item | State |
| --- | --- |
| GPU environment | `gcnenv` on this machine since Nov 19 (it replaces `gnnenv`); versions and smoke runs in "Workspace on this machine" |
| GPU path, `--seed`, `--out` and AUPRC reporting in the ESO-GAD driver | **rebuilt Nov 19** in `main_ec_sbwgnn.py` (also `--fix_eta`, `--eta_lr`). The Nov 17 commit `85e68f9` stayed on the earlier machine |
| ESO-GAD on Amazon and YelpChi, seed 2, 500 epochs | done, but under the repository's protocol (40% labels, E2-only calibrator); not comparable with the paper's tables |
| Synthetic checks of the theory | 27 of 27 passed on the earlier machine. **`theory_check.py` was rewritten on Nov 19** from the paper's statements; it now has 35 checks. Its full run (40 graphs) has not been done on this machine yet |
| Oracle diagnostics on Amazon and YelpChi (`theory_check.py real`) | done: a preview of B3, B4, B7, C1, C2 and C4 with a stand-in calibrator |
| Check of the theory-first track against the paper, the code and short local runs | done Nov 18: 14 problems found and corrected in this file; see "Verification of the track". Probe runs are in `runs/plan_check/` |
| Theory in the paper | done: Section 5 and Appendix A. It contains no measured number yet |
| GADBench protocol and the paper's calibrator in the source | **rebuilt Nov 19** from `BWGNN_codebase`: `gadbench.py`, `main_gadbench.py`, and the paper's rule, the soft target and exact overlap statistics in `edge_calibration.py` (E0.0–E0.2, E0.4, E0.5, E0.8). Tested for execution on a small synthetic graph only; **no run on a real dataset yet** (author, Nov 19: no experiments for now). See "Rebuilt on Nov 19" |
| Everything in blocks A to G under the GADBench protocol | not started; the code for A1 exists, the rest waits for E0.4 and the decisions below |
| Vast.ai | **Not used for now (author, Nov 19): experiments run on the local GPU; Vast.ai only for runs that need more memory than its 24 GB.** Account ready (SSH key registered, \$9.82 of credit left after the DP0 rental, no billing method); the CLI is not installed in `gcnenv` |
| Code repository | Since Nov 19 one repository holds everything: `/home/gnn/icml2027-esogad` (remote `git@github.com:cuongtv312/eso-gad-v1.git`), local branch `phase1`. Tracked: the ESO-GAD source `wsdm-2026/BWGNN_codebase/*.py`, `theory-phase1/` and the paper sources in `icml-paper/icml_ESO_GAD/`; everything else is ignored. Commits use the author's identity only and go to `phase1`. The old `src/` clone (branch `icml2027`) and `runs/` stayed on the earlier machine |

Less than 1 of the 100 GPU-hours has been used.

## Workspace on this machine (Nov 19)

The work moved to a new machine on Nov 19. In this file, `theory.md` and `learn_eta_vs_fixed_eta.md`, read `src/` as `wsdm-2026/BWGNN_codebase/`, `data/gadbench/` as `wsdm-2026/GADBench-master/datasets/` and `gnnenv` as `gcnenv`. Entries dated Nov 14–18 describe the earlier machine and are kept as the record of what was run.

| Item | Here |
| --- | --- |
| Source | `wsdm-2026/BWGNN_codebase/`, **the original code base of the WSDM paper** (author, Nov 19): `ec_sbwgnn.py`, `edge_calibration.py`, `main_ec_sbwgnn.py`, the three `run_*.py` sweeps, and BWGNN's own `BWGNN.py`, `dataset.py`, `main.py`, under BWGNN's protocol (random stratified split by `--train_ratio`, selection by validation macro-F1). The Phase 1 infrastructure was rebuilt on top of it on Nov 19 (next section) |
| Data | `wsdm-2026/GADBench-master/datasets/` (override with `ESOGAD_DATA_DIR`). `dataset.py` now reads all seven graphs from there; `Dataset(name, source='dgl')` keeps BWGNN's original loaders |
| Environment | conda `gcnenv`: Python 3.8.18, torch 2.0.1+cu117, dgl 0.9.1post1, scikit-learn 1.3.2, scipy 1.10.1, numpy 1.23.1, sympy 1.12, xgboost 2.0.3, pandas 2.0.3. No `vastai` |
| GPU | one RTX 3090, 24 GB |
| LaTeX | **the paper does not build here.** Only `texlive-base` and `texlive-latex-base` are installed; `microtype`, `booktabs`, `mathtools`, `cleveref`, `xcolor`, `caption`, `subcaption`, `multirow`, `dsfont`, `eso-pic`, `forloop` and `placeins` are missing and installing them needs root. `main.pdf` in the paper folder was built on the earlier machine |

**The Nov 14–18 code and outputs stayed on the earlier machine:** `theory_check.py`, `gadbench.py`, `main_gadbench.py`, `dp0_eta_check.py`, the Nov 18 version of `edge_calibration.py`, the `--seed` and `--out` flags, and every `runs/` folder (`plan_check`, `dp0`, `flat_eta`, `gadbench`, `theory_real_*.json`). The author decided on Nov 19 to rebuild from `BWGNN_codebase`; the code is rebuilt (next section). **The numbers quoted from the old runs are not reproduced yet**: every result in this file, `theory.md` and `learn_eta_vs_fixed_eta.md` dated Nov 14–18 is from the earlier machine and has to be re-run with the rebuilt code before it is used.

**Found here: the runs behind the WSDM tables.** `BWGNN_codebase` is the original code base. The tables were produced under the GADBench protocol by its port in `wsdm-2026/GADBench-master/` (third-party benchmark folder, not tracked): the `ECHOGAD` detector in `models/detector.py` and `models/gnn.py`, the drivers `echogad_*.py`, `rq2_*.py`, `rq3_*.py` and `runtime_*.py`. This closes point 1 of "Five things to settle" below. Read from that code on Nov 19:

- **Protocol.** GADBench's own: mask columns 10–19, at most 200 epochs, patience 50 on validation AUPRC, Adam at 0.01, class-weighted cross-entropy, seeds 3407 + 10 × trial. It matches the "Protocol" section.
- **Graph.** The GADBench files as shipped, with `add_self_loop(remove_self_loop(·))`. Weibo, Tolokers and Questions are stored directed and were **not symmetrized**; the other four are symmetric. All seven files already contain one self-loop per node and no parallel edges. This answers the question raised under the E0 smoke runs.
- **Calibrator.** `echogad_cross_fitted_edge_calibration` is a port of the tracked `edge_calibration.py` with the same rule: stratified folds by labeled node, fold model $j$ trained on labeled edges with both endpoints outside fold $j$, an edge scored by the models whose held-out fold contains an endpoint, all $K$ models when no endpoint is labeled, self-loops set to 0. So Section 4.1 of the paper describes the rule the WSDM code ran (point 2 of "Five things to settle"). The loss is BCE with `pos_weight` plus Brier on $\mathcal E_2$, plus `lambda_p` times the same on $\mathcal E_1$, each term averaged over its own edges. `lambda_p` defaults to 0 and was swept per dataset by the `echogad_*_lambda_p_10trial.py` scripts; **which value produced each row of the WSDM table is not recorded in the folder** and needs the author.
- **Overlap statistics.** The same 5M-edge guard: Salton and Jaccard were zero on Amazon, YelpChi and T-Finance in the WSDM runs too (E0.8, M6).
- **$\eta$.** The default start is 0.5 with $\eta=\mathrm{sigmoid}(\cdot)$; `echogad_eta_init_10trial.py` and `echogad_eta_investigation.py` varied it. The detector also has what E0.4 asks for: fixed $\eta$ (`learn_eta=False`), the sign / reweight / prune variants, a uniform gate, per-channel $\eta$ and temperature scaling of the scores.
- **Results.** The per-run CSV files of the WSDM tables are not in the folder; only `results/*.xlsx` (120 files) and the runtime CSVs are.

**Smoke runs here (Nov 19, about 1 GPU-minute in total).** They show the code runs and give the timings on this GPU; none is a result.

| Dataset | Directed edges | Tracked source, 40% labels, `lambda_p` 0.5, 5 epochs: Phase 1 / Phase 2 / peak GPU memory | WSDM code, GADBench split 0, default configuration: test AUPRC (WSDM mean of 10) |
| --- | --- | --- | --- |
| Reddit | 168,016 | 0.8 s / 0.21 s / 0.2 GB | 4.0 (5.5) |
| Weibo | 416,368 | 1.9 s / 0.17 s / 2.0 GB | 81.9 (86.2) |
| Tolokers | 530,758 | 1.0 s / 0.16 s / 0.2 GB | not run |
| Questions | 202,461 | 0.8 s / 0.29 s / 0.8 GB | not run |
| Amazon | 8,847,096 | 6.2 s / 0.38 s / 4.4 GB | 80.7 (83.7) |
| YelpChi | 7,739,912 | 6.3 s / 0.45 s / 4.5 GB | 25.5 (28.9) |
| T-Finance | 42,484,443 | 25.6 s / 1.42 s / 16.2 GB | not run |

- Phase 1 used 30 calibrator epochs per fold; the default is 150. At 40% labels the calibrator trains on millions of edges, so Phase 1 times and memory are far above the 100-label numbers in the E0 table.
- The original BWGNN driver (`main.py`, homogeneous and heterogeneous) and `run_weak_edge_confirmation.py` also run.
- **The local GPU is now the card the rental rules name**, and T-Finance fits (16.2 GB at 40% labels). **Author, Nov 19: experiments run on the local GPU; Vast.ai is kept for runs that need more than its 24 GB.**

## Rebuilt on Nov 19

Everything below is in `wsdm-2026/BWGNN_codebase/` and tracked. **It was tested for execution only**, on a 400-node synthetic graph on the CPU (every option of the driver, the three analysis scripts, and a 6-graph pass of `theory_check.py synthetic`). **No experiment has been run with it** (author, Nov 19). Outputs go to `output/phase1/` (untracked), which replaces the old `runs/`.

| File | Content | Replaces |
| --- | --- | --- |
| `gadbench.py` | The GADBench protocol: graph files with one self-loop per node, the shipped splits (columns 10–19), seeds 3407 + 10 × trial, AUROC / AUPRC / Rec@K, random draws of a GADBench-style labeled set, the mask of nodes with a known label. **Weibo, Tolokers and Questions are symmetrized by default**; `--no_symmetrize` keeps them as the WSDM runs used them | E0.1 |
| `main_gadbench.py` | Driver: Phase 1, then Phase 2 with at most 200 epochs and patience 50 on validation AUPRC. Options: fixed $\eta$ (`--fix_eta --eta_init`), $\eta\in[0,\eta_{\max}]$ (`--eta_max`), a separate learning rate for $\eta$, `--variant sign / reweight / prune / none` (`none` is the feature-only network with the same encoder and head), `--q_mode calibrated / oracle / shuffled / flipped / product`, `--oracle_flip`, uniform gate, `--save_q`, `--analyze` (score diagnostics with all labels), `--out` (one JSON line per trial) | E0.1, E0.4, E0.5 |
| `edge_calibration.py` | Added to the original file, whose defaults are unchanged: `edge_target='e2'`, `'e1e2'` (the rule of Section 4.1: one loss over $\mathcal E_2\cup\mathcal E_1$) and `'soft'` (unlabeled endpoint imputed by a cross-fitted feature-only node posterior); `pos_weight=False` (the paper's unweighted loss); `train_on_self_loops=False`; exact Salton and Jaccard above the 5M-edge guard, in chunks on the GPU (`exact_overlap`). `main_gadbench.py` uses the paper's rule, unweighted, without self-loops, by default; `--edge_target legacy --lambda_p … --edge_pos_weight 1 --train_on_self_loops 1` gives the original calibrator | E0.2, E0.8 |
| `ec_sbwgnn.py` | Fixed or learned $\eta$, $\eta_{\max}$, the weight variants, a uniform gate, evaluation of a trained model under another $\eta$ or other scores (for C12), the encoder output (for C9, C13). The powers $L_\eta^jH^{(0)}$ are now shared across channels (item F1 of block F): $P$ sparse products per forward pass in place of $(P+1)P$, same output | E0.4, F1 |
| `main_ec_sbwgnn.py` | `--seed`, `--out`, `--fix_eta`, `--eta_lr` (for the 40%-label runs of `plan_check`) | E0.0 |
| `diagnostics.py` | Score-level quantities with all labels: frustration and label frequency (Theorem 2), its minimiser over $\eta$, reliability slope and curve, edge AUROC / AUPRC / Brier / calibration error and mean score by edge type, $\tilde h_u$, margin, effective degree and node factor $\chi_u$ (Theorem 3), node-wise lift and the first-order term of Theorem 3(d) | the analysis parts of `theory_check.py real` and of the `runs/flat_eta` script |
| `theory_check.py` | `synthetic`: 35 checks on random two-class graphs (every statement of Section 5 except Proposition 3, plus the identity $\sum_k\mathcal W_k=\frac{P+1}{2}I$ of N9). `real --dataset …`: the oracle facts of B7 and the three stand-in calibrators (E2 only, the paper's rule, E1+E2 with true labels) under 100 labels | `theory_check.py` |
| `dp0_eta_check.py` | DP0: learned $\eta$ from starts 0.2 / 0.5 / 0.8 without early stopping, the fixed-$\eta$ grid in steps of 0.125, slope and label-frequency minimiser, with a `--summarize` mode | `dp0_eta_check.py` |
| `flat_eta_check.py` | The two diagnostics "Why the loss is flat in $\eta$" and "What would remove the flat region": $\chi_u$ by $\eta$, lift, weights at $\eta=1$, oracle signing and pruning, and the paper / soft / product targets next to the feature-only references | the scripts of `runs/flat_eta` |

Three things differ from what the Nov 14–18 text describes and matter when the numbers are compared:

- **Not the same script, so not the same numbers.** The old `theory_check.py` had 27 checks and gave the ratios quoted in Appendix A of the paper (0.16, 0.06, 0.06). The rewritten one has 35 and will give other ratios; `proof.tex` has to be updated from its full run.
- **Definitions fixed here.** The reliability slope is the least-squares slope through the origin of the true heterophily indicator on the score, $\sum\hat qt^\star/\sum\hat q^2$. On Amazon every quantity that uses all labels is computed on the subgraph of the 8,639 labeled nodes (the first 3,305 nodes have no label). With these two conventions the oracle facts quoted in `theory.md` for Amazon and YelpChi are reproduced exactly (checked on Nov 19 from the graph files, no training involved).
- **The calibrator default of `main_gadbench.py` is the paper's rule, not the WSDM code's.** The WSDM runs used the class-weighted loss, trained on self-loops, weighted $\mathcal E_1$ by `lambda_p` and did not symmetrize; each of these is one flag.

Order in which the old results have to be re-run before anything new (all local, minutes each except the last two):

1. `python theory_check.py synthetic`, then update the ratios in `proof.tex` and the check count in `theory.md`.
2. `python theory_check.py real --dataset amazon` and `--dataset yelp` (replaces `runs/theory_real_*.json`).
3. `python main_gadbench.py --dataset all --trials 0` (the E0 smoke table).
4. `python dp0_eta_check.py --datasets amazon,yelp --out …` (DP0; 34 minutes on a rented 3090 on Nov 18).
5. `python flat_eta_check.py --datasets amazon,yelp --out_dir …` (the two diagnostics).
6. The 18 runs of `plan_check` with `main_ec_sbwgnn.py` (40% labels, three starts, two learning rates).

## Summary

The main table stays (seven GADBench datasets, 100 labels, 10 splits). What changes is everything around it: the baselines it is compared with, the direct evidence that the mechanism works, and the controls that attribute the gain to the signed operator. Since the theory is now in the paper, a second job has been added: each theorem needs its measured counterpart (next section).

**Two project phases (author, Nov 18).** Not to be confused with Phase 1 and Phase 2 of the method. The first project phase is the theory-first track. The main experiments are not re-run in it: the main table keeps the WSDM numbers for now. The second project phase adds datasets and baselines and re-runs the main table then (block A, with the regression check E0.6).

**Order changed on Nov 17: theory first.** The next experiments are the ones that check or extend the theory ("Theory-first track" below). The main table, the baselines and the large graphs come after. Large-VRAM runs (T-Social) are skipped for now.

**Do we need more baselines?** Yes, six, and two of them change the headline.

- **RF-Graph and XGB-Graph are mandatory** (Reviewers 2 and 4). With them, ESO-GAD is no longer "best on six of seven": on the WSDM numbers it would be best in AUPRC on four (YelpChi, Tolokers, Reddit, Weibo), about level on Amazon, and behind on T-Finance (73.6 against 81.1) and Questions (9.8 against 11.3 for plain RF).
- **FAGCN and H2-FDetector** are needed for the novelty claim (Reviewer 2): they are the existing signed-propagation methods.
- **SpaceGNN and PMP** are the recent methods reviewers named (Reviewers 4 and 2).
- GREET (unsupervised) and NSReg (open-set) are different settings: discuss them in related work, do not run them.

**Do we need more datasets?** Not to support the claims, but two should be added to close Reviewer 2's objection: DGraph-Fin (safe) and T-Social (stretch, memory permitting). That gives the same nine datasets SpaceGNN uses. Expect little or no gain there: with 100 labels on a sparse graph the calibrator sees a few hundred edges, and the theory predicts $\eta$ shrinks toward the unsigned backbone. Report them as scalability and limits, not as wins.

**What the claims should become.** The defensible contribution is at operator level: a signed operator that improves its unsigned spectral backbone on every dataset, for reasons the theory states and the experiments verify, and that is competitive with tree ensembles. Experiment D8 (signed aggregation inside XGB-Graph) is the cheapest way to turn the tree baselines from a threat into supporting evidence.

**Five things to settle before any long run.**

1. ~~The code behind the WSDM tables is not in this workspace.~~ **Found on Nov 19** in `wsdm-2026/GADBench-master/` (see "Workspace on this machine"). The tracked source still uses BWGNN's protocol. What is left to settle: the GADBench protocol and the operator variants are brought into the tracked source (from the WSDM detector, or from the Nov 18 files of the earlier machine), and the author says which `lambda_p` each WSDM row used.
2. ~~Section 4.1 of the paper now describes the cross-fitting rule as `src/edge_calibration.py` implements it. Confirm that the WSDM experiments used the same rule, or correct the text.~~ **Confirmed on Nov 19** by reading the WSDM detector: same folds, same training edges per fold model, same scoring rule.
3. `theory.md` lists two method issues (M1: the E1 target is biased toward anomaly–anomaly edges; M2: a label leak in cross-fitting). Decide whether the fixes become the default or stay ablations. Until then the paper states both openly and claims neither fix.
4. On YelpChi with 100 labels a stand-in calibrator reaches edge AUROC 0.60 and flipping edges *raises* the label frequency. If the real calibrator behaves the same, the mechanism claim for YelpChi needs rewording (decision point DP1).
5. The learned $\eta$ follows its start value: on Amazon it ends at 0.20, 0.54 or 0.84 from starts of 0.2, 0.5 or 0.8 (Nov 18 probes). Every experiment that reads the learned $\eta$ waits for a local check under 100 labels (decision point DP0, "Verification of the track").

## What the paper now commits us to

Section 5 states results without data. Each row is a statement, the experiment that must stand behind it, and where the result goes.

| Paper statement | What it says | Experiment | Lands in the paper as |
| --- | --- | --- | --- |
| Theorem 2, Corollary 1 | The label frequency is twice the frustration; a calibrator with positive lift lowers it for every $\eta\le\frac12$ | C1 | Main-text figure in Section 5.1 or 6: label frequency against $\eta$, learned $\eta$ marked |
| Corollary 2 | Signing moves label energy from the high-pass to the low-pass channel | C3 | Appendix figure, with the learned gate |
| Theorem 3 | Camouflaged anomalies sit on the wrong side under the unsigned operator; signing beats pruning by $(1-\tilde h_u)^{-1/2}$ | B7, C2 | Table of per-dataset facts; figure of gains by local heterophily |
| Theorem 4, Corollary 3 | Soft signs are optimal; the learned $\eta$ is the reliability slope of the scores | B2, C4, D3 | Scatter of learned $\eta$ against the slope; oracle rows in the sign / reweight / prune table |
| Remark 2 | The signed operator has a contrast pathway and a bias pathway | D2 | Attribution table |
| Proposition 2, Corollary 4 | The E1 target estimates whether an edge touches an anomaly; the rule helps anomalies with $\tilde h_u>\frac12$ | B3, B4, B7, D4 | Edge-type breakdown; target comparison |
| Proposition 3 and the rule in Section 4.1 | Scores of a node's edges do not use its label, up to cross-fold labeled pairs | E0.3, B5, D5 | Appendix table; the text must match the code that is run |
| Theorem 5 | Degree-free stability in $\eta\,\varepsilon_{\mathrm{loc}}$ | C5 | Robustness figure redrawn with $\eta\,\varepsilon_{\mathrm{loc}}$ on the axis; $\gamma$ reported |
| Theorem 6 | The Brier score bounds the distance to the oracle operator | C7 | Appendix figure |
| Theorem 7, Corollary 5 | The representation and the node alignment are stable in the local error | C5, C8 | Appendix |
| Table 7 (appendix) | Expected numbers of doubly- and singly-labeled edges at 100 labels | B7 | Replace the expected counts by the measured ones under the GADBench splits |

Two sentences in the paper depend on results and are still the WSDM wording: "best on six of seven" in Section 6.2 (wait for A2) and "the KL loss stabilizes training" in Section 4.2 (wait for D7).

## Theory-first track

These are the experiments to run before anything else. Each one either checks a statement of Section 5, tests an assumption behind it, tries to break it, measures how tight a bound is, or scouts a result that is not proved yet. The last column says what the outcome can add to the theory. Ids B, C, D, G refer to the blocks further down; C9 to C20 are new and are defined here.

"Needs" says what has to exist: *scores* means saved calibrator scores only (Phase 1, seconds per split); *encoder* means the encoder output of one trained model; *training* means extra training runs.

### 1. Do the assumptions hold on real data?

| ID | Checks | What is measured | Needs | What it can add to the theory |
| --- | --- | --- | --- | --- |
| C9 | Assumption 1 (class-centred features with independent noise) and Theorem 3(a) | On the encoder output: size of the class-independent offset against the class direction; noise variance along the class direction per class; correlation of the residuals of neighbors; predicted node error $\bar\Phi(\chi_u\lVert\mu\rVert/\sigma)$ against the realized error of the linear readout, by bins of $\chi_u$ | encoder | If predicted and realized error agree, Theorem 3 becomes a quantitative law for real graphs, not a model. If neighbor residuals are correlated, it calls for a version with correlated noise (candidate N7 in `theory.md`). The offset size says how much weight the bias pathway of Remark 2 carries |
| C10 | Assumption 2 (reliability curve, independent neighbor signs) | Reliability curve of the scores and its distance from a straight line through the origin, **overall and separately for edges of anomalies and edges of normal nodes** (Theorem 4 conditions on the node's class, so the two curves must coincide); over-dispersion of the number of heterophilic neighbors of a node against the binomial value implied by its scores | scores | A near-linear curve makes Corollary 3 apply as stated. A curved one says which recalibration family the theory should cover (candidate N5). Over-dispersion above 1 means anomalies cluster and the variance term of Theorem 4 needs a dependence factor (N7) |
| C11 | Proposition 2 on real data | Score of the calibrator trained on singly-labeled edges **without the class weight** against the predicted $\beta\iota/(\beta\iota+1-\iota)$, with $q^\star$ and $P_{aa}$ estimated from all labels in bins of the score. The class-weighted loss of the code does not estimate $\mathbb E[t\mid\boldsymbol z]$, so its scores cannot be compared with this formula | scores | Turns Proposition 2 from a statement about the target into a measured fact about the trained model |
| B5, D5 | Premise of Proposition 3 | Sharpness of the scores on training nodes' edges against test nodes' edges; learned $\eta$ with in-sample, cross-fitted and pair-excluded scores | scores, training | Shows whether leakage moves $\eta$, which is the link between Proposition 3 and Corollary 3 |

### 2. Do the identities show up as the regularities the paper claims?

| ID | Checks | What is measured | Needs | What it can add to the theory |
| --- | --- | --- | --- | --- |
| B7 | Theorem 3(b), Corollary 4 | Per dataset: edge heterophily, $\tilde h_a$, $\tilde h_n$, share of anomalies with $\tilde h_u>\frac12$, anomaly–anomaly share, measured E1 and E2 sizes | labels only | The table that makes the camouflage regime a measured property of the benchmarks |
| C1 | Theorem 2, Corollary 1 | Label frequency against $\eta$ for unsigned, pruned, reweighted, signed and oracle operators; the minimiser against the learned $\eta$ | scores | The main figure of Section 5.1. If the learned $\eta$ sits at the minimiser, the label frequency is the quantity training optimises |
| C2 | Theorem 3 | Node factor $\chi_u$ by class under the four operators, for one step and for the model's low-pass channel $\mathcal W_0$ (part (c); row norms by random probes, scores only); gain of ESO-GAD over the $\eta=0$ model by bins of $\tilde h_u$ | scores, training | Shows the gain comes from camouflaged anomalies, as part (b) predicts |
| C20 | Corollary 4 | Change in the rank of each anomaly between the $\eta=0$ model and the **$\eta=1$ model** (both in D1), and between $\eta=0$ and the default run, by bins of $\tilde h_u$, for the paper's target. The E2-only and soft-imputed targets wait for D4 | training (reuses D1; D4 later) | The corollary is stated at $\eta=1$ and predicts a sign change at $\tilde h_u=\frac12-\frac1{2\tilde d_u}$ under the paper's target and none otherwise. For $\eta\le\frac12$ the idealized operator helps every anomaly, so the default run alone cannot show the change. Amazon has about 14 anomalies below the threshold and YelpChi about 400; B7 says which datasets have enough |
| C3 | Corollary 2 | Share of label energy per Beta channel, unsigned against signed, with the learned gate | scores, training | Gives the gate a measured role, or shows it has none |
| C4 | Corollary 3 | Learned $\eta$ against the reliability slope, over datasets and splits, from starts 0.2, 0.5 and 0.8. A split counts only if the three end within 0.05 of each other. Next to it, the best fixed $\eta$ on the grid of D1 | scores, training | Correlation across the splits that pass. The stand-in slopes at 100 labels are 0.81 (Amazon) and 0.39 (YelpChi), far enough from the start 0.5 to tell a learned value from an unmoved one |
| C14 | Corollary 3, by intervention | **Dropped after DP0 (Nov 18):** the learned $\eta$ does not move, so no intervention on the scores can be read through it | none | none |
| C12 | Theorem 3(d) | Per-node change of the logit margin of **one trained model evaluated at $\eta$ and $\eta+0.1$ with all weights frozen** (no retraining, which would mix in a different training path) against the predicted $4\sqrt{\tilde d_u}\,\tilde h_u(1-\tilde h_u)(\bar q^{\mathrm{het}}_u-\bar q^{\mathrm{hom}}_u)$ | a trained model, no extra sweep | Tests whether the one-step theory predicts the trained model node by node |
| C13 | Theorem 4 | Realized deflection of one-step aggregates of the encoder output under unsigned, pruned, reweighted, signed ($\eta$ grid) and recalibrated soft-sign weights $1-2\hat p(\hat q)$; the gap of the non-negative schemes against the predicted $\sum_{p>1/2}c_v^2/\varsigma_v^2$. The deflection is pooled over the nodes of a class with rows normalized by $\sum_v|\vartheta_v|$; fix this before coding, since Theorem 4 is per node and scale-free | encoder, scores | If the recalibrated soft sign wins and the gap matches, Theorem 4 is confirmed on real representations |

### 3. What must fail if the theory is right

| ID | Prediction | Test | Needs |
| --- | --- | --- | --- |
| D9 | No gain without alignment | Shuffle the scores over edges: the label frequency is not lowered and the gain disappears | training |
| C15 | Negative lift raises the label frequency (Corollary 1) | Replace $\hat q$ by $1-\hat q$. Exact part, scores only: the label frequency rises with $\eta$ on $[0,\frac12]$. Trained part: $\eta$ moves down from its start. This is not a must-fail test of the gain: $1-\hat q$ carries the same information as $\hat q$, so the bias pathway of Remark 2 survives; D9 is the control that destroys it | scores; training |
| D1 | The operator, not the head, carries the gain | $\eta$ fixed at 0, 0.25, 0.5, 0.75, 1 inside the same code | training |
| D3 | The calibrator is the bottleneck (Theorem 1) | Oracle scores for the signed operator at fixed $\eta=1$ and for the pruned one: oracle signing should be best and far above the calibrated one. Oracle reweighting is the same operator as oracle pruning, and oracle signing with $\eta$ at its start 0.5 is oracle pruning too. If both oracle rows reach the ceiling, add rows with a share $\epsilon$ of the oracle scores flipped (the generator of G1) | training |

### 4. How tight are the bounds?

| ID | Bound | What is measured | What it decides |
| --- | --- | --- | --- |
| C5 | Theorems 5 and 7 | $\lVert\Delta L\rVert_2$ and the change of the representation against $3\delta+\delta^2$, with $\gamma$ and $\varepsilon_{\mathrm{loc}}$ of the learned operator | Whether to quote the bound as informative or as a guarantee only |
| C7 | Theorem 6 | $\frac1n\lVert L_\eta(\hat q)-L_\eta(t^\star)\rVert_F^2$ against the node-averaged Brier score | Whether "Brier controls the operator" holds with a useful constant |
| C8 | Corollary 5 | Share of nodes with a misaligned low-pass label response against the bound, with the distribution of the per-node error $\bar e_u$ | Whether Corollary 5 stays in the main text. Expect not: $\varepsilon_{\mathrm{loc}}$ is a maximum over nodes, and from the preview numbers the bound is far above 1 on both graphs (see Verification) |

### 5. Scouting results that are not proved yet

| ID | Candidate result (`theory.md`) | Experiment | Needs | Go on to prove it if |
| --- | --- | --- | --- | --- |
| C16 | N1, average-case stability | Add independent zero-mean noise of size $\varsigma$ to the scores (one draw per undirected edge, **not clipped**: most scores are near 0, so clipping turns noise into a positive shift), and separately a constant shift $b$; measure $\lVert\Delta L\rVert_2$ and the per-node row perturbation against degree, on the real graphs and on regular graphs of varying degree | scores | noise scales as $\varsigma/\sqrt{d}$ and bias as $b$. That would also explain why the paper's robustness figure, which uses independent noise, shows such small operator changes |
| C17, C6 | N2, multi-hop behaviour | $k$-hop label coherence $\tilde y^\top N^k\tilde y/\lVert\tilde y\rVert^2$ for $k\le6$ under unsigned, calibrated and oracle operators, against $(1-2\mathrm{Fr})^k$; AUPRC gain for $P\in\{1,2,3,4\}$ | scores; training for C6 | the coherence decays geometrically at a rate set by the frustration |
| C18 | N3, plug-in rate | Edge Brier score against the label budget (20 to 1000 labels), and the Frobenius distance to the operator built from a calibrator trained on all labels | scores | the distance is linear in the excess Brier score, as Theorem 6 allows |
| C19 | N6 (new), the global $\eta$ as a compromise | Per-node optimal strength $\eta^\star_u$ from Theorem 4 with the measured reliability curve; its distribution by class and by $\tilde h_u$; the single $\eta$ that maximises the average deflection against the learned one | scores, encoder | anomalies and normal nodes want clearly different strengths. That would give a proposition on what the single $\eta$ optimises and a principled case for a class- or degree-dependent strength |
| G1, G2 | Theorems 3–4, Corollary 3 under control | Camouflage block model with noisy oracle scores: phase diagram over $(\tilde h_a,\epsilon)$; recovery of a known slope from scores drawn with a designed linear reliability curve $p(q)=\eta_0q$ (not from rescaled scores, see C14) | training (small) | used as the controlled counterpart of C2, C13 and C14 |

### Two stages

1. **Stage T1, local.** Amazon and YelpChi, labeled sets drawn like GADBench's (20 anomalies and 80 normal nodes, several draws), as `theory_check.py real` already does. Everything marked *scores* or *labels only* can run now by extending that script. The *encoder* and *training* items need E0.4 (fixed $\eta$, operator variants, score transforms) and a hook that saves the encoder output. T1 steers the theory and is the local test that the rental rule requires; its numbers do not go into the paper.
2. **Stage T2, seven datasets, official splits.** Needs E0.1 and the five other datasets. The *scores* items take minutes. The training sweeps are the batches V1 to V4 in "Budget", run locally since Nov 19.

Estimated training cost of the whole track on seven datasets and 10 splits: 20 sweeps (D1: 5; the default run from starts 0.2, 0.5 and 0.8: 3; D3: pruning at three thresholds, oracle signing, oracle pruning: 5; D9: 1; C15: 1; C6: 3; C14: 2), about 10 GPU-hours at the local speed. The count was corrected on Nov 18 (next section). D2, D4 and D5 are not in it: they stay in steps 3 and 4 of the order of work, so DP1 and the target comparison of C20 close only there.

### Verification of the track (Nov 18)

The track was checked against Section 5 of the paper, the code in `src/`, the existing runs, and 18 short local training runs (`runs/plan_check/`, about 10 GPU-minutes, repository protocol with 40% labels, seed 2).

**What holds.**

- Every statement of Section 5 has an experiment, and the theorem numbers used here match `theory.tex` (Theorems 1–7, Propositions 1–3, Corollaries 1–5, Remarks 1–2).
- `theory_check.py synthetic` passes 27 of 27 again.
- The cost basis is right: a 100-epoch run takes 13–15 s plus 7–8 s for Phase 1 on Amazon and YelpChi, which gives about half an hour per sweep.
- The rental arithmetic (hours, prices, the \$0.5 floor and \$1.5 cap) is right.
- The preview numbers in this file match `runs/theory_real_*.json`.

**The main problem: the learned $\eta$ is not yet a quantity that can be read.** Final $\eta$ from three starting values (the code starts at 0.5):

| Setting | Amazon, start 0.2 / 0.5 / 0.8 | YelpChi, start 0.2 / 0.5 / 0.8 |
| --- | --- | --- |
| The code's optimiser (Adam, 0.01), 100 epochs | 0.21 / 0.52 / 0.82 | 0.36 / 0.60 / 0.83 |
| Same, 500 epochs | 0.20 / 0.54 / 0.84 | 0.60 / 0.61 / 0.78 |
| Learning rate 0.1 on $\eta$ only, 100 epochs | 0.34 / not run / 0.90 | 0.87 / not run / 0.90 |
| Learning rate 0.3 on $\eta$ only, 100 epochs | 0.20 / not run / 0.94 | 0.91 / not run / 0.92 |
| Reliability slope of these scores (E2 calibrator) | 0.50 | 0.62 |

- **Amazon:** $\eta$ ends where it starts, at any length and learning rate. Test AUPRC is 0.896 to 0.907 across the six runs with the code's optimiser, so the loss is almost flat in $\eta$. The agreement "learned 0.54, slope 0.50" reported on Nov 14 comes from the start value 0.5.
- **YelpChi:** with the code's optimiser two of three starts reach 0.60 after 500 epochs and the third is still moving toward it. That supports Corollary 3. With a larger learning rate on $\eta$ every run ends near 0.9, so the end value depends on the optimiser and a larger learning rate is not the fix.
- **Reach:** with Adam at 0.01 the logit of $\eta$ moves by about 0.01 per step, so in 100 epochs $\eta$ can travel from 0.5 to 0.27 or 0.73 at most.
- **Consequence:** C4, C14, C15 and G2 all read the learned $\eta$. Before any of them is rented, Stage T1 has to show, with 100 labels, on which datasets runs from different starts agree. Where they do not, the evidence for Corollary 3 is the fixed-$\eta$ grid of D1 (which $\eta$ is best, against the slope), not the learned value.

**All problems found, and what was changed in this file.**

| # | Item | Problem | Change |
| --- | --- | --- | --- |
| 1 | C4 | The learned $\eta$ depends on its start (table above) | Three starts; a split counts only if they agree within 0.05; best fixed $\eta$ reported next to it |
| 2 | C14, G2 | Rescaling the scores by $c$ cannot test anything: the model depends on $\eta\hat q$ only, so the optimum moves as $1/c$ whatever $\eta$ means. With $c=0.5$ the target would also be above 1, outside the sigmoid's range | C14 uses $\hat q^{\,a}$, which changes the reliability curve and is not a rescaling. G2 draws scores with a designed slope |
| 3 | V1, V2 | Two duplicate sweeps. The paper's Reweighting, $W=1-\hat q$, is the signed family at $\eta=\frac12$, already in D1. Oracle reweighting equals oracle pruning because oracle scores are 0 or 1 | Both removed |
| 4 | D3 | Oracle signing with learned $\eta$ starts at 0.5, where it is oracle pruning, and $\eta$ barely moves | Oracle signing runs at fixed $\eta=1$ |
| 5 | D3 | Pruning selects $\tau$ on validation, so it is one sweep per threshold. It was counted as one | Three thresholds assumed; the WSDM grid is not in this workspace |
| 6 | V1–V4 | The default run (learned $\eta$) was in no batch, although C2, C3, C4, C9, C12, C13 and C20 need it. D2 (needed by DP1), D4 (needed by C20) and D5 were not in the count either | Default run added to V2. D2, D4, D5 stay in steps 3 and 4, said openly |
| 7 | C12 | Two separately trained models differ by more than the operator | One model, weights frozen, evaluated at $\eta$ and $\eta+0.1$. Saves a sweep |
| 8 | C20 | Corollary 4 is stated at $\eta=1$. For $\eta\le\frac12$ the idealized operator raises every anomaly's margin (by $4\eta\tilde h_u/(2\eta+(1-2\eta)\tilde d_u)$), so no sign change can appear near the learned $\eta$. The threshold is $\frac12-\frac1{2\tilde d_u}$. Amazon has about 14 anomalies below it | Compared at $\eta=1$ from D1; power to be read from B7 |
| 9 | C10 | Theorem 4 takes moments given the node's class, so Assumption 2 needs one reliability curve for both classes. Under the paper's target, anomaly–anomaly edges score above anomaly–normal ones (0.79 against 0.30 on Amazon, 0.24 against 0.17 on YelpChi), so for an anomaly a higher score means a *more* homophilic edge | Curves measured by class of the node. Expect the assumption to fail for anomalies where they link to each other |
| 10 | C11 | The code's loss is class-weighted, so its scores do not estimate $\mathbb E[t\mid\boldsymbol z]$ and cannot match Proposition 2's formula | Use the calibrator trained without the class weight |
| 11 | C15 | Flipped scores keep all the information, so no statement of Section 5 says the gain must vanish | Reworded: exact part on the label frequency, direction of $\eta$ as the trained part. D9 is the must-fail control |
| 12 | D3 | Oracle scores reveal the class through the bias pathway ($\nu_u=1-2\tilde h_u$), so all oracle rows may reach the ceiling and hide the ordering | Flipped-oracle rows as a fallback |
| 13 | C16 | Zero-mean noise clipped to $[0,1]$ is not zero-mean when most scores are near 0. The paper's robustness figure has the same clipping | No clipping; one draw per undirected edge |
| 14 | C8, C2 | Corollary 5's bound uses the worst node's error. With the preview scores ($\bar e_u$ about 0.6 for a camouflaged anomaly), $\delta$ is about 1.2 at $\eta=1$ and the bound is about 25, against a trivial value of 1; it is larger at smaller $\eta$. This is an estimate, not a run. Separately, Theorem 3(c), the only multi-step separability statement, had no real-data item | C8 reports the distribution of $\bar e_u$; C2 also covers the low-pass channel |

Two things outside this file follow from the first problem and are for the author:

- The paper says, after Proposition 3, that "$\eta$ estimates the reliability slope of the scores", and Section 6 says the learned values "fall near the empirically favorable region". Neither is supported until the runs from different starts agree under the GADBench protocol.
- The learned $\eta$ reported for the WSDM figure may also reflect its start value. The WSDM detector starts at 0.5 by default (read on Nov 19); whether the figure used the default is for the author to confirm.

One naming hazard remains: `theory.md` uses C1–C14 for the checks of `theory_check.py`, and this file uses C1–C20 for experiments. C14 means Proposition 3 there and the intervention on $\eta$ here.

### DP0 result (Nov 18): the learned $\eta$ cannot be read, and the fixed-$\eta$ grid is flat

Run on a rented RTX 3090 with `src/dp0_eta_check.py`; results in `runs/dp0/`. Setting: Amazon and YelpChi, 10 random draws of a GADBench-style labeled set (100 training nodes with 20 anomalies, the same again for validation), the paper's E2 + E1 calibrator targets with the class-weighted loss the code had at the time, selection by validation AUPRC. These are random draws, not the official splits, and the default configuration.

| | Amazon | YelpChi |
| --- | --- | --- |
| Draws in which starts 0.2, 0.5, 0.8 end within 0.05 of each other, at epoch 100 and at epoch 500 | 0 of 10, 0 of 10 | 0 of 10, 0 of 10 |
| Mean distance of $\eta$ from its start after 500 epochs (largest) | 0.018 (0.060) | 0.028 (0.070) |
| Training loss at epoch 100 | 0.006 | 0.0001 |
| Reliability slope of the scores, mean (sd over draws) | 0.42 (0.13) | 0.63 (0.06) |
| $\eta$ that minimises the label frequency, mean | 0.63 | 0.67 |
| Test AUPRC at fixed $\eta$ = 0 / 0.25 / 0.5 / 0.75 / 1 | 0.806 / 0.787 / 0.809 / 0.817 / 0.798 | 0.238 / 0.240 / 0.239 / 0.232 / 0.228 |
| Best fixed $\eta$ against $\eta=0$: mean gain, draws won, paired Wilcoxon $p$ | +0.012 at 0.75 and 0.875, 6 and 7 of 10, 0.43 | +0.002 at 0.25, 5 of 10, not significant |
| Correlation over draws of the best fixed $\eta$ with the slope | −0.21 | 0.07 |

- **DP0 fails on both datasets.** $\eta$ stays where it starts. The reason is now measured: with 100 labels the training loss is near zero by epoch 100, so the gradient on $\eta$ vanishes.
- **The fixed-$\eta$ grid has no peak either**, so the fallback evidence for Corollary 3 is not there on these two datasets in this setup. The spread over draws (0.02 to 0.06) is larger than any difference between values of $\eta$.
- **The $\eta=0$ control equals GADBench's BWGNN row** (0.806 against 81.7 on Amazon, 0.238 against 23.7 on YelpChi), which says the pipeline is sound. **The signed operator's gain is not reproduced**: at most 1.2 points on Amazon, not significant, and none on YelpChi, where the WSDM table has 28.9 against 23.7. This is a preview of D1 and the most important outcome of the run.
- **What could explain the gap to the WSDM table**, none of it checked: the WSDM row was selected over hidden size, order, $\alpha$ and $K$ while this run uses one configuration; official splits against random draws; the class-weighted calibrator loss; Salton and Jaccard missing on YelpChi; 100 epochs against GADBench's 200 with patience.

Consequences, applied below: C14 is dropped; C4 reports the slope and the fixed-$\eta$ grid, not the learned $\eta$; V2 keeps one default run. D1 on the seven datasets with the official splits (batch V1) is now the experiment that decides whether the operator-level claim stands, and should run first.

### Why the loss is flat in $\eta$ (diagnostic, Nov 18)

Three draws per dataset, 100 labels, 100 epochs, same code as DP0; script and output in `runs/flat_eta/`. The node factor $\chi_u$ is the one-step quantity of Theorem 3(a), computed with all labels.

| | Amazon | YelpChi |
| --- | --- | --- |
| Mean $\chi_u$ of anomalies with the calibrated scores at $\eta$ = 0 / 0.25 / 0.5 / 0.75 / 1 | −9.9 / −9.9 / −7.1 / +0.9 / +2.0 | −7.4 / −7.5 / −7.4 / −6.0 / −4.3 |
| Share of anomalies still on the wrong side at $\eta=1$ | 22–25% | 73–84% |
| Mean $\chi_u$ of anomalies with oracle signs / oracle pruning | +12.8 / +4.0 | +12.0 / +5.0 |
| First-order term of Theorem 3(d), anomalies / normal nodes | −0.24 / +1.23 | −0.61 / +0.76 |
| $\bar q^{\mathrm{het}}_u-\bar q^{\mathrm{hom}}_u$, anomalies / normal nodes | −0.03 / +0.40 | −0.07 / +0.13 |
| Mean weight at $\eta=1$ on heterophilic / homophilic edges | −0.11 / +0.70 | +0.39 / +0.65 |
| Edges negative at $\eta=1$: heterophilic / anomaly–anomaly / normal–normal | 54% / 92% / 3% | 27% / 42% / 12% |
| Test AUPRC: $\eta=0$ / calibrated, $\eta=1$ | 0.823 / 0.824 | 0.236 / 0.258 |
| Test AUPRC: oracle signs at $\eta=1$ / oracle pruning | 0.990 / 0.862 | 0.990 / 0.464 |

Reading:

- **It is not the datasets.** With the true edge signs the same model reaches 0.99 on both graphs, and signing beats pruning by a wide margin on YelpChi, as Theorems 1 and 3 say. (The oracle operator also reveals the class through the bias pathway, so 0.99 is a ceiling, not a fair score.)
- **Flat near $\eta=0$ is what Theorem 3(d) predicts for these scores.** The first-order effect needs a node's heterophilic edges to score above its homophilic ones. For anomalies they do not (the difference is about zero or negative, Proposition 2: anomaly–anomaly edges score highest), and normal nodes, where the difference is positive, are already far on the right side. Below $\eta\approx\frac12$ no edge changes sign and the degree normalization cancels a near-uniform down-weighting.
- **For larger $\eta$ the operator prunes, it does not sign.** Heterophilic edges get weights near zero (Amazon) or still positive (YelpChi). Anomalies move from $\chi\approx-10$ to about $+2$ on Amazon, which is what the unsigned high-pass channel already gives ($2\tilde h_u\le2$), and stay on the wrong side on YelpChi.
- **The training loss is flat for a separate reason:** the Beta channels sum to the identity for every $\eta$, so the model always contains the feature-only network and fits 100 labels whatever the operator.
- **So the bottleneck is the calibrator at 100 labels**, in particular its inability to tell an anomaly's heterophilic edges from its homophilic ones. Deciding whether an edge is heterophilic amounts to deciding which endpoint is an anomaly, from the same 100 labels the node classifier has.

### What would remove the flat region (diagnostic, Nov 18)

Three draws per dataset, 100 labels, 100 epochs, unweighted calibrator loss; `runs/flat_eta/lift_*.json`. The node-wise lift is $\bar q^{\mathrm{het}}_u-\bar q^{\mathrm{hom}}_u$, the factor of Theorem 3(d). "Soft" imputes the unlabeled endpoint of a singly-labeled edge with a cross-fitted node posterior from the features alone; "product" uses $\hat\pi_u+\hat\pi_v-2\hat\pi_u\hat\pi_v$ with no edge model.

| | Amazon: paper / soft / product | YelpChi: paper / soft / product |
| --- | --- | --- |
| Mean node-wise lift of anomalies | −0.18 / +0.26 / +0.45 | −0.07 / −0.06 / −0.03 |
| Anomalies with positive lift | 9% / 73% / 78% | 18% / 29% / 40% |
| Mean $\chi_u$ of anomalies at $\eta=1$ | −6.2 / +0.7 / +3.8 | −6.1 / −3.3 / −3.0 |
| Test AUPRC at $\eta=0.75$ ($\eta=0$: 0.822 and 0.236) | 0.828 / 0.821 / 0.764 | 0.261 / 0.296 / 0.276 |
| Test AUPRC of the node posterior alone (features, no graph) | 0.819 | 0.285 |

- The soft target turns the lift of anomalies positive on Amazon, as Proposition 2 says it should, and not on YelpChi, where the node posterior is too weak.
- On Amazon the feature-only model equals the graph model (0.819 against 0.822), so no operator can show a gain there at 100 labels. On YelpChi the unsigned graph model is below the feature-only model (0.236 against 0.285) and signing recovers that loss (0.296 with the soft target, ahead of $\eta=0$ in all three draws) without clearly passing it.
- Three draws only. These are previews for D4, not results.

Theory additions these point to are listed in `theory.md` as N8 to N10. The whole question (learned against fixed $\eta$, and ESO-GAD against feature-only models) has its own file, `learn_eta_vs_fixed_eta.md`, with the fixes F1–F6 and the experiments X1–X9.

## Review points and where they are answered

| Review point | Reviewer | Answer |
| --- | --- | --- |
| Theory does not explain why signing improves separability | R3 | Theorems 1–4 (in the paper); verified by C1–C4 |
| Evaluate the calibrator directly: edge AUROC/AUPRC, Brier, calibration error, breakdown by edge type, how many edges turn negative | R1, R2, R3 | B1–B3 |
| "Unlabeled endpoint is normal" is a strong assumption; E2-only against E1+E2 | R1, R3 | Proposition 2, Corollary 4, Table 7 (in the paper); B4, D4 |
| Calibrator is frozen; errors only damped by $\eta$ | R1 | Corollary 3 (in the paper); C4, D6 |
| Label leakage; is cross-fitting grouped by node | R2 | Section 4.1 and Proposition 3 (in the paper); B5, D5 |
| No $\eta=0$ control; edge scores as features | R2 | D1, D2 |
| Gate ablation effect is small | R2 | D7, C3; drop the claim if it stays small |
| Novelty against H2-FDetector, FAGCN, GREET | R2 | A3, related work |
| Missing RF-Graph, XGB-Graph, SpaceGNN, PMP, NSReg | R2, R4 | A2, A3, related work |
| Baselines untuned, no significance tests, ablations on few datasets | R2 | protocol below, A4, all ablations on seven datasets |
| Large GADBench datasets omitted | R2 | A5 |
| Runtime on T-Finance | R4 | F1, F2 |
| Limited novelty over BWGNN | R4 | D8 (the operator is backbone-agnostic), Section 5 |

## Protocol

- **Setting.** GADBench semi-supervised: 100 labeled training nodes (20 anomalies), the 10 splits shipped with the datasets, AUROC / AUPRC / Rec@K. Read from the GADBench repository on Nov 18 and implemented in `src/gadbench.py`: mask columns 10–19 of each dataset file; a validation set of 100 nodes (20 anomalies); **at most 200 epochs, stopped when validation AUPRC has not improved for more than 50 epochs**; test scores at the best validation AUPRC; Adam at 0.01; class-weighted cross-entropy; seed 3407 + 10 × split; mean and population standard deviation over splits. The plan had assumed 100 epochs.
- **Two comparisons, both reported.**
  - *Default against default.* The WSDM baseline rows are GADBench's default-hyperparameter numbers (they match its Table 11 to the decimal). ESO-GAD must also be run with one fixed configuration on all datasets ($P=2$, hidden 64, $K=5$, $\alpha=0.01$).
  - *Tuned against tuned.* Same random-search budget for ESO-GAD and the strongest baselines, selected on validation AUPRC.
- **Significance.** Paired Wilcoxon signed-rank test over the 10 splits per dataset (ESO-GAD against each baseline, Holm-corrected), plus average rank over datasets with a Friedman test. Mark wins, ties and losses in the tables.
- **Seeds.** One model seed per split, fixed and logged. Split 0–9 only; no split selection.
- **Oracle labels** (all ground-truth labels) are used only in blocks B and C for analysis, never for training or selection. Say so in the paper.

## E0. Infrastructure

E0.1, E0.2 (without soft imputation) and E0.7 were done on Nov 18 on the earlier machine and never committed. **On Nov 19 E0.0, E0.1, E0.2 (now with soft imputation), E0.4, E0.5 and E0.8 were rebuilt in `wsdm-2026/BWGNN_codebase/`** (see "Rebuilt on Nov 19"); they are code-tested on a synthetic graph and not yet run on real data. E0.3 and E0.6 are open. The notes in the table are the Nov 17–18 record.

| ID | Task | Notes |
| --- | --- | --- |
| E0.0 | Re-apply to `src/` what the old folder had: the GPU path, `--seed`, `--device`, `--out`, AUPRC reporting, and `theory_check.py` | Done on Nov 17 and committed as `85e68f9`: diff identical to the old commit, 27 checks pass, the 3-epoch Amazon smoke run reproduces the Nov 14 numbers |
| E0.1 | GADBench protocol in `src/` | **Done Nov 18.** `gadbench.py` (loader, splits, seeds, scores) and `main_gadbench.py` (ESO-GAD driver with early stopping). Smoke results below |
| E0.2 | Calibrator targets: E2 only, E1+E2 modal (paper), E1+E2 soft imputation | **Paper rule done Nov 18** and now the default of `cross_fitted_edge_calibration`: targets on $\mathcal E_2\cup\mathcal E_1$, folds by node, unweighted BCE + Brier, self-loops not trained on. `--edge_target e2` and `--edge_pos_weight 1` give the old behaviour. Soft imputation is still open |
| E0.3 | Cross-fitting as Section 4.1 describes it, and the pair-excluded variant of Proposition 3 | 10 fold models for $K=5$ in the pair-excluded variant |
| E0.4 | Operator variants behind one flag: signed, reweight, prune, fixed $\eta$, $\eta\in[0,2]$, shuffled $\hat q$, oracle $\hat q$ | **Built Nov 19** in `main_gadbench.py`: `--variant`, `--fix_eta`, `--eta_max`, `--q_mode`, `--oracle_flip` |
| E0.5 | Save $\hat q$ per dataset and split | Blocks B and C read it; no retraining. **Built Nov 19**: `--save_q` |
| E0.6 | Regression check: reproduce the WSDM rows for BWGNN and ESO-GAD within one standard deviation | **Moved to the second project phase** (author, Nov 18) with the rest of the main experiments |
| E0.7 | Datasets | **Done.** On this machine the seven GADBench files are in `wsdm-2026/GADBench-master/datasets/` (untracked), with Elliptic next to them, and `dataset.py` reads them from there. T-Social is only a zip in `wsdm-2026/BWGNN_codebase/dataset/`, not extracted |
| E0.8 | Overlap statistics above 5M edges | The current driver silently zeroes Salton and Jaccard on Amazon, YelpChi and T-Finance. Compute them in chunks. **Built Nov 19**: `--exact_overlap` (off by default, so the default still matches the WSDM runs); equal to the sparse product on a test graph, not yet timed on the three large graphs |

**Smoke runs of `main_gadbench.py` (Nov 18, split 0 only, default configuration, the earlier machine's 4060 Ti; the Nov 19 smoke runs on this machine are in "Workspace on this machine").** One trial is not a result; the last column is only there to show the code runs in the right range.

| Dataset | Directed edges used | Peak GPU memory | Phase 1 | Phase 2 | Time per epoch | E2 / E1 edges | Test AUPRC, split 0 (WSDM mean of 10) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Reddit | 168,016 | 0.3 GB | 1.4 s | 0.7 s | 0.014 s | 6 / 2,500 | 3.7 (5.5) |
| Weibo | 762,947 | 8.0 GB | 1.5 s | 0.8 s | 0.016 s | 96 / 17,630 | 87.7 (86.2) |
| Tolokers | 1,049,758 | 0.4 GB | 1.4 s | 1.2 s | 0.018 s | 84 / 19,798 | 32.9 (40.4) |
| Questions | 356,001 | 2.9 GB | 1.4 s | 1.3 s | 0.021 s | 4 / 1,372 | 4.5 (9.8) |
| Amazon | 8,847,096 | 6.3 GB | 1.8 s | 13.5 s | 0.088 s | 660 / 156,308 | 81.4 (83.7) |
| YelpChi | 7,739,912 | 6.9 GB | 1.6 s | 6.3 s | 0.086 s | 50 / 37,242 | 26.4 (28.9) |
| T-Finance | 42,484,443 | 13.5 GB | 11.9 s | 45.9 s | 0.406 s | 358 / 241,262 | 56.7 (73.6) |

What the smoke runs change in this plan:

- **Cost per sweep is about 0.25 GPU-hours, not 0.5.** Ten splits of all seven datasets take about 15 minutes at these speeds if every run uses all 200 epochs (T-Finance about 10 minutes of it). The budget tables below still use 0.5 and are therefore upper bounds.
- **T-Finance does not fit a 4070.** It peaks at 13.5 GB, so it needs the 3090 (or the local 16 GB card).
- **Weibo, Tolokers and Questions are stored as directed graphs** in GADBench (each edge once; Weibo is partly reciprocal). The signed operator needs a symmetric graph, so the loader symmetrizes them, which doubles Tolokers and Questions. GADBench's own BWGNN row uses the directed files. The BWGNN re-run of A1 must use the same symmetrized graph. The WSDM runs used the directed files with self-loops, not symmetrized (read from the WSDM detector on Nov 19), so on these three datasets the weight matrix of the WSDM runs was not symmetric, while Section 5 (from Proposition 1 on) assumes it is.
- **Salton and Jaccard are still zero on Amazon, YelpChi and T-Finance** (E0.8), in these smoke runs too.
- **T-Finance, Questions and Tolokers are well below the WSDM row on split 0.** With one split, default hyperparameters and the points above, this is not a failed regression check. The check itself (E0.6, ten splits) is in the second project phase, so until then the theory-track experiments run on code whose match to the WSDM numbers is unverified; their comparisons are within this code (for example a fixed $\eta$ against $\eta=0$) and do not depend on that match.

## A. Main comparison

**Second project phase (author, Nov 18).** Nothing in this block runs during the theory-first track. More datasets and more baselines are added here when the second phase starts.

| ID | Experiment | Datasets | Priority | State |
| --- | --- | --- | --- | --- |
| A1 | ESO-GAD with one default configuration, and BWGNN re-run in the same code | 7 | P0 | not started |
| A2 | RF-Graph, XGB-Graph (also plain RF, XGBoost) from GADBench | 7 | P0 | not started; published numbers below |
| A3 | FAGCN, H2-FDetector (a multi-relation method, so heterogeneous Amazon and YelpChi only; check whether GADBench already ships it), PMP, SpaceGNN, each from its public code under this protocol | 7 | P1 | not started |
| A4 | Tuned against tuned: 10 random-search trials on splits 0–2, then 10 splits, for ESO-GAD, BWGNN, GHRN, AMNet, ConsisGAD and the two tree models | 7 | P1 | not started |
| A5 | Large graphs: DGraph-Fin (3.7M nodes, 4.3M edges) and Elliptic. **T-Social (5.8M nodes, 73M edges) is skipped for now** with all large-VRAM tests (author, Nov 17) | +1 or 2 | P2 | deferred until after the theory-first track |

For reference, AUPRC of the tree baselines in GADBench's semi-supervised table (default hyperparameters) against the WSDM ESO-GAD row. The GADBench numbers come from an automated extraction of the paper; its BWGNN and GCN rows match our table exactly, but check them against the PDF before quoting.

| | Amazon | YelpChi | T-Finance | Tolokers | Reddit | Weibo | Questions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RF-Graph | 70.7 | 23.6 | **81.1** | 35.8 | 4.5 | 73.8 | 10.1 |
| XGB-Graph | **84.4** | 24.8 | 78.3 | 34.1 | 4.1 | 75.9 | 7.7 |
| RF | 81.9 | 27.4 | 74.9 | 33.1 | 4.3 | 61.6 | **11.3** |
| BWGNN | 81.7 | 23.7 | 60.9 | 35.3 | 4.2 | 80.6 | 6.5 |
| ESO-GAD (WSDM) | 83.7 | **28.9** | 73.6 | **40.4** | **5.5** | **86.2** | 9.8 |

Notes on A5:

- T-Social does not fit the preferred GPUs: full-batch training with hidden size 64 keeps roughly ten $5.8\mathrm M\times64$ activations for the backward pass (about 15 GB) on top of a 146M-edge graph. It is out of scope until large-VRAM tests are allowed again.
- DGraph-Fin needs about 10 GB for the same activations, so it is a 3090 job (24 GB), not a 4070 one (12 GB), and only after a local memory test.
- In GADBench all models score about 2% AUPRC on DGraph-Fin, and tree models are far ahead of GNNs on T-Social (51.3 against 15.9 for BWGNN) and Elliptic (about 80 against 43).
- With 100 labels the calibrator sees about 230 singly-labeled edges on DGraph-Fin and Elliptic and about 2,500 on T-Social.

## B. Direct evaluation of the calibrator

All on saved $\hat q$, evaluated on edges with both endpoints outside the labeled set, for seven datasets and 10 splits. Cost is Phase 1 only.

| ID | Analysis | Output | Priority | State |
| --- | --- | --- | --- | --- |
| B1 | Edge AUROC, AUPRC, Brier, expected calibration error against true heterophily | Table, one row per dataset | P0 | not started |
| B2 | Reliability diagram and reliability slope $\eta_0$ | Figure (appendix), slope used in C4 | P0 | not started |
| B3 | Breakdown by edge type: mean score and error on normal–normal, anomaly–normal, anomaly–anomaly edges; share of edges with negative weight at the learned $\eta$; flip precision and recall | Table | P0 | preview on Amazon and YelpChi with a stand-in calibrator |
| B4 | Target comparison: E2 only (where E2 has enough edges: Amazon, T-Finance, Weibo, Tolokers), E1+E2 modal, E1+E2 soft imputation, and E1+E2 with true labels as the ceiling | Table | P0 | preview on Amazon and YelpChi, without soft imputation |
| B5 | Leakage test: edge AUROC and Brier on training nodes' edges against test nodes' edges, for in-sample scores, the K-fold rule of Section 4.1, and pair-excluded folds | Table (appendix) | P1 | not started |
| B6 | Descriptor ablation: features only, structure only, both | Table (appendix) | P1 | not started |
| B7 | Per-dataset facts the theory needs: edge heterophily, $\tilde h_a$, $\tilde h_n$, share of anomalies with $\tilde h_u>\frac12$, anomaly–anomaly share, measured E1 and E2 sizes | Table | P0 | Amazon and YelpChi done; expected E1 and E2 sizes for all seven are in the paper (Table 7) |

The preview numbers are in `theory.md` ("Numerical verification") and `runs/theory_real_*.json`. Prediction from Corollary 4 to check across datasets: the gain of the paper's E1 rule over the unsigned backbone grows with $\tilde h_a$ and turns negative where anomalies are mostly linked to each other.

## C. Verifying the theory on real data

| ID | Prediction | Test | Priority | State |
| --- | --- | --- | --- | --- |
| C1 | Theorem 2, Corollary 1: signing lowers the label frequency in proportion to calibrator quality | Label frequency for unsigned, prune, reweight, signed at the learned $\eta$, and oracle; curve against $\eta\in[0,2]$ with the learned $\eta$ marked. Main-text figure | P0 | preview on 2 datasets |
| C2 | Theorem 3: gains come from camouflaged anomalies | Node factor $\chi_u$ by class for the four operators; test recall and AUPRC of anomalies binned by local heterophily $\tilde h_u$, ESO-GAD against BWGNN | P0 | preview on 2 datasets (oracle operators only) |
| C3 | Corollary 2: label energy moves from high-pass to low-pass channels | Share of label energy per Beta channel, unsigned against signed; compare with the learned gate | P1 | not started |
| C4 | Corollary 3: the learned $\eta$ tracks the reliability slope | Scatter of learned $\eta$ against $\eta_0$ (B2) and against the label-frequency minimiser (C1), 70 points | P0 | two data points |
| C5 | Theorems 5 and 7: the new bounds are informative | Measured $\lVert L_\eta(\hat q)-L_\eta(q^{(\epsilon)})\rVert_2$ against $3\delta+\delta^2$ and against the old degree-dependent bound; report $\gamma$. Redraw the robustness figure with $\eta\,\varepsilon_{\mathrm{loc}}$ on the axis | P1 | not started |
| C6 | Open item N2 of `theory.md`: the benefit decays with the filter order when signs are noisy | AUPRC gain over BWGNN for $P\in\{1,2,3,4\}$ against edge AUROC | P2 | not started |
| C7 | Theorem 6: the Brier score bounds the distance to the oracle operator | Scatter of $\frac1n\lVert L_\eta(\hat q)-L_\eta(\boldsymbol t^\star)\rVert_F^2$ against the node-averaged local Brier score over datasets and splits, with the bound drawn | P1 | not started |
| C8 | Corollary 5: few nodes are misaligned when the local error is small | $D^\star$-weighted share of nodes whose low-pass label response has the wrong sign under the calibrated operator, against the bound and against edge AUROC | P2 | not started |

C7 and C8 need only the saved scores and one sparse pass each, so they add no measurable cost. C9 to C20 are defined in the "Theory-first track" above, where C2, C4 and C8 also carry the corrections of Nov 18. The two data points of C4 are not evidence: on Amazon the learned $\eta$ stays at its start value.

Preview, from `runs/theory_real_*.json` and the two training runs (repository protocol, 40% labels):

| | Amazon | YelpChi |
| --- | --- | --- |
| Anomalies on the wrong side under the unsigned operator | 98.3% | 94.1% |
| Mean anomaly node factor: unsigned / oracle prune / oracle sign | −9.9 / 4.0 / 12.8 | −7.4 / 5.0 / 12.0 |
| Label frequency: unsigned, then best reachable with an E2 calibrator at 40% labels | 0.102, 0.026 | 0.451, 0.234 |
| Learned $\eta$ / reliability slope / label-frequency minimiser | 0.541 / 0.496 / 0.55 | 0.606 / 0.623 / 0.71 |

## D. Attribution controls and ablations

All on the seven datasets and 10 splits, default configuration. None is started. D1–D3 are the ones Reviewer 2 asked for by name.

| ID | Control | What it shows | Priority |
| --- | --- | --- | --- |
| D1 | $\eta$ fixed at 0, 0.25, 0.5, 0.75, 1, and learned | $\eta=0$ is the unsigned backbone inside the same code, gate and head. Separates the operator from everything else. Theorem 3(d) predicts a gain at small $\eta$ wherever the calibrator has positive local lift | P0 |
| D2 | Edge scores as features: BWGNN with the neighborhood mean of $\hat q$ and the net sign ratio $\nu_u$ appended to the input. Also ESO-GAD with the bias pathway removed (encoder output centred before propagation) | Contrast pathway against bias pathway (Remark 2) | P0 |
| D3 | Signed, reweighting, pruning with the same $\hat q$, extended from four to seven datasets, plus oracle signing at $\eta=1$ and oracle pruning. Reweighting is the $\eta=\frac12$ run of D1; pruning is one sweep per threshold | Theorem 4 on real data; the oracle rows show the headroom of Theorem 1 | P0 |
| D4 | Calibrator target: E2 only, E1+E2 modal, E1+E2 soft imputation | Node-level effect of B4 (Proposition 2, method issue M1) | P0 |
| D5 | Cross-fitting: none (in-sample), the K-fold rule of Section 4.1, pair-excluded | Does leakage inflate $\eta$ and hurt the test metric (Proposition 3, method issue M2) | P1 |
| D6 | Frozen against relaxed calibrator: jointly fine-tuned in Phase 2; two-parameter Platt map in place of $\eta$; $\eta\in[0,2]$; calibrator loss with and without class weighting | Answers "errors are never corrected" (Corollary 3, method issues M3 and M4) | P1 |
| D7 | Gate: uniform, learned, learned with KL, on seven datasets, with $\alpha\in\{0,10^{-3},10^{-2},10^{-1}\}$ | If the differences stay inside one standard deviation, describe the gate as a design choice and drop the stability claim in Section 4.2 (method issue M5) | P1 |
| D8 | Backbone generality: the signed operator inside a one-layer GCN and inside XGB-Graph's neighbor aggregation (signed mean in place of mean) | The contribution is the operator, not BWGNN | P1 |
| D9 | Shuffled $\hat q$ (same distribution, alignment destroyed) | The gain needs the scores to be on the right edges | P1 |

## E. Sensitivity

| ID | Factor | Values | Datasets | Priority |
| --- | --- | --- | --- | --- |
| E1 | Label budget | 20, 50, 100, 200, 500, 1000 labels at a fixed anomaly share | Amazon, YelpChi, Tolokers, Questions | P1 |
| E2 | Filter order $P$ | 1, 2, 3, 4 | 7 | P2 (shared with C6) |
| E3 | Folds $K$, hidden sizes | $K\in\{3,5,8\}$, hidden $\in\{16,32,64\}$ | 4 | P2 |

E1 reports edge AUROC, learned $\eta$ and node AUPRC against the label budget, for ESO-GAD and BWGNN. The theory predicts the gap opens as the calibrator improves and that $\eta$ rises with it.

## F. Efficiency

| ID | Task | Priority |
| --- | --- | --- |
| F1 | Share the powers $L_\eta^jH^{(0)}$ across channels. The current `SignedBetaWaveletConv` recomputes them per channel, $(P+1)P$ sparse products per forward pass in place of $P$ | P1 |
| F2 | Runtime table with Phase 1 and Phase 2 separated, time against edge count, and peak memory, on the same GPU for all methods | P1 |

Rebuilding $W^{(\eta)}$ and $D^{(\eta)}$ costs $O(m)$, the same as one sparse product, so after F1 the overhead over BWGNN should be close to $(P+1)/P$ plus Phase 1.

## G. Synthetic

Camouflage block model with a controllable calibrator: anomalies attach to normal nodes with probability set by $\tilde h_a$, and the scores are the true indicators with independent flip noise $\epsilon$.

| ID | Experiment | Priority |
| --- | --- | --- |
| G1 | Phase diagram of AUPRC over $(\tilde h_a,\epsilon)$ for unsigned, prune, reweight and signed, with the boundary predicted by Theorems 3–4 overlaid | P2 |
| G2 | Recovery of the recalibration slope: scores drawn with a designed reliability curve $p(q)=\eta_0q$ for several $\eta_0$, learned $\eta$ against $\eta_0$ from two starts (Corollary 3). Scaling one set of scores by a factor would be an identity of the model | P2 |

## Budget

One sweep is one method on seven datasets and 10 splits. The estimates below assume about 0.5 GPU-hours per ESO-GAD sweep (from the Nov 14 runs: 0.13–0.15 s per epoch on Amazon and YelpChi, 100 epochs, T-Finance at four times Amazon). The Nov 18 smoke runs under the real protocol give about 0.25 GPU-hours per sweep (see E0), so everything below is an upper bound.

| Block | Content | GPU-hours |
| --- | --- | --- |
| E0 | Port, regression check | 2 |
| A1–A2 | Default-configuration table (tree models run on CPU) | 1 |
| A3 | Four new GNN baselines | 7 |
| A4 | Tuned against tuned | 18 |
| A5 | DGraph-Fin and Elliptic (T-Social skipped) | 3 |
| B | Calibrator evaluation, Phase 1 only | 2 |
| C | Theory verification on saved scores, C5 to C11 and C16 to C19 included | 2 |
| C, training part | C14, C15, C6 and the default run from two more starts (8 sweeps; C12 needs none, and $\eta\in[0,2]$ is in D6) | 4 |
| D | About 30 variants (pruning is one sweep per threshold) | 15 |
| E | Label budget, order, folds | 8 |
| F, G | Runtime, synthetic | 3 |
| Reserve | Reruns and failures | 20 |
| **Total** | | **85** |

The theory-first track is about 10 GPU-hours of training (part of the D and C rows) plus the score-only analyses. Used so far: less than 1, the 10 minutes of the Nov 18 probe runs included.

### Vast.ai rentals

**Changed by the author on Nov 19: experiments run on the local GPU (RTX 3090, 24 GB). Vast.ai is used only for runs that need more GPU memory than that** (T-Social, and DGraph-Fin if the local memory test fails). Rules 1 to 4 below, the batch sheet and the planned batches V1 to V4 are therefore suspended: V1 to V4 become local sweeps in the same order, and the 100 GPU-hour budget counts local hours. The text is kept as the record and for the large-memory rentals, where rule 3 (the author approves each rental) still applies.

Rules from the author (Nov 17):

1. **Local smoke test, full run on Vast.ai** (changed Nov 18). An experiment that would take more than 5 minutes on the local GPU is not run in full locally. One smoke test of it runs locally (for example one seed, one value, one dataset); this plan is then updated with what the smoke test showed (time per run, peak GPU memory, anything that changes the design or the estimate); the full run goes into a Vast.ai batch. Experiments of 5 minutes or less may run locally in full.
2. **Each rental costs between \$0.5 and \$1.5 in total** (updated Nov 17; it was \$1 to \$2). **Prefer several small batches to one large one**: size a batch near the low end of the range. A batch above \$1.5 is split; a batch below \$0.5 is merged with the next one (or run locally if it takes 5 minutes or less).
3. **Each batch is checked against this plan before launch**, with the batch sheet below, and the author gives the go-ahead.
4. **GPU: one RTX 3090 or one RTX 4070.**
5. **No large-VRAM tests for now.**

Prices on Nov 17 (single GPU, verified hosts, reliability above 0.98, on demand; they drift, so search again before each rental):

| GPU | VRAM | Offers | Cheapest | Typical cheap offer | Hours for \$0.5 to \$1.5 at the typical price |
| --- | --- | --- | --- | --- | --- |
| RTX 3090 | 24 GB | 25 | \$0.153/h | \$0.19/h | 2.6 to 7.9 |
| RTX 4070 | 12 GB | 8 | \$0.088/h | \$0.126/h | 4.0 to 11.9 |

The 4070 is cheaper per hour and the 3090 has twice the memory. Use the 4070 when the local run peaks under 11 GB of GPU memory, the 3090 otherwise. Pick hosts with at least 30 GB of system RAM: the neighborhood-overlap statistics run on the CPU. The account has \$10.00 of credit and no billing method, which is at most six rentals at the \$1.5 cap, or up to twenty at the \$0.5 floor.

Batch sheet, to fill in before every rental:

| Line | Content |
| --- | --- |
| Runs | the list of commands, with dataset, split and variant |
| Local test | smoke test passed (one seed, one value, one dataset); wall-clock time and peak GPU memory of that run, and the full-run time extrapolated from it |
| Estimated hours | sum of the extrapolated local times, times a speed factor. Use 1.0 (no faster than the local 4060 Ti) until a rental has measured the real factor; record it then |
| Overhead | about 15 minutes per rental for the environment and the data upload |
| Cost | (estimated hours + overhead) × the price of the chosen offer. Must lie between \$0.5 and \$1.5, and nearer \$0.5 when the batch can be split |
| Hard stop | wall-clock limit set to \$1.5 divided by the price; the instance is destroyed when the batch ends or the limit is reached, whichever comes first |
| Results | copied back to `runs/` before the instance is destroyed |

Planned batches for the theory-first track (Stage T2). Each batch is five sweeps: 2.5 hours of runs plus 0.25 of overhead, at the local speed.

| Batch | Content | Sweeps | Hours | 3090 at \$0.19/h | 4070 at \$0.126/h |
| --- | --- | --- | --- | --- | --- |
| V1 | D1: $\eta$ fixed at 0, 0.25, 0.5, 0.75, 1. The 0.5 run is also the Reweighting row of D3 | 5 | 2.75 | \$0.52 | \$0.35 |
| V2 | The default run (start 0.5); D3: oracle signing at $\eta=1$, oracle pruning; D1 at $\eta$ = 0.125 and 0.875 | 5 | 2.75 | \$0.52 | \$0.35 |
| V3 | D3: pruning at three thresholds; D9 (shuffled scores); C15 (flipped scores) | 5 | 2.75 | \$0.52 | \$0.35 |
| V4 | C6 (orders 1, 3, 4); D1 at $\eta$ = 0.375 and 0.625 | 5 | 2.75 | \$0.52 | \$0.35 |

The contents were corrected on Nov 18, and again after DP0 failed: the runs from other starts and C14 are gone, and the freed sweeps complete the fixed-$\eta$ grid in steps of 0.125. The hours and costs in the table still assume 0.5 GPU-hours per sweep at the local speed; with the measured 0.25 hours per sweep and the 3090's speed factor of 1.75, a five-sweep batch is about 45 minutes and \$0.2, so the batches have to be merged to reach the \$0.5 floor. That resizing is not done yet.

- **On a 3090, each batch is its own rental**: four rentals of about \$0.52, \$2.09 in total. This is the smallest split the \$0.5 floor allows, and the one the plan assumes.
- **On a 4070, a five-sweep batch is under the floor**, so the batches go in pairs: V1 + V2 and V3 + V4, two rentals of about \$0.66 (5.25 hours each), \$1.32 in total. Cheaper overall, but the batches are twice as long.
- The \$0.5 floor is a planning rule. If the rented card turns out faster than the local one, a rental may finish below it; that is fine, and the measured speed factor then resizes the next batches.
- V1 goes first: it contains the $\eta=0$ control, and its timing gives the speed factor for the other three.

**Scenario asked for on Nov 17: the theory verification needs 200 GPU-hours on a 4070.** This is 20 times the estimate above (10 hours per sweep in place of half an hour) and twice the 100-hour cap of `todo.md`.

| | Cost |
| --- | --- |
| GPU time at the typical price, \$0.126/h | \$25.20 |
| GPU time across the usable offers seen on Nov 17, \$0.115 to \$0.168/h | \$23.00 to \$33.60 |
| Overhead of 15 minutes per rental: 18 rentals at the \$1.5 cap, 45 at about \$0.6, 54 at the \$0.5 floor | \$0.57, \$1.42, \$1.70 |
| Total at the typical price | \$25.80 to \$26.90 |
| Planning figure with storage, data transfer and price drift | about \$28; up to about \$36 if the cheap offers are taken |

The \$10.00 of credit covers about 79 of the 200 hours, so the scenario needs roughly \$18 more. It is 8.3 days on one instance or about 2 days on four in parallel; only seven usable 4070 offers matched on Nov 17, so running several at once pushes the average price up.

Rental log:

| Date | Batch | GPU, price | Rented | Run time | Cost | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Nov 18 | DP0 check (Amazon and YelpChi, 10 draws each, 240 training runs) | RTX 3090, \$0.232/h with storage | 47 min | 34 min | \$0.18 | Under the \$0.5 floor, at the author's direct request. 12 minutes of setup: `data.dgl.ai` was unreachable from the instance too, so the environment was built from the explicit package list of the local `gnnenv` (conda-forge) and the two datasets came from the CARE-GNN repository. Speed factor against the local 4060 Ti: 1.75 (0.049 against 0.086 s per epoch). Peak GPU memory 6.9 GB. Instance destroyed, \$9.82 of credit left |

Later blocks get their own sheets when they come up. The tuned comparison (A4, about 18 hours) needs at least three rentals or has to be cut. The same sweeps also fit on the local GPU in about half a day at no cost, so renting buys time, not capability.

## Order of work and decision points

0. **Settle the four points in the Summary.** No GPU time. The answers fix what E0 has to build and what Section 4.1 of the paper may say.
1. **Stage T1 of the theory-first track, local.** Amazon and YelpChi, score-only items first (B7, B3, B5, C1, C10, C11, C16, C17, C18, C5, C7, C8), then E0.4 and the training items on the same two graphs. No rental. The first training item is the check that the learned $\eta$ can be read: 100 labels, starts 0.2, 0.5 and 0.8, several draws, next to a fixed-$\eta$ grid in steps of 0.125.
   - **DP0 (Nov 18): decided, failed.** Runs from different starts never agree on Amazon or YelpChi, and the fixed-$\eta$ grid is flat (see "DP0 result"). C14 is dropped, C4 reports the slope against the fixed-$\eta$ grid, and the two sentences of the paper about the learned $\eta$ need rewording. Open for the author: how to present $\eta$ (a set value, or a learned one with the start value stated).
   - **DP1.** If on some dataset the edge AUROC is near 0.6 and flipping raises the label frequency, yet node AUPRC still improves, the gain there comes through reweighting or the bias pathway. D2 tells which, and D2 is in step 3, so this decision closes there unless D2 (two sweeps) is moved forward. Reword the mechanism claim for that dataset.
   - **DP5.** For each scouting experiment (C16 to C19), decide whether the candidate result is worth proving (N1, N2, N3, N6 in `theory.md`). For C9 and C10, decide whether Theorems 3 and 4 need the refinement N7.
2. **E0.1 and the five other datasets, then Stage T2.** Score-only items locally; the training sweeps as the rental batches V1 to V4 (about \$0.52 each on a 3090), each after its batch sheet is checked and approved.
   - **DP2.** If soft imputation and pair-excluded folds improve calibration and do not hurt AUPRC, make them the default, update Section 4.1, and rerun the default sweep. Otherwise keep them as ablations; Propositions 2 and 3 already state the limitations.
   - *Paper edits this step releases:* the figure for Theorem 2, the table of per-dataset facts for Theorem 3, the scatter and the intervention for Corollary 3, measured counts in Table 7 (task T2 of `theory.md`), and any new result that DP5 selected.
3. **The second project phase starts here (author, Nov 18).** A1, A2, D2, D4 (P0 for the main table). D2 and D4 are controls, not main experiments, and can be pulled into the theory track if DP1 or C20 needs them.
   - **DP3.** With the tree baselines in the table, fix the headline claim (see Summary). If D8 shows signed aggregation helps XGB-Graph, lead with that.
   - *Paper edits this step releases:* the main table and its text in Section 6.2, the attribution table, the "six of seven" sentence (task T3 of `theory.md`).
4. **A3, A4, D5–D8, E1, F** (P1).
   - *Paper edits this step releases:* the redrawn robustness figure (task T5), the gate wording in Section 4.2 after D7 (task T3), the runtime table.
5. **A5** (DGraph-Fin, Elliptic) and the P2 items if budget remains. T-Social stays out until large-VRAM tests are allowed.

## Tables and figures for the paper

| Main text | Appendix |
| --- | --- |
| Main results with tree and signed baselines, default configuration, significance marks (A1–A3) | Tuned-against-tuned table (A4) |
| Calibrator quality and edge-type breakdown (B1, B3) | Reliability diagrams, leakage test, descriptor ablation (B2, B5, B6) |
| Label frequency against $\eta$ with learned $\eta$ marked; learned $\eta$ against reliability slope (C1, C4) | Channel energy and gate (C3, D7), bound tightness and Brier scatter (C5, C7, C8) |
| Per-dataset heterophily facts and node factors (B7, C2) | Measured E1 and E2 counts replacing Table 7 (B7) |
| Attribution table: $\eta=0$, scores as features, sign / reweight / prune with oracle rows (D1–D3) | Targets, cross-fitting, frozen against relaxed calibrator, shuffled scores (D4–D6, D9) |
| Gains by local heterophily of the anomaly (C2) | Label budget, order, runtime, large graphs, synthetic (E, F, A5, G) |
