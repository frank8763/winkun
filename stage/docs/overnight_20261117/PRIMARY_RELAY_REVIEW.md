# Primary-source review for a bounded Stage relay

2026-11-17. Research only: no native calls, optimization, runtime writes, source synchronization, or launches. Suggestions below require a separately locked protocol and resource ledger; this note does not extend any budget.

## Evidence and scope

Retained T608 is one causal120 Actor with true GRU carry, development34/49 and field13/24. T626 gained2/lost2; T627 meanSGD gained2/lost2 and CAGrad gained0/lost1. Field4's sampled losses and first-order predictions improved while deterministic execution failed. These observations support testing distribution coverage, not another claim that gradient alignment guarantees retention. See [prior primary review](../day_20261117/t627/PRIMARY_RETENTION_REVIEW.md) and [completed findings](../day_20261117/t627/FINDINGS_AND_NEXT.md).

[T628's plan](../day_20261117/t628/PLAN.md) already compares fresh per-policy four-task sampling with fixed versus annealed exploration, common old33 function memory, deterministic sentinels, and conditional49. T628S is unfinished for this review: no incomplete outcome is used or predicted. Its eventual sealed, audited result must govern relay admission. Full-history BPTT is already correct; no new Critic is justified by the recent ranking evidence.

Read local constraints, migration/context/progress documents, current status preamble, and the three requested T627/T628 documents. New reading covered the method/algorithm and relevant limitations in DAgger, DART, CLEAR, EPOpt, Tamar's CVaR estimator, and IMPALA, plus specified author-code excerpts. This is selected primary-source reading, not an exhaustive search or complete historical-chat audit. AAAI's direct PDF endpoint returned access-denied; Tamar's arXiv PDF was accessible. No Phase material was used.

## 1. Verified recovery on candidate-induced old-task histories — first priority

**Admit when:** a completed candidate adds an original-start success but loses a previously successful old task, with drift outside old successful histories. Parent success from the original start does not establish recovery from that drifted state.

[T614 already passed three recovery cases and a new 7.6-second-prefix handover while losing the same two old original-start tasks](../day_20261117/t614/FINDINGS_AND_NEXT.md). This branch is admissible only if its histories come from those lost tasks under the actual candidate and add uncovered states; repeating T614 handover/constraint success is not a new mechanism test.

DAgger labels states visited by the learner and aggregates them, rather than only cloning expert occupancy. Its theorem assumes a suitable expert, learning assumptions, and sufficient iterations; it does not certify this nonlinear recurrent navigation policy. [Ross, Gordon & Bagnell, AISTATS2011, Algorithm3.1/§4](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf).

**Bounded adaptation:** use one predeclared completed candidate, the two known vulnerable old IDs, and two checkpoints per ID, chosen by a fixed physical-time/drift rule before continuation outcomes. At each checkpoint compare candidate continuation against frozen T608 continuation, including exact replays. Only a native safeGOAL within the original remaining clock qualifies that parent's recovery suffix as supervision. If no checkpoint qualifies, stop this mechanism: do not label failed parent outputs as corrective expertise or move checkpoints until one works.

Restore complete simulator/controller context and the actual candidate prefix. Recompute each network's own hidden state from that same causal observation/actual-action history; never transplant teacher hidden, reset time, or use an original-parent trajectory's hidden for the candidate prefix. Save continuation ownership explicitly. A successful switched trajectory proves only training-label availability. Its suffix is supervised data, not on-policy student PG.

Use one small, fixed supervised update tranche on verified corrections plus unchanged old-memory/new-success anchors. Each network processes the real mixed-prefix recovery history under its own parameters. Finish with teacher-free original-start candidate tests; no deployed switch, router, map, waypoint, or teacher. The learned student must itself reach any useful recovery region. Successful labels and lower imitation error cannot replace complete49 acceptance.

**Discriminating measurements:** same-state candidate/parent success table; recovery-label availability; prefix and suffix hashes; action errors on old-parent versus candidate-induced histories; autonomous recovered-old and retained-new IDs. Distinguish “parent cannot recover,” “student cannot reproduce recovery,” and “recovery works but original-start trajectory still fails.” This separates label validity from occupancy transfer.

DART offers complementary evidence: perturbed supervisor demonstrations expose corrections near the supervisor's occupancy, and noise is calibrated to learner error. It is different from obtaining labels at the actual candidate's states. Its appendix also finds appropriately scaled random covariances can compete, limiting claims about sophisticated covariance fitting. Do not turn this into another sigma search or import a different supervisor. [Laskey et al., CoRL2017, §4/Appendix4.4](https://proceedings.mlr.press/v78/laskey17a/laskey17a.pdf).

## 2. Broader task-balanced fresh RL — second priority

**Admit when:** four-task sentinels pass but final49 loses other old IDs, recovery labels are unavailable, or deterioration spans tasks omitted from fresh PG. T628 already refreshes its four tasks: this branch must change old-task coverage, not rename that intervention.

CLEAR explicitly contrasts simultaneous all-task experience against sequential training and combines replay RL with behavioral cloning. IMPALA demonstrates shared-parameter multitask learning with V-trace correction for actor lag. These support the coverage hypothesis, not guaranteed compatibility between this vehicle's tasks. [Rolnick et al., NeurIPS2019, §4.1–4.2](https://proceedings.neurips.cc/paper/2019/file/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Paper.pdf), [Espeholt et al., ICML2018](https://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf).

**Bounded adaptation:** from one declared retained parent, include all34 successful IDs plus the two declared new IDs. Two fresh complete episodes per task permit the existing peer-return baseline: 72fresh+72exact per round, at most two rounds and two updates per round. Lock half the objective mass to old tasks, shared equally across34, and half to the two new tasks, shared equally. Average within task before combining, so long episodes cannot silently dominate. Keep reward, architecture, exploration, regularizer, and optimizer fixed within this branch; export/resample the actual accepted policy each round. This is a proposed stratified objective, not an IMPALA/CLEAR reproduction.

If that worst-case cost cannot fit the relay, do not silently replace all34 with convenient sampled old IDs and claim broad coverage. A smaller fixed subset is a different, explicitly partial hypothesis. Fresh old RL supplies task-return information absent from a zero-initial-gradient function penalty, but old13's prior PG participation already disproves automatic protection.

**Measurements:** per-task sample counts, successes/contact/timeouts, return spread, weighted gradient contribution, action saturation, actual displacement, and stochastic-versus-deterministic outcome gaps. Final old-ID preservation tests whether broader coverage helps; improved mean reward alone does not. Uniform sampling and equal task weights do not eliminate gradient conflict.

## 3. Explicit neighborhood robustness — conditional third mechanism

**Admit when:** original starts improve yet a predeclared lawful small neighborhood shows brittle closed-loop failures. Without that signature, robustness training spends scarce interaction on an unestablished explanation.

EPOpt samples model instances and emphasizes low-return trajectories; its appendix reports unstable exploration when aggressive tail selection starts too early and worse results without appropriate baselines. Its locomotion/model-ensemble results do not establish navigation retention. [Rajeswaran et al., ICLR2017, arXiv1610.01283v4, Algorithm1/AppendixA.5–A.7](https://arxiv.org/html/1610.01283v4).

**Practical overnight limit:** one fixed, already-valid Stage start-perturbation set around the two old/two new conflicts; no new dynamics, sensor corruption, or perturbation search. Train fresh trajectories with fixed stratum weights and impose an empirical requirement that every protected original-start and neighborhood case passes. Lock the finite set and one update tranche before observing outcomes. This changes the trained occupancy support and acceptance condition; exact replays are fidelity checks, not independent robustness samples. Report it as a finite-neighborhood pilot, not full EPOpt or distributional certification. Keep complete49 as a separate final gate.

**Do not improvise CVaR:** Tamar's likelihood-ratio estimator subtracts the quantile threshold and analyzes continuity/smoothness assumptions. Applying ordinary PG only to the worst samples can give the wrong direction. Tied timeout/terminal returns and tiny batches need an explicit atom/tie treatment; the existing peer RTG cannot simply be reused after tail filtering. A true CVaR branch therefore remains deferred unless its estimator and sampling adequacy are independently validated. [Tamar, Glassner & Mannor, AAAI2015, Eq6/AppendixD–E](https://arxiv.org/pdf/1404.3862).

**Measurements:** per-perturbation safeGOAL, worst-stratum return/contact rate, original-start gains/losses, paired parent differences, and sampled history coverage. Small action KL on nominal histories cannot establish trajectory robustness. Reject if robustness costs erase the new capability or any protected original success.

## Implementation provenance and relay decision

- DAgger: algorithm-level adaptation; no original author-code version/license verified. DART author's [repository](https://github.com/BerkeleyAutomation/DART/tree/c6b2a860f393d274da20c0df8be06154b99070f8), commit `c6b2a860f393d274da20c0df8be06154b99070f8`: README inspected; GitHub license endpoint returned404. No code copied.
- IMPALA author's [experiment.py](https://github.com/google-deepmind/scalable_agent/blob/6c0c8a701990fab9053fb338ede9c915c18fa2b1/experiment.py), commit `6c0c8a701990fab9053fb338ede9c915c18fa2b1`, Apache2.0: assigns levels round-robin; this is not equal episode exposure when episode lengths differ. No distributed actor-critic port proposed. CLEAR source-code provenance was not independently verified.
- EPOpt author's [algos.py](https://github.com/aravindr93/robustRL/blob/729e1ada566bbdb243b76fdfcaa212950bdc813b/robustRL/algos.py), commit `729e1ada566bbdb243b76fdfcaa212950bdc813b`: percentile selection inspected; no license detected/API404. This snapshot fits the baseline on `paths`, whereas paper§3.1 describes fitting the selected subset. Do not assume repository head exactly reproduces the final paper. Tamar author-code/version/license not verified.

Preprogram admission conditions and at most three distinct mechanisms, never seed/lr/noise grids. Start each independent hypothesis from its declared parent; chaining successful branches changes attribution and must be called an engineering relay. Require old34 IDs all preserved and ≥1 new original-start success before retaining a research model; 49 repeated development cases establish no generalization. Stop early on absent signal, missing recovery labels, failed acceptance, network/capacity failure, or insufficient audit reserve. The Nov18 09:00 endpoint includes verification and reporting; it is not permission to launch a stage that cannot finish. Preserve every rejected candidate and failure. This research changes no live protocol.
