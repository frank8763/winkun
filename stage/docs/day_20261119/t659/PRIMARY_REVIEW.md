# Frozen-continuation gate: primary review

2026-11-19. Research review only. Today's 17:00 science authorization has expired. No native simulation, optimizer/Actor/Critic update, GPU work, or scientific launch was performed. This review does not authorize one. T501 remains retained.

## Algebra: subtraction alone contributes nothing

At fixed external history/horizon, let candidate coordinates be z, prediction f(h,z), label Y(h,z), and fixed baseline b(h). With identical samples and weights,

`L_delta = sum_i w_i [(f_i-b_i)-(Y_i-b_i)]^2 = sum_i w_i (f_i-Y_i)^2 = L_value`.

This is a samplewise identity; wherever derivatives exist, every derivative of these losses agrees. Also `partial_z(f-b)=partial_z f` when h and b are fixed. Thus subtracting the same baseline on both sides supplies neither a new objective nor a better policy derivative. At candidate-dependent histories, b(h(z)) has a total derivative, although the loss identity still cancels it. Binary deterministic outcomes can be discontinuous: differentiability cannot be assumed.

A separately parameterized residual network, altered clipping/normalization, uncertainty model, or recursive loss changes the experiment and needs its own matched control. For stochastic paired labels, `Var(Y_c-Y_r)=Var(Y_c)+Var(Y_r)-2Cov(Y_c,Y_r)`; variance reduction is conditional, not automatic. Deterministic exact replays provide no sampling-variance reduction. Any contribution must demonstrate an additional estimator, coverage rule, or compute benefit.

## What the first panel can establish

Freeze T653mean and T656 with declared reference/candidate roles. At each original start or valid forced-reference prefix, distinguish three outcomes under identical scene, restored physical/sensor state, goal, remaining native steps, and terminal rules:

1. reference throughout, `Y_r`;
2. candidate's first decision/hold, then reference throughout, `Y_one`;
3. candidate throughout, `Y_c`.

Report `Y_one-Y_r`, `Y_c-Y_r`, and `Y_c-Y_one` separately. The last isolates replacement of the post-first-action controller for that case. A discrepancy falsifies equality of the one-action proxy and full-candidate outcome. It neither measures all terms of a Bellman expansion nor proves that old-Q guidance always fails. Zero discrepancy may be uninformative when all arms share the same outcome. Do not select additional prefixes after seeing these labels and call them held out.

Define “first action” as the actual registered control hold, including native termination within it. Prefix duration consumes the original horizon; it must not reset the deadline. Administrative interruption is unknown/censored. Prefixes already terminated cannot define a continuing branch.

## The performance-difference lemma survives

[TRPO v5, §2, Eqs.1–4 and §3, Theorem1](https://arxiv.org/pdf/1502.05477v5) expresses candidate performance through **old-policy advantages accumulated along candidate occupancy**. Its old-occupancy surrogate matches first-order performance locally; monotonic bounds require further conditions. Therefore an old-continuation Q is a legitimate ingredient in policy improvement. A single first-action return is not the entire accumulated identity. For this finite-horizon recurrent task, the corresponding argument uses time and complete causal history as state, with common terminal rewards. This is an adaptation, not a theorem proved here. An average-return guarantee does not certify every retained ID. KL measured only on old recorded histories cannot establish the theorem's maximum-over-states condition; deterministic clipped actors also require care before importing stochastic KL bounds.

## Causal carry is part of the intervention

Replay each actor from its declared reset through the same **externally applied** observations/actions, computing its own carry. Candidate proposed-but-unapplied actions must not enter history. In the switched arm, the reference processes the candidate action actually applied and the resulting observations; never splice candidate carry or nominal reference carry after divergence. Restore full simulator/integrator and observation-history state, not position alone. For a future differentiable policy family, prefix carry also depends on its coordinates and cannot silently be treated as constant.

Forced prefixes measure conditional recovery. They do not demonstrate that the candidate autonomously reaches that history. Original-start outcomes remain separate.

## Evidence and fairness limits

Twelve fixed tasks, known old02/26, correlated prefixes, and deterministic exact replays are a falsification set, not random IID observations. Mark known cases as selected diagnostics; “currently unmeasured” is not automatically representative or independent. Count unique task/prefix/policy units separately from replay commands. Report per-case discrepancies and label support, not population confidence intervals or calibrated safety probabilities. Insufficient positive/negative variation means inconclusive.

Future predictors need frozen thresholds and whole-task/prefix-family/candidate-policy holdouts before Actor use. Two policy endpoints cannot validate coordinate gradients, interpolation, or new directions. Explicit whole-policy-conditioned values already appear in [Bohlinger/Peters v1, §2, Algorithm1](https://arxiv.org/html/2502.11949v1); policy conditioning is not novelty.

Direct paired rollout acceptance must receive the same proposal stream, cached results, physical/command budget and original-start acceptance rules. Charge all label collection, prefixes, replays, fitting, rejected proposals and audits to the proposed method. Labels already measured cannot simultaneously count as independent test data or rollout savings. Compare retained improvements as well as false acceptance and coverage: rejecting everything is not progress. This first diagnostic panel cannot establish sample efficiency, anti-forgetting, full387/407 retention, or generalization.

Read scope: repository AGENTS, current RESEARCH_STATUS opening entries, migration/context/progress records, innovation_reset proposal/audit, and T656 findings/assessment. Primary browsing was limited to TRPO v5 §§2–3, EPVF v1 §2/Algorithm1; R2D2's OpenReview page/PDF returned browser verification and supports no claim here. No author code, complete proofs, exhaustive literature/history, or Phase evidence was read.
