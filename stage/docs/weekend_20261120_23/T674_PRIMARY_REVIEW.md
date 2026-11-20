# T674 primary review: one fixed displacement projection diagnostic

2026-11-21. Stage only; selective mathematical review. This note executes no simulator, scientific calculation, Actor optimizer, or learning and authorizes no launch.

## Evidence and decision

T673's represented all407 first-fresh-Adam proposal has ten negative mixed-original margins; the field-only proposal has 19. Only 44/384 original tasks have nonzero sampled outcome gradients; 340 are explicitly **sample no-signal**, not protected. Both proposals have positive field-mean directional derivatives, but nonlinear old-task decreases remain. [Direction summaries](../../runs/t673_all_task_signal_gate/RESULTS.json), [registered diagnostic](t673/PLAN.md). T673 COMPLETE/audit PASS is supplied by the supervising task; this review inspected only the requested direction summaries, not its raw computation or audit.

One independently bounded **zero-Actor-optimizer, zero-native diagnostic** is warranted: project the existing represented all407 displacement once against those 44 old-outcome halfspaces. Do not substitute field-only weighting, sweep margins/step sizes, or automatically start learning after a numerical PASS.

## Projection and dual

GEM already defines nearest Euclidean projection onto remembered-task gradient halfspaces and reduces it to a nonnegative quadratic dual. It explicitly relies on local linearity and representative memory. This is established prior art, not a new projection algorithm. [Lopez-Paz and Ranzato, §3, Eqs.7–11](https://proceedings.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf).

With maximization gradients as positively normalized rows of B, use

```text
min_d  1/2 ||d-d0||²,  subject to B d >= 0.
L(d,lambda) = 1/2 ||d-d0||² - lambda.T B d.
d = d0 + B.T lambda.
min_lambda>=0  1/2 ||B.T lambda + d0||².
```

The last objective differs from the negative dual only by a constant: this is NNLS with matrix `B.T`, target `-d0`. These signs and equivalence are our direct derivation. Positive row normalization preserves the exact feasible cone. Zero is feasible; useful field ascent is not assured. The strictly convex primal has a unique displacement even when dependent rows allow nonunique multipliers. Convex KKT conditions suffice for optimality. [Boyd and Vandenberghe, §5.5.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).

Lawson–Hanson's original NNLS source specifies nonnegative least squares, passive/zero sets, dual termination checks and an iteration-limit failure. A NumPy passive-set implementation is a numerical implementation of established methodology; a finite cap is not evidence of convergence. [Author-distributed source, NNLS routine](https://netlib.org/lawson-hanson/all). Require independent finite/rank checks, primal feasibility, lambda nonnegativity, stationarity, complementarity and duality gap; retain residuals and failure outputs. Exercise active, inactive and rank-deficient fixtures plus an independent small-case exhaustive cross-check. Do not replace failed KKT checks with a solver success flag.

## Nonlinear and interpretation gates

Check the represented after-minus-before displacement after parameter casting, not merely the float64 solution. Report removed displacement, changed direction, all 44 margins and field24 mean ascent. Evaluate the fixed nonlinear surrogate using each candidate's complete causal carry: every old44 change must be at least `-1e-12`, field24 mean must increase, and the registered actual-sigma latent-KL statistic must be ≤.01.

Binding tangent constraints can fail at every positive step: for `F(x,y)=x-y²`, the origin's direction `(0,1)` has zero gradient margin yet `F(0,t)=-t²`. This counterexample is our derivation. CPO separately acknowledges approximation/sampling errors and checks surrogate constraints after its approximate solve; its expected-cost guarantees do not transfer to this Euclidean projection. [Achiam et al., §§5–6](https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf).

Gaussian exploration gradients and finite-history latent KL certify neither deterministic `clip(mu)` success nor old-task retention. The 340 no-signal originals remain unprotected. Only genuinely changed, feasible, useful and nonlinear-passing directions justify considering a separately registered learning protocol; eventual acceptance still requires original384 per-ID preservation, an original-start gain, complete407 and independent audit.

Scope: repository instructions/status/context/progress/migration records, the two preceding reviews, T673 plan/direction summaries, GEM §3, convex KKT, original NNLS source and CPO §§5–6. No exhaustive bibliography, full proof/code reproduction, raw-trajectory audit, Phase research or complete historical-chat review.
