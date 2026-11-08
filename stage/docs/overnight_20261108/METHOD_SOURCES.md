# Function-space retention after T431: primary sources and bounded next question

Read on 2026-11-08. Stage only; Phase research, models and results were not consulted. This is a literature/design note, not a registered experiment or permission to launch. No simulator, Actor forward/backward, optimizer or solver was run for this note. Existing models, failures and protocols were not changed.

## Decision supported by the sources

The next useful question is whether the **existing frozen representation and its 66 output parameters admit a single proposal that fits the seven actual recovery histories while satisfying separate old-history action constraints**. Repeating aggregate replay/KL with another weight does not answer that feasibility question. This is an application of established constrained/function-space learning ideas, not a new algorithm claim. Preserve the single Actor120 with its existing carry contract at deployment; memory, labels and constraints below are training-time objects only.

Local context: `../RESEARCH_STATUS.md` records T431 complete, audited and remotely backed up, `six_gate=False` and no eight-metre start. The latest parent-supplied diagnosis is five fixes with six lost old successes after the fixed 120-step output-only fit, despite 211 self-location and 59 own-history aggregate supports. Those numbers motivate this question; this note does not independently re-audit the native results. Sources below do not establish Stage navigation performance.

## Four primary papers actually read

1. **Lopez-Paz and Ranzato, Gradient Episodic Memory (NeurIPS 2017).** Read §3, pp. 3–4, equations (5)–(8) and Algorithm 1 in the [author paper](https://arxiv.org/pdf/1706.08840). GEM stores bounded memories by task and formulates a separate loss inequality for each previous task. Its practical projection replaces these with gradient inner-product constraints. The text explicitly assumes local linearity and representative memory; it describes the resulting old-loss increase as unlikely, not impossible. **Relevance:** retain separate old-task diagnostics/constraints instead of letting improvement on easy histories pay for damage on one critical history. **Limit:** gradient feasibility is first order; even the exact empirical memory objective does not imply retained closed-loop success or unseen-task safety.

2. **Chaudhry et al., Efficient Lifelong Learning with A-GEM (ICLR 2019).** Read §4, pp. 4–5, equations (6)–(11) in the [author paper](https://arxiv.org/pdf/1812.00420). A-GEM deliberately replaces GEM's previous-task constraints with one constraint using a reference gradient sampled from pooled memory. It protects average episodic-memory loss and reduces compute. **Relevance:** pooled retention and taskwise retention are different contracts, even with the same examples. **Limit:** an average constraint permits individual tasks to worsen. Neither taskwise means nor an averaged gradient bounds the maximum action error at a rare, decisive state. This is a reason to test a different constraint geometry, not evidence that A-GEM or replay generally fails.

3. **Farajtabar et al., Orthogonal Gradient Descent (AISTATS 2020).** Read §2–3, pp. 2–3, equations (4), (7)–(8), Algorithm 1 in the [proceedings paper](https://proceedings.mlr.press/v108/farajtabar20a/farajtabar20a.pdf). OGD distinguishes prediction Jacobians from loss gradients and projects updates orthogonal to stored prediction directions. §3 explicitly approximates the current Jacobian by an old one in a nearby parameter neighborhood, and acknowledges information omitted by keeping a subset. **Relevance:** constrain output changes directly, rather than assuming that a small old loss gradient preserves the function. **Limit:** for a changing nonlinear network this is local and memory-limited. The exact frozen-affine case below is a separate algebraic observation, not a general OGD theorem or a safety certificate.

4. **Titsias et al., Functional Regularisation for Continual Learning with Gaussian Processes (ICLR 2020).** Read §2, §2.1–2.4, equations (1), (3)–(5), in the [author paper](https://arxiv.org/pdf/1901.11356). FRCL stores inducing inputs and posterior distributions over their function values; KL terms regularize later learning in function space. It uses neural features and task-specific Bayesian linear outputs, and discusses selecting a compact memory. **Relevance:** what is retained is behavior/function information at meaningful inputs, rather than proximity of all weights. **Limit:** a finite inducing summary and soft KL do not preserve every input or every navigation trajectory. Its task-specific GP heads are not an eligible deployment replacement for the one Stage Actor. Borrowing its motivation would not constitute an FRCL reproduction.

## Exact algebra available with the current output-only restriction

The following is our derivation, conditional on verifying the actual numerical inference path before any implementation. Let `q` be all 66 unfrozen parameters and let `h_i` be the complete fixed causal input/history and recurrent carry at a recorded decision. If only the last affine layer changes, write the two pre-clipping locations as

`z_i(q) = c_i + A_i q`.

The feature vectors and fixed residual `c_i` must include the real Actor120/carry path. Freezing a representation is insufficient if some supposedly fixed feature actually depends on `q`, or if teacher carry was substituted for student carry.

Then `z_i(q0 + d) - z_i(q0) = A_i d` is **exact in real arithmetic at these fixed histories**, with a constant Jacobian. A projection satisfying `A_old d = 0` preserves those locations exactly, without a small-step Taylor assumption. This does not establish feasibility: old feature rows may span every allowed direction, leaving only `d=0`. Report rank, singular values, constraint residuals and the best attainable new-label residual; infeasibility is a meaningful outcome. A tolerance or truncated SVD changes an exact equality into an approximation and must be reported that way.

There is a narrow deterministic-trajectory implication: matching every executed action on every decision of a complete recorded successful trajectory reproduces that trajectory from its **same exact initial simulator state and history**, provided dynamics, termination and frozen recurrent transition are deterministic and numerically identical. This follows by induction through the identical actions/states, not from the four learning papers. It proves nothing about nearby starts, numerical perturbations, unrecorded decisions or novel tasks. Approximate action matching has no such exact guarantee without an additional closed-loop robustness bound.

## Executed-action boundary constraints

T431 source `stage/experiments/t431_actual_recovery/t431_learning.py` explicitly checks executed actions against `clip(locations,-1,1)`. With componentwise hard clipping, equality of raw locations is stronger than needed at saturation. For a target executed component `a`, its exact raw preimage is:

| Executed target | Allowed pre-clipping location `z` | Convex squared fitting penalty |
|---|---|---|
| `-1 < a < 1` | `z = a` | `(z-a)^2` |
| `a = +1` | `z >= 1` | `max(0,1-z)^2` |
| `a = -1` | `z <= -1` | `max(0,z+1)^2` |

These are elementary clipping identities, not an attribution to GEM/OGD/FRCL. The hinge permits a boundary label to stay on its executed-action plateau without forcing an arbitrary raw magnitude. Use the actual executed label, not an invented equality to the clipped target's raw value. Do not replace this by a tanh identity; smooth squashing has a different preimage. Keep the existing native action mapping unchanged.

## Smallest reviewable next proposal

Before any native launch, form fixed feature matrices from already retained complete histories, with training memory indexed by original case and current source-policy hash. Test one predeclared, regularized constrained least-squares/QP proposal in the existing 66-dimensional affine space: fit recovery executed actions with the boundary penalty above; impose pointwise old executed-action constraints; optionally express a predeclared nonzero tolerance as linear interval constraints. Do not pool old cases into one soft average. For saturated old actions use their half-space preimages, preserving executed actions while allowing more freedom than raw equality.

This proposal needs a separate bounded protocol: fixed data, fixed regularizer/tolerances, one solve, numerical residual gates, no rank/seed/weight sweep, independent model readback, and unchanged complete old/new native acceptance if an eligible candidate exists. No candidate passes solely on replay loss, KL or linear feasibility. Preserve the rejected proposal and infeasibility evidence. If exact full-history constraints remove all useful freedom, report that limitation of this frozen 66-parameter class; do not silently relax protection or claim that a larger network, another representation, or the physical task is impossible.

At deployment there is still one Actor120 and its existing carry; there is no memory lookup, task route, map, external planner, waypoint or runtime teacher/controller introduced by this proposal.
