# T653: two bounded afternoon discriminators

2026-11-19. Research only: **0 native, 0 optimizer, no launch**. Recommendation: first test missing vulnerable-task outcome coverage; then, only if informative, test one constrained update on exactly the same data. Two hypotheses are a ceiling, not an automatic relay or grid. Reserve final evaluation/audit time before the user's 17:00 Beijing stop.

## Evidence that the protocol must preserve

[T650](../t650/FINDINGS.md) repaired canonical initialization. [T651](../../../runs/t651_canonical_outcome_pg/pg/summary.json) then reached 380/407 against T501's 384, losing six original successes despite full384 replay. [T652](../t652/FINAL_REPORT.json), changing only trainable scope to 11,906 adapters, recovered four losses but still collided on **5 m LEFT `dev_task6_00` and `dev_task6_03`**; field24 was 6/24, and full407 was not run. Neither candidate qualifies. Small nominal command losses or latent KL do not certify deterministic closed-loop retention.

The existing actual old5m PG group was `dev_task6_24`, not those two resets. Its four failed stochastic episodes had zero leave-one-out advantages. Thus replaying all old functions and supplying an informative old outcome gradient are different interventions.

## H1: add actual vulnerable-task outcome information

Collect exactly **16 stochastic T501 episodes per vulnerable reset: 32 fresh plus 32 exact regenerations**. Lock canonical scene, initial120, integration, goal, behavior-policy and density hashes. Both old tasks must contain success **and** failure; otherwise perform **zero new learning** and report insufficient outcome information. All-equal binary outcomes yield zero task-LOO PG here; do not manufacture labels or keep sampling until success.

Reuse corrected T65176 explicitly: **108 behavior trajectories over 21 tasks**, not 102; exact regenerations add no independent samples. The old 76 are reused evidence, not new independent data. Exact T501 initialization makes them initial-behavior replay; repeated updates are clipped-surrogate optimization, not eight newly on-policy rounds. Compute task-LOO using each task's actual sample count, excluding its own outcome; retain episode-score scaling by fixed512.

Candidate A keeps T652's adapters, fresh Adam, 3e−5, eight attempts, terminal objective, full causal histories, full384 keep term, clipping and rollback. Predeclare task weights; adding two task means changes original task weights. The comparison with audited T652 tests specified coverage/weighting, while A versus B isolates update handling.

## H2: constrain the actual proposed displacement

Candidate B restarts exact T501/fresh Adam, with A's identical data/objective and eight-attempt budget. Project the actual Adam displacement by minimizing `||Δθ−Δθ_Adam||²` subject to two `g_old · Δθ ≤ 0` constraints. Enumerate zero/one/two active constraints with fixed pseudoinverse tolerances. [GEM §3, Eq(7–8)](https://proceedings.neurips.cc/paper_files/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf) supplies the closest-feasible-gradient principle under local linearity and representative memory; this actual-displacement adaptation is **not exact GEM**.

Test only fixed scales **1, .5, .25, .125**, accepting the first passing existing trust checks and finite old2 sample-PG-surrogate nonincrease, tolerance 1e−10. Otherwise reject the entire proposal and stop. Commit proposed Adam moments when a projected step is accepted; restore all moments and parameters on rejection. Log proposed/applied displacement, actual gradient dots, finite losses, command change and KL. Zero gradients/steps are not evidence of protected learning.

[CAGrad §3](https://proceedings.neurips.cc/paper/2021/file/9d27fdf2477ffbff837d73ef7ae23db9-Paper.pdf) optimizes local gradient tradeoffs; its theorem concerns average-loss stationarity under stated assumptions, not per-task navigation returns. [T627](../../day_20261117/t627/FINDINGS_AND_NEXT.md) already failed to translate favorable local task losses into retention. This projection is a standard constrained-update hypothesis, not renamed CAGrad/OGD novelty or a return guarantee.

## Acceptance and the deferred alternative

Use the same predetermined screen and exact checks for both final candidates; advance only qualifying candidates to canonical full407. Require **every original T501 success ID (384) plus at least one additional legacy-field safe GOAL**, with no contact or relaxed timeout. Totals, the two recovered resets, or field6 alone are insufficient; unexecuted items remain null. No extra seed, coefficient, step-count or mechanism follows failure.

[TaSIL §3 and Appendix C](https://arxiv.org/html/2205.14812v2) motivates matching feedback derivatives under explicit stability/regularity assumptions. Stage has not verified them. A later directional-response test must regenerate physically consistent sensor/history inputs and recompute recurrent carry; independent edits to correlated observation columns are not a physical state derivative. Such matching is a separate hypothesis requiring derivative-compute accounting, not an afternoon add-on when outcome gradients are absent.
