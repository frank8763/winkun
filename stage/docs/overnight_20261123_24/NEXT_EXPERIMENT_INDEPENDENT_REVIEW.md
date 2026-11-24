# Independent review: next experiment after T710/T711

Reviewed 2026-11-24. Stage only. This note changes no experiment, protocol, model or raw result. No model forwards, fitting or native integration were run. It recommends a separately authorized bounded study, not a registration or launch.

## Recommendation

Prioritize a small **real recovery qualification**, followed only if it passes by one matched same-capacity comparison of nominal versus verified-recovery supervision. Do not automatically repeat induced-output matching, refit the same head or sweep another width.

A fixed set of roughly12–24 learner-induced states just before contact or during a stalled timeout would answer the next missing question: can the existing retained teacher actually recover those physical states? Fix the learner, teacher (for example T698), states, selection rule, remaining horizon and success gate before native continuations. Restore complete causal observation/action history and reconstruct the teacher's own carry; do not invent teacher-action history. Save failures as failures. A teacher query alone supplies an action label, not evidence of successful correction.

This is a proposal, not an executable protocol: counts, feasibility, success gate and all resource/physical accounting still require a new bounded registration. Failed T668 and consumed T710 cannot be restarted or silently extended.

## Correct the baseline lineage first

B0 already trained its entire108-hidden,20,198-parameter residual head: T699 trained it, and [T701 refine](../../experiments/t701_compact_retention/fit.py) sets the complete head trainable and passes all head parameters to LBFGS. “Unfreeze the full head” is therefore not a new baseline direction.

Moreover, the [Monday review](../overnight_20261122_23/MONDAY_REVIEW_20261123.md) already reports audited108/434/1736-width replay models and a second1736 initialization. Their old403 losses were16/35/43/98 and fresh78 successes52/41/39/54. Capacity was not monotonically beneficial under that finite training rule. Repeating a generic width sweep would not create a new scientific question. These pilots still do not establish the best possible BC result.

The existing [T701 fit summary](../../runs/t701_compact_retention/fit/summary.json) records80 gradient evaluations,38 accepted outer steps, budget exhaustion, and last evaluated gradient infinity norm0.000759 versus stopping tolerance1e-12. Evaluated trial losses fall from0.00492265 to0.00480606; these are not a certified accepted-final loss or proof of stationarity. A properly optimized baseline remains necessary for a paper, but this does not justify serial ad hoc same-head rescues.

## What the latest completed evidence says

The audited [T711 findings](t711/FINDINGS.md) give:

| Model | Fresh same-scene safe /96 | Historical old403 lost | Historical source safe /7 |
|---|---:|---:|---:|
| parent | 32 | 0 | 0 |
| compact | 32 | 0 | 7 |
| B0 ordinary replay/BC | 39 | 11 | 7 |
| constrained | 37 | 12 | 6 |
| induced coverage | 38 | 17 | 6 |

B0 has the best aggregate fresh result among these compact models and fewer historical losses than constrained or induced coverage. Thus extra retention machinery has not earned priority. Compact preserves the old set and source demonstrations while matching the parent's32 fresh successes; finite compatibility alone is not broader control competence. The96 related starts are not independent worlds, and T711 is now exposed.

All five models have40 collisions; equal counts do not establish identical collision states or a common causal mechanism. T710 supplies no missing tie-break: it [failed](t710/FINDINGS.md) in shadow construction before exported audited models or native evaluation. Its [ledger](../../runs/t710_whole_head_replay/LEDGER.json) records live320/shadow285/independent0 charged gradient calls,2,388.31 seconds, physics0/commands0; the [traceback](../../runs/t710_whole_head_replay/fit_main.log) ends in its deadline guard. That is an execution-budget failure, not a negative navigation result or capacity lower bound. Charged attempts are not all completed closures.

## Primary-source check and counterevidence

Three targeted sources were inspected; this is not an exhaustive survey.

* Ross, Gordon and Bagnell, AISTATS2011, [DAgger paper, Sections2.2–4 and algorithm](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf), aggregate expert labels on successive learner-visited distributions. Their task-cost discussion makes recovery important and explicitly permits poor worst-case cost amplification. A fixed102-history parent-output snapshot is not iterative DAgger. Stage raw-action MSE has no established task-cost guarantee.
* Laskey et al., CoRL2017, [DART paper, introduction and method](https://proceedings.mlr.press/v78/laskey17a/laskey17a.pdf), deliberately disturb a supervisor to collect corrective examples. The paper also reports computational and training-performance costs of DAgger in its domains. This supports asking whether correction demonstrations exist, not assuming noise solves Stage. Their [author repository README](https://raw.githubusercontent.com/BerkeleyAutomation/DART/master/README.md) identifies the original simulation scope and old MuJoCo dependency. Only the README was inspected for implementation; two guessed script URLs returned404, so no executable DART/DAgger reproduction is claimed.
* Buzzega et al., NeurIPS2020, [DER paper, introduction and Section3](https://proceedings.neurips.cc/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf), report strong simple replay and explain that stored predictions can be biased. The [author DER++ observe implementation](https://raw.githubusercontent.com/aimagelab/mammoth/master/models/derpp.py) combines current supervision, stored-output MSE and replayed supervised labels. This supports keeping ordinary replay competitive. Counterevidence: continual classification results do not transfer a collision-safety guarantee to Stage continuous control. Mutable master code was inspected, not copied or executed.

These are existing-method motivations, not an originality claim or instructions to combine DART, DAgger and DER.

## Why recovery qualification comes before recovery learning

[T668](../weekend_20261120_23/t668/FINDINGS.md) achieved only2/4 true teacher takeovers and failed its predeclared correction-learning admission gate. T691's induced supervision also failed its numerical gate. These are direct counterevidence to treating frozen-parent outputs on learner states as universally valid recovery targets.

A new qualification must differ substantively from retrying T668: explicitly motivate its teacher/state distribution using current evidence, freeze a finite selection rule, and prohibit takeover-time/teacher/seed searches after observing outcomes. A newer teacher may be better, but that is a hypothesis. Even successful continuations show only local recoverability, not global reachability or learner success.

If the gate passes, compare nominal versus verified-recovery supervision using the **same teacher in both arms**, same learner capacity, old-task weights and finite optimizer rules. Account for added examples, label queries, native continuations and training compute; otherwise improvement may simply reflect more data or a stronger teacher. Keep failed qualification states visible and do not turn failed continuations into successful demonstrations. Use complete old/source outcomes and predeclared genuinely unexposed validation against B0 and compact/T703; T711 is an exposed regression set. A trained student must execute autonomously, with no runtime teacher, router, planner or waypoints.

If qualification fails, stop this recovery-supervision branch. The missing ingredient is autonomous successful behavior at those states; extra parent-output labels do not supply it. T698 stays retained, and no automatic follow-on experiment is authorized by this review.
