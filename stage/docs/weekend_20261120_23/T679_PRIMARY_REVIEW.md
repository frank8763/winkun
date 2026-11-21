# T679 primary review — episode parameter exploration

2026-11-21. Recommendation: admit one separately registered **data-only diagnostic**, conditional on calibration, resource and replay checks. No Actor optimizer, learned candidate, center promotion, amplitude/seed rescue or automatic learning follows.

## Why pivot

T678 both arms completed/audit PASS but failed the deterministic screen: executed lost dev_task6_00/02/03/26; self lost00/13/20. Each gained four IDs and reached field7/24; neither ran full407; T501 remains retained. Lost00/02/03 (executed) and00/13 (self) nevertheless had positive sampled terminal surrogates. These finite reused-data surrogates do **not** establish improved stochastic population return. They motivate checking exploration/controller mismatch, not another reference-weight grid. [Screens and fits](../../runs/t678_reference_target_pilot/), [findings](t678/FINDINGS.md).

Historical T207 already tried parameter PG: six rank3×2 coefficients, frozen/learned LEFT samples each0/64; learned center timed out, old8 each7/8. Current full11,906 adapters and carry64 policy differ, so that result neither settles this diagnostic nor permits novelty claims. [T207 evidence](../results_20261031_parameter_pg/findings.md).

## Primary evidence and limits

[Sehnke et al., PGPE, Algorithm1](https://archive.air.in.tum.de/Main/Publications/Sehnke2010a.pdf) explicitly evaluates symmetric parameter pairs and differences their returns for the mean update. Its indexed algorithm text was available; full PDF retrieval failed, so this is not a complete paper audit.

[Plappert et al., §3/AppendixC.2](https://arxiv.org/pdf/1706.01905) keep perturbed parameters fixed throughout an episode and calibrate policy distance in action space on stored states. This supports temporally consistent **conditional** controllers, not a claim that every action varies slowly. For recurrent policies consistency conditions on the complete history/carry. Their adaptive rule and layer normalization are not reproduced here. [Author Baselines implementation](https://github.com/openai/baselines/blob/master/baselines/ddpg/ddpg_learner.py) independently perturbs a copy for distance measurement and resets exploration parameters at episode boundaries.

[Salimans et al., §§2–3.2](https://arxiv.org/pdf/1703.03864) distinguish parameter-smoothed return from action-smoothed return, use antithetic samples, and discuss dimensionality costs. [Author code](https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py) evaluates plus/minus policies and differences processed returns; its rank shaping changes the objective. Here preserve raw terminal Y=safeGOAL−collision. For fixed scale s, the mean-gradient estimator would be `mean[(Y+−Y−)epsilon/(2s)]`; a common pair baseline cancels exactly. This estimates the Gaussian-smoothed objective, not deterministic-center performance or per-ID retention.

## Single-scale diagnostic

Freeze T501 and its backbone; perturb every existing adapter coordinate with **unnormalized IID Gaussian** epsilon. Use eight fixed calibration directions, disjoint from four fixed evaluation directions. On eight fixed complete stored external causal histories, compute the true network JVP of raw locations with respect to adapters. Set `s=.05/sqrt(mean(J_epsilon²))`, with the mean explicitly task-balanced, then time/action-coordinate averaged. Verify finite positive denominator and all eight actual evaluation variants' stored-history RMS in [.025,.1]; otherwise cancel without rescaling.

This is a finite Monte Carlo trace calibration, not exact expected variance. Cached frozen recurrent features are valid only after proving adapters cannot alter carry on the fixed history; otherwise unroll the candidate's carry from reset. No noise enters120 inputs. Neither JVP nor RMS differentiates simulator states, matches clipped-action distributions, or guarantees closed-loop safety. Independent calibration preserves Gaussian evaluation draws conditional on the chosen global s; normalizing individual epsilon vectors would change that distribution.

Evaluate four antithetic pairs across eight fixed tasks: old00/02/13/20 and field sectors0-offset3/1-offset1/7-offset1/7-offset2. Preserve physical initialization, goals, horizons and boundary rules. Each variant executes deterministic actions throughout. Budget64 fresh+64 exact complete episodes, all replay physics charged, hard stop≤1h after explicit worst-case reservation.

Report every ID's GOAL/contact/timeout/boundary counts, Y variance, pair differences, and fixed-history command-offset RMS/lag-one coherence. Compare T672 IID sigma.05 histories descriptively with their original4/8 denominators: reused controls are not fresh independent controls or an equal-budget superiority test; these field tasks are previously observed.

Predeclare support PASS only with nonconstant Y on≥2old and≥2field IDs plus exact replay/calibration PASS; otherwise stop. Report missing positive-success support explicitly. Four shared directions cannot certify gradient quality: our linear-Gaussian calculation gives relative gradient MSE `(d+1)/n` with d=11906,n=4. PASS establishes collection/support feasibility only; T501 and all failures remain preserved.
