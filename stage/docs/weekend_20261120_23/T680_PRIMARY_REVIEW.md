# T680 primary review — parameter-direction support

2026-11-21. Admit one separately registered **zero-native, zero-optimizer diagnostic**. T679 completed 64 fresh full episodes and 64 exact replays with independent audit PASS; T501 remains unchanged. Its collection-support PASS does not establish a usable update direction. This diagnostic is specified after observing T679 outcomes, not a blinded confirmation. [Protocol](t679/PLAN.md), [raw results](../../runs/t679_episode_parameter_exploration/summary.json).

[Sehnke et al., PGPE §2.3, Eq12](https://archive.air.cit.tum.de/2018/Main/Publications/Sehnke2010a.pdf) use symmetric parameter-return differences for a distribution-mean gradient. [Salimans et al., §§2–3.2](https://arxiv.org/pdf/1703.03864) derive the Gaussian-smoothed objective and use antithetic sampling. These justify, for standard IID epsilon, fixed scale s and raw terminal Y=safeGOAL−collision:

`g_task = mean_four_pairs[(Yplus−Yminus) epsilon/(2s)]`.

This scores the **parameter distribution**, not Gaussian action log probabilities, clipped-action densities or deterministic-center performance. Preserve raw Y amplitude; no rank shaping. The [author ES implementation](https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py) differences processed paired returns, so its ranking variant is not the raw-Y objective here. A shared pair baseline cancels algebraically.

Compute old4, field4 and joint8 with equal task weights, retaining every zero difference. Verify source/model SHA, parameter ordering, seeds and plus/minus mappings against all original full records. Independently reconstruct every raw direction and gradient, hash arrays, and reproduce the loss/return estimates. Report pair contribution norms, total norms and cosines. Delete each pair with denominator **three**; compare the fixed first-two versus last-two Gaussian blocks with denominator **two**. Shared directions couple tasks; exact replay contributes no additional independent sample. Zero-vector cosine is undefined, recorded null.

Predeclare the conjunction: at least two field tasks each have ≥2 nonzero paired differences; nonzero finite field mean; every deletion leaves a nonzero mean with cosine≥.5 to the full field mean; fixed split cosine≥.5. These are engineering requirements, not a paper-derived confidence theorem. Keep the threshold after failure; no resampling, scale rescue, learned-span normalization or best-variant promotion. [T680 protocol](t680/PLAN.md).

Support already exposes failure: field1_offset1 differences are `[-1,0,0,0]`; field7_offset1 `[-1,0,0,1]`; field0_offset3 and field7_offset2 are zero. Thus only one field task meets the two-pair requirement. Old02/20 each give `[0,2,0,0]`, old00 `[0,0,1,0]`, old13 zero. Preserve 7 collisions/1 timeout at old00, 1 safe/7 collisions each at old02/20, all collisions at old13/field0_offset3, field1’s 1 safe/7 boundary timeouts and field7_offset1’s 2 safe/6 boundary timeouts; field7_offset2 has no success.

Four directions span at most four dimensions of 11,906: do not invert them as an identifiable full gradient. Our linear-Gaussian reference relative MSE `(d+1)/4` is a mathematical illustration, not this environment’s variance estimate.

After FAIL, stop this update route. A useful next finite protocol could validate the fixed new-source-positive frozen variants on full407, reporting capability and retention tradeoffs for every variant without score selection or renaming a variant the learned center. Any further acquisition needs a distinct evidence question and bounded registration, not another seed grid or BC revival.
