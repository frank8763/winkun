# T676 primary review: recorded boundary severity before learning

2026-11-21. Stage only. Support the separately registered zero-native, zero-Actor/Value/optimizer diagnostic; no reward change or learning follows automatically. [Proposed contract](NEXT_T676_BOUNDARY_SIGNAL_SPEC.md). T675 found three varying radial-progress tasks whose eight samples all crossed the boundary; its corrected identical-failure field count is15, not16. [Evidence and correction](t675/FINDINGS.md).

For each existing24field×8 complete history define `C=max_t max(0,abs(x_t)-7,abs(y_t)-7)/7` using recorded `poses`. Applicability comes from `request.job.fullfield_start_pose`, not a task-name shortcut; nonfield8m tasks are exempt. Independently verify canonical boundary bool, first crossing, peak and exact reconstruction. Report final `next_pose` separately without changing the historical rule. This maximum excludes unrecorded native positions.

Our distinction: `B=1{C>0}` measures the recorded boundary event; `C` measures its maximum severity within an episode. Optimizing `E[C]` does not optimize worst-case trajectories or necessarily reduce `P(B=1)`: severity can decrease while crossings become more frequent. Neither statistic measures collision clearance or arrival. CPO defines constraints on expected discounted additive costs, explicitly distinguishing its theoretical bounds from practical approximations. This episodic path maximum is not automatically that CMDP objective; a running-maximum state augmentation could represent it, but T676 needs no new state/component. [CPO, §4/§5.3](https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf).

Our likelihood-ratio derivation, assuming independent same-policy episodes within each task, parameter-independent environment dynamics and regular differentiation/integrability:

```text
Y_i=-C_i; b_i=sum_(j!=i)Y_j/(n-1)
s_i=sum_t grad_theta log p_theta(a_it|H_it) at theta0
g_k=mean_i[(Y_i-b_i)s_i]; g_field=mean_(24 tasks)g_k
```

Hold observed poses, applied actions, costs and LOO baselines fixed while differentiating the policy likelihood. Do not differentiate through the environment, absolute values or maximum. Each evaluated policy reconstructs its own complete causal GRU carry from reset; stored behavior carry cannot replace that derivative. Under the assumptions, `E[s_i]=0` and independence give `E[b_i*s_i]=0`; including the same sample in its baseline instead changes this calculation. Constant observed costs yield zero LOO signal, not evidence of safety or a zero population gradient. Keep all24 tasks in the mean and preserve the registered scale; do not normalize by episode length or only varying tasks.

Clipped Gaussian applied actions have interior density and boundary probability atoms, rather than a renormalized truncated Gaussian. Preserve recorded atom flags and sigma=.05; use stable log-tail evaluation. CAPG derives the tail scores; its author implementation switches between log-CDF, log-survival and interior log-density. Neither validates our GRU implementation. [CAPG, §3/Eqs.9–10](https://proceedings.mlr.press/v80/fujita18a/fujita18a.pdf), [author source, `log_prob`](https://github.com/pfnet-research/capg/blob/master/clipped_gaussian.py).

Use fixed numeric tolerance `abs1e-8+rel1e-4`. Recompute LOO separately for first4/last4 and every omission; report zero directions, concentration, inner products and cosines. Require at least two varying boundary tasks with positive half-gradient inner products and nonzero full gradients before considering separately registered learning. This engineering gate is neither a confidence interval nor independent replication; exact reconstructions add no samples.

Evaluate only the analytical first-fresh-Adam proposal with fixed nominal387/weight10 and registered clipping/lr/epsilon; ordinary fresh ascent gives `d_j=lr*h_j/(abs(h_j)+eps)` after clipping. Include representation rounding, negative-cost surrogate change and all old44/field-mean margins. No optimizer executes; later stateful steps differ. [Adam, Algorithm1](https://arxiv.org/pdf/1412.6980). Same-batch surrogate improvement gives no new-policy occupancy or deterministic-retention guarantee. Do not repeat canceled projection or rescue thresholds/weights/seeds. Scenario-conditioned cost applicability is training/evaluation semantics, not Actor task routing. Retain original384+gain+complete407/audit acceptance for any later candidate.

Selective scope: repository instructions/status and cited local notes; fresh primary-paper sections and author source above. No raw-data audit, simulator execution, complete historical-chat review, Phase research or other-file edits.
