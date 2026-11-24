# Stage self-audit and next decision — 2026-11-24

Planning only: no experiment, model forward, fitting, native integration or old-job restart was performed in this review. T698 remains retained.

## Corrections and evidence

B0 is not a frozen-hidden/readout-only model: T699 trained its complete residual MLP and T701 `refine` optimizes all `head.parameters()` (20,198 parameters). T705/T709 fix that hidden representation and fit the final readout. T710 was additional whole-head optimization/coverage from B0; its deadline failure supplies no completed new native comparison. Earlier chat wording suggesting the first complete-head training was still absent was inaccurate.

T699 already evaluated widths108/434/1736 and a second1736 initialization. All four known suites have audit PASS. Increasing width in those finite schedules did not monotonically improve retention. This does not prove sufficient optimization or a capacity lower bound. See `../overnight_20261122_23/MONDAY_REVIEW_20261123.md` and the actual `t699_unexecuted_continuation/known_replay_*/summary.json` and audit files; original per-ID records control any count discrepancy in prose.

T711 audited fresh96: parent32, compact32, B039, constrained37, inducedcoverage38. Old403 losses:0,0,11,12,17 respectively. Root recomputed fresh paired transitions from saved case files: constrained versus B0 gains1/losses3; coverage versus B0 gains1/losses2. Five collision counts are each40, but only36 IDs collide under all five models. Equal totals are not identical failures. These starts are now exposed same-map regression data.

T703 already provides useful finite compatibility compression:157684 added tensor bytes,396core,old403loss0,source7safe7. It is an important reference, not evidence of independent-map transfer or algorithm novelty. The current constraints/induced-label additions have not earned priority over ordinary replay.

T710 performed320 completed live gradients,160 nominal and124 completed induced shadow gradients (one additional induced shadow BEFORE call was charged), then hit its deadline. Export followed both live/shadow arms; no new audited candidate was retained. New protocols need measured fit+rebuild+audit admission, per-arm provisional export, and explicit final acceptance. No old consumed work may be replayed under a supposed engineering continuation.

## Decision

Do not launch another same-head extended fit, simple width sweep, tighter old-state constraint or unverified parent-output replay as an assumed breakthrough. A stronger optimization/capacity baseline remains required for a paper, but repeated finite fits do not resolve whether supervision actually contains recovery competence.

The next finite scientific question should be: **can an already audited autonomous teacher actually recover specific states reached by the current compact learner before collision or timeout?** This is a new learner/state contract; the original T6682/4 failure remains unchanged and must not be rerun or relabeled.

1. Before launch, fix a small12–24-state development diagnostic, learner SHA, selection rule, teacher SHA, restoration semantics, continuation horizon and admission criterion. Select across multiple parent tasks; do not scan takeover times or teachers after failures. Reconstruct the teacher's own carry from the actual observation/action history, never copy learner hidden state or invent successful corrective labels. Verify physical restoration and preserve every attempted failure. No planner, waypoints or deployment teacher are admitted.
2. Use frozen retainedT698 only as a proposed training-data teacher, subject to this native qualification. If it cannot supply genuine complete safe recoveries across the declared cases, stop this route and identify autonomous successful behavior acquisition as the gap. Do not merely add its unsupported outputs to another regression objective.
3. Only if recovery qualification passes, admit a separate matched same-capacity supervision comparison: ordinary nominal replay versus verified-recovery augmentation, using the same teacher information access, initial learner, optimizer, old/source weighting policy and declared data/compute budgets. Where lengths differ, report actual decision/FLOP/collection cost instead of asserting exact equality. Measure the full old403, sources and a new prospectively frozen test set; do not tune on T711 and call it blind. Final scientific assessment includes acceptable forgetting, paired safe gains/losses, collisions, bytes and deployment latency; zero forgetting is not a prerequisite for running all evaluations.
4. A promising point then needs independently trained seeds and genuinely separate scene geometry under a verified environment contract. This diagnostic or a small same-map improvement alone does not establish paper-level superiority.

## Primary-source support and limits

[DAgger, Ross et al.2011](https://proceedings.mlr.press/v15/ross11a.html) treats the action-dependent observation distribution through iterative expert-labeled learner-distribution data. Stage fixed historical parent-output matching is not that algorithm and has no inherited closed-loop guarantee.

[DART, Laskey et al.2017](https://proceedings.mlr.press/v78/laskey17a.html) motivates corrective demonstrations collected while the supervisor is disturbed; [author repository](https://github.com/BerkeleyAutomation/DART) documents that original scope. It does not establish that the Stage teacher recovers arbitrary learner states. Recovery-data supervision is an existing idea, not a novelty claim.

Readiness, measurable control improvement and later scientific novelty are separate questions. This decision prioritizes testing the missing recovery assumption before spending another night fitting labels whose competence has not been established. No run is registered or launched by this document.
