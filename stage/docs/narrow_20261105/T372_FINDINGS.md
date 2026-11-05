# T372: nonlinear expansion did not localize the terminal repair

2026-11-05, Stage only. COMPLETE456.463939main seconds,66allcommands,1,272,408all native steps.300live+300exact shadow supervised Adam calls; independent300discarded historical-call audit PASS. Zero new RL updates or native training collection.32saved-model native readbacks match all arrays/history/hidden/integration. No T372process remains.

| Autonomous original-start group | Frozen T369parent baseline | T372final |
|---|---:|---:|
| Fixed old20 |10success /7contact /3timeout|12success /8contact /0timeout|
| Wide1.17m/80s, five starts |3 /2 /0|0 /2 /3|
| Original narrow/56s, two centres |1 /1 /0|1 /1 /0|
| New0.70m/80s, five starts |0 /4 /1|1 /4 /0|

Initial32are explicitly historical T369/T371 baseline reuse, not32new independent measurements. One fresh centre run reproduced complete initial arrays exactly. New narrow success is only `narrow070_dev_2`,64.304s, distance.199837528m. Centre contacts40.410s, distance2.454491454m. Other new starts contact32.690/34.768/34.612s. Original right centre is retained, but original left fails. T372lost old `dev_task6_13` and all three original wide successes. Retention and required centre/dev2/dev3 recovery gates fail; candidate is research-rejected, not promoted. It does not improve on retained T365's2/5different-start development set.

Fullold275/originalsix54/fresh16/release78 were not executed/null; do not calculate their failure rates. No7/8m/global navigation or qualified six-metre video. SafeT255/T353/T365, human routes, fallback and failed candidates remain intact. The currently tested new0.70m/80s scene is distinct from original narrow/56s.

## What this test rules out and leaves open

The73parameter zero-output16→4ReLU→1steering-logit residual initially reproduced the frozen parent on all fixed histories exactly. Entire parent network remained byte-identical. Therefore this failure is not evidence of accidentally updating the GRU/drive/old body. The initial ReLU preflight failure and its deterministic training-feature bias centering were explicitly preserved before0native/0Adam; the final preflight made8discarded clone calls. No initialization/seed/width grid was run.

On fixed T370neutral successful-branch states, five-label steering RMSE falls .434449→.062566, but non-intervention steering changes with RMSE .070815. The centre's first steer changes .095323615→.135298565 at0s; positions diverge>5cm at3.2s and>10cm at5.0s. Contact40.410s occurs before the intended49.6s correction. This again demonstrates off-target preceding-route changes rather than a terminal repair being applied to an unchanged approach. Final constant logit bias is−.0028701; nonlinear contribution RMS is.0845718, so blaming only the output bias is unsupported without a separate ablation.

T371affine and T372small nonlinear repairs both fail under this data/objective/initialization. This does not establish that every nonlinear head is incapable, that more capacity alone is sufficient, that the scene is impossible, or that catastrophic forgetting is the unique cause. Do not append fit steps, widths, loss weights or seeds to the ended pilot.

## A protection-support gap, checked without fitting

The219protected full histories are the neutral branch,211older recorded routes and7T366factual routes. Parent outputs were correctly computed on them. Exact whole-history state-SHA membership shows **none of the14actual successful T369parent baseline routes is one of those219histories**. Baseline compact records preserve full-array hashes/actions but not full120state sequences of those14routes. This says neither that every individual state is absent nor that the older reference trajectories are worthless. It identifies a missing direct protection set for the actual parent successes we claim to retain.

Consequently a better next prerequisite is to reconstruct those precise14historical successful trajectories from the frozen parent, matching their existing complete-array/integration hashes, rather than blindly increasing adapter capacity. Then a separately bounded learning design can assess worst-state action deviations on the **actual** protected histories, explicitly preserve the original approach before terminal correction, and retain true native acceptance. Near-zero mean error on different histories remains inadequate; T284/T285also showed sparse first-order constraints can miss closed-loop failures. No finite-data match is a general unseen-state safety certificate.

Evidence: [locality](T372_POSTHOC_LOCALITY.json), [support membership](T372_PROTECTION_SUPPORT.json), [audit](T372_AUDIT.json), [next fixed-data recovery draft](T373_FACTUAL_PARENT_SUPPORT_DRAFT.md). Read-only posthoc analyses used0native/gradient/optimizer/parameter fits.

## Closure and backup

Original driver stopped during a TLS failure reading a Release segment; original error/log retained. One transport-only retry completed all61raw files via2RAMsegments with full binary and every-member SHA verification, archive `bb8ca6054044398f25db91243bee746930d95102c25e0f62439ae87a7a9d7ff7`. No training/native/audit restart. All61local members rechecked and all main/worker/driver/inhibitor PIDs absent. Global logical1,067,836,493B<1GiB, own1,509,395B<6.5MiB, free>=40GiB; all evidence retained. [Closure](T372_CLOSEOUT.json), [receipt](../long_distance_20261101/release_receipt_T372.json).
