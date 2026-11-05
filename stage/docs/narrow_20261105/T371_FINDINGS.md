# T371: the correction fitted, but it changed the route before the correction

Completed 2026-11-05. Stage only; all source models, human evidence and failures retained. This is a rejected research candidate, not a stable six-metre result.

| Autonomous original-start group | Initial T369 future parent | Fitted T371 |
|---|---:|---:|
| Fixed old20 |10 success /7 contact /3 timeout|11 success /9 contact /0 timeout|
| Wide 1.17m /80s, five development starts |3 /2 /0|0 /1 /4|
| Original narrow /56s, two centres |1 /1 /0|0 /2 /0|
| New 0.70m /80s, five development starts |0 /4 /1|2 /3 /0|

New narrow successes are `narrow070_dev_2` and `narrow070_dev_4`, not the centre. This 2/5=40% is a small, repeatedly used development set, not an unseen-success estimate or improvement over retained T365's different 2/5 success set. T371 lost five initial successful IDs: one old task, one original-six right centre and all three wide successes. The aggregate old count increasing does not satisfy ID retention. Keep the fitted model as a rejected research record and the initial fallback; do not promote either as a newly validated controller.

The centre changed from an80s no-contact timeout to contact at33.826s, remaining2.345599m. Identical start drive is unchanged, but steer changes from .095323615 to .158138573 at decision0. Actual positions diverge by >5cm at3.2s and >10cm at6.0s. It never reaches the49.6s repair state. Thus this experiment does not demonstrate a failed repair applied at the same terminal state; its trained steering change has already altered the preceding trajectory.

## What was learned and what changed

300live+300exact independent shadow Adam calls updated only17effective parameters of the existing affine steering row. A separate300discarded historical-Adam reconstruction passed. No new RL update or native training trajectory was collected. On the fixed successful T370 neutral branch, the five explicit intervention-label steering RMSE decreased .434449→.061474, while all other branch states changed with RMSE .065536. Drive on identical recorded histories is exactly unchanged; the frozen drive row cannot prevent drive changing when the car visits different histories.

Read-only, zero-native/gradient/parameter-fit SVD of the248prefix hidden-feature design [h,1] gives rank17, singular values26.0302…1.33635, condition19.4786, nullity0. Consequently no nonzero change to this affine steering row can preserve all248prefix logits exactly at the reported finite-precision tolerance. This is finite-data algebra, not a proof that approximate preservation cannot work or that parameter count uniquely caused collision. The current training uses soft mean preservation, not an exact prefix constraint.

A separate fixed-representation check compared all five repair states with30,971protected factual/old history states. No incompatible exact duplicate was found at1e-12 hidden-coordinate tolerance. The closest repair/protected hidden-RMS separation is .03210, between repair252 (target0) and subsequent253 (parent target.55171). This warns that a steep local change may be needed. Absence of exact duplicates does not establish sufficient capacity, reliable labels outside the intervention, or coverage of new learner states.

Evidence: [posthoc locality](T371_POSTHOC_LOCALITY.json), [linear-head rank](T371_LINEAR_HEAD_CAPACITY.json), [representation check](T371_FROZEN_REPRESENTATION.json), [primary-source assessment](T371_NEXT_METHOD_PRIMARY_SOURCES.md).

## Verification and closure

97all commands /1,866,642all native steps /648.043782main seconds; two full centre trajectories and62compact records with full-array hashes. All32final saved-model independent native repeats match every array, hidden/history and final native integration. Independent audit PASS. Repeats and historical initial-parent matches are not new independent training examples. All five process IDs are absent. Full old275/originalsix54/fresh16/release78 remain unexecuted/null, not zero-percent failures; no7/8m or new qualified six-metre video.

Original driver stopped at a TLS failure during raw backup; its error and subsequent failed transport logs remain. Archive-only curl fallback uses configured transport, passes credentials by pipe, validates signed redirect hosts, and preserves the original member/SHA checks. All90raw files completed two-RAM-segment remote binary readback and every-member SHA verification, archive `ce7cb1dbc86e67f86e4e5cc66ca88cede7301710c30f9029195ae57c8c671f09`. No native, learning or independent audit was restarted. Local90members rechecked; own1,574,762B and global1,066,327,098B at backup completion; free>=40GiB. [Closure](T371_CLOSEOUT.json), [remote receipt](../long_distance_20261101/release_receipt_T371.json).

Next is a distinct nonlinear correction-capacity draft, not another T371 fit or a launched run. It must first demonstrate exact zero-output initialization and honest source/target separation, then test autonomous retention. Neither adapters nor rehearsal is original here or a safety guarantee.
