# T375: current observations improve offline locality, but the joint gate still fails

2026-11-05, Stage only. COMPLETE28.588349main seconds;300live+300exact shadow supervised Adam calls. Independent300discarded historical-call reconstruction PASS;8preflight discarded calls separate. Parent bytes frozen. Twenty raw files completed full remote binary/every-member SHA verification. No simulator worker spawned:0nativecommands/0physics, as the predeclared joint gate failed. These metrics are fixed-history diagnostics, not navigation results or success rates.

| Metric | T374historical | T375 | Fixed upper limit |
|---|---:|---:|---:|
| Five-label steering-location RMSE |.219129|.190643|.062566|
| Centre248prefix max executed-steer difference |.197954|.172085|.083702|
| First20centre max executed-steer difference |.090872|.019196|.014328|
| Actual14success histories max executed-steer difference |.121462|.045002|.061292|

Only the factual-history maximum criterion passes. All four diagnostics improve relative to historical T374, but correction, centre prefix and first20checks still fail. The current-observation-plus-history branch is promising for this finite-data locality comparison, not a demonstration of closed-loop performance. Information path, parameter count and corresponding first-layer initialization dimension change together; no unique causal claim about representation or capacity is warranted.

The model is one120input/trueGRUActor with a zero-initialized136→4ReLU→1residual on current steering logits,553parameters. The same14factual data, five neutral-intervention labels, losses/300steps/learning rate/schedule/gate were kept. Preflight verified exact initial parent actions/hidden/tape,14factual actions and nonzero direct-current-observation gradients after cloned updates, with no cross-row future dependence in that direct branch. Drive on identical histories remains exact. No camera, planner, waypoints, teacher/router or explicit time/distance action rule.

This fit only corrects steering at five terminal-label states, while holding the parent's other route outputs. It does not test full-route human driving supervision or learned drive corrections. A larger raw-observation branch did not solve the existing five-label/locality problem in300steps, but that is not a convergence/impossibility proof or evidence that every current-observation policy fails. Do not append steps, change loss weights or relax the failed gate.

The rejected candidate and fallback are retained. Centre/native readback/actual14native retention/old275/originalsix54/fresh16/release78 were all unexecuted/null. Six-metre stability has not advanced; no7/8m or new qualified video. Historical safeT255/T353/T365/human routes and failures remain intact.

## Closure

Independent audit PASS reconstructs final Actor/Adam and exact metrics/gate/source data. Sole driver completed audit/backup normally; allmain/driver/inhibitor processes absent. All20original raw files rechecked locally after full remote per-member readback. Archive `0456bf1756e42bce37fab683a94693b05a4ae034b9752ab4d0e9a66695fa84c0`; original source remote `0918fa3c5fdedc20d67b61371bb4bf3337df12c6`.

Global logical1,071,810,042B<1GiB; own762,092B<2.5MiB; remaining1,931,782B, physical free>=40GiB. No evidence deletion or source-data copying. [Audit](T375_AUDIT.json), [closure](T375_CLOSEOUT.json), [remote receipt](../long_distance_20261101/release_receipt_T375.json). [Next design](NEXT_AFTER_T375.md) is not registered or started.
