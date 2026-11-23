# T701R original fixed comparison completed or explicitly missing

State: COMPLETE. Original T701 helper failure unchanged; fitted models and independent audit reused by SHA, zero new training.

| Block | Audit | Success | Core | Lost403 |
|---|---|---:|---:|---:|
| known_quant32 | PASS | 410/431 | 396 | 0 |
| new_quant32 | PASS | 45/78 | 0 | None |
| known_replay_refined108 | PASS | 399/431 | 387 | 11 |
| new_replay_refined108 | PASS | 53/78 | 0 | None |

Compare T696 historical396core/0old loss/47new, small replay historical382core/16old loss/52new. New78 is now an exposed same-map regression set, not blind or new-map generalization.
Original T701 fit metrics: quant32 added1320120B, offline403+7 owncarry commands exact; refined108 added161584B,80 full-batch gradient evaluations, budget reached rather than proven converged. See original fit/audit and DAY_T701_PRIMARY_0 for verified model evidence.
Same finite retention requirements and all failures reported; no capacity lower bound, universal BC failure, method novelty or top-journal claim. T698 still retained. New native raw stays local because transport is limited; no original model/raw deletion.
