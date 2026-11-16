# T588: actual-trajectory finite support audit

Read-only recomputation of12 real recorded trajectories: three tasks times old success, actual failed handoff, full-teacher success, short-teacher failure. All support arrays and model/inputSHA independently reproduced;0native/0optimizer/0gradient.

For field1/4 actual failed handoffs, compact correction branches were active for only about0.48%/0.51% of decisions and inactive at takeover; normalized minimum distances were3.229/3.189 radii. Their old successful histories had100% active coverage. This exposes limited off-trajectory support. Field2 had no active branch even on its successful record, so support loss is not a sufficient explanation of every failure. A short handback point on field1 was inside support yet later collided; pointwise support is not a safety guarantee. No causal rescue or improved navigation policy was tested.

Audit PASS;7 raw members220118B fully remote readback/memberSHA. Archive2e10f41907b5c834721f279429c63c7fa4e0aa6aaee62234c6155bea8c4440c1. All PIDs stopped.

The old preservation set covers27 of39 known-success task IDs. Missing12 full histories will be captured under separately locked T589 with no learning. Completion alone cannot solve forgetting: five of seven lost T584 successes already had protection histories. Existing histories preserve provenance and are not all current-policy rollouts.
