# Camera integration timing: recommendation, not an active experiment

The user reports an onboard camera and asks whether to add it now or wait for full global LiDAR-only stability. Recommendation: first establish a stable current six-metre LiDAR baseline with retained old-task ability; then register a separate matched LiDAR-versus-LiDAR+camera comparison. Waiting for every global distance to pass is unnecessary. No camera training/input/calibration/image-recording pipeline has been authorized or launched by this recommendation. T367 continues unchanged under its locked120-causal-input protocol.

Current evidence shows immediate action sensitivity to small reset offsets, incomplete recovery-state coverage and old/wide-task regression. T366's limited input-history comparisons found distinguishable observations, but this does not establish general observability or prove sensing is sufficient. No evidence yet identifies absent vision as the primary cause of current failures. A camera cannot be assumed to fix the learning/retention problem.

Potential camera benefit should be tested when observed failures involve information outside the LiDAR sensing plane, ambiguous visible free space, or genuinely indistinguishable LiDAR histories requiring different safe decisions. At that point it can be brought forward before all longer-distance baselines are complete. Verify actual onboard viewpoint, calibration and timestamp alignment; a third-person visualization camera is not automatically a deployable robot observation. Do not add rendering/vision/storage costs to the active protocol.

Future fusion study: same original starts/vehicle/contact geometry/goals/caps, matched native training budget and explicit model-capacity accounting; measure success/contact/timeout, preservation, latency and robustness to missing/occluded sensors. Keep one causal Actor with training-only auxiliary supervision as applicable; no map/planner/waypoint/runtime teacher/router. Any image collection gets a separate strict disk budget; raw image/video volume is not silently added to the current1GiB runs limit. Fusion alone is an established design, not a journal novelty or safety guarantee.

Primary sources checked2026-11-05:

- [Learning Navigation Behaviors End-to-End with AutoRL, RA-L2019](https://research.google/pubs/learning-navigation-behaviors-end-to-end-with-autorl/): LiDAR-input point-to-point policies evaluated in simulation and on robots. This demonstrates feasibility in their setting, not proof that our current observation/architecture is sufficient or justification for their large hyperparameter search here.
- [Surmann et al.2020](https://arxiv.org/abs/2005.13857): map/planner-free proof-of-concept using2Dlaser+RGB-D/goal orientation, with richer3Dobstacle sensing. RGB-D is not interchangeable with an unspecified RGB camera, and their results do not guarantee transfer to our steering vehicle.

Keep Stage fully isolated from Phase. This document records advice and future design criteria; no change to the active experiment or release gates.
