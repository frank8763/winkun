# After T374: test a direct-observation correction path, not more of the ended fit

2026-11-05. Hypothesis/design only. No new code/protocol/preflight/optimizer/native/launch beyond completed T374. Stage only; no Phase material, maps, planner, waypoints or runtime teacher/router.

## Evidence and competing explanations

T371affine and T372small nonlinear edits on frozen GRU16features changed the approach before a terminal correction could be reached. T373recovered actual14parent successful histories. T374then used those histories and max-state losses but failed both correction and protection screens; it did not enter the simulator. Adding this data/objective did not suffice under one fixed300step fit. It remains possible that the feature class is restrictive, that nonconvex optimization/relative objectives are inadequate, or that learner-state correction coverage is insufficient. None is uniquely established by these runs.

The existing120causal observations include current perception, final-goal-relative information, vehicle/actuator state and executed-action history. The correction branch so far only receives16features produced by a frozen encoder/GRU. A direct information path is a concrete next hypothesis; neither added memory nor more iterations is automatically justified by these results.

## One proposed function-class change

Freeze the same T369future parent. Keep one Actor with existing120observations and true recurrent history. Feed a neural steering-logit residual both the frozen16history features and the existing normalized/clipped120current observations:136→4ReLU→1. It has553new trainable parameters (544weights+4biases+4weights+1bias), versus73previously. Fixed width4is not a width search. Zero final layer preserves initial parent function globally; deterministic fixed initialization and gradient/step-sequence contracts must still pass numerically.

This is a function-class/information-path and parameter-count change, not a matched-capacity causal proof. No new sensor, handcoded bearing/distance turn rule, time window, task identifier, action table lookup, planner, route selector or separate deployment controller is introduced. The original observation packet already has causal time/history fields; using an unconditional learned branch does not rule out timing overfit, so later unseen-start validation remains essential. A single success would not establish generic driving or novelty.

Keep factual14data, old211schedule, five intervention-label provenance, fixed optimizer budget and the T374objective/gate unchanged if this design is registered. Initialize from the immutable parent, never continue rejected T374Adam. Independently reconstruct the fit/shadow and maximum per-route executed-action deviations. The T374negative result is historical comparison, not a fresh independent control. Do not silently add current observation bypass and also change teacher labels/reward/scene/speed/loss weights.

Screen offline before a bounded centre-first native trial. Failure stops the new mechanism; no threshold relaxation, seed/width/LR grid or appended fit. If the branch meets offline locality but reaches unsupported histories during native control, inspect actual observations and prior genuine human corrections; richer features do not supply expert labels for arbitrary new states. Human recovery labels may then be the justified next data source. No new user operation is requested here.

## Admission and honest scope

Actual remaining global logical capacity after T374backup is2,693,874B. A next implementation must reserve its full worst case inside that amount, for example at most2.5MiB if code/protocol/metadata/models/conditional raw evidence demonstrably fit, and maintain>=40GiBphysical free space. Recompute at admission; the mere example is not a registered budget. Otherwise a separate verified lossless-storage workflow is needed before launch, with all unique bytes retained. Never expand1GiB or delete unbacked evidence.

No new native result has been obtained since T372; retained-model six-metre status is unchanged. Complete release gates remain unmet, original narrow/56s and new0.70m/80s must stay separate, and7/8mremain unentered. The next protocol must state exact optimizer/physics/commands/wall/disk bounds and source/model SHA, pass zero-step/gradient/rollback contracts, be cleanly committed and exactly remotely synchronized before one unique launch. This note does not itself authorize unbounded work or claim that direct observation fusion is an original algorithm or guaranteed remedy.
