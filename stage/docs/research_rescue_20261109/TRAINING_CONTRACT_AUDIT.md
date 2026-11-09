# Stage 6 m training contract audit — 2026-11-09

Later same-day status clarification: the user explicitly reopened algorithm/network research after the T440 closeout. The stop described below is the historical execution state when audited, not a permanent prohibition on the newly authorized research. This report and the linked rescue design do not start a run or extend an old budget.

Scope: source and existing JSON/text evidence only. No model was loaded, no project/runtime module imported, no Actor forward pass, simulator, optimizer, Python process, package installation, experiment launch, Git mutation or run-output write occurred. The only new artifact is this report. Historical `phase10` filenames below are the verified Stage simulator dependencies, not Phase research. This report is research analysis, not authorization to resume: `stage/docs/RESEARCH_STATUS.md:1` records the stop after T440. T440 produced no candidate/native step/inner Adam/new online RL; its external execution-budget failure did not test method efficacy.

## Current measured facts

| Existing candidate | Wide 6 m development | Old development | Decision |
|---|---:|---:|---|
| T425 A | 54/54; 0 contact, 0 timeout | 264/275; 5 contact, 6 timeout | rejected |
| T437 proposed research policy | 52/54; 0 contact, 2 timeout | 271/275; 1 contact, 3 timeout | rejected; whole Actor/Adam restored |
| T439 proposals | full 54/275 **not executed** | full 54/275 **not executed** | all four 31-task panel proposals rejected |

T425 A counts were read directly from `.result.rounds[0].check` in the losslessly cold-encoded original `stage/runs/t425_offline_day_chain/a/summary.json.cold.gz`; `.accepted` is false. T437 report counts are at `stage/docs/review_20261109/T437_FINAL_REPORT.json:15` and `:23`; detailed evidence/limitations at `T437_FINDINGS.md:3–11`. T439 panel repaired/lost counts are 6/2, 6/1, 5/2, 6/2; 0 retained updates (`T439_FINDINGS.md:5`). Its 16 collection trajectories exactly match T437's first four rounds and its 16 exact replays add no independent samples (`T439_FINDINGS.md:7`). Do not assign T437's 52/54 measurement to T439.

T437 collected 32 exploratory episodes (11 GOAL, 19 contact, 2 timeout), performed 64 live and 64 shadow updates and retained one intermediate research round. Its final full evaluation repaired five but lost four relative to initialization; all four new losses were outside the 31-task panel (`T437_FINDINGS.md:3–7`). Numerical/density/audit success therefore coexists with failed task retention. Neither candidate meets the joint 329-task gate; fresh16/reload78/8 m remain unexecuted.

## Actual return and policy-gradient estimator

The active T437 learner imports `t432_learning` at `stage/experiments/t437_joint_closed_loop_rl/t437_learning.py:15–22`; its PPO path calls `E.L.blocks(q, items, 0, p)` at `:291`. That resolves to `stage/experiments/t392_demo_closed_loop/t392_learning.py:98–109`, not to the older optional reward-to-go path in `t353_offline_night_chain/night_core.py:104–125`.

At native timestep 0.002 s the inherited discount is

`gamma = exp(-0.002 / 60)`

(`stage/experiments/t203_native_exploration/runtime.py:12`). For decision k lasting n_k native steps, `t392_learning.py:70–74` computes the complete observed return backwards:

`G0_i = r0_i + gamma**n0_i * (r1_i + gamma**n1_i * (...))`.

At 100 native steps per hold this is a 0.2 s decision and a discount `exp(-0.2/60)`, with 60 s physical e-fold horizon. Describing the current contract as gamma=.99 at 2 ms or a 0.199 s horizon is incorrect; that concerns early historical experiments.

For the four current-policy episodes from the same focus job (`t437_joint_closed_loop_rl/run.py:143–147`), the coefficient is

`A_i = G0_i - (sum_j G0_j - G0_i) / 3`.

`A_i` is one Python float for the whole episode (`t392_learning.py:103`), broadcast to **every action** via `coefficient = advantage / (4*400)` and a sum of clipped per-decision likelihood-ratio objectives (`t437_learning.py:299–307`). It is not a per-time return-to-go, state baseline, GAE, learned V or bootstrapped target. Terminal GOAL/contact/timeout observations enter the finite observed return, with no continuation V estimate.

The current sampling distribution is an iid Gaussian around the un-clipped location, executed after float32 clipping. Exact boundary atoms use tail masses; interior actions use Gaussian densities (`t392_learning.py:16,25–49`). Frozen old log probabilities and actions are used for ratios. Before fitting, sequence means must match recorded locations within 1e-10 and densities within 1e-9 (`:104–107`); T437 also checks four complete stochastic, unowned trajectories with contiguous actual states (`t437_learning.py:268–282`). Teacher/recovery labels enter a separate auxiliary, never fabricated on-policy likelihoods (`:90–101,214–215,335–338`).

**Interpretation:** complete-episode score-function credit is a legitimate estimator construction; its episode-wide coefficient may have poor temporal discrimination and high variance. This observation is a testable limitation, not proof that PPO is invalid or that temporal credit uniquely caused the present failures. PPO clipping, repeated updates, auxiliary gradients and projection further make this an engineering combined objective, not a claim of an unbiased unclipped policy-gradient update.

## Recurrent and residual architecture

There are two different memory contracts. The frozen safe base evaluates one row at a time with its historical hidden-reset behavior (`t348_causal_sequence_warmstart/t348_core.py:22–23,41–42`). The **outer** encoder/core has a real 16-dimensional GRU carry: encoder120→32, GRU32→16, head16→2 (`:28–30`). The active inherited step passes its hidden state to the next decision (`t375_observation_residual/t375_core.py:28–34`); the actual session stores/updates `self.hidden` at each decision (`t381_native_mppi_teacher/run.py:121–125`). It is incorrect to describe the current entire Actor as resetting its GRU every decision.

The active whole-episode `sequence` starts at episode zero and calls the GRU once over the complete normalized history, without detach/truncation (`t375_core.py:20,25–27`), then the PPO loss backpropagates through that sequence (`t437_learning.py:300–306`). In T437 the recurrent representation is frozen; full history is still recomputed for the trainable readout. In T439 gradients can also propagate through the encoder/core. Step/sequence hidden carry equivalence is explicitly checked by the existing preflight at `t437_joint_closed_loop_rl/contract.py:102–126`; this audit did not re-execute it.

The deployed mean is the nested residual function

`location = 1.25*tanh(atanh(base_action/1.25) + head(h) + adapter([h,s]) + extra([h,s]))`,

where s is the normalized/clipped 120-input vector and `[h,s]` has 136 dimensions. The extra MLP is136→32→2 with ReLU (`t410_day_offline_chain/day_core.py:29–52`), following the inherited base/core/head/adapter topology (`t377_full_human_observation_transfer/t377_core.py:11–19`). The final command is clipped to [-1,1] float32 (`t392_learning.py:42–49`). T437 trains only existing `extra.*` parameters, 4450; T439 trains existing `encoder.*`, `core.*`, `extra.*`, 10722 (`t437_learning.py:40–59`; T439 counterpart same lines). Base/head/adapter remain frozen. T439 deliberately retains mean preservation rather than accidentally switching to the max profile (`t439_learning.py:182–185`; `runner.py:20–35`).

Full BPTT does not ensure useful long-horizon credit. Nor does opening the recurrent representation ensure compatible old/new functions; T439 is a fixed-seed counterexample to that sufficiency claim, not a rejection of all representation learning.

## Preservation, projection, and the zero initial KL gradient

`repair_core.old_kl` is an equal-variance **latent Gaussian** squared-mean penalty averaged over all history positions/action coordinates (`t418_evening_repair_chain/repair_core.py:124–128`), not a closed-loop safety certificate. T437 averages these trajectory losses across old/six supports; its broad support uses one task-balanced historical episode per step (`t437_learning.py:198–224`). These supports are explicitly historical/off-policy function targets (`:139–162`), not all states the modified policy will visit.

The combined preservation gradient supplies one local reference vector. Correction and raw PG vectors are projected only when their dot product with that reference is negative, then all vectors are summed, clipped and sent through Adam (`t437_learning.py:308–324`; `repair_core.py:143–154`). This is a local soft first-order rule. Even a nonnegative dot product with the aggregate reference does not constrain each time, each task, the actual Adam displacement, or future closed-loop state drift.

At an exactly matched anchor prediction, the squared-mean KL gradient is zero. That is the normal derivative at a smooth minimum; `protect_vector` then returns its input when reference norm²≤1e-30. It explains why this first-order reference alone cannot oppose the very first displacement at an exact match. It does **not** establish that every preservation support is initially matched, that the full reference is zero, that later penalties never act, or that zero initial KL gradient uniquely caused failure. T437 observed nonzero first-step PG/correction norms and verified actual updates (`T437_FINDINGS.md:7`). A useful future test would inspect the *actual post-Adam functional displacement* on critical histories instead of assuming the local projection guarantees retention.

## Native actuation and success semantics

The native forward command is exactly `wheel = 5*((drive+1)/2)**2` for clipped drive∈[-1,1] (`stage/runtime/mk-mini/training/e2e_phase10_lidar_v7_curriculum/phase10_strict_mujoco_env.py:39–54`; default limit at :24). It spans zero to five and supplies no reverse command. Zero command is not an instantaneous braking guarantee or zero actual velocity. Steering is the center steering command, converted into Ackermann wheel angles and rear differential commands with wheelbase0.60 m and track0.54 m (`stage9_runtime_fullplant_fix_v5_2.py:34–35,192–235`). The older0.42 m approximation must not be used here.

Native success is `goal_dist <= 0.20 and not collision`; success/contact immediately terminates and timeout is secondary (`phase10_stage1a_dense_env.py:37,306–328`). The current hold loop checks native termination after **every** 2 ms step and breaks immediately (`t381_native_mppi_teacher/run.py:129–138`). Thus GOAL is collision-free first hit, not learned stopping/settling inside the goal. A stopped terminal video frame cannot demonstrate braking.

## What the evidence supports, and what remains a hypothesis

- **Observed:** one wide6 m policy reached54/54 but sacrificed old tasks; the latest full proposal reached52/54+271/275 and lost four successes outside its panel. This demonstrates a current joint-retention failure, not six-metre impossibility.
- **Observed:** the more trainable T439 control retained0 proposals despite identical collected data. Frozen representation alone is not a sufficient explanation or remedy.
- **Hypothesis:** episode-wide credit may dilute the contribution of a short decisive steering interval. Code establishes the estimator; no current matched same-data estimator test establishes this as the cause. Older return-to-go experiments exist and must be reviewed before proposing a duplicate.
- **Hypothesis:** mean action preservation over finite historical supports may allow critical-time/state errors and feedback drift. The missed panel losses support a coverage problem; they do not prove that a particular max-loss replacement or trust-region method will solve it.
- **Hypothesis:** tanh/command saturation can reduce mean sensitivity and produce persistent turning. The architecture permits this, but no population-wide saturation statistics or Jacobian evidence were computed here. `T437_FINDINGS.md:9` explicitly says terminal-only records cannot diagnose the three near-goal old timeouts. Do not infer all failures are stopping, saturated steering, or insufficient time.

A rescue proposal should therefore separate (1) temporal estimator quality, (2) representation capacity and (3) post-update functional retention, with matched data/finite budgets and measured closed-loop outcomes. It must retain the original .20m first-hit contract, original evidence and failed candidates, avoid runtime routing/planning, and obtain explicit new authorization before any renewed Stage experiment. This audit itself licenses no launch.
