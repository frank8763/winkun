# T656: critical assessment and the next scientific decision

2026-11-19. **No new experiment, optimizer, native execution or launch.** T656 is complete: eight actual, eight shadow and eight independent historical updates passed audit; raw remote verification passed. These are execution checks, not a navigation success certificate.

## What the result actually says

[T656](FINAL_REPORT.json) preserves all eight mean field successes and repairs old26, but newly collides on old02; old00/03 still collide. Both its origin mean and T656 lose three required old IDs in the regression screen. There is **no net screen improvement**, full407 is null, and T501 remains retained. We have not solved “repair one, lose another.”

The [learning code](../../../experiments/t655_joint_current_recovery/t655_learning.py) uses terminal task-LOO advantages, clipped ratios, eight passes over 56 recorded trajectories and 11,906 adapters. [Actual comparison](ACTUAL_T656_REVIEW.json) shows old02 command RMS change 0.000156, peak 0.000582, yet paths separate by 2 cm at 29.8 s: candidate contact at 31.67 s versus origin GOAL at 36.902 s. Maximum recorded-history KL is only 1.13e-5. Tiny nominal drift is insufficient protection.

The [new loss audit](KEEP_TARGET_REVIEW.json) exposes an avoidable design problem: keep10 averages 384 old histories while keep10 averages eight field histories. Each field history therefore receives **48 times** one old history's nominal coefficient, before within-trajectory normalization. Five task IDs occur in both sets with identical initial120, hidden state, integration and goal but different command targets; initial steering differences are 0.000312–0.000766. Zeroing both imitation terms is impossible there. Both reference trajectories reach GOAL: this is conflicting exact imitation supervision, **not physical task incompatibility**. Neither the weighting nor conflict alone establishes the cause of old02.

## Is the estimand wrong?

It is legitimate for the declared conditional stochastic problem, but inadequately aligned with deployment. Three separate gaps matter:

1. `P(safeGOAL | fixed mean prefix, Gaussian suffix)` differs from autonomous original-start success. Prefix masking correctly avoids a false gradient, but also means that learning does not optimize reaching those recovery states under the updated policy.
2. Expected success under sigma 0.2 actions differs from success of the deterministic location policy. Success-conditioned exploratory actions may occupy alternate modes; increasing their likelihood need not improve the mean action sequence.
3. At initialization, terminal-return score gradients can estimate the conditional stochastic objective. Later clipped-ratio passes retain behavior-policy states and histories; they do not resimulate the changed controller's state distribution. Eight epochs are not eight new closed-loop training rounds.

[PPO §3/5, Algorithm 1](https://arxiv.org/pdf/1707.06347) alternates fresh collection with surrogate optimization. Clipping changes the sampled surrogate incentive; it does not itself enforce a hard policy-KL bound or guarantee each task's deterministic return. Our episodic adaptation is not a completed standard actor-critic PPO benchmark.

Thus deployment mismatch is a substantial design weakness, **not a proven sole cause**. Finite-sample variance, sparse long-horizon credit, parameter capacity and sensitive closed-loop dynamics remain competing explanations.

## What our sequence has and has not excluded

- T653 rejects its sampled-surrogate protection as sufficient, not constrained optimization generally.
- T654's teacher failed at all three deviations; its supervised control never trained.
- T655 recovery successes were 2/8, 1/8, 2/8; original-start old03 had 0/8. No paired learning comparison occurred.
- T656 rejects this recipe, not joint capability, all RL, or physical solvability. Representation aliasing and the precise regression cause remain unproven.

Our sequence produced useful falsifications, but repeatedly moved from a failed premise into another small surrogate repair. Independent audits cannot compensate for the missing strong learning baseline or the training/deployment gaps. Adding old02 alone would continue that pattern.

## Necessary next baseline, before another mechanism

First fix the reference contract offline: one canonical successful reference per **unique task ID**, explicit per-ID/group weights, and identical-input target-consistency checks. Use T501 for the five already-solved overlapping IDs and mean for its three additional field IDs: 387 unique required successes. Removing duplicate imitation targets must **not remove any of the eight field GOAL requirements**.

Then preregister a **joint original-start learning reference** over a fixed old/new task distribution, holding actor/trainable scope fixed initially. Refresh current-policy rollouts after each update block; specify advantage estimation and batch lifetime. Measure sampled and deterministic execution separately. Include joint supervised consolidation of the cleaned unique-ID union as the fitting/capacity control. Neither low loss nor more data alone is sufficient. [DAgger §3](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf) motivates learner-induced coverage; T654 invalidates assuming an expert can correct these states.

Retain every required old ID plus all eight field IDs in one original-start controller, then assess separately fixed neighborhoods and holdouts. This empirical suite gate prevents deploying observed regressions; it is not a theorem of monotonic control performance.

For a later method, [deterministic policy gradients §3](https://proceedings.mlr.press/v32/silver14.pdf) separate noisy behavior from a deterministic target. A TD3-style joint learner requires held-out validation of target-continuation value/action guidance; existing failed critics cannot supply it by assumption. Binary contacts limit smoothness guarantees. Annealing alone remains unproven; neither candidate establishes novelty.

No new launch today. Preserve 15:30 learning/16:10 native/17:00 all-stop limits. Read scope: cited PPO methods/Algorithm 1, DPG §2–3, DAgger §3; local T656 final/screen/learning, actual/keep-target audits, runtime and T655 learning source. Not complete proofs or all Stage history; no Phase evidence.
