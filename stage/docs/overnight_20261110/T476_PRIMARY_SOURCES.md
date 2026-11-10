# T476 primary-source research: wide6m → wide8m

Research date: 2026-11-10. This file is a research recommendation, not a launch receipt or an experiment result. No simulator, optimizer, or training was run for this note. Only Stage materials were consulted. The coordinating agent reports renewed user authorization after the T475 stop; the older stop record remains preserved.

## Decision

The two simplest additional branches are **a fixed goal-distance curriculum** and **full-episode λ=1 credit assignment**. Keep direct 8m recurrent PPO as the reference branch. Start each branch from the same explicitly identified checkpoint and optimizer contract; do not silently pass a rejected candidate between branches. Defer correlated exploration tonight because correct history-conditioned density adds an avoidable implementation risk. These are established methods and engineering adaptations, not claims of algorithmic novelty or guaranteed improvement.

Task contract supplied by the coordinator: T449 wide6m development 54/54 and old274/275; extend the final goal radially from 6m to 8m while retaining the base6 initial pose and 1.17m aperture; final cap106.668s versus80s. This note does not independently re-evaluate those checkpoint results. None of the recommendations inserts a runtime planner, waypoints, or teacher.

## Sources actually inspected

| Primary source | Relevant evidence | Limit for this task |
|---|---|---|
| Schulman et al., [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438), ICLR2016, §3 Eq16–18 and §5 | GAE interpolates TD residuals and return-minus-baseline; λ=1 removes value-approximation bias in the discounted advantage estimator while increasing variance. | Does not establish that λ=1 outperforms short traces on this navigation task. |
| Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347),2017, §3 and Algorithm1 | On-policy collection followed by limited epochs of a clipped likelihood-ratio objective. | Clipping is not an old-task retention guarantee or a pointwise bound on the first update. |
| Florensa et al., [Automatic Goal Generation for Reinforcement Learning Agents](https://proceedings.mlr.press/v80/florensa18a.html), ICML2018 | Learns a goal curriculum by proposing tasks of suitable difficulty. | A manually fixed radial6.5→7→7.5→8 curriculum omits the generator/adversarial algorithm; call it an engineering goal curriculum. |
| Florensa et al., [Reverse Curriculum Generation for Reinforcement Learning](https://proceedings.mlr.press/v78/florensa17a/florensa17a.pdf), CoRL2017, §1 and §3–4 | Expands feasible initial states outward from successful states; discusses a minimum degree of reversibility. | Goal extension at a fixed start is a different intervention. Ackermann dynamics and inability to reverse make the reverse-start assumption especially uncertain. |
| Pardo et al., [Time Limits in Reinforcement Learning](https://arxiv.org/html/1712.00378v4), §2–3 | Distinguishes a true finite-horizon task from collection truncation; remaining time matters for the former, endpoint bootstrapping for the latter. | A106.668s task deadline must not be treated as an arbitrary rollout cut solely to manufacture optimistic value targets. |
| Tallec et al., [Making Deep Q-learning Methods Robust to Time Discretization](https://proceedings.mlr.press/v97/tallec19a.html), ICML2019 | Shows that physical timestep and algorithmic time scales can materially affect continuous-control learning. | Its main method and collapse analysis concern Q-learning; it is motivation for a physical-time audit, not a PPO failure diagnosis. |
| Hollenstein et al., [Colored Noise in PPO](https://arxiv.org/html/2312.11091v2), AAAI2024, §2–3; [author implementation](https://github.com/jkbjh/cn-ppo-paper-code) | Reports improved benchmark learning with temporally correlated action sampling; best correlation depends on collection setup. | Benchmark gains do not establish exact conditional likelihood correctness for our clipped bounded policy or recurrent history. No local reproduction was performed. |

## Branch A: fixed final-goal curriculum

Implementation recommendation: four finite stages6.5,7,7.5,8m. Each episode has **one final goal fixed before reset**. Preserve the same base6 vehicle pose, map geometry, aperture, action interface, policy observation construction, and success/contact definitions. Recompute the relative final-goal observation using the actual chosen goal; never substitute a subgoal during an episode. This is compatible with a single recurrent Actor.

For a clean distance comparison, use the existing cap rule proportional to distance, rounded to native timestep: approximately86.668,93.334,100,106.668s. Record this explicitly: changing distance and allowed time together is a task-family intervention, not a pure distance-only ablation. Verify each terminal goal is valid without silently retrying a seed and moving the initial pose.

Before launch, give each stage a fixed interaction/update allocation and a fixed advancement rule. Do not train at a stage indefinitely until it passes. If the development advancement condition fails within its allocation, retain that failure and stop this branch; unused later-stage budget is not permission for retries. Include a fixed proportion of old/6m on-policy episodes if retention is needed, declared before launch. Retention data should be the branch's current policy experience; successful old trajectories are not automatically current-policy PPO samples.

This tests whether access to nearer goal successes helps learning the longer task. It confounds the training-goal distribution with the timing of exposure to8m; equalize total interaction/update caps with the direct8m reference where feasible and report actual consumed compute. Curriculum-task success cannot substitute for a frozen full8m evaluation.

## Branch B: full-episode λ=1 with unchanged physical gamma

Local evidence: `stage/experiments/t457_balanced_task_ppo/protocol.json` declares gamma_native0.999966667222216≈exp(-.002/60) and lambda_native0.9995001249791693≈exp(-.002/4). Its `run.py` calls `physical_gae` with actual `native_counts`. Confirm the chosen T476 parent uses this contract before carrying these numbers forward.

For decision duration n native steps, use Γ=gamma_native^n and Λ=lambda_native^n. The inherited residual-trace scale is then1/(1/60+1/4)=3.75s. As a derived illustration, a residual80s away receives coefficient exp(-80/3.75)≈5.43e-10; at106.668s≈4.44e-13. This is not the total information available to the learner: dense rewards and an accurate bootstrapping critic can convey earlier information.

Change only λ_native to1, keeping gamma and task/reward/action/retention contracts fixed. Compute targets over complete episodes before any minibatch or BPTT slicing. With correct terminal masks and zero terminal value, the recursion telescopes to discounted return minus the frozen pre-update baseline. At gamma's60s scale, terminal attenuation remains exp(-80/60)≈.264 and exp(-106.668/60)≈.169. Thus λ=1 removes trace truncation, not discounting.

Use pre-update values and fixed advantages for each PPO batch; do not recompute them after every Actor epoch. Record actual gradient norms, empirical KL, return/advantage spread, value error, successful exploration episodes, and frozen deterministic outcomes separately. Full-return variance may overwhelm a small batch, and a baseline fitted on its own sampled targets can overfit; do not claim the whole practical clipped/normalized PPO update is unbiased. A recurrent policy's long return target also does not increase its BPTT horizon or prove that its hidden state learns a long memory.

The task cap is a genuine failure deadline if the registered objective says so: terminate its return there. A storage/chunk boundary is different and needs the appropriate continuation handling. Preserve any existing time observation; if remaining time is absent, document that finite-horizon partial-observation limitation rather than silently changing the Actor input in this branch.

## Correlated exploration: plausible third mechanism, deferred

Correlated innovations can produce persistent steering deviations that white noise averages away, but this is a hypothesis for this vehicle. A future finite branch should hold marginal scale and physical correlation time fixed and score the **actual conditional behavior law**. For an AR latent process, the conditional law depends on the previous sampled latent and on both current and previous policy locations; recompute those from the same full observation/action history when evaluating a candidate policy. If clipping creates boundary atoms, include their conditional probability masses instead of using an interior Gaussian density. An invertible squash instead needs its Jacobian. These density requirements are mathematical implementation deductions, not claims that the colored-noise paper proves this specific Stage variant correct.

Do not add correlated noise to rollout actions and then keep the existing independent boundedGaussian log-probabilities. Defer this branch until synthetic old/new density equality, conditional normalization including atoms, recurrent sequence/step consistency, and exact rollout replay all pass. The coordinator has chosen no correlated-noise engineering tonight.

## Interpretation and finite stopping

All branches need one registered joint physical/wall-clock/storage bound, including evaluation, exact replay, audit, and backup; hard stop is Nov11 09:00 or an earlier existing bound. This note authorizes no numerical budget expansion. Report the predetermined terminal checkpoint, all rejected/failed branches, old-task regressions, and missing evaluations. Do not pick a best intermediate model after seeing the test panel. Fixed development tasks and repeated deterministic replay are not new independent samples or cross-map generalization. The prior old274/275 result already contains a known failure; preserving that success set and solving8m is a more honest research comparison than silently labeling the parent globally safe.
