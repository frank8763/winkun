# Navigation primary-source review — 2026-11-09

Later same-day clarification: the user explicitly reopened research after the T440 closeout. The historical stop state quoted below remains true for old executions; it does not prevent the newly authorized literature/design work. No new training was registered or launched here.

Research only. No simulator, model load/forward, optimizer, installation, Git operation, or new experiment. This note uses Stage entry points and primary external sources only; it does not use Phase research. Current `RESEARCH_STATUS.md` says T440 ended through execution-budget failure and the Stage line is stopped pending fresh explicit authorization. This note does not authorize a restart. Only this file is written.

## Decision

**Inference:** SRU is credible evidence that a causal recurrent navigation policy can handle goals beyond sensor range. It is not a demonstrated solution to Stage's lidar/Ackermann task or a reason to launch another small-budget cell replacement. The more defensible rescue candidate is a clean recurrent learning baseline with an auditable sequence/action contract, followed by one matched memory ablation only if that baseline learns. This is an adaptation of published methods, not algorithmic novelty or a success promise.

## SRU paper: observed claims and limits

Source: [Yang et al., arXiv 2506.05997v2](https://arxiv.org/html/2506.05997v2), §§5.1–5.6, Table 1, §6, Appendix C.

- Navigation runs at 5 Hz; its linear/angular velocity actions feed a learned 50 Hz locomotion controller. The real robot is Unitree B2W, not an Ackermann car.
- Actor input includes front depth, velocities, gravity and relative goal; the critic additionally receives clean 360° height-scan information. Real deployment uses LiDAR-based state estimation/localization.
- Reported overall success: GRU 61.0%, LSTM 63.5%, SRU-Ours 78.9%. Three-seed learning curves exclude unsuccessful GRU training runs: this qualification matters.
- Training start–goal distance is at most 30 m; demonstrations include goals beyond 70 m and traversal beyond 100 m. Outbound/return goals are separate episodes, with memory reset between goals.
- DML jointly trains policies with mutual distillation; SRU without DML reports 65.7%, with DML 78.9%. A claim of “no distillation” must distinguish this regularizer from a teacher–student pipeline.
- The authors acknowledge exponential memory decay and local mapless scope; these are not kilometer/hour global-navigation guarantees. The inspected text does not establish training GPU count, VRAM or wall time.

## Official code: what is now available

**Observed:** the [official project site](https://michaelfyang.github.io/sru-project-website/#code) links the training, simulation, depth-pretraining and deployment components. The historical Stage note that only the synthetic memory subtask was public is outdated. Availability is not a claim that every paper result has been independently reproduced. The correct deployment repository is [`leggedrobotics/sru-robot-deployment`](https://github.com/leggedrobotics/sru-robot-deployment); `sru-navigation-deployment` returns 404.

### Simulation and goal contract

**Observed in live raw source:** [`navigation_env_cfg.py`](https://github.com/leggedrobotics/sru-navigation-sim/blob/main/isaaclab_nav_task/navigation/navigation_env_cfg.py) sets `PLANNING_FREQ=5`, default `num_envs=2048`, physics `dt=.005`, low-level decimation 4, and 60 s episodes. The goal resampling interval is `1e9`, explicitly disabling automatic goal changes except episode reset. Actor observations include noisy/delayed state, last action, relative target and depth. Critic observations include clean state/depth, normalized time and encoded height scan. Velocity output is consumed by a separately loaded locomotion policy.

[`goal_commands.py`](https://github.com/leggedrobotics/sru-navigation-sim/blob/main/isaaclab_nav_task/navigation/mdp/navigation/goal_commands.py) separates goal resampling from updating the goal in the robot frame. Reset sampling uses terrain-validity masks and spawn clearance. That is privileged training/reset construction; it is not evidence that a runtime route planner supplies intermediate goals.

**Inference:** the fixed world goal with a continuously recomputed relative vector fits Stage's final-goal constraint. The robot, sensors and action interface do not transfer unchanged. A 3-component body velocity head cannot simply become Stage drive/steer while retaining the same policy semantics.

### Training scale and recurrence

**Observed:** [`B2W agent configuration`](https://github.com/leggedrobotics/sru-navigation-sim/blob/main/isaaclab_nav_task/navigation/config/b2w/agents/rsl_rl_cfg.py) sets 16 rollout steps/environment, 15,000 iterations, a 512-unit recurrent layer, actor/critic MLPs `[512,256,128]`, dropout .2, five learning epochs and four minibatches. MDPO and PPO have different configured discount factors (.999 and .995). The [simulation README](https://github.com/leggedrobotics/sru-navigation-sim#training) also shows 2,048/4,096-environment launches; examples are not measured paper run receipts.

**Calculated, not measured:** `2048 × 16 × 15000 = 491,520,000` navigation transitions if the default configuration completes fully. At 4,096 environments the corresponding product is 983,040,000. Neither is a verified total for the paper, an independent-sample count, or a Stage budget recommendation. Physical integration steps are a different unit.

**Observed:** [`actor_critic_sru.py`](https://github.com/leggedrobotics/sru-navigation-learning/blob/main/rsl_rl/modules/actor_critic_sru.py) maintains separate actor/critic recurrent memories and exports an actor carrying hidden/cell state across calls. Exported inference has an explicit reset. Its training memory reset can initialize terminated environments from other batch states, whereas export resets to zero; do not silently copy that difference. [`lstm_sru.py`](https://github.com/leggedrobotics/sru-navigation-learning/blob/main/rsl_rl/networks/sru_memory/lstm_sru.py) implements a learned transformation gate and refined forget gate. This is learned implicit memory, not an explicit geometric map or analytic motion-compensation guarantee.

**Unknown:** an 8 GB training memory peak, a feasible Stage wall-clock budget, or equivalent performance after replacing depth features with lidar tokens. No such measurement was performed here. Live source was read from `main`; the simulation tree observed through the official API was `6a25e724fafe972ff7b7442ee98f6b93be5bca9f`. Moving branches must be pinned before any future implementation.

### Deployment caveat: a video alone does not prove a fixed-goal episode

**Observed:** [deployment README](https://github.com/leggedrobotics/sru-robot-deployment#module-details) describes autonomous goal navigation, smart joystick and waypoint playback. Its hierarchy lists navigation at 5 Hz and locomotion at 50–200 Hz.

The [controller README](https://github.com/leggedrobotics/sru-robot-deployment/blob/main/rl_nav_controller/README.md) documents joystick-defined goals updated at 5 Hz, recorded waypoint sequences, a manual hidden-state reset button, and `ARRIVE_GOAL_THRESHOLD=.75` m. These are supported modes, not evidence that the paper demonstrations used them. Conversely, the existence of autonomous mode does not prove a particular video used an unchanged final goal. Goal-message and mode/reset logs would resolve that. Do not import the .75 m threshold into Stage or call its arrival criterion equivalent without checking the locked Stage task version.

## Ni et al., ICML 2022: a practical recurrent baseline

Sources: [official PMLR paper](https://proceedings.mlr.press/v162/ni22a.html), [author manuscript](https://arxiv.org/html/2110.05038v3), §§4–5, Appendices A/B; [official implementation](https://github.com/twni2016/pomdp-baselines).

**Observed:** the paper compares against specialized methods on 21 environments, reporting stronger sample efficiency/asymptotic performance on 18. Separate actor/critic recurrent encoders outperform sharing in its ablations; off-policy SAC/TD3 often beat tested on-policy recurrent baselines, with exceptions. Context length is task-dependent: 64 is a starting point, not a universal optimum. LSTM versus GRU is usually less decisive. Its replay samples sequences/subsequences, pads short sequences and uses zero initial hidden state. This is not full-history reconstruction or a burn-in guarantee. Input history must contain information needed to infer hidden state.

Reported system examples use roughly 1–1.2 GB GPU memory but take 7–22.5 hours per million environment steps; hardware is an 11 GB RTX 2080 Ti plus a large CPU host. These are benchmark observations, not Stage runtime predictions. The paper does not test lidar Ackermann real-world navigation or establish success after a few thousand updates. Its guidelines support an implementation baseline, not “SAC always wins” or “GRU solves long horizons.”

## Transfer to Stage: a bounded proposal, not an execution protocol

| Question | Evidence-based choice | Status |
|---|---|---|
| Runtime architecture | One causal lidar/state/final-goal Actor with persistent memory and direct Ackermann drive/steer; no runtime teacher, router, waypoint generator or planner | User contract; compatible with recurrent model-free learning |
| Learning baseline | Separate recurrent actor/critic encoders; explicit sequence masks, hidden-state initialization and action/observation timing | Inference from Ni; must be verified against Stage implementation |
| Memory change | Compare an ordinary recurrent baseline with a parameter/computation-matched SRU only after the baseline learns meaningful new closed-loop behavior | Inference; no evidence that cell replacement alone rescues Stage |
| Perception | Use lidar-native features; do not assume a pretrained depth encoder or spatial attention transfers | Unmeasured adaptation |
| Privileged training | Treat SRU's critic privileges as an experimental factor; any Stage training-only access must be explicit and absent from Actor inputs | Different information contract, not a free reproduction |
| 8 GB resource constraint | Admit a measured small sequence/batch configuration before a bounded registered run; estimate total rollout/learning time from throughput | Future requirement, not permission to run now |

**Inference:** weeks of retained-model adaptation do not establish that recurrent navigation is impossible. Equally, these papers do not establish that Stage's current failures are caused by insufficient memory. A useful rescue must first distinguish representation/credit-assignment failure from a learning contract that cannot acquire and retain new closed-loop behavior. Keep the final task's threshold, timeout and collision criterion fixed; count genuinely new interactions separately from exact replay; evaluate held-out start/goal cases with fixed goals and reset logs. If a plain recurrent baseline cannot learn within a newly authorized, adequate but finite budget, do not respond by stacking SRU, attention, DML and new rewards together.

No new algorithm novelty, generalization, real-car readiness, or successful rescue is established by this review. No experiment was launched.
