# T655: current-policy recovery coverage versus original-start coverage

2026-11-19. Read-only preparation: **0 new native/optimizer/launch**. Recommend one matched collection-distribution experiment. It may test a missing recovery capability; it cannot promise to eliminate forgetting.

[T654](../t654/FINAL_REPORT.json) completed exact verification but all three deterministic T501 recoveries collided, so learning correctly remained zero. The teacher premise failed at these anchors. [T653](../t653/FINAL_REPORT.json) already shows that favorable sampled-return constraints and nominal replay can coexist with lost deterministic successes. Repeating their coefficients or projecting another local gradient does not resolve this evidence.

## The structural issue

For a shared policy, improving a scalar average permits task tradeoffs. Small command errors on historical trajectories constrain neither the new state distribution nor every closed-loop outcome. Conversely, observed tradeoffs do not prove that the representation cannot express a joint solution.

Two objectives must remain distinct:

`J_recovery(theta) = E[safeGOAL | fixed recorded mean prefix, stochastic suffix(theta)]`

`J_original(theta) = E[safeGOAL | original start, entire policy(theta)]`.

Neither improved recovery return nor improved Gaussian-policy expected return guarantees success of the deployed deterministic mean. This gap is why final original-start **per-ID retention**, rather than surrogate nonincrease, must decide retention.

[JSRL §4.1–4.3](https://proceedings.mlr.press/v202/uchendu23a/uchendu23a.pdf) motivates training exploration after a roll-in and eventually eliminating the guide. Its guarantees assume useful guide coverage and a learning curriculum; one fixed batch after a failing policy's prefix does not inherit those guarantees. Here the frozen mean supplies an actual state to investigate, not an expert action label or a deployment controller.

[RECALL §1/4](https://arxiv.org/html/2311.11557v1) and [Zhou et al. §4](https://proceedings.mlr.press/v199/zhou22a/zhou22a.pdf) show that perfect historical memory can still leave continual RL vulnerable to imbalance and distribution shift. Their SAC/MPO/CRR value-estimation mechanisms are different from Stage's critic-free update: Q extrapolation is not an established cause here. They support distinguishing memory availability from current-policy coverage, not installing an unvalidated critic.

## One fixed comparison

Initialize both arms from the identical T653 **mean research checkpoint**, with 11,906 adapters, frozen encoder/GRU/head, fresh Adam3e-5. Keep T501 unchanged. Both arms retain the original384 T501 nominal histories at keep10 and all eight mean field success histories at keep10, reusing T654's verified full reconstructions.

- **Original arm:** old00/03/26, each eight fresh mean-policy stochastic episodes from the original task start, plus eight exact regenerations.
- **Recovery arm:** the same three tasks, each eight fresh stochastic suffixes after the recorded mean's fixed decision28/56/71 prefixes, plus eight exact regenerations. No T501 action labels.
- **Shared field group:** each of mean's eight field success tasks gets four fresh current-mean original-start stochastic episodes plus four exact regenerations, reused by both arms.

Total: **80 fresh and 80 exact episodes, 166 shared-stage commands including bookkeeping**. Each arm receives 56 behavior episodes across the same 11 tasks. This changes old-task starting-state coverage only; exploration, task weights, regularization, capacity, update budget and field data match. Average each task's episodes before combining tasks; disclose the fixed 11 task weights and their old/new group totals. Episode duration and differing sample counts must not silently reweight tasks.

Require at least one safeGOAL in every old group in **both** arms, and at least one mixed old group per arm, before either learns; otherwise both perform zero learning and report the coverage result. Mixed outcomes provide terminal-LOO direction. An all-eight-success group is eligible but has zero binary LOO gradient; disclose it. No new seeds or later anchors follow a failed gate. Eligibility conditions sampling, so do not claim an unbiased post-selection estimate.

Each arm gets at most eight clipped episodic-PG proposals: **16 actual/16 shadow/16 independent-history updates total**, zero Critic or extra optimizer preflight. Both fits finish before evaluating either candidate. All 56 trajectories are current-policy samples only at initialization; eight fitting epochs are eight surrogate passes, not eight fresh on-policy rounds. Historical correct76/new32 are provenance and diagnostics, not silently relabeled current-mean PG data.

## Conditional density and deployment checks

For a fixed prefix, the legitimate suffix score is `sum(t >= first) grad log pi_theta(a_t | causal_history_t)`. Prefix actions have no sampled ownership and no score; they still determine observations, action history, clock and carry. Reconstruct each policy's own full causal state. Preserve original scene/initial120/integration203/goal contracts and original remaining time; never reset memory or extend the suffix horizon. Verify clipped-action atom densities and exact regeneration of every array. No successful-action averaging or inferred safe teacher is needed.

Use matched 36-screen/20-exact evaluations, then conditional full407. Require **all original384 success IDs and all eight mean field success IDs**, at least387/407, in one autonomous original-start policy. More total successes cannot compensate for one lost required ID. Missing items remain null. No runtime prefix, policy switch, planner or teacher is allowed. A full-suite pass establishes this finite regression set only; basin enlargement requires separately fixed neighborhoods/holdouts.

## Alternatives and stopping inference

[C-CHAIN (ICML2025)](https://proceedings.mlr.press/v267/tang25g.html) studies output churn and plasticity; its [author driver](https://github.com/bluecontra/C-CHAIN/blob/2f8bedfefb6a0a276d7709447a5cdb75cecfdaad/crl_gym_classic_control/control_train_aligned_c_chain.py) explicitly uses reference batches and adaptive regularization. Stage has not diagnosed NTK-rank collapse; another regularizer would confound today's coverage test. [ARROW2026 §3](https://arxiv.org/html/2603.11395v2) instead trains a world model with distribution-matched replay and imagined actor updates; that is not a validated Stage dependency.

[OWL §1](https://cdn.aaai.org/ojs/20674/20674-13-24687-1-2-20220628.pdf) distinguishes truly conflicting tasks with identical observations from forgetting. Stage supplies goal/history inputs, so this impossibility premise is unproven; its runtime head selection is outside scope.

If recovery sampling lacks successes, this fixed exploration budget did not expose a learnable recovery signal. If both arms fit but fail retention, this batch/capacity/update protocol is insufficient, not proof that all RL fails. If only recovery passes, evidence favors its specific state coverage. If neither passes, stop this mechanism today. Lock physical/storage reserves before approval, retain 15:30 learning/16:10 native/17:00 all-stop limits, and preserve every failure.

Read scope: cited method passages, RECALL introduction/method and selected evaluation passages, ARROWv2 §1–3, C-CHAIN publication abstract/author driver, OWL §1; not complete proofs or exhaustive 2026 literature. No external code copied/executed. Local reads: Stage status, T653/T654 summaries, T654 departure records and learning source; no Phase evidence.
