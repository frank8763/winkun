# T654: one gated actual-deviation recovery-imitation comparison

2026-11-19. Research preparation only: **0 native, 0 optimizer, no launch**. Recommend one matched comparison of nominal supervision versus qualified recovery supervision. This is a standard imitation-learning hypothesis, not a new algorithm or a TaSIL reproduction.

[T653's audited report](../t653/FINAL_REPORT.json) rejects both PG candidates: mean field8/24 lost old00/03/26; protected field6/24 still lost old00/26. Full407 remained null. T501's local old00/03/field5offset1 results were 1/8, 7/8, 7/8. These observations justify testing feedback-state coverage; they do not establish a unique cause or a stable teacher. T637–T649 cross-scene claims remain withdrawn after T650.

## What the primary sources support

- [TaSIL, §3, Eq(14), Appendix C](https://arxiv.org/html/2205.14812v2): matching expert action derivatives or finite differences can control imitation error under explicit incremental stability, smoothness, approximation and coverage conditions. Stage has verified none of those conditions. T501's fragile neighborhood prevents assuming robustness; three successful suffixes would still not establish it. A Jacobian with respect to correlated sensor columns is not a derivative along physical vehicle states. Defer derivative losses.
- [DART, §4–4.2](https://proceedings.mlr.press/v78/laskey17a/laskey17a.pdf): noise broadens the supervisor's visited-state distribution, producing corrective labels; its noise covariance is adapted to learner error. This supports recovery examples, not arbitrary noisy-action cloning or guaranteed safe navigation. T654's fixed actual student deviations are not DART's optimized-noise procedure.
- [DAgger, §3, Algorithm 3.1](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf): obtain expert labels on learner-visited states and aggregate them. A single fixed batch of three deviations is only a restricted analogue, without the iterative coverage or no-regret assumptions behind its guarantees.
- [CLEAR, §3](https://papers.neurips.cc/paper_files/paper/2019/file/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Paper.pdf): combines new/replay actor-critic learning, V-trace, historical policy KL and value cloning. Nominal replay is therefore an essential comparator; replay's empirical effectiveness does not certify closed-loop retention here. T654 is supervised adapter repair, not CLEAR.

## Shared qualification: stop if any teacher fails

Use exact T653 **mean** as the research starting model, retaining T501 unchanged. Fix old00/03/26 and each first physical-time-aligned state at which mean differs from nominal T501 by at least 2 cm or 1 degree. Lock those indices before executing anything; no later-anchor, teacher, noise or seed fallback.

Replay each recorded mean prefix, rebuild T501's own causal carry from legal observations and actual executed actions, then let deterministic T501 finish within the **original remaining episode time**. Require safe GOAL and complete exact regeneration for all three. A known nominal success or a successful stochastic trajectory does not qualify that deterministic suffix.

Check canonical scene, initial120, integration203, goal, clock, wrapped yaw and complete prefix arrays before branching. Preserve actual action-history inputs. Never transplant mean hidden states into T501, reset hidden state at handoff, edit individual lidar columns, or inject future suffix information. During student fitting reconstruct its own carry from the full prefix; mask prefix supervision. Cached frozen-body features require whole-sequence equivalence to uncached forward passes, not just first-state equality.

Reconstruct the five missing full mean field successes plus exact regeneration, matching existing T653 compact hashes; combine with the three already available full successes. Thus shared preparation has **8 unique native episodes plus 8 exact episodes, 22 commands including six pool bookkeeping commands**, before evaluation. Existing successes are reused evidence. No new stochastic-teacher collection is needed to prepare the offline protocol.

## The one factor and finite implementation

Both arms start the identical mean checkpoint/fresh Adam, train the same 11,906 adapters for exactly **32 updates at 3e-5**, with frozen encoder/GRU/head, identical order and task/time normalization. Common losses preserve all original384 T501 nominal histories (keep10) and all eight mean field success histories (keep10). Add three equally weighted repair-task slots:

- **Nominal control:** T501's existing nominal old00/03/26 successful trajectories.
- **Recovery arm:** the three qualified T501 suffixes, with full prefixes used only to reconstruct history.

Only repair supervision support changes; model capacity, common data, labels' policy identity, weights and update counts match. Count 64 actual/64 shadow/64 independent historical reconstructions; no RL, Critic, extra optimizer preflight, grid or automatic second attempt. This directly differs from T653's sampled-return surrogate constraints. The frozen mean origin remains descriptive evidence, not a substitute for the 32-update control.

Use each verified deterministic teacher's executed command labels. Do not average successful stochastic action traces into a claimed safe response: outcome selection, exploration noise and alternate routes can create conflicting labels. Do not fuse distinct histories merely because current observations look similar. Report fitting error and remaining conflicts; perfect action matching is not a safety proof.

## Gates and interpretation

Run the same 36-screen/20-exact gate for both arms, then conditional full407. Retention requires **every original384 success ID and every one of mean's eight field success IDs**, implying at least 387/407, not merely improved totals. Missing evaluations stay null. Every accepted case must be autonomous from its original start, without teacher routing, contacts, relaxed termination or extra time. No generalization claim follows these reused tasks.

If teacher qualification fails, the chosen T501 recovery premise fails at those fixed anchors: perform zero learning. If fitting improves but autonomous gates fail, this batch and budget did not transfer recovery supervision into the required controller behavior. If both arms improve similarly, extra nominal replay remains sufficient explanation; if recovery alone passes, evidence favors this specific coverage intervention. Neither outcome proves global stability or excludes other capacities/budgets. Lock final native/storage/wall bounds before approval; retain 15:30 learning, 16:10 native and 17:00 all-stop deadlines.

Read scope: selected methods/theory passages cited above, not complete proofs or an exhaustive literature search. Author repository inspected at commit `4b10cb71a220dac0cf961618b8d68580c4779722`: [bc.py](https://github.com/unstable-zeros/TaSIL/blob/4b10cb71a220dac0cf961618b8d68580c4779722/imitation_learning/bc.py), [dart.py](https://github.com/unstable-zeros/TaSIL/blob/4b10cb71a220dac0cf961618b8d68580c4779722/imitation_learning/dart.py), and `train.py` configuration. No author code copied or executed; `gym.py` retrieval failed. Local scope: Stage instructions/status, migration/context records, T653 plan/review/final JSON; no Phase evidence read.
