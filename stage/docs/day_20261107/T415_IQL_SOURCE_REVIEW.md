# T415: IQL / AWAC primary-source review

2026-11-07. Literature review only: no registration, simulator calls, optimization, model changes, or launch. Stage only. The coordinator reports T414 completed 800 live + 800 shadow BC steps, rejected both candidates at old8=6/8 (LEFT 2/5 then 0/5), retained safeT255, and made no PPO updates; this note does not independently audit those results.

**Decision:** IQL is a defensible *new candidate mechanism* for learning from successful demonstrations plus failed transitions. It adds multi-step value propagation and advantage-based action selection to imitation. AWAC is a related offline-to-online alternative, but its critic evaluates current-policy actions and therefore retains an extrapolation pathway that IQL avoids during value fitting. Either can deploy one causal Actor with critics discarded. Neither establishes global navigation, safety, missing-data connectivity, or novelty. IQL's AntMaze evidence motivates testing trajectory composition; it is not evidence that Stage's observation/history and datasets permit it. [IQL paper, §§4–5](https://arxiv.org/html/2110.06169), [AWAC paper, §§III–IV](https://arxiv.org/html/2006.09359), [AWAC author project](https://awacrl.github.io/).

## Equations and implementation facts

Let `x` denote an admissible causal information state, `a` the recorded action, `m` the bootstrap mask, and `R, Γ` the interval reward and discount defined below. Let `q̄=min(Q̄1,Q̄2)` and stop gradients through targets/weights.

\[
L_V=\mathbb E_D[|\tau-\mathbf1_{u<0}|u^2],\quad u=\bar q(x,a)-V(x),\quad \tau>1/2,
\]
\[
y=R+m\Gamma V(x'),\qquad L_Q=\mathbb E_D\sum_{j=1}^2(Q_j(x,a)-y)^2,
\]
\[
A=\bar q(x,a)-V(x),\quad w=\min(\exp(\beta A),100),\qquad
L_\pi=-\mathbb E_D[w\log\pi_\theta(a\mid x)].
\]

The sign shown minimizes negative weighted likelihood. Official IQL code uses the minimum of two target critics for V/Actor, clips exponential weights at 100, updates V then Actor then Q, and Polyak-updates the target critic. The actor never supplies next actions for IQL's Bellman target. Its code calls the **inverse temperature** `temperature`: larger means sharper weighting. [critic.py](https://raw.githubusercontent.com/ikostrikov/implicit_q_learning/master/critic.py), [actor.py](https://raw.githubusercontent.com/ikostrikov/implicit_q_learning/master/actor.py), [learner.py](https://raw.githubusercontent.com/ikostrikov/implicit_q_learning/master/learner.py).

AWAC instead estimates `Q^π` with a policy-action Bellman backup and uses `A^π(x,a)=Q^π(x,a)−E_{b~π}Q^π(x,b)`, then minimizes `−E_D[exp(A^π/λ) log π(a|x)]`. Here λ is a **temperature**, reciprocal in role to IQL β. Author code has multiple normalization/entropy/ablation switches; its usual batch softmax of `A/β` is not IQL's cap-100 rule. Select and register one exact objective; copying the generic trainer defaults does not specify a faithful AWAC experiment. [AWAC paper, Eq.13 and Algorithm 1](https://arxiv.org/html/2006.09359), [author-linked trainer, train_from_torch](https://raw.githubusercontent.com/vitchyr/rlkit/master/rlkit/torch/sac/awac_trainer.py).

## Stage data and time contract

These are application requirements/inferences, not empirical Stage findings from this review.

1. **Use the physical clock.** For a hold lasting `k` native steps, use `Γ=γ_native^k` and `R=Σ_{j=0}^{k−1}γ_native^j r_j`, including the actual shortened terminal interval. Equivalently fix `Γ(Δt)=exp(−ρΔt)` with correspondingly discounted interval rewards. Reuse the registered Stage objective/units; do not silently set `.99` per recorded row or discount an already discounted interval twice. The duration-dependent bootstrap is standard SMDP bookkeeping; it introduces no options, hierarchy, planner, or waypoints here. [Sutton, Precup & Singh, 1999, SMDP learning](https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf).
2. **Separate episode boundary from terminal mask.** GOAL/CONTACT are true task terminations. An inherent finite task deadline is also terminal (`m=0`), with causal remaining time represented for a Markov finite-horizon formulation. Administrative wall-clock/storage cancellation is truncation, not failure: bootstrap only if a genuine final observation and coherent continuation objective are available; otherwise exclude the incomplete target and report censoring. Never use reset observations as `x'` or bridge episode/teacher-reset boundaries. Official IQL deliberately separates trajectory splitting from `1−terminals`; this does not decide what Stage TIMEOUT means. [Gymnasium time-limit contract](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/), [IQL dataset_utils.py](https://raw.githubusercontent.com/ikostrikov/implicit_q_learning/master/dataset_utils.py).
3. **Keep histories causal and compatible.** Reconstruct each current Actor history from the recorded observations/actions/clock/reset boundaries; a previous Actor's hidden vector is not the new network's recurrent state. Do not merge incompatible goal, barrier, horizon or reward versions into one unlabeled task. Retain failed trajectories for value learning. Teacher/mixed-control data may be offline replay; they are not PPO on-policy samples. Exact replays share one data family and cannot count as independent successes.

## Reliable Actor targets and the first useful test

The likelihood/regression target is the **recorded action actually associated with that transition**, including learner prefix actions in mixed-control trajectories. It is not a teacher action queried but never executed, an old Actor mean, sampled latent noise, or a critic-maximized invented action. Separate unexecuted expert queries from transition replay. With fixed-variance unsquashed Gaussian likelihood, weighted MSE is equivalent up to constants; with clipping or tanh it generally is not. Official IQL actually chooses a Gaussian with tanh-bounded mean and clips sampled execution actions. For Stage's executed-domain clipped distribution, preserve its interior density plus boundary tail-mass contract; do not invert boundary actions with `atanh` or pretend all boundary samples have ordinary Gaussian density. [IQL actor.py](https://raw.githubusercontent.com/ikostrikov/implicit_q_learning/master/actor.py), [learner.py](https://raw.githubusercontent.com/ikostrikov/implicit_q_learning/master/learner.py), [policy.py](https://raw.githubusercontent.com/ikostrikov/implicit_q_learning/master/policy.py).

Before any new launch, register a fixed dataset/family split, critic/Actor capacities and initialization, reward/time/mask contract, τ/β and cap, exact train/shadow/audit counts, retention rule, simulator/bytes/wall-clock ceilings and stopping gates. Report per-family success/CONTACT/timeout ordering, held-out Bellman/return errors, weight quantiles/cap fraction/effective sample size and old-task retention. Low TD error alone cannot establish useful action ranking. Exponentiation can amplify critic errors; clipping does not guarantee retention, and failed-trajectory prefixes can have useful actions. Compare the one predeclared weighted extraction against matched unweighted BC so another longer fit is not mistaken for an IQL mechanism result. A single Gaussian/mean can still average distinct safe modes into an unsafe command. The final decision must use autonomous closed-loop GOAL and existing old-task gates with the same named Actor; teacher success, training loss and a selected trajectory are insufficient. These are proposed falsification checks, not a registered budget or claim that they have passed.

Scope of source reading: IQL method/analysis/evaluation sections and listed original implementation files; AWAC method/derivation and listed original trainer portions; cited time-limit and SMDP definitions. No claim to a complete repository reproduction, exhaustive literature search, complete historical-chat review, or algorithm originality.
