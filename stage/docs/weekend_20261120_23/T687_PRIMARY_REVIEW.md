# T687 primary review — trainable scope and recurrent gradients

2026-11-21. **Admit one bounded zero-native, zero-optimizer contract; conditionally admit a separately registered one-step scope pilot.** This establishes a legitimate learning implementation, not an innovative algorithm or an explanation already proved.

## Evidence and falsifiable question

[T686](t686/FINDINGS.md) directly observed old-task stochastic safety executed4/16, self5/16 versus baseline7/16. Positive surrogates and high finite ESS did not predict several actual paired outcomes. Neither more OPE nor restoring rejected candidates addresses that result.

Recent [T678](t678/PLAN.md) updates only11,906 adapters. [T665 `features()`](../../experiments/t665_remaining_verification/t665_learning.py) computes normalized observations, encoder, GRU and head under `no_grad`; cached features are valid for that frozen scope. Merely enabling backbone `requires_grad` leaves this learning path disconnected. The distinct hypothesis is that allowing the existing representation and recurrent dynamics to adapt permits a useful direction unavailable to adapter-only optimization. No evidence yet establishes representation insufficiency or that more trainable parameters will preserve old behavior.

## Primary sources and counterevidence

[Heess et al., 2015, §3/Algorithm1](https://arxiv.org/pdf/1512.04455) train recurrent continuous-control policies through complete histories with BPTT. This supports differentiating recurrent state construction. Their learned-Q actor-critic algorithms, tasks and LSTM differ from our critic-free episodic score estimator and GRU; their results do not validate our objective.

[PPO, v2 §3](https://arxiv.org/pdf/1707.06347) defines the clipped per-action probability-ratio surrogate. It does not turn a sampled surrogate into deterministic retention. [Pleines et al., 2022, §§3.1–3.2](https://arxiv.org/html/2205.11104v1) specify episode splitting, correct sequence-start hidden states and padding masks. Their [author reference model](https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt/blob/main/model.py) reshapes recurrent sequences and initializes hidden state; [buffer](https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt/blob/main/buffer.py) preserves sequence order. Their truncated-BPTT option is deliberately excluded: complete-episode differentiation is this contract. Inspected live main source, MIT license; no code copied, commit pin unavailable.

[Kumar et al., ICLR2022 §§3–4](https://arxiv.org/pdf/2202.10054) show full fine-tuning can distort pretrained representations and worsen OOD performance. [Biderman et al., 2024 §§4.1–4.3](https://arxiv.org/html/2405.09673v1) find LoRA learns less and forgets less than full fine-tuning. Vision classification and language-model LoRA are analogies, not evidence about these nonlinear adapters or navigation. Locally, [T210](../results_20261031_full_parameter/findings.md) already produced first-step analytic KL≈.131 and old8 degradation to2/8 and3/8. Its architecture/data differ; it warns against unrestricted first steps rather than disproving current full-scope learning. Viewed cited sections and source, not an exhaustive literature review.

## Contract and admission

Use the identical T501 single120/GRU64 Actor, all existing weights, initial function and T672 sigma.05 complete external histories. Keep T678's exact407 task-balanced LOO terminal safeGOAL−collision objective, per-action ratio clip.2, scale512, plus10 times identical387 task-balanced executed-command hinge/square references divided by.2². No new module, critic, reward, teacher or routing.

Compute locations via autograd-enabled `q.sequence(states)` from fresh h0 for every complete episode, including reference histories. Never detach intermediate carry, reuse carry across episodes, or cache trainable features. Stream whole trajectories, accumulating their exact task/episode weights; padding must be masked. Retained targets remain recorded executed commands, independent of changing predictions.

Require finite full gradients, nonzero backbone/GRU norms, initial locations/densities matching recorded evidence, and **pre-clipping** adapter-gradient slices matching T673/T677 within1e-10. Verify three fixed full-GRU directional derivatives through entire sequences, epsilon1e-6, atol1e-8+rtol1e-4; test reset isolation and restore all state exactly. Record fixed ordering/dtypes and component gradients. Opening a gradient path is not proof of useful features.

Separately calculate the fresh-Adam first-step formula using T678's lr3e-5, eps1e-8, normclip1, with no optimizer call; temporarily represent then restore it. Full-scope clipping changes adapter displacement, so only preclip gradients must match. Semantic PASS and step admission are separate: admit only finite displacement and maximum task-mean latent KL≤.01 on the same44-old+24-field panel. Failure ends this fixed proposal without LR/seed rescue.

Measure bounded runtime/peak memory before reservation. If admitted, register one full-scope step with shadow/independent reconstruction, fixed screen and conditional complete407, using T678 executed as the reused matched control. Same optimizer settings do not match function-step size: Adam geometry is part of this scope intervention. No capacity, safety, generalization or novelty conclusion follows from one candidate; failed admission or screen stops this line.
