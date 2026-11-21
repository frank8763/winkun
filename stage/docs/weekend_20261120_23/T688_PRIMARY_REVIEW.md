# T688 review — optimizer geometry is distinct, but not the next admitted experiment

2026-11-21. **Reject the proposed norm-matched SGD pilot as the next priority. No T688 learning/diagnostic job, code, registration or launch is authorized by this note.** This is a prioritization decision, not evidence that SGD cannot work. T501 remains retained. The stopped fixed full-scope proposal stays stopped: [T687 actual RESULTS](../../runs/t687_full_actor_gradient_contract/RESULTS.json) passed complete64772-parameter BPTT semantics but its fixed first-Adam formula gave max panel latentKL .033027652947>.01 (101/407 tasks over .01), with zero optimizer/native calls. No learning-rate rescue.

## What would actually differ

A scientifically interpretable optimizer comparison would return to **the original frozen-backbone11906 T678 executed arm**, keeping its T501 initial state, all1724 T672 histories, task/episode weights, terminal safeGOAL−collision LOO objective, scale512/ratio clip.2, all387 executed-action references, reference weight10/.2² and sigma.05. It would change only the update map. It must not inherit T687's64772 scope or use self targets. [T678 plan](t678/PLAN.md), [actual learning source](../../experiments/t678_reference_target_pilot/t678_learning.py), [executed summary](../../runs/t678_reference_target_pilot/executed/summary.json).

For the initial joint **loss** gradient g, fresh Adam is δA=−3e−5 g/(|g|+1e−8); g's norm is below1, so norm clipping is inactive. The proposed alternative is δS=−||δA_actual||₂ g/||g||₂, one no-momentum/no-decay SGD step. The normalization is deterministic, anchored to the original represented step, not a searched learning rate. It preserves total parameter-step norm, not per-tensor norm, function drift, KL, task performance or effective objective weights. Coordinate magnitudes remain relevant when |g| is near epsilon; “Adam equals sign” is only an approximation.

Read-only arithmetic on the already stored [T673 gradients](../../runs/t673_all_task_signal_gate/gradients.npz) and [T677 gradients](../../runs/t677_reference_target_contract/gradients.npz), using g=−mean(task_gradients)+10·executed_retention_gradient, gives:

| Quantity | Value |
|---|---:|
| Joint ||g||₂ | .017358705787609386 |
| Original represented Adam ||δA||₂ | .003265833889993327 |
| Deterministic SGD coefficient | .1881380979637591 |
| cos(δA,δS) | .2346890243 |
| g·δA / g·δS | −1.3304673e−5 / −5.6690650e−5 |
| Squared-step energy in final66 adapter parameters, Adam / SGD | .5568% / 93.1021% |

No new gradient, candidate representation, forward evaluation, optimizer call or simulation was performed for these numbers. The stored analytic Adam norm differs from the actual represented norm by about2.6e−18. The last66 parameters are the two32-weight output layers and their two biases. Concentration suggests a materially different function step; **we did not measure or predict its KL**, so neither admission nor failure is claimed. Cauchy–Schwarz makes −g the steepest local decrease of this sampled loss at fixed Euclidean norm. It gives no closed-loop preservation result.

## Primary support and counterevidence

Fresh web research on2026-11-21 inspected [Kingma–Ba, Adam v9, Algorithm1/§2](https://arxiv.org/pdf/1412.6980v9): zero moments and bias correction yield the first-step formula above. Its experiments also support Adam on noisy/sparse objectives; there is no general reason to prefer unpreconditioned gradients here.

[Balles–Hennig, ICML2018 proceedings, §§1.1,2,6](https://proceedings.mlr.press/v80/balles18a/balles18a.pdf) directly separates sign and variance adaptation, supporting geometry as a real variable. Counterevidence is in the same paper: sign descent helps some axis-aligned ill-conditioned problems, and its empirical usefulness depends on the task. Their individually tuned, momentum-based multi-step classification/language experiments are not one-step recurrent navigation or old-task preservation. [Wilson et al., v2 §§2–4](https://arxiv.org/pdf/1705.08292v2) supplies adaptive-versus-SGD generalization counterexamples, not a guarantee that SGD retains existing trajectories. Neither paper matches our local sensors/vehicle,2ms physics, causal120/GRU64 history, clipped continuous commands, episodic outcome, fixed development tasks, or absence of planner/teacher/Q. No robot benchmark was claimed.

Inspected the tagged first-party [PyTorch v2.5.1 SGD source](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/sgd.py) to check the no-momentum/no-decay update semantics; [BSD-style license](https://github.com/pytorch/pytorch/blob/v2.5.1/LICENSE). This is a source reference, not a claim that the local runtime isv2.5.1. No external code was copied. Paper versions/sections above were checked; no exhaustive search or complete reading of every cited work is claimed. The mutable stable documentation redirected to2.15, so the tagged source is the versioned reference.

## Local prior failures and decision

SGD is not new to Stage. [T627 plan](../day_20261117/t627/PLAN.md) used plain SGD lr.001/clip1 against CAGrad on9 tasks,8 steps, a different T608 representation/data/RTG objective and33 references. It did not match original Adam displacement. [Actual findings](../day_20261117/t627/FINDINGS_AND_NEXT.md): meanSGD gained2/lost2; CAGrad gained0/lost1, despite favorable sampled first-order changes. [T628/T628R](../day_20261117/t628/PLAN.md) also used SGD in fresh-data/noise comparisons. [T632](../day_20261118/t632/PLAN.md) used64 supervised SGD steps and still lost an old task; [T633's recap](../day_20261118/t633/PLAN.md) records this. [T580](../day_20261116/t580/PLAN.md) already compared effective SGD step sizes in supervised MER, although it did not norm-match Adam. These are relevant warnings, not exact repetitions of the proposed single step.

The exact direction comparison is therefore legitimate and differentiating, but its positive rationale stops at **improving a sampled joint surrogate**. [T678](t678/FINDINGS.md) lost old tasks even with smallKL and positive surrogates; [T686](t686/FINDINGS.md) measured executed4/16 and self5/16 old stochastic safety versus baseline7/16. Better gradient alignment, high finite ESS and better proxy values have not supplied the missing retention evidence. Parameter-norm matching can additionally concentrate the step into sensitive outputs. Another fixed-KL admission-only failure would spend the next slot without testing a stronger closed-loop mechanism. Do not shrink the norm, switch scope or search seeds to rescue this proposal. No NEXT_T688 experiment spec is produced.

One substantively different direction to consider is **changing the old-task rehearsal state distribution to verified, update-induced student deviations**, rather than changing the optimizer on the same initial-policy histories: this addresses feedback-induced distribution change directly. It is only a research lead, not an admitted collection or learning job. [T668](t668/FINDINGS.md) already found frozen-T501 recovery on only2/4 deviations and failed its3/4 gate; selecting those two cannot silently bypass that failure. A new proposal must establish a concrete difference from T631/T632 and existing recovery work before spending data or optimizer budget. Likewise, “finite-step proximal retention” is not new without distinctions from T606/T606R/T607/T608/T614/T616, T660 and T671. This review does not invent an alternative experiment merely to keep launching.
