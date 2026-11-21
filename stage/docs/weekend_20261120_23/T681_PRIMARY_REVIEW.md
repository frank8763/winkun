# T681 primary review — frozen variant transfer

2026-11-21. Recommend one separately registered, zero-optimizer characterization, conditional on exact preparation bounds and independent audit. T680's gate remains failed: split cosine .0154034, minimum deletion cosine .454311, and only one field task supported by two nonzero pairs. No training, additional pairs, scale rescue or promotion follows. [Evidence](t680/FINDINGS.md).

The distinct question is whether **both exhaustive positive-source T679 variants**, direction00_minus and direction03_plus, succeed beyond their four discovery field tasks. T679 established isolated variant successes and old-task damage, while T501 stayed unchanged. Mapping their fixed task sets can distinguish discovery-only behavior from wider descriptive transfer. It becomes another seed grid if outcomes trigger replacement models, added directions, or adaptive acquisition. [Original evidence](t679/FINDINGS.md).

## Primary support and counterevidence

[Plappert et al., ICLR2018/arXiv v2, §3 pp3 and AppendixA.2 pp12–13](https://arxiv.org/pdf/1706.01905) keep Gaussian-perturbed parameters fixed throughout a rollout; their learned DDPG/TRPO experiments support coherent exploration, not retained-center improvement from an individual perturbation. The [author Baselines implementation](https://raw.githubusercontent.com/openai/baselines/master/baselines/ddpg/ddpg_learner.py), inspected `master` 2026-11-21, functions `get_perturbed_actor_updates`, `adapt_param_noise`, `reset`, perturbs copies and resets exploration after episodes. Its constructor explicitly lacks recurrent support. T681 therefore needs its own variant's causal GRU carry from reset; the code supplies no recurrent equivalence guarantee. No author code is incorporated.

[Salimans et al., arXiv v2, §2 pp2–4](https://arxiv.org/pdf/1703.03864) optimize expected return under a parameter distribution, using antithetic sampling. This foundational distinction counters interpreting one successful member as evidence for the distribution mean, a usable four-pair gradient, or the unmodified center.

[Cobbe et al., Procgen report, introduction/§2 pp1–3](https://cdn.openai.com/procgen.pdf) motivate diverse distributions and separate train/test levels because RL can overfit repeated environments. Deterministic transitions alone do not create independent task evidence. Recent [Chen et al., 2025 preprint v1, §§2.1–3.1.1, Tables1–2/Figure2](https://arxiv.org/html/2510.10541v1) report weak diagnostic separation on reasoning benchmarks and aggregate scores masking cross-difficulty failures. This is LLM evidence, not a navigation effect-size estimate; it supports scrutinizing evaluation construction. Neither source validates Stage's sensors, native action timing, reward, boundary or termination contract.

## Lock and report

Freeze both model SHAs and the canonical24 task manifest before execution. Reuse each model's original4 field records by SHA; run only its20 previously unrun field tasks, full+exact: **40 fresh +40 exact**, without rerunning consumed cases. Preserve seed=None deterministic inference, variant-owned GRU, original starts/horizons and unchanged success/contact/boundary rules. Compare per-ID against T501's existing complete407 original-start provenance; verify task, scene and reset identity before pairing.

Publish both models' complete24-ID tables, separately marking discovery4 and postselection20; include GOAL/contact/timeout/boundary/censoring, completion denominators, T501 outcome, gained/lost/overlap sets, and each model's old4 known outcomes. No best-model selection or union controller. Known retention failures already reject retention: full407=null, not zero success.

Descriptive expansion requires **each** model to add ≥1 safe success on a previously unrun field ID outside original384 and discovery4. Failure denies that criterion only. These20 tasks are a postselection untested-task transfer panel, not untouched statistical holdout or generalization proof: scenes may correlate; N=2 selected variants is not IID. Exact replay verifies reproducibility and adds no independent sample.

Before registration, prove worst-case physics/storage and command accounting within proposed5M physics/90 commands/256MiB, 2400s main+300s audit/root3000s. Lock all source/raw/model SHAs; independent audit must reconstruct all arrays and integral state. No new models, optimizer, teacher, router, novelty or safety claim; preserve every failure and T501.
