# T683 primary review: temporal latent-action density

2026-11-21. **Admit only the bounded zero-native/zero-optimizer contract.** [T682](t682/FINAL_REPORT.json) is COMPLETE/audit PASS/scientific FAIL: four head ablations preserve 0/4, 0/4, 3/4, 0/4 old tasks. T680's four-pair 11,906-dimensional estimator remains disqualified. Neither result proves temporal exploration useful; no parameter/seed/head rescue follows.

[Korenkevych et al., §§4–5, Eqs9–11](https://arxiv.org/pdf/1903.11524) derive stationary Gaussian autoregressive exploration and explicitly differentiate every past/current mean occurrence. This supports the proposed mechanism, not novelty or navigation safety. Inspected author implementation at commit `8bee7f97667dcff5a5240264a74ed70036c23e97`: [distributions.py:19–25/45–53](https://github.com/kindredresearch/arp/blob/8bee7f97667dcff5a5240264a74ed70036c23e97/common/distributions.py#L19) reconstructs past standardized residuals for density/sampling. [arp.py:74–84](https://github.com/kindredresearch/arp/blob/8bee7f97667dcff5a5240264a74ed70036c23e97/arp/arp.py#L74) resets history to zero, creating a transient; T683 deliberately initializes `z0=epsilon0~N(0,I)` instead. Code was read, not executed/incorporated. Counterevidence: [Fujimoto et al., §6.1](https://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf) report no OU exploration benefit over IID Gaussian in their benchmarks; this does not decide Stage.

Let `u_t` be recorded RAW action, `H_t` external causal history, `mu_t=mu_theta(H_t)`, `sigma=.05`, `dt_t=.002*native_counts[t-1]`. Lock tau=2s versus tau=0/IID, with `rho_t=exp(-dt_t/tau)` and tau0 explicitly rho0. Generate `z_t=rho_t*z_prev+sqrt(1-rho_t²)*epsilon_t`, `u_t=mu_t+sigma*z_t`; only the environment clips. The policy contribution to complete-trajectory log likelihood is

`log N(u0;mu0,sigma²I) + sum_t>0 log N(u_t;m_t,v_t I)`,

where `m_t=mu_t+rho_t*(u_prev-mu_prev)` and `v_t=sigma²*(1-rho_t²)`. This is not the conditional density given the entire future external-state path. For fixed sigma/rho, each later score is

`((u_t-m_t)/v_t)^T*(Dmu_t-rho_t*Dmu_prev)`.

Hold recorded external histories/raw/dt fixed, recomputing each policy's own carry. [Schulman et al., Condition1/Theorem1](https://arxiv.org/pdf/1506.05254) permit likelihood-ratio gradients with nondifferentiable downstream clipping/dynamics. Raw actions are stochastic nodes; clipping is a deterministic pushforward. This validates complete-path REINFORCE under usual integrability conditions. It does not validate an observation-only Markov score. Freezing parameter-dependent GRU carry would omit derivatives; current 11,906-adapter-only scope has frozen backbone/GRU, so its recomputed features are parameter independent at fixed histories. Full-GRU learning requires a separate differentiation contract.

**Clipping prohibition.** Given previous raw/history, a scalar `N(m,s²)` clips to lower mass `Phi((-1-m)/s)`, interior Gaussian density, upper mass `1-Phi((1-m)/s)`; their sum/integral is one. [Fujita–Maeda, §§3.1/3.5](https://proceedings.mlr.press/v80/fujita18a/fujita18a.pdf) establish this mixture. Their tail-constant continuation argument does not transfer automatically: raw magnitudes surviving as AR memory affect future actions. Do not replace raw-path scores with products of clipped conditional atoms.

Our two-step scalar counterexample fixes `mu0=mu1=1`, sigma=.05, rho=exp(-.1):

`P(c0=c1=1)=integral_1^infinity N(u;1,sigma²)*Phi(rho*(u-1)/(sigma*sqrt(1-rho²)))du`

`=1/4+asin(rho)/(2*pi)=.4300038747`.

Plugging boundary raw0=1 gives `.5*.5=.25`. A command atom leaves a truncated latent posterior, not one raw representative. Require independent quadrature, including a nonboundary plug counterexample.

Stationarity requires an exogenous clock: check every source `native_counts[:-1]==100`; terminal shortening has no next action. Adaptive counts can destroy stationary Gaussian marginals. Even with fixed clock, sigma denotes unconditional latent stationary scale, not noise conditioned on histories/survival; innovation SD is smaller.

Contract checks actual frozen `Actor.step` wrapper, IID equality, prefix causality/reset, sigma/physical-dt recurrence, mixture normalization, counterexample and analytic/three-direction numeric gradients on real T672 histories; independently reconstruct/hash all arrays. Ceilings: 600s main+600s audit, 1800s root, 32MiB, CPU14, zero GPU/native/optimizer. Old IID records are fixtures, never AR on-policy data. PASS allows only separately registered fresh/exact collection with tau/sigma behavior hash, own-carry reset, actual-dt metadata and independent density audit; collector integration remains unverified. No PPO, Q, teacher, router, new parameters, outcome-weighted experiment, scientific exploration claim or safety claim.
