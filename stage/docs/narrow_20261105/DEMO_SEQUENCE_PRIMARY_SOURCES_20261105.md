# Demonstration transfer: primary sources and one bounded next mechanism

Checked2026-11-05. Stage only. Literature/design note; no experiment registration, native/optimizer calls, source/protocol/run changes, commit or synchronization. T368 is active and its final outcome is unknown here. No Phase research consulted.

## Stage evidence

Current architecture already has true recurrent memory:120 causal inputs→Linear32/Tanh→GRU16→two-output residual, frozen T255 base, carried hidden and current-parameter full-history BPTT;6306 trainable parameters. Supervision still targets one current action pair. [Implementation](../../experiments/t348_causal_sequence_warmstart/t348_core.py), [T368 plan](T368_PLAN.md).

T361 has295 human actions; T364 has20 learner-prefix actions without human labels plus277 human actions. These are two correlated routes, not572 independent recovery demonstrations. [T365 protocol](../../experiments/t365_correction_aggregation/protocol.json).

[T362](T362_FINDINGS.md) fitted demonstrations yet remained0/5 autonomous. [T365](T365_FINDINGS.md) reached2/5 fixed0.70m/80s development successes, centre collision, wide5/5→0/5 and old20=10/20→9/20. [T366](T366_FINDINGS.md) found incomplete factual-wide-history preservation. [T367](T367_FINDINGS.md) had eight fresh centre contacts, two rejected proposals and zero retained updates. Induced-state shift, missing recovery coverage and task interference are supported hypotheses, not uniquely proven causes. Original0.57m/56s acceptance remains distinct.

## Verified primary methods

**DAgger.** Algorithm3.1 executes learner/expert mixtures, asks the expert for actions on visited states, aggregates labels and retrains. Its stationary deployed policy need not contain a teacher. Existing successful tapes do not label arbitrary learner states. No-regret/reduction guarantees require the stated assumptions; action MSE alone does not certify collision avoidance. More DAgger requires additional expert labels, unavailable under the present assumption. [Ross et al., AISTATS2011, §§2–4](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf).

**DART.** Noise is injected during supervisor demonstrations; the supervisor responds to disturbed states, providing recovery examples. Noise covariance is estimated/scaled iteratively. Its theory includes assumptions about the noise-injected supervisor; the authors acknowledge humans can change behavior under noise. Perturbing stored labels does not create valid recovery demonstrations. T368 perturbs the learner without new expert labels, so it is not DART. DART is deferred without additional expert operation. [Laskey et al., CoRL2017, §§3–4](https://proceedings.mlr.press/v78/laskey17a/laskey17a.pdf).

**ACT.** It predicts action chunks, uses a Transformer/CVAE and queries every timestep when temporal ensembling. The ensemble combines overlapping predictions for the *same* execution timestep, not previously executed commands. The paper's exponential indexing starts at the oldest prediction; its positive coefficient does not favor newer predictions. The training-only CVAE encoder is discarded and latent style set to zero for inference. These components exceed ordinary deterministic sequence regression. [Zhao et al., RSS2023, §IV](https://arxiv.org/pdf/2304.13705). Authors' examples use50 demonstrations/task, RGB/joints and50Hz joint-position control, unlike two Stage routes at5Hz. [Project](https://tonyzhaozh.github.io/aloha/). Code confirms timestep queries and masked L1+KL; paper Algorithm1 instead labels reconstruction MSE. [Inference](https://raw.githubusercontent.com/tonyzhaozh/act/main/imitate_episodes.py), [loss](https://raw.githubusercontent.com/tonyzhaozh/act/main/policy.py).

**Diffusion Policy.** Predicts a longer action sequence, executes a specified shorter horizon then reconditions. It retains behavior-cloning limitations with inadequate demonstrations and higher inference cost. Defer this larger architecture/objective/inference change. [Chi et al., RSS2023 v4, §II-C/§VIII](https://arxiv.org/pdf/2303.04137v4).

## Smallest feasible next test

After T368 closes, separately register one matched pair: current-action supervision versus current plus four future action pairs, using existing human data and identical complete factual-success rehearsal. Keep current120-input GRU, plant and0.2s decisions. Five command intervals cover1s; target times are t through t+0.8s. Execute only the first and recompute every0.2s. **This is multi-horizon auxiliary supervision: no ACT temporal ensemble or fivefold decision-horizon reduction.**

Linear16→10 adds136 parameters, totaling6442. Preserve current residual semantics/initial output exactly; future absolute-action heads use current hidden only. Never evaluate the frozen base on unavailable future observations. Both arms should have identical head allocation/data/schedule, with future loss disabled in control; cap each at600 live updates, one fixed horizon/weight, no grid. A full later protocol must lock shadow/audit/native/wall/storage budgets.

Mask shifted targets by actual label and terminal boundaries; T364prefix20 never becomes human supervision. Unroll factual history from time0. Hold all five wide/two narrow success histories and old references equal across arms.

Possible benefit is learning imminent timing. Limits: no direct runtime smoothing, no new recovery labels, no joint multimodal model; future expert commands reflect unavailable future feedback. Extra MSE targets can average modes or harm current precision. Require autonomous centre improvement, actual success-ID retention and saved-model readback; lower loss is insufficient. Stop if unhelpful. No originality or six-metre guarantee.

Reading was targeted algorithm/assumption/inference/limitation sections and cited author code, not an exhaustive survey. No author implementation was executed.
