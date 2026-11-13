# T519：全 Actor 新鲜 on-policy 适应的一手依据与边界

日期：2026-11-13。仅 Stage 文献审阅；本任务未启动原生回合、训练或评估，未修改实验源码。下述方案是待有界验证的工程 pilot，不是原创算法、整体无偏性证明或安全证书。

## 判断

从当前接受的 single120 causal64GRU、field11/24 + old8/8 出发，重新采集固定两任务各四条独立噪声轨迹，允许现有 Actor 的 encoder/core/head 及原适配器共同接受回报驱动更新，是有文献依据的下一项能力检验。它改变了连续离线 BC 的学习信号与可训练空间；论文没有证明八条轨迹足够，也没有证明全网络一定优于残差头。T515–T518 的失败不等于端到端 RL 已被否定。

`.002 old-locationRMSE` 是存储历史输入处的输出距离代理。它既不是旧任务原生成功的定义，也未被本项目证明是保持成功的必要条件。将最终保留依据改为原始起点 old19 逐 ID 原生闭环结果与完整 field24，能够直接回答本轮要保留的行为问题；这仍只是有限已知案例的经验验收，不能推出泛化、安全或 old275/six/eight/fresh 全部保持。父模型独立锁定与失败回退用于保存已接受资产。

项目事实依据：[当前研究状态](../RESEARCH_STATUS.md)、[T518 最终报告](T518_FINAL_REPORT.json)、[T509 最终报告](../review_20261112/T509_FINAL_REPORT.json)。本段为项目证据的解释，不冒充论文结论。

## 阅读的一手来源

| 来源与实际阅读位置 | 支持什么 | 限制与本轮关系 |
|---|---|---|
| Schulman et al., **PPO** (2017)，§2–5、式(1)(7)、Algorithm 1：[全文](https://arxiv.org/pdf/1707.06347) | 交替进行自身策略交互采样及有限次 surrogate 优化；裁剪目标为 `min(r*A, clip(r)*A)`。Actor 参数没有必须只限小残差或输出头的要求。 | 原论文给出的主要实现是 actor–critic，通常使用学习的 V 估计优势。本轮 MC/同行基线、Value 不更新属于改编，称“critic-free clipped policy-gradient pilot”更准确。裁剪的下界针对未裁剪 surrogate，不能解释成真实任务回报下界。 |
| Wang et al., **Truly Proximal Policy Optimization** (2020)，§3：[全文](https://proceedings.mlr.press/v115/wang20b/wang20b.pdf) | 分析并举例说明 PPO 的普通裁剪不能严格约束概率比，也不能自动实施严格 trust region。 | 本轮不能用 clip、较小学习率或样本 KL 代替旧任务原生验收，更不能承诺绝不遗忘。这里未引入该文的新算法。 |
| Pleines et al., **Generalization, Mayhems and Limits in Recurrent PPO** (2022)，§3、§5.3、§6.4、Algorithm 1：[全文](https://arxiv.org/pdf/2205.11104)；作者 [trainer.py](https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt/blob/main/trainer.py)、[model.py](https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt/blob/main/model.py) | 明确强调序列布局、序列起点 hidden state、episode reset、padding loss mask。作者代码将编码器、GRU/LSTM、policy 分支组成可微模型，`loss.backward()` 后更新模型参数，而非只能训练最后一层。 | 作者使用 TBPTT；实验以像素/离散动作任务为主，不能当作本车连续控制的直接复现。刷新 hidden state 的实验未得到显著收益，因此“本轮从完整历史重算”是确保当前参数条件密度一致的工程选择，不应宣传成已证性能改进。 |
| Greensmith, Bartlett & Baxter, **Variance Reduction Techniques for Gradient Estimates in RL** (2004)，§3.5、§5、Theorems 10–11：[全文](https://jmlr.org/papers/volume5/greensmith04a/greensmith04a.pdf) | baseline 是控制变量；最小方差 baseline 受 score 大小加权，简单平均回报乃至真实 value 都未必最优。由此不能认为必须先训练新 critic 才能更新 Actor。 | 论文分析的是特定 GPOMDP/平均回报设定，不直接证明本轮“物理时间对齐三同行 RTG”最优或普遍降方差。leave-one-out、peer 独立性、终止后取值及时间对齐规则均须明确。 |
| Ross, Gordon & Bagnell, **DAgger** (2011)，§1–3、Theorems 2.1–3.2：[全文](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf) | 动作会改变后续输入分布；固定 expert 分布上的监督误差与 learner 实际访问分布不同。支持不能仅靠存储位置的回归误差判定闭环行为。 | DAgger 在 learner 访问状态上查询专家标签；本轮使用自身回报，不是 DAgger。该理论不证明本次 offline gate 失败的唯一原因，也没有给出 `.002` 的必要阈值。 |
| Rolnick et al., **CLEAR / Experience Replay for Continual Learning** (2019)，§2、式 policy/value cloning、§4–5：[全文](https://arxiv.org/pdf/1811.11682) | 将新 on-policy 经验用于适应，旧 replay 的 off-policy 更新与行为克隆用于稳定；论文以任务表现检验遗忘。其稳定性/可塑性冲突与本项目保持问题相关。 | 克隆是训练正则项，不是必须逐状态近乎完全一致的原生成功定义；论文也讨论保护旧分布可能妨碍适应。本轮没有复现 CLEAR，去掉紧输出硬门槛也不保证保持。 |
| Haarnoja et al., **SAC** (2018)，§4–5：[全文](https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf) | SAC 是 off-policy maximum-entropy actor–critic，Actor 更新依赖学习的 soft Q。 | 新 SAC 路线需要另立 critic/target、replay、估值与更新预算，不能把 frozen Value 加 on-policy RTG 更新称为 SAC。 |
| Ball et al., **RLPD** (2023)，§4.1–4.5、Algorithm 1、§6 ablations：[全文](https://proceedings.mlr.press/v202/ball23a/ball23a.pdf)；作者 [sac_learner.py](https://github.com/ikostrikov/rlpd/blob/main/rlpd/agents/sac/sac_learner.py) | RLPD 混合在线/离线采样，使用 critic LayerNorm、ensemble 与多步更新；作者 `update_actor` 通过 critic Q 优化 Actor，`update_critic` 学 bootstrapped targets。它提供另一条利用历史经验的成熟路线。 | 这是 SAC 系 off-policy 机制，需要新估值部件；不能据其基准成绩认定本项目应立即替换现有 Actor。当前 pilot 先检验现有策略能否从自身新鲜回报适应，避免把 critic 学习同时变成未知变量。 |

## 对当前具体方案的推论（工程解释）

1. **更新整个既有 Actor 有合理通路，但收益未知。** 新鲜回报可以调整感知特征、记忆递推及输出，允许固定表示/有界残差不能表达的变化；这一优势只是可训练函数空间更大，不是本项目“瓶颈已定位”的证明。原适配器成熟 Adam 状态保留，首次解冻 encoder/core/head 使用 fresh 参数组，必须记录各组学习率、step 与 moment 来源；这是 warm-start 优化器方案，不是所有参数同历史 Adam 的精确续训，也未被上述论文验证为最优。

2. **密度必须对应实际采样器和因果历史。** 固定 behavior snapshot 与存储 `logp_behavior`；每次计算 numerator 时，用当前 Actor 从该 episode 的合法初始 hidden state 顺序处理完整已存因果输入历史，得到当前 `logp_theta(action_t | history_t)`。不可逐样本清零 GRU、跨 episode 混 hidden state，或在 core 已变化时把旧 hidden state 当作当前 hidden state。若仍采用 AR/相关噪声，必须沿用实际条件创新密度及动作变换合同，不能换成独立 Gaussian。PPO 的逐步 conditional ratio 不是整轨迹 importance ratio；不宣称已经精确修正更新后的状态访问分布。此处是 likelihood 一致性要求，不需对模拟器动力学求导。完整历史前向密度也不等于完整历史 BPTT：若为计算量对 hidden state 分块 detach，须明确梯度已截断。

3. **RTG 与三同行基线不依赖新 critic。** 每任务四条独立探索 rollout 可以构造排除自身的三同行物理时间 RTG 基线；完整奖励、折扣、终止、peer 终止后的值、插值/查找及有限时限处理需预先固定。不能用自己的 action/return 拟合自己的 baseline 后仍无条件声称无偏，也不能只挑成功 rollout 再称原分布无偏 PG。即使基线合法，PPO 裁剪、多次复用、标准化、接受/拒绝及小样本也使整体过程不能简单宣称无偏。

4. **保留与新增分开核验。** 八条探索中的 qualified GOAL 只说明该 stochastic policy 曾到达。学习后必须由同一个冻结 Actor 在原始起点重新闭环测试，逐 ID 保留 old19，并报告完整 field24 的 GOAL/contact/timeout/invalid/censored；已有成功集合与新增成功集合均需明示。精确重生是审计，不算独立样本。离线 RMSE/KL 可以继续报告作诊断，但没有论文允许据此替代原生闭环结果。

5. **Value 冻结须说清对象。** 如果 Value 与 Actor 共享被解冻的 encoder/core，冻结 Value head 和其 Adam 仍会改变 Value 的函数。需确保 Value 模块/输入依赖独立，或准确称为“Value 参数不更新”；本轮不使用它生成优势时，应明确这一点，不能把旧 V 的存在写成 critic 已验证。

结论：文献支持把“新鲜自身交互 + 整个既有 Actor 的有限回报驱动适应 + 原生逐 ID 保留验收”作为可证伪的新 pilot。它没有直接证据保证任务突破；若失败，应区分探索未覆盖、优势/密度实现、参数更新未产生有效行为、真实保持失败及预算删失，保留全部失败证据，再决定下一机制。
