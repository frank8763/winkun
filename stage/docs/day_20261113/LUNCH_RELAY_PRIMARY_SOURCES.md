# 午间有限接力：联合任务 RL 与成功轨迹整合的一手依据

2026-11-13。仅 Stage；本笔记未启动模拟器、优化器、评估或模型处理，未修改 experiments/runs。它为午间离线接力提供文献与解释边界，不替代正式注册协议、容量准入和绝对截止（北京时间 13:17）。不得以剩余时间为理由无限加轮、换 seed 或继续旧实验。

## 当前问题与证据范围

接受资产仍是 single120 / causal64GRU 的 field11/24 + old8/8。T519 的新鲜全 Actor on-policy 更新得到研究候选 field12/24：保留原11个成功 ID，新增 `sector0_offset2`，但 `dev_task6_13` 超时，故整体拒绝并回退。见 [T519完整报告](T519_FINAL_REPORT.json)、[T519复核与下一步](NEXT_AFTER_T519_DRAFT.md)。

本次委托时提供的 T520 暂态证据为：只对旧 `dev_task6_13` 采四条新鲜轨迹（1 GOAL、3 contact），候选已恢复旧8项，但新增 `sector0_offset2` 超时；最终原生核验仍在进行。因此这里不把 T520 写成已完成、已接受或已完成远端备份。正式设计见 [T520协议](T520_PLAN.md)、[T520来源边界](T520_PRIMARY_SOURCES.md)，最终状态须由主管线核账确认。

这组局部证据表明：先只更新新任务、再只修旧任务，当前两个候选产生了真实行为取舍。它不证明存在不可化解的梯度冲突，也不证明单网络无法同时成功。下一步同时向两任务提供新鲜回报，比继续交替单任务修复更直接检验联合适应；但“两个成功标签都存在”也不是联合可行性证明。

## 一手来源及反证

| 实际核对来源 | 可支持的判断 | 不支持的外推 |
|---|---|---|
| Yu et al., **Gradient Surgery for Multi-Task Learning**, NeurIPS 2020，§2.1–2.3、Algorithm 1：[论文全文](https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)；作者 [PCGrad实现与说明](https://github.com/tianheyu927/PCGrad) | 明确用任务分布上的平均损失描述联合目标，分析不同任务梯度的冲突、幅值差与曲率；作者实现接受逐任务 loss。支持保留任务分组、避免一个任务由数据量支配。 | 同时训练本身仍可能发生负迁移；等权不是普适最优，4+4轨迹也无成功保证。论文的任务条件模型及 RL 基准不是本车共享动作头的直接复现。这里不新增 PCGrad 或投影变量。 |
| Rolnick et al., **Experience Replay for Continual Learning / CLEAR**, NeurIPS 2019，§3、§5、Appendix A.4：[全文](https://arxiv.org/pdf/1811.11682) | 新经验用于可塑性，历史 replay 及行为克隆用于稳定；论文将保护与适应视为共同问题，也明确讨论保护历史分布可能妨碍适应。 | 原法为 V-trace actor–critic，含 policy/value cloning；本项目 critic-free peer-RTG + soft reference 不是 CLEAR 复现。旧输出应作辅助训练项，不能自动等同于旧任务成功定义或硬必要条件。 |
| Rusu et al., **Policy Distillation**, ICLR 2016，§3.3、§4.4、Appendix A：[全文](https://arxiv.org/pdf/1511.06295) | 分离多个已学会的教师经验，再通过监督训练整合到学生网络，是已有方法谱系；因此有限的成功轨迹 BC 是另一项合理检验。 | 关键限制：原文多任务 Atari 使用每任务 controller head 和 task ID，训练与评估均切换对应输出；不能据其结果声称已证明本项目无路由、共享2动作头必定整合成功。连续动作回归也只是改编，不能称原算法精确复现。 |
| Ross, Gordon & Bagnell, **DAgger**, AISTATS 2011，§2–3、Theorem 2.1：[全文](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf) | 学生动作改变后续观测分布；教师轨迹上的低监督误差不能代替学生自己的闭环评估。 | 此处只用预存成功轨迹，不在学生新访问状态查询专家，因而不是 DAgger；不能把论文的条件性保证转移到本轮 BC。 |
| Schulman et al., **PPO**, 2017，§3、§5、Algorithm 1：[全文](https://arxiv.org/pdf/1707.06347)；OpenAI [PyTorch实现](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py)，`compute_loss_pi` | 自身交互采样、固定行为 logp、有限次 clipped-surrogate 优化有直接来源；官方实现以 `exp(logp-logp_old)` 计算 ratio。 | 官方实现使用学习的 Value/GAE，本项目同行 RTG 是明确改编。PPO 裁剪不修复旧数据的 on-policy 身份，也不提供旧任务保留证书。 |
| Wang et al., **Truly Proximal Policy Optimization**, UAI 2020，§3：[全文](https://proceedings.mlr.press/v115/wang20b/wang20b.pdf) | 分析普通 PPO 裁剪无法严格限制概率比，也不能自动实施 trust region。 | clip、较小 LR、经验 KL stop 均不能保证32项全部通过；不在本次混入该文的新算法。 |

上述来源以作者论文、会议全文及作者实现为依据；没有把二手总结当全文，也没有把论文基准成绩转成 Stage 成功率预测。

## 有限分支 A：两任务同批的新鲜回报适应

工程推论：从明确锁定的研究初始化出发，在**同一冻结行为策略**下，对 `dev_task6_13` 和 `sector0_offset2` 各采四条新鲜独立探索轨迹；同一批联合更新整个现有 Actor。四条只是有限预算选择，论文没有证明其样本充分性。保留全部 GOAL/contact/timeout，不只筛成功轨迹后再称原分布 on-policy PG。

分组须进入损失的定义，而不只写在采样器里：先按既有物理 RTG、同任务三同行 leave-one-out baseline 和既定组内缩放计算各任务 loss，再作 `L = 0.5 * L_old + 0.5 * L_new`。每组内部的轨迹/有效步权重须在协议固定并如实报告；简单拼接全部决策行会按轨迹长度改变任务权重。同行只在同任务内配对，不能拿不同目标/时限的另一任务回报充当可交换同行。组内 RMS 缩放是工程权重选择，不能额外宣称估计量整体无偏或最小方差。

当前参数的 numerator 必须从合法 h0 按完整因果历史重算；保留实际条件噪声密度、边界动作语义与 behavior logp。旧 T519/T520 轨迹若复用，只能明确作 replay/监督参考，不能混进新鲜 PG 后仍称同一 on-policy batch。Value 不用于本轮优势时应保持这一事实清楚。

不用旧位置动作硬 RMSE 作为接受条件。若保留已有 soft reference，它仅是固定辅助项，并非已证明的恢复专家。最终由**同一冻结 Actor**在原始起点运行完整 old8 + field24：旧8全保、原field11及新增 `sector0_offset2` 逐 ID 全保、完整 contact/timeout/invalid/censored 均报告。门槛和回退由正式协议锁定；精确重生用于审计，不增加独立样本数。任务平均回报提升不能掩盖一个保护 ID 失败。

这项联合测试不会证明场地全覆盖、跨地图泛化、old275/six54/eight54/fresh16/reload78保持，未运行项目必须为未执行。

## 有限分支 B：实际神经成功轨迹的单 Actor 整合

若联合 RL 未通过且原注册时间/容量足够，可在新分支中整合 T520 的旧5m真实成功轨迹和 T519 的新field真实成功轨迹，并平衡保留原有成功经验。先核验 native GOAL、无接触、模型/输入/动作/终止及来源 SHA；必须确有完整因果120输入历史，不能从 compact evaluator 的动作与终态伪造缺失观测。

区分确定性均值成功与带噪探索成功。若源轨迹有噪声，执行动作和 Actor 均值是不同监督对象，须预先选择并记录；不能未经验证将均值动作称为这条成功轨迹的实际成功动作。标签只来自实际神经策略证据，不用失败初始策略在历史状态上的刷新输出冒充修复成功标签。

教师仅负责训练数据来源。学生部署仍是一个120输入、64GRU carry、共享2动作输出的 Actor，无 teacher/router/planner/map/task-ID 切换。GRU 学生对各 episode 从合法 h0 递推，不能直接继承教师隐藏状态并当成学生当前隐藏状态。

两教师各自成功说明存在两个成功行为见证；不说明同一参数、同一有限观测历史可以兼容全部动作。观测别名、标签冲突、学生访问分布偏移和有限优化都可能使整合失败。因此固定有限 BC 次数、完整32项原生自主验收、失败回退和保留失败原件是必要的可证伪安排。训练误差小或两个 source replay 精确，不等于学生闭环通过。

## 条件第三步与报告边界

只有联合 old8 + field24 保留门槛已由同一个模型实际完整通过，且注册预算还包含第三分支、留足最终核验及硬停时间时，才可考虑另一项固定近目标扩展。否则结束并报告取舍；本笔记不授权追加、不给已有失败换名续跑。

新鲜联合 RL、回放保护、策略蒸馏和任务平衡都已有明确前人工作。此次是单一 Stage 网络在有限已知任务上的工程机制检验，不作算法原创、无偏、安全、可行性已证或泛化成立的主张。
