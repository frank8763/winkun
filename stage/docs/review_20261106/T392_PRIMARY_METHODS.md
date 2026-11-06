# T392 一手方法核查：示范辅助的真实循环策略梯度

2026-11-06。仅文献和概率语义核查，不是实验已注册、已启动或结果。范围限 Stage；正式主任务按本轮用户回答为新 **0.70 m 缺口／80 s**，原 0.57 m／56 s 仅另列诊断。部署仍为单个因果 Actor，无运行时教师、规划器、路点或控制器切换。

项目动机来自 [T391_FINDINGS.md](T391_FINDINGS.md)：真实恢复教师从 B 的 2.4 s 偏离态完整到达，学生 1000 次监督后仍于 21.440 s 接触。它支持检验学生自身交互的奖励学习，不证明 BC 是唯一根因，也不保证新方法成功。本文不重新运行任何历史实验。

## 1. 截断 Gaussian：边界必须记尾概率

Fujita 与 Maeda 的 CAPG §3、Eq(4–10) 要求环境转移和奖励只依赖实际截断动作；将截断区间外的 score 换成边界尾概率的 score。§3.2 将结果推广到给定状态条件独立的动作分量。论文的无偏和逐参数方差降低结论有这些前提，不能扩张成 PPO、示范辅助或本项目成功率的保证。[CAPG 全文](https://arxiv.org/pdf/1802.07564)

作者的 `ClippedGaussian.log_prob` 确认：左端为 log-CDF，右端为 log-survival，内部为 Gaussian log-pdf，最后按动作维求和；不是把 ±1 当 Gaussian 内部点求密度。[作者实现](https://github.com/pfnet-research/capg/blob/master/clipped_gaussian.py#L98-L115)

**对拟用 float32 实现的推导，不是论文原文：** 令每维原样本 `u ~ N(mu_theta(H), sigma^2)`，固定 sigma、跨步及维度独立创新；实际命令为 `float32(clip(u,-1,1))`。在默认最近舍入下，记

`E = (1 + float(nextafter(float32(1),float32(0)))) / 2`。

保留增强动作变量 `z`：`|u| < E` 时保留原始 raw64，两个尾部各合并为一个边界标记。其合法混合 likelihood 为

\[
\ell_\theta(z\mid H)=
\begin{cases}
\log\Phi((-E-\mu_\theta(H))/\sigma),&z=\mathrm{left},\\
\log\mathcal N(u;\mu_\theta(H),\sigma^2),&|u|<E,\\
\log\Phi((\mu_\theta(H)-E)/\sigma),&z=\mathrm{right}.
\end{cases}
\]

内部 raw64 再确定性映射到执行命令。因此内部保留 raw64 求 log-pdf 是合法的增强变量表示，**不需要**把每个内部 float32 格子积分；也不能把它误称为执行 float32 命令自身的连续密度。两个边界原像必须与实际 cast／clip 顺序一致；数学连续采样的舍入平局为零概率，有限 PRNG／浮点实现仍须实测断言。现有 [T351 实现](../../experiments/t351_recurrent_recovery/t351_core.py) 已使用这个 E 和 raw／atom 表示，新轮须独立核验，不能只凭历史实现声称通过。

样本 raw／atom 在 score 计算中固定，不让梯度通过 `u=mu+sigma*epsilon` 抵消 score。新旧 likelihood 都按同一保存的 raw／atom、同一 sigma、各自当前参数的完整历史计算；prefix 伪 logp 不进入比例。

**探索反证：** 固定 raw sigma 不能保证执行动作有同量级方差。若 location 远在边界外，仍会几乎全饱和；若在 tanh 前加噪声，饱和区更可能几乎无执行变化。预检应报告实际命令分位数、边界质量及反向命令频率，不能只报 latent 标准差。噪声单位是实际归一化动作域；转成角度须使用真实执行器映射。

## 2. 示范辅助有先例，但固定 MSE 不是 DAPG 原式

DAPG §IV-C Eq(5–6) 先 BC，再将真实策略轨迹的 `grad log pi * A` 与示范的 `grad log pi * w` 相加，经 NPG 更新；论文采用衰减权重 `w=lambda0*lambda1^k*max(A_onpolicy)`，并明确示范优势不能无额外假设或 rollout 直接得到。论文也报告 BC 策略可能因状态分布偏移失败。[DAPG 全文，PDF 第 6 页](https://arxiv.org/pdf/1709.10087)

作者当前公开 `mjrl/algos/dapg.py` 对 on-policy 优势标准化，示范使用常数衰减 `demo_adv`，随后共轭梯度 NPG；此代码与论文显示的 max-A 缩放并非逐字相同。[作者实现](https://github.com/aravindr93/mjrl/blob/master/mjrl/algos/dapg.py#L49-L107)

本轮若使用 `loss = loss_PG + 0.1*demo_MSE`，应称**示范辅助的循环 PG 工程变体**。固定方差、未截断 Gaussian 的均值 MSE 与负 log-likelihood 只差尺度／常数；带边界 atom 时，MSE 不等于边界 NLL。示范动作不是当前策略随机采样，不赋予虚构 on-policy 优势或 PPO 旧概率。两个真实成功数据源提供训练先验，不等于两份独立随机成功实验。

## 3. 前缀课程改变条件起态；完整历史仍属于策略输入

Reverse Curriculum §3–4 通过改变训练起态分布，从近目标逐步扩展，并在原始起态分布评估；其原法使用目标态附近随机游走及成功率筛选，含任意 reset 和可通信性假设。固定成功轨迹的 50／35／20／0 s 前缀课程并非该算法复现，论文也不保证这个有限课程能学会原起点。[Florensa 等全文](https://arxiv.org/pdf/1707.05300)

Heess 等 §2–3 以完整观测／已执行动作历史 H 定义部分可观测策略，通过 BPTT 学习循环摘要；其算法是 RDPG／RSVG，不是本轮无 Critic 的 score-function PG。[Memory-based control 全文](https://arxiv.org/pdf/1512.04455)

**本轮语义推导：** 每级先真实执行固定示范前缀，所有 prefix 物理都计费，保留 native 状态、原任务 clock、传感器历史和真实 previous-action。每次采集及每次优化都从 episode zero 用当前参数重建循环状态；不能重置 hidden／时钟，不能把教师 hidden 或旧参数 hidden 当当前状态。重建时前缀实际执行动作作为历史输入，不能用当前 Actor 未执行的均值替换。

前缀动作由固定数据决定，其行为概率与 theta 无关，故只对随机学生后缀求 score；但后缀策略依赖当前参数对整个前缀的编码，梯度可通过此前缀的循环计算图传播。这不是对示范动作做 on-policy PG，也不通过模拟器／历史观测反传。短后缀通过只评价该条件起态，只有 0 s 原起点自主结果才能报告正式六米通过。

## 4. 四条 LOO、终止与 PPO 的边界

**直接由 likelihood-ratio 恒等式得到的条件结论：** 固定同级前缀及当前行为参数，独立采集 4 条后缀，令完整终端折扣回报为 `G_i`，`b_i=sum(G_j,j!=i)/3`。plain PG 可用

\[
\widehat g=\frac14\sum_{i=1}^{4}(G_i-b_i)
\sum_{t\in\mathrm{student\ suffix}_i}\nabla_\theta\ell_\theta(z_{it}\mid H_{it}).
\]

其他轨迹的回报在给定固定条件后独立于本条动作，因此 baseline 项期望为零。共同任务／同一主 seed 拆成独立子流可行；相同噪声重置、反向配对或复制轨迹不能当独立 peer。独立精确重生只用于审计，不能加入 N=4。若只奖励终端，G 应按预注册的固定绝对／后缀起算折扣定义，一条轨迹内各 score 使用相应一致权重；不要漏掉折扣因子，也不要除以随机终止长度而悄悄改变目标。固定损失尺度可声明；本批标准差归一化不能沿用上述简单无偏结论。

PPO Eq(7) 是 `min(r*A,clip(r,1-eps,1+eps)*A)`，在旧参数处与普通 surrogate 一阶一致。`eps=0.1` 只限制该代理目标的奖励方向，不能保证所有 ratio／KL 都落界、真实回报单调或接触安全。多 epoch、clip、梯度裁剪及 demo 辅助后的总体更新不声称原始 PG 无偏。[PPO 全文 §2–3](https://arxiv.org/pdf/1707.06347)

80 s 若已被定义为正式任务 deadline，其到时是**真实有限时域终止**；50 s 前缀只剩 30 s，不能重新获得 80 s。反之，墙钟／命令／物理额度迫停属于行政删失，不能赋予假 timeout／终端回报。无 Critic 的终端 MC 方案不能靠未知价值补全删失；应记录 censored，并按预注册规则停止该不完整更新，避免只挑先完成轨迹。[Time Limits in RL 全文 §2–3](https://proceedings.mlr.press/v80/pardo18a/pardo18a.pdf)

本轮可检验的是“成功前缀条件下，真实学生后缀奖励是否带来保留的闭环改进”。应把执行噪声、当前历史概率重算、后缀 mask／前缀 BPTT、独立随机子流、完整终止／删失及最终 0 s 验收写入有界协议；不把标准组件组合、代理损失下降或近终点成功称为原创算法／完整导航成功。
