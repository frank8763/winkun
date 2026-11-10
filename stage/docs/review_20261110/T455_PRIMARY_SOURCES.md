# T455：固定表示、仅动作头学习的文献边界

2026-11-10，Stage限定。复用[T454已核验的一手材料](T454_PRIMARY_SOURCES.md)，不新增论文清单、网络或实验。本笔记只说明待检验假设，不报告T455结果；没有native交互或优化器更新。

T454在同一旧task2父history上，warmup→PPO0原始输出差RMS约0.02376，双顺序表示分量约0.020923、head约0.00306；宽RIGHT相应表示分量约0.02314、head约0.00138。这些固定历史分解提示：可独立检验“允许共享表示改变是否是本轮退化的可避免来源”。分量会交互、抵消，以上不能解释为导航损失的因果比例，也不能证明表示改变应当全部禁止。数据与独立审计以T454原报告为准。

| 复用的一手依据及阅读范围 | 对T455的支持与限制 |
| --- | --- |
| Rusu等，*Policy Distillation*，[全文](https://arxiv.org/pdf/1511.06295)，§3.2–3.3、§4.1 | 旧策略输出可作为单独监督保持目标，多任务交错训练已有先例。原文离散Atari、多任务专用输出层不同于Stage连续单头递归Actor；它没有证明仅130个head参数就能修复00并保留原328项，也没有证明固定表示优于全参数学习。 |
| Kapturowski等，*Recurrent Experience Replay in Distributed Reinforcement Learning*，[作者出版页](https://willdabney.com/publication/r2d2/)，仅作者摘要已核验；论文PDF此前被验证页阻挡 | 参数变化引起的representation drift和recurrent state staleness是已有问题，支持把共享表示变化作为可检查因素。本次不声称复核全文或将R2D2的DQN经验直接移植到PPO；冻结encoder＋GRU是否改善Stage仍须实测。 |
| Pfrommer等，*TaSIL: Taylor Series Imitation Learning*，[正式全文](https://papers.neurips.cc/paper_files/paper/2022/file/7f10c3d66c3b7863a9cda255dcac5bb7-Paper-Conference.pdf)，§2–3、Assumptions 3.1–3.2、Theorems 3.1–3.2 | 小的专家分布输出误差并不普遍保证闭环轨迹相近；正面保证需要专家稳定性、策略正则性及足够小的相应误差等条件。固定表示没有自动建立这些条件或碰撞裕度，因此即使保持loss、平均KL和全部数值合同通过，仍须接受完整闭环反证。 |

本地推断：固定encoder＋GRU权重后，同一个完整因果history从相同h0重算的表示应保持一致；这为单因素匹配提供可审计的不变量。**固定权重不等于固定hidden或固定访问分布**：head更新会改变实际动作，继而改变观测、前动作和递归hidden；新的闭环反馈仍可能退化。原有表示也可能缺少修复00所需的可线性读取信息，130参数头容量是否足够尚无证据。

T455因此只把可训练参数限制作为与T453匹配的改变：从同一328研究父模型及成熟Adam起点出发，保留128BC、两轮各4条新学生PPO采集、既定seed/权重/阶段面板和最终329合同。冻结encoder/core参数与其Adam状态逐字节哈希不变属于本地实现合同；文献不提供该合同、这些数值或本任务成功的充分性。阶段面板只观察，不据其选择中间候选；最终是否保留原328并修复00由预定验收决定，不用输出分解代替导航验收。

结果只检验这一受限pilot：通过不证明跨起态泛化、冻结策略普遍有效或算法原创；失败也不否定全部固定表示方法或全参数端到端学习。冻结GRU、策略蒸馏和仅头部微调均不作为新算法主张。
