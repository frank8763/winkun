# Stage失败现象的一手文献对照

2026-11-13。只做阅读与文献核对；未启动模拟器、训练、评估或资源整理，未修改 experiments/runs/协议。现场范围仅为 [T524中期稿](T524_INTERIM_AND_NEXT_DRAFT.md)、[T523解释笔记](T523_FORGETTING_PRIMARY_SOURCES.md)、[午间文献依据](LUNCH_RELAY_PRIMARY_SOURCES.md)；不是原始数组或最终审计的独立复核。

**目前有把握的是：出现了旧能力回退、监督误差与闭环成绩脱节、以及至少两类不同的实际导航失败。文献确认这些都是已有研究问题，但不能据相似症状指定唯一根因，也不能推导本车任务不可解。** T524中期稿记载 old8=8 GOAL、field24=10 GOAL/10 CONTACT/4 TIMEOUT，逐ID终止类别与T523相同；其精确重生和独立audit当时尚未结束。固定初始梯度幅值平衡没有改变本次通过率，仅否定了这个有限改动已解决问题的说法。[现场依据](T524_INTERIM_AND_NEXT_DRAFT.md)

| 一手来源 | 已研究的现象；与Stage的关系 | 适用边界 |
|---|---|---|
| Chiang等，**Learning Navigation Behaviors End-to-End with AutoRL**，IEEE RA-L 2019，[作者机构论文页](https://research.google/pubs/learning-navigation-behaviors-end-to-end-with-autorl/) | 比通用游戏更直接的导航对应：分别学习端到端点到点与路径跟随，输入含噪lidar、输出线速度和角速度；作者明确报告对照深度RL的 catastrophic forgetfulness 问题。 | 原文AutoRL还进行大规模奖励/结构搜索，且速度接口不同于本车驱动/转向；其导航实验不是Stage证据。点到点与路径跟随是分别学习的行为；可作更大导航系统的模块不等于无规划器全图导航已解决。 |
| Rolnick等，**CLEAR / Experience Replay for Continual Learning**，NeurIPS 2019，§3–4.1，[全文](https://arxiv.org/html/1811.11682v2) | 用分别、同时、顺序训练区分任务间 interference 与新任务覆盖旧能力的 forgetting；以新经验、历史replay和策略/价值克隆兼顾适应与保持。T519/T520修此失彼与遗忘风险相容。 | 原文是Atari/DMLab、V-trace actor–critic。Stage未有同等因果对照；行为回退不等于已经证明不可兼容，也不证明CLEAR能修好此车。 |
| Yu等，**Gradient Surgery for Multi-Task Learning / PCGrad**，NeurIPS 2020，§2.2–2.4，[全文](https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)、[作者代码](https://github.com/tianheyu927/PCGrad) | 同时考察负内积梯度、幅值差、高曲率；论文明确冲突梯度单独存在并不必然有害。它提供一种解释联合优化困难的机制。 | T524仅改固定初始幅值平衡，不等于完整测量或排除该机制；平均任务loss改善也不等于每个保护ID通过。论文结果不能证明本共享动作头存在或不存在联合解。 |
| Ross、Gordon、Bagnell，**DAgger**，AISTATS 2011，§2–3，[论文](https://proceedings.mlr.press/v15/ross11a.html) | 学生动作改变未来输入分布；专家轨迹上的监督误差不能直接代表学生自己访问状态上的误差。方法在学生访问分布查询专家。T522低BC loss却闭环退化与此风险相容。 | 仅重复拟合旧成功轨迹不是DAgger；因果GRU历史、可恢复专家及有效反馈均不能由论文代替核验。尚未证实分布偏移是Stage唯一根因。 |
| Laskey等，**DART**，CoRL 2017，[论文](https://proceedings.mlr.press/v78/laskey17a.html) | 通过给教师演示注入与学生误差相配的扰动，让数据包含纠错和恢复；明确研究BC漂离演示后误差累积。它解释了“成功轨迹存在”与“学生会恢复”之间的缺口。 | 实验是MuJoCo步行任务与Toyota HSR杂乱抓取，**不是Ackermann导航**；给学生增加探索噪声不自动成为DART，也不保证教师能从当前学生失败状态恢复。 |
| Pfrommer等，**TaSIL: Taylor Series Imitation Learning**，NeurIPS 2022，§1–3，[论文](https://proceedings.neurips.cc/paper_files/paper/2022/file/7f10c3d66c3b7863a9cda255dcac5bb7-Paper-Conference.pdf)、[作者代码](https://github.com/unstable-zeros/TaSIL) | 连续控制中，动作值相近仍可能产生不同闭环；通过匹配策略高阶Taylor项，在专家满足增量输入到状态稳定性等条件下建立轨迹误差保证。它比“BC还不够低”更明确地指出局部反馈响应/鲁棒性也是问题。 | 保证依赖稳定性、光滑性、近似/数据等假设；不是普通BC loss的保证。原实验为MuJoCo，未证明Stage接触动力学、部分观测GRU和此教师满足假设；本文不建议直接移植或宣称不稳定已证。 |
| Reeds、Shepp，**Optimal paths for a car that goes both forwards and backwards**，1990，[期刊原文](https://msp.org/pjm/1990/145-2/pjm-v145-n2-p06-s.pdf) | 有最小转弯半径、指定起终朝向的车，其可行路径受曲率与前后行驶约束；位置距离小不等于已完成所需运动。对近目标控制失败提供几何背景。 | 原文是理想几何最短路，不含Stage的驱动/制动延迟、障碍接触或奖励；目标若只要求位置范围，不能偷加终朝向条件。不能凭此宣布目标半径小于转弯半径便不可达，也不引入外部规划器。 |

## 对实际失败的解释精度

`sector0_offset2` 在78.8秒的最近**决策端点**为0.347152m，此后49.424秒远离，128.224秒以2.834561m超时；能说的是“到过目标邻域但未按注册规则完成捕获，随后没有恢复”。最近决策端点不是连续原生最小距离；尚无速度、朝向、制动和原生距离时序的本次核对，不能把它直接定性为惯性冲过头、目标半径错误或车体运动学极限。`sector2_offset2` 则在51.122秒、距目标5.264785m接触障碍，属于更早的避障交互失败，不能用终点精度解释全部失败。[现场依据](T524_INTERIM_AND_NEXT_DRAFT.md)

## 小批探索成功不等于确定性策略成功

PPO优化的是采样行为下回报的经验代理；有限批次只是估计，不是每个状态、每项任务的成功证书。[PPO §2–3、Algorithm 1](https://arxiv.org/pdf/1707.06347) 若四次带噪探索中一次GOAL，1/4描述的是这四个实际采样事件；它不等于“冻结Actor的均值动作闭环有25%成功率”，也不能证明均值动作会复现那条成功轨迹。推理依据是：执行噪声改变动作并继而改变后续状态，非线性闭环的成功事件不能通过对动作取均值保留。[DAgger对输入分布依赖的定义](https://proceedings.mlr.press/v15/ross11a.html) 此处不提供独立同分布置信区间，因为本笔记没有核验四轨迹独立性及同分布合同。

另外，多训练seed重复、同策略独立探索、逐任务固定确定性验收、同轨迹精确重生回答的是不同问题；精确重生用于复核，不能增加独立成功样本数。RL方法/环境的方差会使少量结果易被误读，已有专门的一手实证研究，但不能拿其他基准方差替代Stage自己的数据。[Henderson等，Deep Reinforcement Learning that Matters，AAAI 2018](https://arxiv.org/abs/1709.06560)

本次文献检索没有找到可直接断言“同一Ackermann模型、同一目标判据、同一GRU观测合同”的已发表对应案例。AutoRL提供导航遗忘的直接对应；遗忘/联合优化和闭环分布偏移另有机制证据，近目标超时的物理原因仍未确定。本笔记仅解释问题和证据边界，不授权修复、追加预算或启动草案。
