# 11月2日后续一手检索与机制选择

检索日期2026-11-02。用户明确授权继续Stage、必要时借鉴论文或改算法。本轮查正式论文页、可访问正文的方法/限制及作者README，未声称穷尽互联网、复现外部模型或下载执行外部代码；Phase完全隔离。六米最新完整仍T2911/54、旧274/275，safeT255保留。

## 信用估计先做可证伪检验

[Greensmith/Bartlett/Baxter，JMLR2004](https://www.jmlr.org/papers/v5/greensmith04a.html)把baseline和actor-critic视为控制变量，并指出真实价值函数也不保证最优梯度方差。[Chung等，ICML2021](https://proceedings.mlr.press/v139/chung21a.html)进一步指出baseline可能改变策略优化动态，不能只用曲率/噪声解释成败。[GAE论文](https://arxiv.org/abs/1506.02438)提供基于学习价值的优势估计近邻，但本项目尚无验证可靠的状态V，直接引入bootstrap/GAE会同时改变偏差、奖励传播和信用估计，当前不盲接。

所以T292先用固定非线性ridge的状态V控制变量，与既有time-LOO对照：按整条轨迹、跨四起态的同seed家族划分，真实Gt目标、只读已有120因果状态；独立比较留出MSE与实际全部参数score-gradient Gram。不是新神经Critic/TD训练，不是将失败预测事件头当Q排序，不因拟合好就自动接Actor。T272/T274动作盲负结果针对事件Q排序，并不否定state-only baseline。

## 如果信用路线失败，保留两类更结构性的方向

[Contrastive RL，NeurIPS2022](https://arxiv.org/abs/2206.07568)把带动作轨迹的对比表示与目标条件价值联系起来；[Single-Goal CRL，ICLR2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c34ddd05eb089991f06f3c5dc36836e0-Abstract-Conference.html)研究单个困难目标下从自监督学习涌现探索。[ICLR2026机制分析](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cbaffeeda13dbd8bf9489feb3f198ff4-Abstract-Conference.html)提供近邻，不据标题认定当前数据满足其假设。[HER原论文](https://arxiv.org/abs/1707.01495)适用于off-policy重标记，当前on-policy条件AR轨迹不能改目标后直接当新PPO样本；历史120观测含目标历史、首次命中和接触优先也必须重建。T275/T277已有数据语义合同，T272/274/276头未过筛选，不换名重复预测头。后续需真正goal-conditioned action-value与安全终止/数据支持检验，未见目标/动作不能凭未来正例制造可靠负标签或可达性证明；现在未接Actor。

[SRU正式IJRR论文](https://journals.sagepub.com/doi/10.1177/02783649251401926)研究通过乘性交互增强循环记忆的空间配准，并配合注意力和训练正则。此方向贴近长距无显式地图导航，但传感器/机器人与当前Stage不同，尚未证明当前失败源自记忆。[预印本正文](https://arxiv.org/html/2506.05997v2)与[作者仓库](https://github.com/leggedrobotics/sru-pytorch-spatial-learning)已核对；公开实现明确主要对应附录空间时间记忆子任务，不是可直接运行的整套导航。不能把网站README的早期日期当正式出版日期，正式页标在线2025-12-13/卷期2026。当前部署GRU每步清零；若以后改变记忆路径，必须独立单因素协议、完整序列/密度/状态恢复合同和旧能力验收，不能在当前诊断中顺手打开hidden。

带外部路径、子目标、图规划的长距方法不直接套用本线，因为本线约束单Actor无地图/规划器/人工路点/运行时教师或路由。空间记忆和CRL本身都是已有方法，不当作本项目原创。

## 论文主张当前能到哪里

可报告的事实是：旧硬guard确会拒绝更新，但取消它后仍只恢复同历史成功点并丢失旧案例；小的源状态动作误差不足以保证闭环成功。尚无支撑六米稳定/全地图/顶刊新算法的结果。创新候选必须解释怎样扩大真实成功状态覆盖并保持旧闭环，再做匹配消融、多个训练seed/未见地图和成本报告。T292门槛是进入下一机制设计的筛选，不是显著性、安全证书或论文创新证明。
