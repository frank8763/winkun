# T299期间补充的一手文献与适用边界

2026-11-02阅读以下官方摘要/作者实现链接；不把只读摘要说成完整复现。资料整理不改变正在运行的T299源码、损失、目标、预算或门槛，Stage/Phase完全隔离。

- [GCSL原文](https://arxiv.org/abs/1912.06088)：自身轨迹可作为到达其终态的示范，迭代采集与目标条件动作学习；[作者训练源码](https://raw.githubusercontent.com/dibyaghosh/gcsl/master/gcsl/algo/gcsl.py)调用条件NLL。[作者公开网络](https://raw.githubusercontent.com/dibyaghosh/gcsl/master/gcsl/algo/networks.py)中可见离散及逐维离散随机策略实现。当前Stage连续截断AR、历史观测及native firsthit都不是无修改复现。T299仅检验对应辅助loss数值路径，0新RL。
- [Foster/Block/Misra，NeurIPS2024](https://arxiv.org/abs/2407.15007)：对logarithmic-loss模仿学习的时域依赖给出条件性理论，涉及可实现性/收益范围/策略类复杂度限制。这支持认真区分logloss和动作MSE，但不保证本项目有限数据或确定性读取在长距离成功，也不覆盖未证实可观测性和碰撞闭环保持。
- [Tang/Kucukelbir，AISTATS2021 hEM](https://proceedings.mlr.press/v130/tang21b.html)：从目标条件RL的变分下界/EM框架解释hindsight与监督M-step。仅阅读官方摘要；不能据此将任意goal重标记MSE/NLL称已实现hEM或原6m目标无偏优化。当前T295目标来自已消费轨迹，必须分清原6m失败和辅助目标正例。
- [Matthews等，ICML2026 LEO](https://proceedings.mlr.press/v306/matthews26a.html)：官方摘要提出并行输出所有目标的价值和动作，以降低all-goals relabel计算成本，报告Craftax/连续控制任务结果。只读官方摘要，未读完整PDF或运行其代码。它属于多目标结构化输出的新路线，不等价当前单Actor直接控制，也没有解决本项目的first-hit安全性/AR密度/旧能力保持；不在T299混加网络。
- [Pfrommer等，NeurIPS2022 TaSIL](https://papers.neurips.cc/paper_files/paper/2022/file/7f10c3d66c3b7863a9cda255dcac5bb7-Paper-Conference.pdf)：已有先前研究记录；理论依赖增量稳定性等条件。可以启发对状态敏感度及闭环误差的检验，不把有限旧任务检查或很小动作MSE称为稳定性证书，也不能将标准Jacobian保持包装成原创。

当前可检验的问题是：在同参数/目标/样本/500步及固定初始梯度尺度下，修正辅助loss能否改善确定性原6m成功覆盖并保持旧275。NLL变好、辅助goal自主成功、原6m完整安全成功是不同层级。某个pilot失败不能否定GCSL/HPG/CRL/hEM整体，某个pilot多一例成功也不能证明原创新算法或顶刊水平。
