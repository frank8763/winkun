# 复杂场景端到端强化学习导航：与Stage的对应关系

检索日期：2026-11-06，北京时间。用户问题：是否有相似论文用强化学习做复杂困难场景的端到端导航？本次只查资料，不新增仿真、优化器、训练或模型下载。Stage与Phase完全隔离。不是穷尽文献检索；下列事实按正文、出版社、作者项目及作者代码分别核验，不拿摘要替代方法核验。

## 结论

有。直接雷达到连续速度的去中心策略、视觉空间记忆驱动的长距离导航、规划器训练指导的窄空间脱困均有明确实例。其中部分研究在部署时仍使用子目标、动作安全掩码或多控制器切换，应和Stage要求的单因果Actor直接动作分开。端到端部署、无环境地图输入、无定位依赖、纯RL训练是不同条件。

经典雷达论文详细核对见同目录 `CLASSIC_LIDAR_RL.md`；下面补充长距离、窄空间和容易误判的对照。

## 1. SRU：长期目标最接近，正式IJRR论文

Fan Yang、Per Frivik、David Hoeller、Chen Wang、Cesar Cadena、Marco Hutter，Spatially-enhanced recurrent memory for long-range mapless navigation via end-to-end reinforcement learning。

- 出版社页面显示2025-12-13首次在线发表，卷期为IJRR Volume45 Issue11（页面标2026）。不要只说未审稿预印本，也不要混用预印本年月与卷期年。[正式出版页](https://journals.sagepub.com/doi/10.1177/02783649251401926)
- 本次具体方法和数字核对版本为[arXiv v2，2025-09-04](https://arxiv.org/html/2506.05997v2)，查看§3、§4.1/4.4–4.6、§5.1–5.6和Appendix C；未逐项比较正式版与v2所有数值是否完全一致。
- 输入为前向双目深度及本体状态、相对目标；SRU空间记忆、注意力、循环PPO，训练使用非对称Critic，Critic额外获360度height scan。采用时序一致dropout及两策略互学习。导航策略输出速度命令，不生成运行时显式地图或路径点。底层另有50Hz学习运动控制器，不能表述为深度直接输出所有腿轮执行器。[方法§4.5–4.6、平台§5.1及Appendix C](https://arxiv.org/html/2506.05997v2#S4)
- 测试包括迷宫、随机柱体、楼梯、坑洞。Table1中SRU-Ours总体78.9%、LSTM63.5%、GRU61.0%；Table2为另一平台/基线比较，Ours78.3%，不可把两表混成同一实验。Table1选各单元最高回报模型，正文注明GRU只纳入成功训练运行，不能当所有种子无选择均值。[§5.1–5.3](https://arxiv.org/html/2506.05997v2#S5)
- 距离分组仿真实验：原60秒最大时限下50米内SR超过80%；将时限延长到120秒后，120米内超过70%。这些是本文特定仿真平台/环境/成功判据结果，不是原始六米Stage验收率或120米真机统计。[§5.3，Figure10](https://arxiv.org/html/2506.05997v2#S5.SS3)
- 真机为Unitree B2W、ZEDX、Jetson AGX Orin；用LiDAR状态估计提供速度/重力/相对目标，环境感知由前向深度提供。办公室临时封堵通道要求回退绕行，另测大厅、露台、森林。不能称全系统只有相机、无定位模块。报告有超过70米目标、超过100米行进的真机演示，不以演示替代总体成功率。[§5.6.2](https://arxiv.org/html/2506.05997v2#S5.SS6.SSS2)

### 当前开源状态纠正与边界

本次通过GitHub连接器实际读到作者组织的 [sru-navigation-learning README](https://github.com/leggedrobotics/sru-navigation-learning/blob/main/README.md)，已确认有导航网络与PPO/MDPO训练框架，并非只能看到空间记忆合成子任务。这个事实应更新此前仅查看独立记忆仓库的认识。

但该README明确不包含仿真任务、奖励/观测/动作任务定义、机器人模型、底层运动策略和深度预训练。其指向 `leggedrobotics/sru-navigation-sim` 的README链接，本次GitHub API返回404（可能不存在、未公开或路径不同，未判定原因）。[作者项目页](https://michaelfyang.github.io/sru-project-website/)列出多个仓库，不代表每个资源均实际可访问。没有宣称完整开箱复现，也没有克隆/安装/下载大模型。README把新增MUON明确标成非原论文组件，不能倒推论文使用了MUON。

**Stage借鉴判断：** 值得作为后续长距离空间记忆和循环训练对照。雷达替换深度、车动力学替换腿轮及记忆接口需要受控验证，不能把换GRU为SRU当原创或保证当前窄口成功。既有Stage循环试验失败也不能证明SRU必无效。

## 2. Embodied Escaping：窄空间任务最接近，但有关键额外能力

Han Zheng等，Embodied Escaping: End-to-End Reinforcement Learning for Robot Navigation in Narrow Environment，IROS2025。[作者单位正式介绍](https://news.sjtu.edu.cn/jdzh/20250715/212892.html)确认题名和会议。本次方法核对[arXiv v1正文](https://arxiv.org/html/2503.03208v1)，主要§III–IV，未核验最终会议版全部修订。

- SAC、Transformer编码，输入500点360度雷达、相对目标及有效动作分布；输出42种离散的线速度/角速度组合。是感知到控制的学习策略，不是我们现有连续drive/steer分布的直接替换。[§III-A](https://arxiv.org/html/2503.03208v1#S3.SS1)
- 训练期用A*可行路径反算动作指导，困难情况下仍由RL行动；正文明确A*仅用于训练，测试由RL执行。**测试仍使用由几何运动推演得到的动作安全掩码**，所以单Actor毫无外部动作约束的Stage部署合同不能直接照搬。[§III-B/C](https://arxiv.org/html/2503.03208v1#S3.SS2)
- 包括倒车、原地转向与固定曲率转弯。真实机器人35×35厘米方形差速平台；与Stage约61厘米有效碰撞宽度、无倒车/实际执行器动力学不同。不能据其窄口成功推断Stage原窄口可走或同算法同成功率。[§III-A2、§III-C式5、§IV-C](https://arxiv.org/html/2503.03208v1#S4.SS3)
- 先随机生成机器人可行运动路径，再在轨迹包络外加障碍；1800训练、360测试场景，特征涵盖窄出口、受限空间、稀疏障碍和长走廊。20,000训练episodes。TableII总成功率75.8%，TableI不同特征组合分别计数，不能选最高99.5%代表全部困难任务。真机有狭口、长廊、椅子等演示，正文没有足够分母支持把真机演示称总体100%成功率。[§IV-A–C和TablesI/II](https://arxiv.org/html/2503.03208v1#S4)

**Stage借鉴判断：** 已验证可行的场景生成、训练期指导、可执行动作表达和转向时机很相关。可借训练阶段机制；若使用运行时掩码或倒车，必须明确改变部署/动作合同并重新验收，不能偷偷加入后仍声称合同不变。

## 3. DRL-VO：T-RO的困难场景实例，但整套系统不符合无路径点合同

Zhanteng Xie、Philip Dames，DRL-VO: Learning to Navigate Through Crowded Dynamic Scenes Using Velocity Obstacles，IEEE Transactions on Robotics39(4):2700–2719，2023。[出版社](https://doi.org/10.1109/TRO.2023.3257549)，本次方法核对[arXiv v2](https://arxiv.org/html/2301.06512v2)§III、IV和V。

- 短雷达历史、行人位置速度和子目标输入，PPO学习连续前向/角速度；VO用于回报设计。行人信息来自RGB-D检测和跟踪，不是仅原始雷达。[§III-B/F](https://arxiv.org/html/2301.06512v2#S3)
- 仿真测试最多55行人，并有真机与高度受限BARN场景。**子目标由名义路径上的pure pursuit取前方2米点**，已知地图AMCL提供定位；不能作为无全局路径/无路点的全程端到端成功证据。[§III-B3及§IV-C](https://arxiv.org/html/2301.06512v2#S3.SS2.SSS3)
- 它说明强化学习可用于困难动态避障控制；不能把其总体系统直接叫作Stage同合同算法。作者公开代码[TempleRAIL/drl_vo_nav](https://github.com/TempleRAIL/drl_vo_nav)，本次仅查看公开仓库说明，不宣称完成源码级复现。

## 4. 两个应排除的“名称很像”例子

- [Context-Aware Deep Reinforcement Learning…，CoRL2023](https://proceedings.mlr.press/v229/liang23a.html)：作者[代码README](https://github.com/marmotlab/Context_Aware_Navigation)明确学习图上的下一路点，维护逐步扩展的partial map/graph。无先验地图不等于无在线显式地图，不属于我们单Actor直接动作合同。只作长期规划参考。
- [Learning Robust Autonomous Navigation and Locomotion for Wheeled-Legged Robots](https://junja94.github.io/learning_robust_autonomous_navigation_and_locomotion_for_wheeled_legged_robots/assets/paper/AOW_preprint.pdf)：系统含大尺度规划和分层RL，公里级演示不能当单无地图Actor全程输出动作的同条件比较。本次只确认系统结构，不列其性能数字作Stage目标保证。

## 和昨晚真实证据的关系

Stage T388D在新0.70米/80秒任务中已有一条完整原生教师GOAL60.516秒，并精确重生；T387学生开发0/5，原六米两个中心0/2，完整54/16留出/78重载未执行。[本地原始验收](../overnight_20261105_revision/MORNING_REVIEW_20261106.md)。教师成功与学生失败支持“该新场景存在这条可达轨迹、当前成功控制尚未有效转移进自主Actor”。它不证明唯一根因为遗忘、容量、空间记忆或RL本身失效，也不证明原0.57米合同可达。

近期优先级判断（推断，非已执行实验）：先做教师/学生动作和完整历史一致性及首次闭环偏离诊断，再检查实际learner恢复状态的标签覆盖。空间记忆可作为后续受控变量，不同时把相机/SRU/Critic/安全层全加上。端到端、无地图和强化学习本身已有大量先例；顶刊贡献需要明确的新机制、对应消融、跨场景/种子与真机证据，不能以六米跑通或模块组合本身声称创新。
