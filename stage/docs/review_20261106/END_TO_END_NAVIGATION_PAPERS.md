# 纯端到端深度强化学习导航：一手论文核验

核验日期：2026-11-06。仅 Stage 文献笔记；本次没有启动训练/仿真，没有修改运行协议、实验源码或既有结果。阅读范围是下列原论文的方法和相关实验章节，以及注明的作者页面/仓库；不是部署源码的完整审计，也不是穷尽检索。

## 判定口径

需要分别回答三个问题：①部署时是否由观测与最终目标直接输出动作；②训练是否从随机策略开始，仅用 RL 而不用示范；③训练奖励/目标输入是否借助特权几何或定位。训练期示范或最短路奖励不等于运行时混合导航，但也不能叫“完全不借助规划知识、从零纯 RL”。下列“无运行时规划器”按论文明确给出的策略接口与执行流程判定，不额外推断未审计的软件封装。

## 1. Tai、Paolo、Liu，2017

[Virtual-to-real Deep Reinforcement Learning: Continuous Control of Mobile Robots for Mapless Navigation](https://arxiv.org/pdf/1703.00420)，IROS 2017。

§III-C：14 维输入为稀疏激光 10 维、上一步动作 2 维、相对目标距离/角度 2 维，直接输出连续线速度与角速度。ADDPG 从零训练、没有示范；部署策略不需要障碍地图或外部路径规划。§IV-B 包含未见仿真环境及 Turtlebot 真实办公室、狭窄通道实验。[原论文 §III-C、§IV-B](https://arxiv.org/pdf/1703.00420)

**范围限制：**仿真和实机路线按预设的多个目标依次执行，实机为图 9 的 0–9 号目标；不能把整条多目标路线宣传为“只给一个远处最终目标，无中间目标完成复杂迷宫”。Env-1 模型实机失败也有报告。它支持无示范的端到端点到点控制，不证明任意长程任务已解决。[原论文 §IV-B2/B3、图 7/9](https://arxiv.org/pdf/1703.00420)

## 2. Long 等，2018 版本

[Towards Optimally Decentralized Multi-Robot Collision Avoidance via Deep Reinforcement Learning](https://arxiv.org/pdf/1709.10082)，此处核验 arXiv v3，2018-05-20；首次预印本为 2017 年。

§IV-A/B：输入为最近三帧 512 束激光、相对目标、当前线/角速度，输出连续线/角速度。共享策略用 PPO 和多场景课程训练，集中采集多机器人经验、分散执行；没有专家示范预训练。训练课程从简单随机交互扩展到含障碍场景，不是运行时教师或路径路由。[原论文 §IV-A–C](https://arxiv.org/pdf/1709.10082)

§V-B4 的两组机器人在含两障碍的狭窄走廊交换位置，最终策略通过；另有交叉、交换、异形机器人及 100 机器人泛化实验。**这些是仿真多机器人避碰证据，不能改写成实机结果，也不能等同于单车在任意静态迷宫中寻找远处出口。**[原论文 §V、图 7/9](https://arxiv.org/pdf/1709.10082)

## 3. Pfeiffer 等，2018

[Reinforced Imitation: Sample Efficient Deep Reinforcement Learning for Map-less Navigation by Leveraging Prior Demonstrations](https://arxiv.org/pdf/1805.07095)，RA-L 2018；[作者实现及书目信息](https://github.com/ethz-asl/rl-navigation)。

§III-B：1080 束激光经最小池化变 36 维，加相对目标，直接输出线/角速度。主方法为 move_base 生成训练示范、BC 初始化、随后 CPO；部署时是单策略。部分训练奖励用 Dijkstra 最短可行距离，论文明确该距离只用于训练奖励，策略不可见。另有零示范、随机初始化的纯 RL 对照。**主方法是部署端到端的 IL+RL，不能列为从零纯 RL。**[原论文 §III-B/D、§IV-B、表 I](https://arxiv.org/pdf/1805.07095)

§IV-C 在两个未见 10×10 米地图（迷宫、薄墙与杂乱障碍）各取 100 对随机起终点，碰撞即终止；最佳 c1000+CPO 成功率 79%。实机表 II 的最佳模型五次路线零碰撞、零摇杆救援，其他模型存在人工干预；实机图 7 亦为多个编号目标。因此随机单目标迷宫测试应引用仿真评估，不能把所有实机模型都称完全自主成功。[原论文 §IV-C/D、图 5–7、表 II](https://arxiv.org/pdf/1805.07095)

## 4. Chiang 等，AutoRL，2019

[Learning Navigation Behaviors End-to-End with AutoRL](https://arxiv.org/html/1809.10124v2)，[Google 作者页：RA-L 4，2007–2014，2019](https://research.google/pubs/learning-navigation-behaviors-end-to-end-with-autorl/)。

必须拆开 P2P 与 PF。P2P 用激光与相对最终目标的短历史直接输出连续线/角速度，DDPG 从随机策略训练，无示范；外层搜索奖励和网络规模。其定位依赖传统方法，不能称完全无定位辅助。§IV-C2 的三栋未见建筑、5–10 米起终点测试平均成功率 89%；§IV-F 有约 13.4 米实机 P2P。[原论文 §II、§III-A/B/E、§IV-A/C/F、附录 B](https://arxiv.org/html/1809.10124v2)

**PF 使用外部给定路点，实机 80.6 米长路线属于 PF，不能作为无路点 P2P 的长程证据。**论文把 P2P 定位为可绕角的局部行为，明确不解决走廊迷宫寻房、跨房间的大尺度局部极小问题；P2P 随目标距离增加而退化。计算也包含大量独立训练与超参搜索，不能按单个最终策略的训练量估算。[原论文 §III-B/C、§IV-B/D2/F、§V](https://arxiv.org/html/1809.10124v2)

## 5. Wijmans 等，DD-PPO，ICLR 2020

[DD-PPO: Learning Near-Perfect PointGoal Navigators from 2.5 Billion Frames](https://arxiv.org/pdf/1911.00357)，本次核验v2的§4、§6、表1、附录C与图13–14，及[作者机构页面](https://research.google/pubs/dd-ppo-learning-near-perfect-pointgoal-navigators-from-25-billion-frames/)。这是一个最终目标坐标的PointGoal任务，未知室内场景中没有提供障碍地图或外部路径规划器；视觉编码器＋循环LSTM策略输出动作，PPO＋GAE训练。最强Depth分支使用深度图、GPS+Compass更新的相对目标及上一动作，动作是前进0.25米、左右转10度、停止，不能等同于本项目车体连续drive／steer动力学。

论文评估长／难目标任务，并在20–25米距离分组分析性能；采用25亿**智能体交互步**、64块GPU，不能把这些交互步和MuJoCo积分物理步等量比较。训练奖励含测地最短路径进度和SPL，属于训练期几何信息；无示范PPO不等于训练阶段也完全没有地图几何。无GPS+Compass的RGB设置仍远弱于主设置，论文没有证明无定位长程导航同样解决；附录后的高层控制器迁移分支也不能冒充基础单策略PointGoal。这里只引用基础PointGoal设置作无运行时地图规划的长程端到端证据。[原论文§4、§6与图13–14](https://arxiv.org/pdf/1911.00357)

以上DD-PPO段由主代理补充，未改任何实验源码／协议或runs，未执行训练，也未在运行中的T398期间提交／同步源树。

## 对当前问题的回答

**有从零 RL、观测/目标直接到控制的实验论文。**若严格要求无运行时规划器、无路点，最清楚的连续控制实例是 Tai 的单目标控制接口、Long 的分散避碰策略、AutoRL 的 P2P 分支；各自的目标序列、动态多机器人任务和局部导航范围必须保留。Pfeiffer 的随机单目标迷宫实验更贴近静态复杂障碍，但最佳方法有训练期专家示范和最短路奖励。

以上没有一篇能直接推出“当前 Stage 车体、间隙、时限和观测下，单一无示范策略应当在现有预算内成功”。也不能从它们推导“纯端到端不可能”。后续若选基线，需要先对齐最终目标是否唯一、连续动作与车体动力学、训练来源、定位信息、地图/难度、成功/碰撞/超时及训练预算；本笔记不授权或注册新实验。
