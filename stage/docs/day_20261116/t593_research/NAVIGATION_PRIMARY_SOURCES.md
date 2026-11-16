# T593：无地图导航的一手论文与作者实现核验

核验日期：2026-11-16。只处理 Stage；本项只有文献/源码阅读，未训练、未步进模拟器、未变更模型或运行目录。本次读到的是指定本地资料与公开论文，不是全部历史聊天。

结论：存在可借鉴且有作者代码的方法；最贴近的是“先示范再自主 RL”和“容易/前沿/随机任务共同训练”。它们支持继续研究，但没有一篇证明能直接修复本项目的保持—恢复冲突。下列主选是 **2 篇期刊论文 + 1 篇明确标注的会议方法对照**，并非三篇期刊。

## 1. Reinforced Imitation：最贴近训练示范与单策略自主部署

**书目与归属事实。** Pfeiffer、Shukla、Turchetta、Cadena、Krause、Siegwart、Nieto，*Reinforced Imitation: Sample Efficient Deep Reinforcement Learning for Map-less Navigation by Leveraging Prior Demonstrations*，IEEE Robotics and Automation Letters，2018，3(4):4423–4430，DOI [10.1109/LRA.2018.2869644](https://doi.org/10.1109/LRA.2018.2869644)。作者[论文全文](https://arxiv.org/pdf/1805.07095)第 2 页脚注直接指定 [ethz-asl/rl-navigation](https://github.com/ethz-asl/rl-navigation)，不是凭仓库名称猜测的“官方实现”。仓库 README 同时给出期刊书目。

**方法与证据事实。** 先由示范做 IL 预训练，再用 TRPO/CPO 自主强化学习；传感器与相对目标直接映射为运动指令。示范由训练阶段的 ROS move_base 产生。论文的机器人试验比较碰撞与人工干预；部分模型存在干预，不能将整篇真机证据概括为全程零接管。论文也报告 CPO 初期成功率下降及训练图只展示多次 RL 中最佳一次。[全文 §III、§IV、表 II](https://arxiv.org/pdf/1805.07095)

**部署与代码边界。** 学得策略不要求运行时 planner/教师；但原论文确实使用训练阶段规划器。如果用户的“禁止外部规划器”覆盖全部阶段，就不能原样照搬其示范生成。官方代码采用 TensorFlow、Python 2.7、ROS Indigo/Kinetic、stage-ros，是旧依赖实现；本次没有执行或宣称已复现。[作者代码与运行要求](https://github.com/ethz-asl/rl-navigation)

**对 Stage 的建议（推论）。** 可迁移的是“合格完整示范提供初始化，随后在学生真实到访状态上自主学习”的训练分工；可以只使用已核验的 Stage 自身成功/恢复数据，继续保持单 Actor 部署。它不是“只要 BC 拟合 3 条恢复就能保住原 39 项”的证据，也不提供历史所有任务保持的保证。

## 2. NavACL-Q：最贴近稀疏成功、精细操控与保持课程

**书目与归属事实。** Xue、Hein、Bakr、Schildbach、Abel、Rueckert，*Using Deep Reinforcement Learning with Automatic Curriculum Learning for Mapless Navigation in Intralogistics*，Applied Sciences，2022，12(6):3153，DOI [10.3390/app12063153](https://doi.org/10.3390/app12063153)。[期刊正文的 Supplementary Materials/Data Availability](https://www.mdpi.com/2076-3417/12/6/3153)直接链接作者 [ai-lab-science 仓库](https://github.com/ai-lab-science/Deep-Reinforcement-Learning-for-mapless-navigation-in-intralogistics)。

**方法与证据事实。** 分布式 SAC 配合成功概率预测，混采 easy/frontier/random 初始任务；easy 用于减轻遗忘。NavACL-Q 在几何任务特征中加入初始 Q。部署由 LiDAR/图像等观测直接输出连续速度，无运行时规划器。论文的主验证在 Isaac Sim，不能当真机部署完成。作者明确指出：负样本过多会让课程失效，较大目标夹角下仍受部分可观测性限制。[期刊 §3.4、§4、§5.3–5.4](https://www.mdpi.com/2076-3417/12/6/3153)

**代码边界。** 仓库公开 SAC、网络、任务类、课程代码；[nav_acl.py](https://github.com/ai-lab-science/Deep-Reinforcement-Learning-for-mapless-navigation-in-intralogistics/blob/main/nav_acl.py)能核对三类采样与 Q 特征。该文件还导入 `utils`、`isaac_q_helper_functions`、`per_buffer`，本次打开的仓库根目录未见这些文件；没有核验完整环境/资产可开箱恢复，不能称完整可复现包。

**对 Stage 的建议（推论）。** 最值得迁移的是：自主训练任务持续包含已知成功原起点、恢复前沿和随机覆盖；不能只让“原成功轨迹上的动作保持”代替原起点闭环保持。当前价值网络尚未证明可靠，不应因为论文名称带 Q 就立刻加 Q 课程；先用冻结真实成败分组可减少新增机制。训练只在回合起点选择任务，不在部署途中给路点或切换教师。

**与原 NavACL 的区别。** 原 NavACL 是 Morad 等的 RA-L 论文，DOI [10.1109/LRA.2020.3048662](https://doi.org/10.1109/LRA.2020.3048662)。其[作者全文](https://arxiv.org/pdf/2009.05429)展示 RGB、LSTM、离散运动原语以及地面机器人/无人机实验；训练课程使用测地距离等几何信息。本文只把它作为 NavACL-Q 的来源，不将后者仓库冒充原 NavACL 官方代码。本次未找到经原作者论文/主页直接确认的原 NavACL 完整公开实现。

## 3. RLPD：有官方代码，但不是期刊或真机导航解决方案

**书目与归属事实。** Ball、Smith、Kostrikov、Levine，*Efficient Online Reinforcement Learning with Offline Data*，**ICML 2023 会议**，PMLR 202:1577–1594；[正式出版页](https://proceedings.mlr.press/v202/ball23a.html)。[论文](https://proceedings.mlr.press/v202/ball23a/ball23a.pdf)直接链接作者 [ikostrikov/rlpd](https://github.com/ikostrikov/rlpd)。

**机制事实。** 在线 RL 使用每批约 50% 既有数据、50% 在线 replay；critic LayerNorm 控制值外推，配合 ensemble/较高更新数据比。原设计无离线预训练/BC 保持项，也不靠从示范状态 reset。评估是 D4RL、Adroit、V-D4RL 等基准；AntMaze 不能当现实复杂无地图车辆导航验证。[论文 §4–5](https://proceedings.mlr.press/v202/ball23a/ball23a.pdf)

**可核代码与边界。** [train_finetuning.py](https://github.com/ikostrikov/rlpd/blob/main/train_finetuning.py)给出 `offline_ratio=0.5`、默认 0 次离线预训练和离线/在线分开采样；[rlpd_config.py](https://github.com/ikostrikov/rlpd/blob/main/configs/rlpd_config.py)给出 10 个 Q、2 个 min-Q、critic LayerNorm。部署不需运行时规划器；迁移需重新建立奖励、终止、动作语义与数据接口。

**对 Stage 的建议（推论）。** 若将来重新进入自主 RL，可把真实历史成功/失败/恢复转移与新学生转移放入分离的池，检验既有数据是否帮助探索。它不能替代当前的初始化保持审计；Q 有界也不等于动作排序正确、碰撞安全或旧 39 项不退化。不能把“增加 replay 比例/LayerNorm”声称为项目原创或已经有效。

## 未入主选的候选核验

- **APPLR / APPL：不符合当前部署约束。** [作者项目页](https://www.cs.utexas.edu/~xiao/Research/APPL/APPL.html)明确 APPLR（ICRA 2021，会议）在运行时输出传统运动规划器的参数。其期刊统一框架 APPL 是 Robotics and Autonomous Systems 154 (2022):104132，[作者全文](https://www.cs.utexas.edu/~xiao/papers/applx.pdf)。尽管用示范、纠正、RL，完整部署仍包含传统规划器，不能当本项目允许的纯单策略方案。
- **AutoRL：期刊是真的，名字相似的仓库不能算其实现。** [Google 作者出版页](https://research.google/pubs/learning-navigation-behaviors-end-to-end-with-autorl/)确认 *Learning Navigation Behaviors End-to-End with AutoRL* 是 RA-L 4 (2019):2007–2014，训练层搜索 reward/网络。点到点策略与路径跟随、长距离 PRM-RL 必须区分；[作者博客](https://research.google/blog/long-range-robotic-navigation-via-automated-reinforcement-learning/)的长距结果结合采样规划，不符合当前部署约束。本次未确认 2019 导航论文完整作者公开代码；[google/brain_autorl](https://github.com/google/brain_autorl) README 列的是另外两篇算法/架构搜索论文，不能拿来充数。其大规模搜索也不适合直接套入当前有界窗口。
- **“Learning to Walk and Drive with Reinforcement Learning”：本次按此完整标题未核验到匹配的一手论文。** 不猜作者、出版场所或官方仓库，不作为证据；需要精确标题或 DOI 才能进一步消歧。

## 结合 T591/T592，下一步应解决什么（项目建议）

本地入口是 [RESEARCH_STATUS.md](../../RESEARCH_STATUS.md)、[T591 报告](../t591/FINAL_REPORT.json)、[T592 报告](../t592/FINAL_REPORT.json)。按当前任务给定结果：T550 保留；T591 恢复接管 3/3，但原 49 为 34/49，原 39 成功丢 5 且无新增；T592 指出其中 4 项在 T562 初始化时已经失败，另 1 项是学习后 success→contact。因此不能把五项全归因为 600 步遗忘，或把接管成功当原起点能力提高。

建议首先要求候选学生在任何恢复学习前，以真实 carry/history 从同一原起点保持全部已知成功；这处理的是实际初始化缺口，不是再加一个保持权重。之后才有资格分别检验示范初始化、自主数据课程或混合 replay。三篇方法都没有豁免原起点闭环验收，更没有保证本项目通过。本文不授权新实验，不续跑 T591/T592，也不推荐无界参数搜索。
