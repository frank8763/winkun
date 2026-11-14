# T552 三臂联合监督、真实RL与局部响应保持：一手来源复核

检索/阅读日期：2026-11-14。Stage only；本次仅阅读与文档写入，0模拟器、0训练、0优化器、未修改实验代码。研究对象为冻结T550父策略上相同固定容量、零初始化平滑神经残差的三臂工程对照，每臂至多4轮；正式预算、截止与接力规则以锁定协议为准。本页不是启动授权。

## 结论与最接近工作

当前方案只能称为**联合监督恢复/保持、真实策略梯度、有限邻域响应保持的工程对照pilot**。恢复6项旧成功、保持23个已知成功家族、加入4条真实扰动后的自主恢复history，可以检验有限开发集合上的修复与非退化；不能直接证明算法新颖性、全域安全、跨场地泛化或长期持续学习能力。

- **A：联合监督恢复与保持。** 与历史经验重放、自蒸馏、DART式受扰恢复示范相近。已有成功history可以作动作监督；T551外部强制转向帧须排除动作标签，之后真实自主恢复帧可以作为有明确来源的监督数据。
- **B：A加新策略真实RL。** 最接近CLEAR的稳定性/可塑性组合原则，以及2026年CIDER的在线新任务RL加冻结历史教师行为保持。Stage不用Critic、HIL运行时教师或视觉任务预处理，不能称复现CLEAR/CIDER。
- **C：B加有限差分响应保持。** 最接近TaSIL附录C.2的成对动作增量匹配；与CAPS相似但不等同：CAPS压低相邻状态动作差，TaSIL匹配教师与学生的动作响应差。若惩罚的是残差增量，则它等于保持冻结父策略的局部增量响应；并不自动学到纠错响应。

A/B/C须共享监督history、预定配对状态、容量、初值和评估协议。B与C的自主采集会随各自策略改变，这些属于各臂实际on-policy分布，不宜为追求“相同数据”而把另一臂或旧策略轨迹冒充本臂on-policy。A与B比较新增RL机制，B与C比较响应项；多轮路径差异应作为整体机制结果报告。

## 一手来源与实际阅读范围

### CAPS：直接平滑约束及其代价

Mysore, Mabsout, Mancuso, Saenko，ICRA 2021；阅读[arXiv 2012.06644v2全文HTML](https://arxiv.org/html/2012.06644v2)第III节公式(1)–(3)、第IV节Table I及Gym/Sim-to-Real讨论，以及[作者项目方法页](https://ai.bu.edu/caps/)。没有执行其代码或重现结果。

CAPS在策略目标中加入时间相邻动作差和状态邻域动作差；论文使用欧氏距离及按噪声/容差设置的高斯邻域。其目标是控制信号平滑。Table I和Gym讨论明确存在更平滑但到达较慢、部分任务数值回报下降的情况。因此，C项权重不能因平滑代理改善就在线加大；真实到达、碰撞、超时仍是验收依据。[方法与结果](https://arxiv.org/html/2012.06644v2)

Stage原始history包含射线、目标、车体及因果历史特征；任意独立高斯扰动可能产生不可能观测。优先用已真实执行、物理有效且输入历史一致的状态对，是本项目的数据有效性选择，不是CAPS的理论保证。仅平滑残差也不能保证父网络或闭环整体平滑。

### TaSIL：成对响应匹配的精确近邻及前提

Pfrommer, Zhang, Tu, Matni，NeurIPS 2022；阅读[arXiv 2205.14812v2](https://arxiv.org/html/2205.14812v2)第2–4节的问题设定/稳定性前提、第5节实验，以及附录C.1–C.2有限差分条件；未逐行验证所有证明。

附录C.2直接匹配两种策略在成对点上的动作增量，即“教师在邻点与原点的动作差”减“学生同样的动作差”。这与拟议C项非常接近。定理依赖专家闭环局部incremental input-to-state stability、可控的高阶光滑性/余项及足够小的模仿误差；有限差分讨论还涉及方向张成、矩阵条件及近似误差。几条自然轨迹对只约束少数方向，不能据此声称完整Jacobian已匹配或继承TaSIL保证。[理论及附录C](https://arxiv.org/html/2205.14812v2)

作者实现已定位：[unstable-zeros/TaSIL](https://github.com/unstable-zeros/TaSIL)，2026-11-14 API读取默认分支commit `4b10cb71a220dac0cf961618b8d68580c4779722`，BSD-3-Clause。实际查看`imitation_learning/bc.py`：由expert rollout构建数据集并调用policy_loss训练；未完成整个损失实现审计，未复制或运行其代码。不能把该文件的阅读称作TaSIL实现复现。

### CLEAR：重放稳定性与新鲜RL可塑性

Rolnick等，NeurIPS 2019；阅读[正式论文](https://proceedings.neurips.cc/paper/2019/file/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Paper.pdf)第2–3节及第4.1节开头，重点核对损失与历史分布处理；未重现其Atari/DMLab实验，未审计作者实现。

CLEAR混合新经验与历史重放，在actor-critic/V-trace训练外对历史策略分布和历史value输出进行保持。其历史数据明确是off-policy，需要相应校正；行为克隆本身不是on-policy策略梯度。对Stage的直接启示是分别标记监督与新鲜RL来源，并报告按ID丢失/新增，不能以净总分抵消遗忘。[CLEAR方法](https://proceedings.neurips.cc/paper/2019/file/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Paper.pdf)

### CIDER：2026年直接近邻与适用范围

Li等，**arXiv预印本2608.21899v1，2026-08-22**；阅读[一手HTML](https://arxiv.org/html/2608.21899v1)第3–4节、第5.3节、第6节及附录C.1–C.2。不将预印本称为已同行评审结果；没有在所读正文中确认可供审计的作者训练仓库。

CIDER在新任务在线HIL actor-critic学习期间，交错使用冻结累计教师在历史档案状态上的reverse-KL保持，采用任务均衡采样及不同梯度路由。正文重点在六个短时程真实机械臂任务，局限段明确指出长时程、稀疏奖励和延迟失败仍有开放问题。Stage的critic-free导航与其传感器、动作、时标和监督方式不同。[方法与局限](https://arxiv.org/html/2608.21899v1)

附录C.2还明确：虽然Actor不直接输入任务ID，系统依赖当前场景信息选择相机裁剪及归一化配置。因此不能将其结果引用为完全任务无关或感知混淆已解决的证据。[部署说明](https://arxiv.org/html/2608.21899v1#A3.SS2)

### DART：受扰后的真实恢复数据

Laskey等，CoRL 2017；本轮阅读[作者项目页](https://berkeleyautomation.github.io/DART/)及[作者仓库README](https://github.com/BerkeleyAutomation/DART/blob/master/README.md)，并定位[正式论文](https://proceedings.mlr.press/v78/laskey17a/laskey17a.pdf)。本轮没有完整阅读论文方法和证明，不能声称全文复核。

其已确认原则是给监督者执行加入扰动，让监督者示范恢复；它学习扰动尺度以近似学习者误差。Stage固定4条扰动恢复不是完整DART：未做其噪声分布优化，也没有通过这些4例建立鲁棒覆盖。外部强制动作与之后自主恢复必须分账。[作者说明](https://berkeleyautomation.github.io/DART/)

## 裁剪高斯、真实on-policy与RTG/RLOO合同

Fujita与Maeda，ICML 2018 [Clipped Action Policy Gradient正式论文](https://proceedings.mlr.press/v80/fujita18a/fujita18a.pdf)：本轮阅读第2节、第3.1–3.5节，重点核对Eq.(4)–(10)、独立动作分量条件、裁剪分布以及奖励依赖未裁剪动作时的例外；未审计全部附录证明。

对执行动作 `a=clip(u,l,h)`，概率必须区分下界原子、内部密度、上界原子：分别为 `Phi((l-mu)/sigma)`、`NormalPDF(a;mu,sigma)`、`1-Phi((h-mu)/sigma)`。边界动作不能使用普通高斯在边界处的PDF。多维独立条件下可加各维log概率；这要求核对实际条件分布。裁剪高斯、截断高斯与tanh变换高斯不是同一分布。上述结论要求转移和用于该估计的奖励只通过裁剪后动作依赖原抽样，否则须按论文3.4处理。[CAPG第3节](https://proceedings.mlr.press/v80/fujita18a/fujita18a.pdf)

作者代码已实际查看：[clipped_gaussian.py，commit 787aabce0b580eb6a8469d427695178dfb79f96e](https://github.com/pfnet-research/capg/blob/787aabce0b580eb6a8469d427695178dfb79f96e/clipped_gaussian.py)，MIT；`sample`裁剪，`log_prob`在上下边界用logCDF/logSF，内部用Gaussian logPDF，最后按动作维求和。只读核对，未复制运行。

若探索是AR相关噪声，简单把各时刻边缘Gaussian/裁剪原子相乘一般不是整段真实似然。必须核对条件历史、隐含未裁剪噪声与实际生成图；可保留真实latent完整score，或实现与裁剪历史相符的条件分布。CAPG的独立分量例子不自动证明任意AR裁剪边缘化正确。

本项目实施含义（推论，不声称论文已验证Stage）：

1. 先完成本轮监督更新，再从实际采集Actor/hash出发收集新鲜随机完整轨迹。旧6/23/4监督history、强制动作及教师轨迹不能伪作本轮on-policy PG。
2. 真实RTG使用所执行动作后的奖励，保留终止、超时、采集删失差异与物理时间折扣；删失不是完整回报。密度须用相同因果输入/历史与采集策略重算。
3. 同一批采集后第一次更新之后，后续参数已经改变。若重复使用旧批，须明确冻结behavior分母和importance-ratio/clipped surrogate；不得称每一步都是新的无偏on-policy REINFORCE。PPO比值裁剪也不是全局信赖域或安全保证。
4. 控制因子裁剪、梯度范数裁剪、PPO概率比裁剪是三件不同的事，协议应明确各自使用处。CAPG降低估计方差的结论不等于提高本任务成功率。

RLOO参考：Ahmadian等，[Back to Basics，arXiv 2402.14740](https://arxiv.org/html/2402.14740)，本轮阅读第2.2–2.3节公式(6)–(9)及问题语境。RLOO对同一条件下独立采样的多个完整输出，使用其他样本均值作leave-one-out baseline。Stage时序RTG版是改写：同行baseline不得含本条动作/未来信息或依赖本条动作的筛选；独立噪声家族、同条件、时间对齐、终止后mask与stop-gradient均须在实现中核验。LLM的整段终端回报实证不证明物理闭环RTG版会降方差，不能把“无Critic”本身作为新颖性。

## 为什么有限邻域保持不等于全局安全或泛化

以下为结合来源与Stage契约的推论：

- 有限history上输出相等，只保护实际约束点；闭环进入新状态后误差可累积。TaSIL正是为此额外引入稳定性和邻域高阶误差前提。T550已出现净成功数提高而6个旧ID丢失，直接要求逐ID非退化验收。
- 局部动作响应小不等于能及时转向/制动。近碰撞、接触切换和任务阈值可能需要敏感响应，平滑损失可能增加延迟；CAPS已报告到达变慢的权衡。
- 冻结骨干减少旧参数变化，却不能修复特征混淆。如果两个物理/历史情境映射到同一残差输入但需要相反纠错动作，任何确定性残差头都无法同时满足。若残差另接原始history，须核对实际输入后再判断；不能笼统称冻结特征消除了aliasing。
- 固定有限容量可能发生真实任务冲突/表示不足；监督零训练误差、局部响应误差或单次接受都不证明不存在冲突。CIDER采用任务可区分的视觉布置并承认系统级预处理选择，不能填补Stage观测可区分性的证据缺口。
- 同布局、同开发家族、已消费扰动上的重复通过属于开发保持/恢复。独立未消费扰动、新起终点、跨布局与实体新场地仍需预先封存评估；本轮至多提供下一轮资格，不能直接晋升为普遍鲁棒。

采用：共同监督基线、真实新策略采集、有限物理有效配对响应消融、逐ID闭环验收。排除：把强制帧当专家动作、把历史监督当新鲜PG、任意无效观测扰动、测试失败后追加方向/幅度/轮数网格、将近邻组合包装为新算法。暂缓：全域稳定性证书、全Jacobian匹配和跨场地泛化主张。

阅读边界：本页没有声称穷尽全网、读全历史聊天、完成全部论文证明/代码审计或重现。对DART明确只读作者说明；CLEAR只核对关键方法部分；其余范围逐项列出。没有借用Phase规划研究。
