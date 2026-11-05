# T381之后：人控动作序列初始化的滚动训练教师

2026-11-05。仅Stage文献与既有证据阅读；本说明不注册、启动、续跑任何任务，不执行native、Actor或optimizer。T381已经结束，原始失败、模型、人工记录和归档不改。下一项应检验**从真实learner状态出发、以人控动作序列为nominal、每次只执行首个短控制再重规划**的教师机制；不追加T381的seed、候选网格或12秒开环执行次数。

## 已有证据与要排除的机制

[T381注册合同](T381_PLAN.md)、[原始summary](../../runs/t381_native_mppi_teacher/summary.json)、[独立审计](T381_AUDIT.json)和[事后只读动作支持检查](T381_POSTHOC_SUPPORT.json)表明：

- 教师起点是冻结T379真实前8秒的完整状态；nominal为恒定当前mode `[0.8007401228, 1.0]`。33条候选、12秒预测、drive/steer标准差`[.20,.40]`、每5个`.2s hold`保持一次噪声、裁剪`[-1,1]`，均为原预注册配置。
- 33条预测全部CONTACT、0条GOAL；实际执行的候选steer全部为正，最小`+.081195`、p05/p50/p95为`.338224/1/1`，首控制范围`+.096998..+1`。因而原有限采样没有检验负转向恢复；不能从latent Gaussian推断执行动作覆盖充分。共同`1000*CONTACT`只是softmax共同偏移而抵消，碰撞惩罚并未漏加。
- 人控同一时刻steer为`-.1`，但人控与learner位置差`.127950m`、yaw差`8.187581°`。这是nominal方向的线索，**不是该learner状态的专家动作标签或可恢复证明**。
- 一次加权序列被开环执行，任务时刻`16.742s`接触，未完成计划12秒窗口；终止后没有Actor续行。第二pass是精确重生，不是第二独立样本。两pass合计77命令/364274物理步，0Actor/optimizer；没有成功训练候选。

这些结果支持改变nominal及执行反馈合同的下一项工程pilot；尚不能把“饱和nominal”认定为唯一根因，也不能断言MPPI无效、场景无解或人控后缀必能恢复。

## 四项一手来源及直接对照

检索/读取日期2026-11-05。以下为定向阅读范围，不声称逐页通读、穷尽检索或运行作者代码；未把摘要当全文证据。

| 一手来源与实际读取范围 | 可采用的具体机制 | 对本项目的边界 |
|---|---|---|
| **[1] Williams等，Information Theoretic MPC for Model-Based Reinforcement Learning，ICRA 2017，[作者托管PDF](https://nolanwagener.github.io/media/mppi/paper.pdf)。** 定向阅读§III末的均值匹配限制、§IV-B与Algorithm 2、§V-C及AutoRally设置段。 | §IV-B/Alg.2从当前状态预测，只执行`u0`；未执行序列移位，作为下次重要性采样nominal。§V-C将短预测与持续重算列为缓解多步模型误差的机制。 | T381一次加权加长开环执行，与上述闭环合同不同；其裁剪/块噪声/自定义cost也不能自动等同论文重要性加权推导。论文指出一般非线性/多峰情形的均值匹配不足可能性；有限样本不保证最优或安全。赛车实验另有全局cost-map，不是无地图导航证据。 |
| **[2] AutoRally作者实现，[固定commit `dc265a22f982c4f65be7a3b5758ce5fac8803a9b`](https://github.com/AutoRally/autorally/tree/dc265a22f982c4f65be7a3b5758ce5fac8803a9b)。** 读取controller声明/许可证头、`computeControl`完整函数及主控制循环相关段；未编译/执行。 | [`mppi_controller.cut` L400–460](https://github.com/AutoRally/autorally/blob/dc265a22f982c4f65be7a3b5758ce5fac8803a9b/autorally_control/include/autorally_control/path_integral/mppi_controller.cut#L400-L460)取首控制L439–442，移位及尾部初始化L447–455；[`run_control_loop.cuh` L69–85](https://github.com/AutoRally/autorally/blob/dc265a22f982c4f65be7a3b5758ce5fac8803a9b/autorally_control/include/autorally_control/path_integral/run_control_loop.cuh#L69-L85)刷新状态、求控制并发布。 | 这是机制交叉核验，不是Stage移植或复现。代码的模型、CUDA、平滑和发布频率不匹配本native车辆；不复制其plant状态推进逻辑。源码带BSD许可证头，若以后复用须保留相应声明。 |
| **[3] Zhang/Kahn/Levine/Abbeel，Learning Deep Control Policies for Autonomous Aerial Vehicles with MPC-Guided Policy Search，arXiv `1509.06791v2`（2016-02-16），[全文PDF](https://arxiv.org/pdf/1509.06791v2)。** 阅读引言训练/部署合同、§III/Alg.1、§IV-A/B与Alg.2相关段、§VI限制。 | MPC在全状态可见的训练环境产生数据，神经策略只接传感器观测；以当前状态滚动求解。它对教师与学生的行为分布施加一致性目标。 | 原方法用离线iLQG参考及相应surrogate cost解决短视野的长程障碍规避；并非只加collision penalty就保证成功。人控动作nominal+native采样并未实现其DDP/BADMM/分布匹配，因此只能借鉴训练教师路线，不能称完整GPS复现。 |
| **[4] Levine/Finn/Darrell/Abbeel，End-to-End Training of Deep Visuomotor Policies，JMLR 17(39)，2016，[全文PDF](https://jmlr.org/papers/volume17/15-522/15-522.pdf)。** 定位阅读§3.2、§4.3与§7相关段；不声称读完40页或复核全部实验。 | 轨迹教师在full state上优化，学生在对应真实observations上监督学习；§3.2说明直接监督的小误差会造成状态分布漂移，交替优化用于缩小教师/学生分布差异。 | 不能将另一条轨迹同一时刻的动作直接配给learner观测。论文对显著环境变化的泛化和训练期full-state要求也有明确限制。Stage单条教师恢复及动作拟合改善均不足以推出闭环成功或跨场景泛化。 |

## 下一项建议：仅形成新协议，不在本文启动

以下是结合本地失败与上述机制的**项目推论**，不是论文已证明的Stage结论。改变nominal与闭环执行两项，只能作为组合工程可行性pilot报告；若要归因于任一单项，需以后另有事先锁定的对照，不能事后补网格。

1. **固定一个真实learner偏离点。** 沿用原8秒开发状态的完整native积分、速度/执行器、previous实际动作/时间、完整当前120及四帧history、真实Actor hidden。以现有成功人控的对应**动作后缀序列**为初始nominal；不移植人控pose、history或hidden，不清零实际车辆状态，不加中间路点。人控同时时钟只规定初始索引，不证明状态对齐。
2. **原生模型检验整个候选动作序列。** 所有候选从该次真实当前snapshot起步，经已核验native步进与逐物理步接触判据。教师优化使用当前full state及当前观测，学生记录使用实际新观测；不能只拿已存人控控制当反馈策略。下一次nominal用未执行序列移位并按预定规则补尾部，不每次重新冻结为饱和learner mode。
3. **执行前明确可行性拒绝。** 候选预测若接触即不进入“可执行教师”集合；所有候选均拒绝时记`NO_FEASIBLE_TEACHER`并停止，不把一组全碰撞控制再平均后当教师。若生成加权序列，须从同一完整snapshot额外做原生校验；校验拒绝也停止，不能看到结果后改选best sample。该门只提供有限horizon、精确模型下的拒绝证据，不保证未来可达或实际系统安全。
4. **预测长度与执行长度分开。** 一次最多执行首个`.2s hold`，再取真实新状态/观测重新规划；保留前一次未执行nominal。short execution不能简单等同short prediction：短视野可能漏掉后续障碍或陷入局部选择，必须预先固定预测窗和终端规则，不能逐次看结果调窗。Actor hidden及观测中的previous action始终跟随实际教师动作更新。
5. **先验证教师，再讨论学习。** 新协议只允许一个固定开发起态、一条预选人控nominal来源、一组固定采样与最多重规划次数、一条实际轨迹及一次精确重生；0Actor/optimizer，不同时训练新模型。独立锁定全部候选/校验/执行/重生的命令、物理步、墙钟、完整存储最坏字节与全局1GiB容量后才可注册，不能继承或续用T381预算。无可行控制、接触、超时、预算删失分别保留，不补seed/幅度/轮数。
6. **教师成功仍只算训练辅助轨迹。** 只有实际无接触GOAL才可另立纠偏监督协议；在每个实际learner/teacher状态上配对当前观测与该状态求得的教师动作，ownership与Actor proposal/logp分存，不能冒充on-policy RL。未来部署仍是单Actor直接控制；必须另做冻结学生闭环、旧任务保持及独立重载/留出验收，不能把教师辅助成绩写成Actor自主或稳定六米。原0.57m/56s与新0.70m/80s课程继续分列，7/8米未进入。

本文只提出上述有界机制方向；没有实施参数搜索、native预测、学生蒸馏或新的成功结论。标准MPPI/GPS及训练期规划辅助本身均不构成原创性证明。
