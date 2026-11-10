# T453：学生早期状态纠正与独立新交互PPO的一手依据

2026-11-10。仅Stage文献复核；没有原生交互、训练、优化器更新或修改既有证据。结合[当前状态](../RESEARCH_STATUS.md)、[T451复盘](T451_FINDINGS_AND_WINDOW_REVIEW.md)及[T452原报告](T452_FINAL_REPORT.json)。这是一项有限联合工程pilot的依据，不是算法创新或成功保证。

T449 recovery研究父模型为宽54/54、旧274/275，合328/329；T451整条恢复轨迹400BC后为322/329，PPO为0。T452证实同一学生自主4秒后T255接管可以安全GOAL，12/24秒接管均碰撞。故可检验早期学生访问分布上的纠正，不能把4秒以后或整个早期区间宣布为已验证的教师正确域。

## 本次实际查看的一手材料

| 材料与范围 | 支持及反证边界 |
| --- | --- |
| Ross、Gordon、Bagnell 2011，PMLR正式[全文](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)，§2–4、Algorithm 3.1、Theorem 4.2 | DAgger采集当前学生或混合策略访问的状态，查询专家标签并聚合；原算法没有“只保留成功轨迹”的条件。支持优先覆盖学生实际状态，而非只回归教师接管后状态。有限前21状态掩码、神经网络有限Adam及单任务修复不满足完整定理合同；专家可恢复性、损失与无后悔条件不能由一次接管成功代替。 |
| Schulman等2017，[PPO原文v2](https://arxiv.org/pdf/1707.06347v2)，§3、§5、Eq.7、Algorithm 1 | 每轮先运行行为策略，再用所采样动作、优势和固定旧策略密度优化若干epoch。论文讨论了递归网络的分段采集/截断优势，但没有证明Stage的GRU实现或2轮×4轨迹样本量充分。裁剪的是代理目标中的项，不能据此承诺真实成功集合不退化。 |
| OpenAI Baselines作者机构实现，固定commit `ea25b9e8b234e6ee1bca43083f8f3cf974143998`：[runner.py](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/runner.py#L17-L62)、[ppo2.py](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/ppo2.py#L138-L164)、[model.py](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L41-L83)、[LSTM](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/models.py#L148-L166)；[MIT许可](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/LICENSE) | runner保存实际动作、行为neglogp、done及采集段起始hidden；递归minibatch按环境整段抽取，保留时间顺序，使用起始hidden和reset mask。该实现会复用采集段起始hidden，并非每次从真实h0重算；Stage全前缀重建是更严格的本地计算合同，不能冒称原实现已经这样做。本次只检查实现，没有移植代码。 |
| Rusu等，[Policy Distillation全文](https://arxiv.org/pdf/1511.06295)，§3.2–3.3 | 支持将已有策略函数作为单独监督目标及交错多任务训练。原文主要为离散Atari/Q输出且多任务使用任务输出头，不能直接证明Stage单输出连续递归Actor的218+10锚点能保持闭环能力。 |

## 对本轮有限合同的解释

1. **状态归属和动作归属分开。** 原学生从h0真实走出的索引0…20对应0…4秒，共21个访问状态；索引20的状态仍由学生前缀诱导，但该处执行动作已经是首次教师动作。BC掩码应按这21个状态定义，不能直接用“执行动作为学生”掩码替代。教师标签查询使用各状态的真实因果输入及T255既有历史语义；学生训练从其完整前缀h0重建，不输入教师hidden。4秒分支成功只验证这次接管及其后续，不能逐一认证之前20个反事实纠正动作。此处是结合T452的研究推断，不是DAgger定理结论。
2. **纠正与保持分别归一、分别报告。** 前21状态教师标签可形成有限masked BC；若协议保留教师诱导后缀，须预先固定一个小总权重并与边界标签去重，避免后缀时长决定实际权重。218条旧成功history与10条自身成功history循环均衡保持时，目标应是本轮更新前328研究父模型的函数，在同一完整history上分别从各自h0计算；不要求父模型复现历史采集动作，不把00失败函数作为保持目标。有限锚点上的小误差仍须接受完整闭环反证，T451已经提供这种反例。
3. **BC结束后重新采集PPO数据。** 两轮各4条新学生探索轨迹及其exact，在各轮采集前固定该轮行为Actor。仅实际学生随机动作及真实奖励/终止进入PPO/GAE；碰撞和超时保留，exact不是新增独立样本。教师标签、恢复混合轨迹、旧history和保持目标只能进入另列监督项，绝不能伪造学生logp混入PPO比率。第二轮必须来自第一轮更新后的学生。该边界来自[PPO采集更新算法](https://arxiv.org/pdf/1707.06347v2)及[实际runner](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/runner.py#L17-L62)。
4. **每次策略计算重建真实递归历史。** 旧密度对应冻结行为参数及其完整历史；当前密度/BC/保持函数对应当前参数从真实h0重算的history。保留序列、终止/reset和实际采样动作；不能随机打散为独立状态、每步清零，或把旧参数中段hidden当作当前参数的准确hidden。辅助BC/保持损失可越过PPO裁剪项改变参数，因此4epoch是更新上限，不能当信赖域或不退保证。密度、历史、数值和预算合同仍须通过；不再用BC拟合阈值阻断预注册的新交互PPO。
5. **结论只由预定验收决定。** 固定一次机制、seed、权重与BC上限，最多2轮×4新轨迹、每轮最多4PPO epoch；无拟合失败后加步数或新网格。最终无教师学生跑完整329，必须保留父模型原328成功ID并修复00。16fresh与78独立原生重载本轮不自动消费，须另有完整预算；开发329通过也不等于泛化。此联合pilot同时改变纠正覆盖和实际RL执行，无法将收益单独归因于任一组件，失败也不否定全部DAgger或PPO。

采用：有限学生状态纠正、独立新交互PPO、标准递归历史与函数保持。排除：整条混合恢复伪装自主成功/新on-policy数据、失败后补时刻或seed、拟合门槛造成再次0PPO却称RL失败。上述21/218/10、2×4和4epoch均由Stage证据与预算选定，文献不提供这些数值的充分性保证；标准组件组合不称原创。
