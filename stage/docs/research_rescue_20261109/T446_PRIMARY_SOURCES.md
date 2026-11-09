# T446：干净单Actor联合迁移的一手依据与合同边界

2026-11-09。仅Stage；本次只查作者论文正文及作者源码，0模型前向、0原生交互、0训练。项目事实以[T445结论](T445_FINDINGS_AND_NEXT.md)为准；论文不是本项目通过证据。

## 有限一手核对（四篇）

- **Policy Distillation，Rusu等，ICLR 2016。** 正文§3.3/§4及附录明确用多个教师产生监督目标，将行为整合进学生；原多任务实验有按任务切换的输出头，不能当作本项目共享两维动作头的直接验证。论文还分别测教师起态之外的泛化，提示低监督误差与导航表现须分开。[作者正文](https://arxiv.org/html/1511.06295v2)
- **Actor-Mimic，Parisotto等，ICLR 2016。** 正文§3.1以多个专家的策略目标监督单网络；教师分布与学生分布采样均讨论过，实证中学生采样更好。因此本轮固定成功教师轨迹预热有先例，但不等于已经处理学生分布偏移。原方法是离散动作softmax交叉熵，本轮连续命令回归只是相关工程路线，不能称原算法复现。[作者正文](https://arxiv.org/html/1511.06342)、[作者实现](https://github.com/eparisotto/ActorMimic)
- **Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs，Ni等，ICML 2022。** 正文§4/§5.2比较独立/共享循环编码器及历史长度；独立编码器在其测试中有优势，最优上下文长度随任务变化。它支持先建立普通循环基线，不保证GRU解决本场景；其主体为off-policy actor-critic，也不直接证明独立V的PPO优越。作者实现将历史按序送入RNN并在单步部署传递hidden。[作者正文](https://arxiv.org/html/2110.05038v3)、[作者循环Actor源码](https://github.com/twni2016/pomdp-baselines/blob/main/policies/models/recurrent_actor.py)
- **Proximal Policy Optimization Algorithms，Schulman等，2017。** 正文式6–7和Algorithm 1先由旧策略采样，再用同一采样策略分母构造比率；§5给出V及截断GAE。BC教师历史不能因放入同一损失就变成当前学生on-policy PPO样本。作者runner同时记录采样动作、旧logp、V和终止mask，再逆序计算GAE。[作者论文](https://arxiv.org/pdf/1707.06347)、[作者PPO runner](https://github.com/openai/baselines/blob/master/baselines/ppo2/runner.py)

## 本轮应锁定的工程合同（以下为本项目推导，不是论文保证）

1. **部署单Actor。** causal120→128 Tanh→GRU64→两维`1.25*tanh`，所有权重可训练；不冻结legacybase、不叠残差、不按任务选教师。教师只提供训练数据；部署hidden从回合起点独立递推。教师hidden不能作为学生hidden，完整历史必须用当前学生参数重算；顺序、padding mask、回合reset及全历史BPTT须一致。
2. **直接动作目标及截断边界。** 标签为实际执行命令，须区分教师原始均值、潜变量和clip后动作。若`a=clip(z,-1,1)`，观察到`a=1`只能约束`z>=1`，不能反推`z=1`；`a=-1`同理。内部平方误差、边界单侧hinge可作为有限监督代理，但它不是截断高斯分布KL、潜变量真值或校准似然，也不识别边界外均值。不得对边界动作直接`atanh`后声称取得正确高斯目标。拟合RMSE仍应另按实际执行动作计算，报告逐组及最坏误差，不能只报hinge趋零。
3. **BC与PPO的数据账分离。** 1500次联合BC，旧组总权重.5、宽六米总权重.5，每批8旧组序列+2宽组序列；它是固定示范辅助预热。十组各自RMSE≤.03且旧LEFT5m首10步最大误差≤.10才进入预注册的两轮、每轮4条新采集学生PPO，否则PPO为0。若PPO中保留BC，监督项单列，不给历史教师轨迹伪造当前logp/优势/importance ratio。对新学生高斯探索，保存实际采样的clip前潜变量及采样时旧logp，重算同一变量的当前条件密度；若改用clip后动作密度，边界必须按尾概率质量处理，不能套连续高斯点密度。
4. **独立V与物理时间GAE。** V不反向改变Actor参数；新学生轨迹的终止、截断和bootstrap语义分别锁定。若每条控制命令跨越不同原生步数，折扣与GAE时间单位必须预注册并一致；不能把控制命令序号折扣默称物理时间GAE。旧成功教师回报不是当前学生V/优势的无偏标签。
5. **有限结论。** 有限历史拟合通过只允许进入有界学生闭环检查，不能推出闭环保持、分布外泛化或硬件安全。最终同一模型329固定任务分别报告旧275与宽六米54；本轮不含fresh16/独立reload78，不能补写通过。PPO clip不是安全证书，T445未通过的几何净空重构不能进入安全奖励或证明。联合迁移、MLP/GRU、BC、PPO与独立V均有先例；本轮是训练/架构联合pilot，不能单独归因容量，不能声称创新或从零纯RL。
