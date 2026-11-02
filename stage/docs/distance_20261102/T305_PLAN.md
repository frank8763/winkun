# T305：目标条件学习最小实现合同，零导航训练

2026-11-02用户已授权准备今晚有界实验。T305仅工程验证，0native/0Actor/0Critic Adam；没有将T304失败值头接入Actor，也不是完整在线SGCRL已经运行。Stage/Phase隔离。

复用T295八条已实际原生验证的完整目标历史与first-hit证据：逐行用真实四历史pose重建全部13目标字段/107非目标字段、alternate current/next120及最后不足100物理步hold；原行为AR mean/logp必须逐位一致。旧轨迹不算新样本，新goal不复用旧logp作on-policy。

新实现明确有限任务的折扣**决策端点占用**语义：每hold时长折扣gamma^n分配到该hold结束的已观察状态；安全原GOAL后同坐标吸收，CONTACT优先进入碰撞token，真实task cap剩余质量进入timeout token，提前collector中断的质量是未知而非失败/负例。不是连续2ms路径占用、0.2m首次安全到达概率或原论文继续MDP的无偏保证。记录完整质量为1的合成闭式例子；collector未知既不作正例也不作对比负例，排除后的学习分布/偏差仍须在训练协议披露。

实现Critic的行为120+前latent2+动作2输入，查询为每个anchor自己的body-frame坐标+终止token；跨样本负query先按当前anchor实际pose变换，禁止把不同body原点的相对向量直接混排。查询绝对坐标只作离线标签变换，不进入Actor/网络全局地图。初始随机Critic只用于InfoNCE+.01LSE、梯度与序列化检查；没有训练、没有价值有效性门槛通过。

原有Gaussian-copula AR条件采样保留前均值导数，用标量有限差分核验当前/前均值梯度。Actor对query目标使用完整重建120，Critic行为context仍原goal；这是**行为续行到新查询policy之间的待验证迁移**，不能用代码跑通当作同policy目标Q恒等式已成立。若今夜选择标准独立动作探索以形成明确新CRL基线，必须另注册分布变化/初始函数/数据支持与方法比较，不能悄改旧协议、不能把旧AR轨迹logp当新iid数据。

32真实记录状态上的随机Critic→Actor可微接口只求梯度、不作优化器步。Actor/随机Critic权重前后SHA相同，不能据随机梯度方向自动启动学习。真实控制门槛仍false，T304反例保留。

producer600秒/audit600秒，own1MiB（数组≤128KiB，余协议/日志/报告），global1GiB/free40GiB。source/inputSHA锁定、零步预检、精确远端树后唯一launch；结束父/timeout退出后独立零步重算全部数组/梯度/合同，全原始备份。没有新6m/7/8训练，旧五米模型/失败/HD视频不动。

近邻：本轮复核[SGCRL正文§3](https://arxiv.org/html/2408.05804)与[作者critic/actor源码](https://raw.githubusercontent.com/graliuce/sgcrl/main/contrastive/learning.py)、[CRL原论文页](https://arxiv.org/abs/2206.07568)；作者的InfoNCE+LogSumExp与目标采样提供参考，但无限时域/完全状态/标准采样假设不自动成立。只复用数学定义，未复制运行作者JAX库/安装新依赖，不将数据合同称原创方法。
