# T448：完整经验梯度累积与原循环小批量的固定对照

用户2026-11-09“下一步”授权，先完成T447零原生审计：旧LEFT5对联合梯度余弦+.2367、成熟Adam局部代理变化-8.72e-6，未支持它被完全压住；误差分布于first2s、2–20s和末段，不据此加首2s特殊权重/PCGrad。T447独立10梯度/分段误差重算及远端逐SHA已通过，T446原失败不覆盖。

独立新问题：原每步10条轮转history的梯度估计，与对同一固定219条history的完整分组经验梯度累积，能否在有限优化中产生不同联合拟合与闭环结果？不是无界继续旧预算。本试验有一条明确新机制及匹配旧机制对照；标准full-batch/Adam不称原创。

两臂都从T446同一BC末Actor、V及成熟Actor/V Adam原字节初始化，结构、120输入/GRU64carry、所有数据、group权重、lr3e-4/梯度clip1和64次Adam相同。A为原cyclic_control，使用原轮转游标1500–1563，每步10history；B为full_empirical，每次固定参数上计算全部219history的分组经验损失梯度。B用8条一microbatch只为显存，**完整219条累积后一次clip、一次Adam**，不在microbatch间更新/改变hidden语义。旧8组各1/16，宽左右各1/4；每组每回合先时序均值再回合均值。

每臂64真实Actor Adam＋64独立shadow，独立audit另重构64，各自固定64末模型评估完整329。V不更新，PPO=0；不拿此次BC研究声称完整RL训练。两臂唯一历史样本来源相同、Adam次数匹配，但每步数据呈现/计算更多，**不匹配计算量或总样本呈现数**，不把差别单独归因消除噪声/效率。固定有限对照不增加seed/网络/LR网格，不挑中期最好模型。

PPO准入和新奖励不改动，目的是检验全量联合拟合优化。完整开发分别旧275/宽54，接触/超时/invalid/censored/null分开；即便329全过，本轮没有预留fresh16/78独立原生模型读回，six_gate=False，不自动八米/全图；研究模型不替换安全T255，原窄暂停。部署仍一个plainActor，无地图/规划器/路点/运行教师/路由，Phase完全隔离。

预检4丢弃Actor Adam（2臂各live/shadow1），两路成熟Adam回退/读回一致；full经验损失及梯度范数与T447逐组独立计算的解析总和在1e-12/1e-10内一致；真实单步最大时长测64双臂live+shadow估计须<1600秒。8原生worker首物理前取消/退出24命令，复用原T446冻结初始checkpoint，不另存预检模型。

预算own11MiB：2模型×1.5MiB＋660compact×4KiB＋metadata/protocol/logs5MiB<11MiB，所有最终动作/步/终止和完整数组hash保留；紧凑验收缺完整120的限制明确，独立audit不伪称重建658个全观测。global1GiB/free40GiB不放宽，不删唯一模型/轨迹。本轮无新full训练数据或教师查询。最多14M全部物理、720全部命令，实际最多698（preflight24＋READY/关闭16＋658验收）；worst按原cap锁定。

main3600秒、前2100秒允许学习、audit1200、RAM备份600、source与余量，all6000秒启动落绝对deadline，更早从严。8CPU并行仿真/GPU学习，离线科学不依赖网络/LLM；driver唯一finalizer，数值/协议/空间失败保留停止，不重启或借号续旧预算。父及全worker退后独立128历史Adam/trace/Actor/V/双Adam/动作count终止hash/预算审计，原文件RAM全远端逐成员SHA再精确source树。主流程结束前不外写runs/并发审计或source同步。

工程预检在完成4丢弃Adam及全损失/梯度范数断言后，动态模块worker不可pickle，0native退出；原源/protocol/error在preflight_failed_001保留。仅改本脚本top-level worker入口，finalize新增0Adam、一次完整梯度复核和8worker零物理取消通过；保留全部原错误，无科学配置改变。为原失败协议/源证据预启动metadata3→4MiB、own10→11MiB，仍完整global1GiB/free40GiB，物理/命令/训练/墙钟上限不变。

最终存储重核：源成熟checkpoint未压缩序列化1423007B，后续不增加任何参数/优化器状态键；模型单文件硬上限收紧1.5MiB，2模型共3MiB，meta预留5MiB以完整保留原失败协议和源副本，own11MiB不再扩大，均在launch前落协议。
