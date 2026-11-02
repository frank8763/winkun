# T298：自身辅助目标监督整合，条件通过后一次真实6m RL

Stage-only，Phase完全隔离。用户2026-11-02继续实验授权。T295/T296已结束，辅助goal数据可用但长前缀HPG权重塌缩，不截平/替换密度拿来做PPO。safeT255五米基线、原失败和两HD视频保留；六米最新完整T291 1/54且旧274/275，不晋升。

## 明确算法及范围

参考[GCSL](https://arxiv.org/abs/1912.06088)的自身经验目标条件监督学习思想。固定T295八条真实first-hit成功前缀、两个历史选择的辅助goal，全部120/13目标历史/107其余字段/短hold终态已原生核验；数据全部已消费开发，不是新导航测试。监督阶段按8条轨迹各8行等权采样，每步另按旧8任务各4个诱导状态保持。loss为辅助mode-action MSE +100×旧mode-action MSE，原baseLR3e-6、残差LR1e-4、clip1，500步真实监督Adam+500匹配shadow，所有参数仍同一594628可训练Actor；hidden每步清零，不加网络或Critic。监督不是RL/HPG，不称纯从零DRL或标准GCSL完整复现。

监督结束必须同时满足：固定全辅助数据等轨迹MSE降到初值一半以下；真实旧20全部安全GOAL；两辅助goal中心在相同原生初态自主单Actor导航都安全GOAL。辅助goal仅作为独立训练/诊断回合目标，不在原六米导航中切换，不是运行时路点/teacher/规划器/地图/路由。任一门槛失败则停止采集学习，仍导出研究候选与完整旧275/六米54验收，不将失败候选替换safeT255。

全过后丢弃监督Adam、fresh RL Adam；至多一轮4固定起态×4条件AR探索=16条真实新policy轨迹，各自再完整12数组/积分原生精确重生，合计32回合。使用原eta.1 reward、时间对齐MC RLOO、条件AR/previous mean正确梯度、PCGrad cap32、soft100、真实旧20硬接受、原数值信赖/LR/三步长候选[1,.25,.0625]。**最多1次live RL Adam+1次shadow**，每候选真实旧20检查、同policySHA缓存只供shadow复用，非独立物理重复；拒绝/中断ActorAdam回退。监督后的同seed轨迹不当旧T287 on-policy数据，首批means/logp须按新policy真实重算。仅500监督/一轮一次RL提案，不追加seed/噪声/权重/模型/轮数网格。

这是监督+真实RL联合工程pilot，历史T287/T291只作已消费描述性控制，非并发匹配因果消融/创新证据。完整旧275和六米54须全安全，才16fresh再78独立原生读回；辅助goal/探索/中心成功不能称六米稳定。本协议不自动训练七八米或晋升模型。

## 有限预算与验收

600全部命令/16.8M全部物理步，main3600秒、前2200学习、末1400验收，外部3600秒TERM/1秒KILL更早从严，不延长。own17MiB/global1GiB/free40GiB。16完整state-chain192KiB、524紧凑10KiB、ActorAdam7.25MiB、metadata1.5625MiB，总17752064B<17825792B。紧凑5数组可精确还原rewards=environment_rewards-eta*aux_costs；没有丢弃唯一观测训练数据。JSONL以gzip逐行保存，消费者显式解码。最坏紧凑20预检旧+2辅助+3×20候选旧+12轮中心+275旧+54新+16fresh+78reload+4中断=521≤524。命令/物理均按上限核算。T297原日志69件先远端完整读回/原SHA无损编码释放7772236B，完整新预留核算后才启动，不删除视频/唯一资料或增quota。

原预检完成8个丢弃clone Adam/0native：2步监督live+shadow4次、RLlive+shadow2次、拒绝1次/中断1次，更新和ActorAdam回退全部通过；随后metadata估算1594511>1572864拒绝。原源/协议/8次调用/错误保留preflight_failed_001，不重复克隆。启动前在同17MiB内metadata+64KiB、紧凑520→524、RL提案2→1（减少科学努力，非续旧预算），其余不变。finalize复用8次已过合同，新增0优化器/0native，重新核算最坏容量/函数依赖/命令物理通过。不是训练中改协议。

源码本地提交+GitHub完整root-tree精确核验后唯一launch。运行中Stage-native-exclusive，禁止外部stage/runs写入和source同步。所有原始/partial/失败保留。结束确认父/4worker/timeout/inhibitor全部退出，独立500监督live+500shadow及至多1RLlive+1shadow精确重构最终ActorAdam（最多1002丢弃调用、0native），全部源SHA/模型SHA/12数组/完整命令物理和有效/碰撞/超时/缺测/删失核账，RAM原始远端逐成员SHA全读回，再同步源码。
