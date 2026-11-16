# T598 完整冻结T550与独立循环恢复pilot

单臂工程试验，不做循环结构单模块因果归因。T550完整encoder/core/head/全部修正及normalization冻结；新120→32 tanh encoder、32 GRU、32→2 readout和同初值冻结reference，输出旧location+(g_theta-g_theta0)。真实额外双32隐藏状态逐决策记录，原64隐藏保持原存储；所有流在回合重置，不在决策间清零，不使用taskID路由。训练新分支全部参数；不安装包/用GPU/占用Phase。

共同保留39条T550成功历史初始原location、3条T591已实际成功的学生恢复后缀。外部强制前缀只输入，不当专家标签。原120当前history完整BPTT；旧location平方误差、恢复边界hinge监督，分别等任务平均后各半权重，任务内半均值/半worst10%，128固定fullbatch Adam lr3e-4、clipnorm1。相较T596同时改变表示和训练目标，不称单模块因果；0新RL/critic，不能称本轮新DRL收益。

3原前缀自主接管+3紧凑exact必测；至少1safe才完整49项+4关键紧凑exact。历史完整T596零残差T550接管0/3只作明确历史控制，不计新物理独立重复。不用拟合loss阻挡必测3接管。全部39初始成功ID保持且新增至少1接管成功才保留研究副本，否则整ActorAdam回退T550。研究保留不等于新增原起点任务或全场地晋升；无407/fresh16/reload78/RL接力。

不追加步数/seed/噪声/旧作业。正式source/协议/输入SHA锁定后唯一启动，主程序退完才独立128次历史Adam重构、全数组/额外双记忆/精确native重放审计；全部原始RAM远端读回逐成员SHA。main1500/学习900/audit600/backup180/source120秒，19:35停学、19:45停native、20:00全部停止及相对deadline更早从严；若完整最坏时间不够不启动。own12MiB/global1GiB/free40GiB，CPU14/15总200低优先/GPUoff。具体预算见BUDGET，等待独立额度确认；尚未注册/未启动。
