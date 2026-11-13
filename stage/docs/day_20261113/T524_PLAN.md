# T524 固定初始梯度范数平衡的配对控制

2026-11-13用户“检查、分析原因、继续实验”。T523完整32+32精确已完成：旧8=8/8、field10/24，原研究12丢失sector0_offset2和sector2_offset2。全部结束/audit/远端逐SHA是正式注册及启动前提，未满足不启动。

T523首次实际PG两任务梯度范数old1.115794163349819/new0.44879499433644554，cos0.02123513851237517。首次不是负向冲突，等权回合目标没有均衡有效梯度幅值，old约2.486倍new。只是一批的描述，不能排除其他步冲突/闭环分布偏移/小动作敏感性，不能称灾难性遗忘唯一根因。sector0_offset2最近决策端点.350126m后驶离最终2.84254m超时，sector2_offset2剩5.56068m接触，失败发生阶段不同。

先前考虑的Adam历史清零仅准备草案，未注册/运行；依据刚落盘范数证据，本次优先改变更直接的任务PG幅值。唯一实验因素：任务PG scale固定sum(norms)/(2*norm_i)，使用T523首更新已测范数，首批两任务贡献等范数；不是每步自适应，不改soft保持权重或自己的history保持项，不扫权重。对照T523原scale=[1,1]。T519同一Actor权重和完整两组Adam原样继承，训练初始joint_hash完全一致；Value及Adam冻结。

两实验同T522已采8轨迹、同精确行为policy/合法h0/历史/条件噪声密度核验；是共享样本的配对优化方法对照，不是新采样或独立重复。此前T523用过这批数据须披露；本轮重新从同源初始化，不用变化后策略冒称当前数据。保留全部4成功3接触1超时，任务内物理RTG同行基线/原sigma.2/奖励/时限/完整history BPTT/16max/clip/.01KL/soft19全部不改。16live+16shadow，独立audit最多16历史丢弃，预检4合成丢弃Adam0native，原错误/模型不删除。

唯一候选old8+field24原起点自主全部+32精确；old8全GOAL、T51912field成功ID全保、field>=12/contact<=9/无无效删失才接受，否则整ActorAdam回退原T519。原接受最佳T507 field11+old8冻结；old275/six54/eight54/fresh16/78未执行null，不用代理误差冒称全场地通关。部署single120/64carry Actor，无地图规划路点运行教师路由。标准任务梯度平衡非原创/安全证书。

预算68全部命令、零新训练采集、两次32物理上界锁BUDGET，main1200/学习600/audit180/备份360秒。完整最坏余时不足不启动，13:06原生/13:17全部截止不延长。own4MiB/global1GiB/free40，CPU14/15总1核idle/nice19GPUoff，Phase完全隔离。不同因素的明确配对检验，不是T523续轮数。

一手依据及边界：Yu等 Gradient Surgery for Multi-Task Learning §2分析梯度幅值差/曲率/冲突的组合，不称本轮固定归一化为PCGrad复现：https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf 。Adam历史机制见 https://arxiv.org/abs/1412.6980 ，本轮不改该变量。现场结果决定报告，不能预言范数平衡必成功。
