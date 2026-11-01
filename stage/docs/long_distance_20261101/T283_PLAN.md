# T283：有限幅度任务平衡候选

**19:30前更新**：GitHub认证实际恢复；T280的1064原文件已完整远端读回逐成员SHA通过，见release_receipt_T280.json。下述“等待恢复/不启动”是早期认证阻断期间的历史状态，现已解除；仍须精确同步全部注册源后才唯一启动。新检索与当晚执行理由见../research_20261101_horizon/EVENING_REVIEW_AND_DECISION.md。科学协议、输入hash及预算均不变，尚未实际launch。

本项是新机制试验，不续T280。T281已测得新代价下局部1m组参数梯度比LEFT弱约362/484倍，PCGrad本身未消零；T282已实现并通过有限幅度平衡的零原生/优化器数学合同。现在准备真实端到端Actor学习pilot，但用户GitHub重新认证选择“稍后处理”，所以仅做本地注册、代码与预检，不启动导航。

## 科学设计

一个新candidate，冻结安全T255s2_reset+freshAdam起步；同T280 bounded_endpoint_cost的四个reset（原六米左右、固定六米终点余1m左右完整snapshot）、eta.1奖励、原条件AR、common seed2026111801、2轮每轮4×4采集、每轮最多2live提案/三回退factor[1,.25,.0625]、PCGrad、soft100与原生fixed20保持。120输入单Actor hidden每决策清零，不换Critic/BC/教师/头/地图/规划/路点/路由。

唯一新机制：每轮起点由真实四组任务梯度范数计算常量权重，参考最大范数，非零任务最多放大32倍；norm≤max(reference×1e-8,1e-12)权重1，不放大零信号。用同批PCGrad原/加权向量范数之比作为公共系数，匹配起点task-direction范数。该权重和公共系数在整轮包括各提案/步长评估内冻结，加权代理与梯度保持一致；oldsoft保持项不放大。不能推导整体学习率、整个参数路径或最终任务期望目标与原算法完全一致。

对照明确使用**已消费的T280 bounded_endpoint_cost历史结果**，不是新并发对照/独立重复。科学机制选自这些开发数据，不称盲选。两者同安全父模型/同32训练回合和四live提案上限及随机种子；新候选实际首16轨迹全部十二数组及积分必须与历史逐位相同，回报排序门槛也仍须过，才更新。之后由各自Actor产生不同on-policy数据。需报告全部真实物理/候选拒绝差异，不声称两者实际消耗恰好相同、墙钟公平或统计显著。

若候选不能保旧275/改善六米54，报告失败，不继续加cap/seed/轮数网格或降低old20要求。任务平衡有效性需实际导航验证，不把Gram一阶改善当通过。只有旧275+6m54全过才16未消费fresh与78完整保存重载逐数组/积分精确，再决定7m；本项无7/8自动训练。

## 资源与原始证据

新run唯一stage/runs/t283_capped_task_balance。最多32采集+32全部原生重生、最多12candidate×20旧检查、24逐轮中心、329最终旧/六米开发、94条件fresh/独立重载，共751命令≤800、最坏751×28000≤22.4M物理；实际行政prefix也计账并先保存。wall4800秒，前3600学习、末1200留保存/验收；总绝对截止从main登记，不延长。

own35MiB完整预留：32×384KiB full120十二数组，700×12KiB紧凑payload，一份最终Actor+Adam8MiB和5MiB metadata，共34816000B<36700160B；metadata另按全部JSON行、候选、预检错误及worker日志最坏计算。global1GiB/free40GiB不变，不删未备份唯一资料。与T280相同受控写入互斥、case全扫描和热路径O(1)wall/physics/free守卫。

零native预检只用16历史真实样本检查canonical density、一次克隆live/shadow（同ActorAdam）、拒绝/中断回退，最多4丢弃Adam调用；synthetic old20不是真实安全。结束独立audit最多8丢弃调用精确重建候选各轮ActorAdam，新增0导航物理。所有调用分列，原数据模型不改。

## 来源与限制

[GradNorm正式摘要](https://proceedings.mlr.press/v80/chen18a.html)支持梯度尺度作为研究问题；[CAGrad正式摘要](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html)提醒梯度操纵与非线性冲突限制；作者README已在T280核对，本轮未移植代码、不宣称复现。固定cap32是根据已消费T281开发诊断预声明的单一工程设置，不是新颖方法或无偏/安全证书，也非多seed/跨地图证据。

先检查T283是否已有launch/status，禁止重复；本地预检PASS后仍须GitHub认证恢复、T280原始资料备份完整读回与所有注册源精确同步，才允许唯一启动。用户已选择稍后处理授权，本轮不追问、不自动生成新设备码或启动原生。模型及失败全部保留，Stage/Phase完全隔离。
