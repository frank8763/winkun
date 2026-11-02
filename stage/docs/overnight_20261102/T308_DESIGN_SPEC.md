# T308待实现设计：单一iid目标条件价值基线与条件Actor试验

本文件是本夜下一项的具体实现规格，**不是已锁正式协议/不是已启动**。须先T307工程合同+独立audit+备份PASS、实际容量核算，再代码/源输入SHA/正式protocol/零步/提交/精确远端树后唯一launch。若实现合同不成立，先报告/修工程，不绕开值头控制门槛。T305/R、T304失败不重启，不拓网格。

## 科学变量与边界

从冻结安全T255确定性函数开始，同120输入、hidden每步清零、same body/encoder/native head，无新地图/规划器/路点/运行时教师/模型路由。明确改为rho0 iid bounded Gaussian探索，sigma沿原(.035,.03)，本身是分布变化；结合新目标条件价值作为一个工程pilot，不把结果单独归因CRL或声称原创。旧AR记录只作历史控制，不能把其logp作新数据。

Critic参考SGCRL的future-state InfoNCE+.01 LogSumExp，使用T307可微接口与终止语义。目标是本任务有限时域**决策端点占用密度比**的近似，不是安全first-hit概率；goal-dependent行为context与query policy迁移、部分观测/有限数据都是限制。使用每anchor独立SE2 body-query矩阵，原始绝对位姿只用于离线label/坐标变换，不进部署Actor。重复相同coordinate或同terminal token需处理等价正例，不能将同一query强行当不同负例。

## 固定数据与一期价值训练

候选数据只采当前父policy的原6m LEFT/RIGHT reset13与01四个固定开发病例，各4条固定seed轨迹，共16，逐条精确重生16。reset13两侧共8条用于拟合，reset01两侧8条只用于验证；均非最终16fresh导航留出。seed在正式协议锁定，不搜索。此iid数据没有旧AR历史重复要求，若任何旧轨迹数组恰相同如实记重合。

采集端每个决策保存真实四history pose/goal（T295 Witness近邻），包括末态，完整120和所有12数组state-chain无损。iid真实边界动作/边际密度独立重算，不能为数值失败截CDF或替换logp。native cap仍28000×2ms、100步hold、GOAL半径.2，CONTACT优先，collector部分记录保留。future law按实际native_counts折扣，safe GOAL同坐标吸收、CONTACT token、真cap timeout token、collector未知掩码，各项语义先合同过。

两个同初始/同容量Critic：action vs blind置零**新动作两列**（前动作仍在原120中，不能偷偷删除），T307兼容124→64→32与query6→32→32；原前latent两列恒0。固定1500个64样本batch/头、lr1e-3/clip5/seed一次锁定，各3000新headAdam，独立audit3000丢弃重建。训练按episode等权、anchor等权、未来按law采样，重复goal/token为等价正例；不选最佳checkpoint/改长度/权重/种子/模型。训练-only规范化或固定原观测尺度须在protocol明确，不能测试后调。

验证至少同时包括整reset家的future对比loss/等价tie-aware预测，以及**真实动作控制作用**。行动对照不能再用T304的6状态/18标签，也不能把其确定性续行Q充作当前iid Q。

## 当前policy下的真实控制检查

用上述8条验证轨迹的真实原始起态，每条中心轨迹已采且作为center控制；两个追加分支只修改**第一步**同drive创新、steer创新为原值±1sigma，此后仍同iid父policy、相同后续创新流，在各自真实观测闭环续行。新增16分支各再精确重生16，center复用16采集中的8条但不能重计新增物理重复。全部controls从原6m起点走完，native原始到达/接触/超时保留。

与T304区别：本项真实当前iid续行、单primitive动作与Critic的两动作输入一致，Q不混三步open-loop宏，不把后续随机噪声当不同策略。强制第一创新只用于诊断，不混入onpolicy RL训练数据。每条center的20%/50%/80%决策端点位置固定作离线query（碰撞终态不选；若长度/几何不足则按正式协议预先写明invalid/停止），只用于评估值，不在回合内切目标。以同finite endpoint law加固定宽度.2m的高斯坐标核计算真实折扣占用（非原环境奖励/firsthit）；其与连续密度的平滑近似限制要披露。

预注册的控制筛选建议一并落正式protocol：action heldout NCE≤blind95%、每侧≤blind102%；每侧至少6个真实控制差>1e-8的可排序pair，action预测符号正确率≥90%，最高值选择平均真实核占用高于全平局随机选择。若指标定义需调整，必须在正式注册/采样/训练前一次决定；正式启动后不得再改阈值。仅此有限验证不是通用Q证书；任何一项不过就停止Actor阶段。

## 只有价值阶段通过才允许一期Actor更新

只允许**一次**新Actor Adam提案+完全相同shadow，不能网格试步长。以已训练冻结Critic的当前动作价值作目标，iid重参数动作，固定新目标/真实原目标各一半的64样本；所有新目标当前与历史13字段用真实pose重建，不复用旧logp。新目标已在当前.2m内的状态按预注册规则不参与Actor学习，不伪造安全正例。数据/随机创新先锁定，Critic不随提案更新。

Actor loss为负value加显式iid边际熵项（系数.01）与原old mode软保持100；原学习率body3e-6/residual1e-4与gradclip1保持，不能硬mode clipping作为value梯度路径。旧211状态每8任务4行参考仍保留，并披露关键case覆盖不是均匀保证。Adam前完整快照，提案/影子ActorAdam必须精确；真正old20全部安全才保留，否则整ActorAdam回退并停止，不试第二次。保存全部候选hash/优化器/原始数据，拒绝提案可独立重建，不能因未保存独立大文件删掉失败证据。

若Actor从未调用或完整回退为safeT255，历史同policy验收可只标“既有结果复用、非本轮物理验收”，本轮完整275/54/fresh16/78写未执行，不能称本轮已通过。若有保留新Actor，必须完整old275+原6m54全测，全安全才16fresh再78独立原生重载逐数组/积分相同。最终原子保存/读回ActorAdam、CriticAdam/源SHA并独立audit；无论何种结果，本项不自动进入7/8，只有完整六米真门槛与备份核验后另立新级。

## 完整最坏预算建议（正式注册前必须实算）

一期最多32个unique full（16新采集+16控制）及32精确重生，64命令；加old20提案/回退、8old中心+2真6m中心、完整275/54/fresh16/reload78和partial，总≤600命令、16.8M全部物理、14400秒（最后1800留最终验收，且夜间绝对截止更早从严）。整夜剩余2200/60M/21600包含本项全部成本；此前T305/R/306/307均0native。

own建议26.5MiB：32full各≤256KiB含history、4partial各256KiB、最终ActorAdam≤8MiB、两个CriticAdam各256KiB、evidence≤256KiB、最多500compact各10KiB、metadata2MiB，余约1.6MiB buffer。以上必须用真实序列化/数据上界预检核验；如果超出，注册前缩减协议或继续真正无损冷整理，不能先跑再扩大/删除唯一资料。T306后约28MiB可用，T307会再占少量，不能以建议当实际预留。

单writer/native run不得外写runs或源同步，明确父/4worker/watchdog/inhibitor绝对截止；模型/部分数组/动作/奖励密度/命令物理源hash/训练与shadow调用全计数。结束全部进程停，独立source/数据/控制值/3000头Adam+至多2ActorAdam精确重建，全部原始RAM小分包完整远端二进制读回逐成员SHA，最后精确root-tree同步。任一gate失败本夜不扩该机制预算；结果如实报告，科学准备不冒充训练。
