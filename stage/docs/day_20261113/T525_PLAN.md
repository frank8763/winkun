# T525 实际学生偏离状态恢复＋真实成功序列保持

2026-11-13用户确认应用已有研究思路，并明确另开15:30全部停止窗口。仅Stage，与Phase完全隔离。上一T524已完成/audit/完整远端逐成员SHA/父worker及driver退出，不重启。无训练结果保证，不将标准DAgger/CLEAR理念称原创。

## 学生、教师与数据资格

从已接受T507单Actor（T508 old8/8、field11/24）初始化，不续被拒绝T524。保留19条同一模型真实完整成功序列及原locations，8旧＋11field，逐SHA与从h0重算函数精确，不再刷新旧7/8父函数冒称成功专家。T519研究模型仅训练期神经教师，其原起点sector0_offset2确定性成功已证，但从新学生状态恢复尚未证明。

固定任务sector0_offset2，学生历史基线最近决策端点.483351675m在77.8秒。因此唯一接管时刻为提前8秒＝69.8秒、第349决策；由已见失败选择，探索性诊断，非盲泛化。真实原起点前缀全部native推进，goal/history/时钟/执行器/积分/学生hidden不重置，不teleport。教师独立从合法h0在实际学生每个观测上同步递推自己的64hidden，不拷学生hidden，也不在接管时清零。前缀逐数组/积分初态须匹配已审计T508基线。

1. 固定教师接管到终止，若无接触/边界GOAL才合格；完整120历史、学生hidden/tape、教师实际action、ownership保存，精确原生重生。
2. 教师合格才固定纠正4秒（20决策），之后交回原学生。学生在教师动作期间仍接收实际观测更新自己的hidden。全部精确重生。只有合格教师与此短纠正交回均安全GOAL、且至少有一个自主决策，才允许学习；否则0优化器停止，明确缺少合格恢复数据，不扫更多时点/教师/种子。
3. 资格全过，原完整教师恢复轨迹中owner1的真实执行动作作为标签，从学生实际因果观测h0全历史BPTT；前缀不冒充教师标签。内点平方动作误差、边界可行集hinge；半均值＋半top20%，32固定监督Adam＋32shadow，19真实成功序列等家庭soft保持。独立audit再32丢弃历史重构。Value权重与Adam冻结，0新PG。这是恢复辅助学习pilot，不是完整DAgger/CLEAR或从零纯RL。
4. 全程无教师的更新学生自主完整old8＋field24，全部32精确原生重生。旧8全保、原field11逐ID全保、field>11、接触不超9、无无效删失才接受，否则整个ActorAdam回退训练初态；接受原件独立冻结。恢复混合GOAL不能计新自主成功。

本轮不把多个算法全堆叠；新鲜学生RL只能在后续独立注册中使用自身数据，不能把教师混合轨迹塞进on-policy PPO。

## 优化器及资源

Actor权重与T507逐位相同；原adapter Adam1025/385完整保留，lr明确改3e-5，encoder/core/head开放并新增空Adam组，故训练初态joint与源原Adam不同，实际hash锁定。原模型不修改。single120因果输入/64GRUcarry直接驱动与转向；部署无地图规划路点运行教师/路由。

主管线最多3000秒、学习1800、独立audit600、备份600；14:...相对上限与绝对学习15:00/native15:10/all15:30取更严。最多82全部命令（预检6、两次pool会话8、teacher/交回4、32验收64），全部物理精确上界BUDGET，所有prefix和精确重放计费。最多2条full≤1MiB、32compact8KiB、模型2MiB、metadata1.5MiB，own6.5MiB/global1GiB/free40。无预算/容量就拒绝，不改门槛。CPU14/15总CPUQuota100%idle/nice19，GPUoff、Mesa llvmpipe见证，不查看Phase进程。

运行单writer，唯一driver负责main结束所有native退后audit、manifest、RAM远端逐成员SHA和精确source。网络失败明确REMOTE_PENDING，不以网络等待阻断科学。失败raw/error完整保留，不重复启动结束目录。完整old275/six54/eight54/fresh16/78未安排null。

## 存储整理

T525_STORAGE_LEDGER：只删全远端逐SHA的废弃合成预检clone；两份相同零初始化模型改显式可逆reference，完整canonical本地保留、原gzip/raw/Adam字节一字不丢，原远端保留，无symlink绕预算。科研候选/轨迹/高清视频不删，预计毛释放3752439B。新收据/参考记入版本库，原receipt不覆盖。恢复旧裸路径时须按reference核SHA并守容量。

文献依据及限制：EXISTING_SOLUTION_METHODS.md、RELATED_FAILURES_PRIMARY_REVIEW.md；DAgger学生状态反馈 https://proceedings.mlr.press/v15/ross11a.html ，CLEAR历史保护 https://arxiv.org/html/1811.11682v2 ，当前只借鉴组织原则，不移植原V-trace/Critic/安全证书。
