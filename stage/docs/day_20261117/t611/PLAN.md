# T611：唯一7.6秒实际偏离点的成功父策略恢复诊断

用户11月17日“下一步开始”授权执行已锁定的单接管点恢复诊断，沿用日22M物理/660cmd/9000main/50MiB，11:30停学、12:30原生、13:00全部停，不扩大窗口/学习额度。Stage与Phase隔离，CPU14/15总200%低优先、GPU关闭。

T610已完整原生/精确复验同一起点父成功、候选超时；首次XY差>1cm为第38决策即3800物理步/7.6秒，此位置已在执行前NEXT_HANDOVER_SPEC锁定。不能用父的物理状态覆盖候选，不另扫时刻。固定候选原起点→第38决策前候选自身动作→之后冻结成功父策略动作；物理环境、目标、时钟、执行器history全不重置。为排除借用hidden，父策略从第0条实际120输入开始旁路前向，每一步保留自己的64hidden＋32extra＋32anchor记忆；第38条实际输入时直接用该真实carry继续。旁路前向断言原生积分不变。候选也持续真实前向记录，所有执行者归属明确。

一条mixed轨迹＋一次逐数组/积分精确原生重生，最多106668物理/6命令/900main秒/300audit秒/5MiB，0Adam/0RL/0新增seed/时刻/动作网格，不替换34/49研究模型。它若safeGOAL，只能称该候选实际状态被成功父策略恢复，不能称单Actor部署或八米/全场地通过。教师动作与完整真实carry逐行重算核验，真实前38步的全部120、pose、history、actions、native_counts、hold摘要与T610失败候选逐位一致，接管输入也一致。

原生保存完整双策略记录：candidate全tape/hidden、教师action/location/64hidden/extra32/anchor32及末态、完整120/pose/history/积分。所有proposal与所有执行动作区别记录；混合rollout的policy_hash字段仅指记录的candidate Actor，另显式标记teacher SHA、mixed_teacher_diagnostic=True、rollout_is_single_actor=False、deployment_evaluation=False，绝不把teacher后缀当onpolicy RL数据。只有真实GOAL且无接触/不越界、完整审计和精确重放全过，才具备后续监督纠正数据资格；本诊断不自动训练。

独立audit在全部原生父worker退后逐数组重算candidate和teacher的整条实际history、自己的全部循环memory、动作/奖励/物理计数和精确重生。主协议已捕获双模型记录路径，教师附加数组容量由原失败完整序列的实际dtype/shape与最长cap外推；full单文件3MiB，own5MiB。原始失败保留，不清理未备份唯一资料。全部收尾后RAM远端逐成员SHA和精确source根树。

任何起始/前缀/carry/密度/奖励或证据合同失败则停，原件保留，不临时增加budget或新编号重复。13:00硬截止前完整最坏准入。既有T608/T609/T610不重启、不再算新独立数据。
