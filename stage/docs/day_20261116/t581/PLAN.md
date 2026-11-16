# T581：从T550原函数直接做一次真实闭环残差RL

用户2026-11-16在T580三组拒绝后明确“下一步开始”。本轮独立新机制，不续T580的MER预算。12:00全部停止，11:15学习、11:30原生，更早main1800/learning900秒从严。CPU14/15总200%低优先、GPU关闭、Stage/Phase隔离，own22MiB/global1GiB/free40GiB。

T579证明T550四条任务成功但弱学生初始化即失败。T580三组都损两项，MER无新增；小步长ER八米RIGHT13恢复但不能保留。因此更换学习初始化契约：完整T550冻结原函数＋零输出184→32→2神经残差，输出为原location+.1*tanh(newhead)。既有神经基函数层保留在同一个Actor；5986参数可训，真实120维因果输入/64GRUcarry不变，无运行时教师/规划器/路由/路点/地图。残差与PPO都不是原创，也不能保证不遗忘。

先在27既有历史＋T579两失败完整history上逐位核验原函数；真实step动作/hidden/密度和因果性、合成4次丢弃Adam/live-shadow/成熟Adam回退、模型读回和full codec通过后注册。原生先两项零残差完整复现T579（LEFT6及field6offset3），全部数组/积分精确才学习；两项属于历史重复不是新增独立任务。

单个训练目标固定为T550已知失败wide117_80_dev6_LEFT_00。4条新鲜iid原始CAPG bounded Gaussian sigma[.2,.2]真实轨迹及各1精确重生；只当前行为数据做PPO，原物理时间peer RTG/奖励/终止保持，不把示范当on-policy。本轮0新教师、0BC、0Critic。最多8次Actor Adam，lr3e-5、KL>.01止步、clip .8/1.2、grad1；27历史T550原location保持weight100，同时学习新任务，单次提案，完全相同shadow和独立audit历史重构。没有新seed/噪声/轮数网格。

验收固定49已消费开发任务：T580原37，加T550历史已成功old/six/eight各4项。每家族按原407顺序在未选成功项中等距选取，不看新候选结果。训练前以同模型T550原407锁定49上的全部原成功ID。所有原成功逐ID保留且至少新增1项安全GOAL才研究保留，否则整Actor+Adam回退，冻结T550始终保留。四关键六/八米保存full并exact，其余45compact只有RAM完整因果检查与进度记录；不能称全部49完整pose离线审计。共55主记录（2历史基线、4训练、49自主）+8exact+4生命周期=67全部命令，上限70；物理按各step_cap求和锁定。

本轮完整407/fresh16/reload78均未执行null，不将49成功当六米全过/全场地或泛化。只有主进程与所有worker退后driver唯一audit、manifest、RAM全部成员远端SHA核验、精确源码同步。无网络则REMOTE_PENDING、原件保留，绝不伪称备份完成。不自动启动下一轮。
