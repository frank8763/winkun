# T675测量合同通过，但不准据此奖励径向接近

COMPLETE/PROGRESS_MEASUREMENT_PASS、独立1724结果逐值auditPASS，9.363main+9.413audit秒，0物理0Actor/Value/optimizer。全407/1724 pose链/native计数/末端距离合同过，最大距离误差1.7764e-15m。15个同终态失败field中仅3个达到预设progress range>=.01：sector0_offset2(.01221)、sector1_offset1(.03666)、sector7_offset1(.03001)。这些是测量差异，不是可用PG或导航提升。

追加只读复核原T672 summary发现：这3任务全部8条均timeout且field_boundary=True。故直接奖励径向接近可能强化已经无效的越界路径；不定义/启动该径向奖励学习，不称policy-invariant shaping。10个全碰撞field的径向range均<.01，不能称连续进展解决了碰撞探索。

范围更正：前述T673/675文字称16个同终态失败field，是把all407的11全碰撞都算进field；实际10个field全碰撞+5个field全零=15，第11个全碰撞是非field dev_task6_01。原all407的341safe/50mixed/11collision/5zero计数正确；原锁定源/协议/文献笔记保留，不覆盖。T675代码按真实task前缀独立统计15，门槛和全部407/1724未改。

下一只准备T676边界代价测量/score支持合同：按既有fullfield_start_pose场景适用边界(7m)、自身历史重建，先核对原boundary bool与记录poses越界峰值，不向8m非field旧任务错误施加7m边界，不把task条件reward当Actor任务路由。若有连续越界程度信号再独立零优化器梯度审查与有限pilot，原384+新增+同候选完整407/audit不变。
