# T305当前目标与初始历史帧目标的工程恢复

原T305已FAILED/8.194秒，0native/0Adam/0Actor更新，原源/协议/失败目录不动、不重启。失败发生在所有8个真实alternate current/next120重建和旧AR检查之后、原goal恒等式汇总处；未达到随机Critic/Actor梯度步骤。原失败记录不能写成整体合同PASS。

原因：T295的原生witness明确保存了历史frame实际生成时的goal，factory初始四帧可能来自更早goal。T305把`history_goals[k][0]`误当当前active goal，于首行造成1.60/1.28的当前goal列误差。这不是原导航Actor训练bug，也不是T295数据坏；T295原代码验证identity时用的是当前env.goal_xy与每个历史帧各自goal。

单独T305R工程恢复只补`source.original_goal=原已锁T295 source.goal_xy`，恒等式重建当前五列用actual active goal，历史八列仍用原witness逐帧goal。查询goal的全部13字段、107其他字段、firsthit/partial hold、密度、future终止语义/采样/阈值/随机Critic及梯度不改。原接口函数/原生环境不改，不增导航数据/优化器/模型网格；不重新调用已结束main。

新output为`t305_goal_contract_recovery`，独立注册源/输入SHA、零步预检、精确源码同步后唯一launch。wall≤300秒且不晚于原T305 absolute deadline扣30秒，工程audits各≤600秒，own1MiB/global1GiB/free40GiB。原T305失败原始先完整备份；恢复结束再独立audit/全原始备份，不能将恢复通过覆盖失败或当真正控制门槛通过。
