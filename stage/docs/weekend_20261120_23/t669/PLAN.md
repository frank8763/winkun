# T669：四个旧反例的原起点真实on-policy结果支持

T668条件恢复只有2/4成功，未达到预先三至四项可纠偏准入，不进行该BC对照。本项检验尚缺的真实旧任务RL信号：四个实际退化ID(dev_task6_20/21/23、hold_task6_06)均不在此前fresh32任务内，虽然被387名义行为回放覆盖，但没有其自主随机访问状态的完整结果梯度。本项不保证弥合这个缺口即可不遗忘。

冻结T501原权重，sigma=.05、原起点完整因果carry，各任务固定8条独立声明的学生探索种子，共32full+32exact。只生成sigma metadata正确的模型副本，不学习、不改架构。全部成功/碰撞/超时轨迹都保留，不挑成功、不替换seed，不把exact重放算独立样本。teacher/query为false、所有ownership0，自主前缀到终态，不强制恢复或改目标、碰撞、20cm判据、时限和canonical reset。

1493376物理最坏(64×23334)/70命令含worker6/0优化器/128MiB，1800主管秒+600独立audit秒，单job60分钟内硬停，周末总余额内独立登记。CPU8/10/12原生14主管、400%Nice19、4GiB无swap、GPU关闭。真实原始states/actions/raw_actions/atom/logp/locations/carry完整保存，独立全历史重构、密度、计数与32配对exact逐数组/积分SHA核对。科学结束再串行备份/source，raw未远端核验前不删。

报告每ID safe/碰撞/超时与binary和collision-cost(-1碰撞,1安全到达,0其他)结果变化，而非只平均回报。0新Actor/Value/Critic；不自动训练。有至少两旧ID成败变化才进一步零更新算逐任务结果score梯度及其与新场地改进方向的冲突，决定下一一次软结果优化vs逐旧任务结果约束更新的匹配pilot，具体协议先行。不将小样本约束当CPO定理或每ID确定性安全保证。若结果全单一/无梯度支持，报告缺测信息并复盘，不扫sigma/seed/采样量去救同轮。
