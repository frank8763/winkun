# T430 完整碰撞体静态核查

0物理积分、0 Actor/Adam；不是新的导航或恢复实验。独立静态重建 audit PASS。

| 已有轨迹终态 | 静态非地面接触 |
| --- | --- |
| student_original | real_circle_obs_9 / chassis_collision (-0.0004582m); real_circle_obs_9 / chassis_collision (-0.0004582m) |
| neural_teacher_recovery | real_circle_obs_9 / chassis_collision (-0.0003861m); real_circle_obs_9 / chassis_collision (-0.0003861m) |
| fixed_feedback_recovery | real_circle_obs_9 / None (-0.0000927m); real_circle_obs_9 / None (-0.0000927m); real_circle_obs_9 / None (-0.0000927m); real_circle_obs_9 / None (-0.0000927m); real_circle_obs_9 / None (-0.0000927m) |
| successful_reference | 无 |

只核查保存的终态几何；没有重放原生求解器上一积分时刻的contact，不把当前静态查询当独立导航验收或可恢复性证明。模型按当前factory MJB及T326完整静态碰撞几何核对，不声称重新构造所有运行时动力学参数。

远端原始备份状态：FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED。
