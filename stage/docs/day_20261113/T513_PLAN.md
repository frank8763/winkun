# T513：同一37.2秒学生状态的固定目标反馈恢复教师

Stage only，Phase完全隔离。用户已授权训练期自动仿真教师，11月13日要求白天逐轮推进并启动下一轮。部署仍单120因果Actor真实64GRUcarry，不引入地图、路径、路点、外部规划器或运行时教师切换。

T512实际4条iid随机恢复均超时，最近决策距离.2267–.2340m，0接触；不证明不可恢复。原确定性轨迹37.2秒距目标.3273m、车头到目标方位误差.8864rad，39.2秒最近.2312m后驶离。这是行为定位，不是唯一因果证明。本轮与T512同一任务、同一真实前缀决策186，不再添加种子/噪声网格。

冻结当前接受field11/old8父模型。完整真实确定性前缀计费，保留原动力学/目标/时钟/执行器/hidden/history。决策186后独占训练期教师控制，到原生GOAL/CONTACT或原cap34697；教师使用当前真实pose和最终goal，不读未来轨迹、地图/障碍或人工路点。

固定局部目标曲率2sin(方位误差)/max(距离,.2)，物理steer=clip(atan(wheelbase*curvature/yaw_gain),±native maxsteer)；名义command速度min(.08,native vmax)*min(1,d/.5)*max(.25,cos(error))。通过原_policy_to_commands的精确逆（含command_v下界.05）映射归一化float32 drive/turn并clip。只减少速度，不修改原速度上限；.2是固定曲率数值下界，不改变原生成功半径。body-pose近似点目标曲率，不声称精确Ackermann后轴跟踪、障碍安全或稳定性保证。

一次固定教师全余程+一次全数组积分精确原生重生，共2次。成功须原生≤.2m GOAL无接触且±7.5边界合格，才提供下一轮真实恢复标签；不成功照实记录，不追加增益/速度/窗口网格。教师成功不是学生自主成功，无学习/Actor/Value/Adam/新PG；old275/six54/eight54/field24/fresh16/78未执行null。父模型、原成熟Adam与旧最佳结果不动。

记录完整120/实际GRUcarry/原学生proposal tape、动作/ownership、teacher_actions、原车speed_gain/maxwheel/wheelbase/yawgain/maxsteer参数、history/奖励/创新与积分。proposal密度只是冻结Actor输出，不是强制教师行为密度，不能作整条on-policy PPO数据。ownership前186为0，之后全部1；离线独立审计逐步重算反馈teacher动作与proposal/hidden/原奖励和所有源SHA，精确重生非独立样本。

预算2*34697=69394全部物理、6main+6零物理preflight=12命令，native600秒/audit300秒/backup600，prepare锁原生1500秒与全部3000秒上限，own6MiB/global1GiB/free40GiB。CPU14/15合计1核idle/nice19/GPU禁用Mesa，不读改Phase进程。一个writer；active不改科学源/协议，不并发audit/归档/source。真实预检0native0Adam（方向/边界/速度规则+2worker取消）通过、提交及远端源码根树精确后唯一启动；失败目录不重启。完整audit、原始RAM远端逐成员SHA后报告，再根据真实结果决定后续学习。
