# T500：专攻旧任务和宽六米的原因对照与成功源重建

用户最新授权集中旧275与宽六米；T499同一模型旧233/275、宽六米53/54（LEFT18接触）、八米T49854/54保持，合383已知任务340成功43碰撞。此前329为0训练冻结验收，不能把这次测出缺陷归因于T499训练或最后128更新。

已运行零原生/零Adam反馈命令 `./run_stage.sh stage/experiments/t500_joint_repair_sources/review.py`，复现raw43失败及其固定来源：42任务在T449 recovery模型有真实历史GOAL，唯一dev_task6_00有本地完整实际成功“原学生5决策前缀+safeT255接管”轨迹。原modelSHA/完整数组SHA已验证，混合成功是训练源，不是纯学生通过或新onpolicy。学生在旧成功历史上转向RMSE约.047–.058，说明修复不能直接假定仅需油门；相关误差非唯一因果。

三项固定假设和判别：
1. 后加速度适配器伤害旧避障：仅归零drive_adapter最后weight/bias（其余Actor/Value/两Adam字节不变），固定旧task4_00、task6_00、宽六米LEFT18三任务原生+exact，对比T499已有全history。若恢复说明适配器对这几个初态有因果贡献；仍失败说明仅移除适配器不足。不能从3项推广全部43或整体骨干。
2. 原八米骨干已经有旧任务动作误差：若归零仍失败，后续学习应允许纵向与转向共同修复；新神经残差/骨干可训范围尚须独立合同，不能临时改本轮零训练协议。
3. 成功覆盖不足：42旧模型已实际成功+原43失败的同任务对应，须重建完整成功120/hidden/history并精确原SHA/native积分；任一不符立即停止共享证据链，不放宽阈值或重跑原目录。成功源全就绪后才为下一联合修复提供监督。未知仍需学生访问状态恢复覆盖，专家history低loss不能保证闭环。

T500只两项预注册phase：ablation3+3exact，source42+42exact；原dev6_00成功轨迹仅只读全history独立审查，不新仿真。source42全部须匹配T449原policyhash/全数组/native终态hash并真实GOAL，之后42exact；形成43任务成功来源集合，绝不按任务ID部署选择教师。没有Actor/Value/预检Adam、没有新学生训练或383验收，不称修复成功。下一学习应以这些43成功标签＋T499旧/六米286和T498八米54共340真实学生成功未裁剪location保持为起点，单Actor两动作适配，独立注册/预算/预检后启动，不自动续旧预算。

预算：2092052全部物理（ablation164002、sources1928050），104全部命令（6取消预检+10ablation+88source）；main600+2400<=3000秒，独立audit120+300秒，备份/源预留1500，all5400。原生绝对2026-11-12T15:31:14.134884+08:00，全部2026-11-12T16:01:14.134884+08:00，相对更早从严。own48MiB、shape+metadata最坏27813552B、global完整预留878639971B<1GiB/free40GiB。CPU14/15总quota100%=1核idle/nice19/weight1，GPU禁用，Mesa实际llvmpipe断言；Phase完全隔离。

预检0native/0Adam，验证只归零两个最终张量、全其他Actor/Value/Adam不变、现有dev6_00完整混合成功history行为/奖励核验，以及两模型worker物理前取消6命令。源码提交+GitHub全根树精确后唯一launch，不能以本计划杜撰运行。

结束每phase所有native父worker退出后唯一独立audit、全文件manifest，再RAM全原始远端二进制逐成员SHA，Git仅代码/协议/汇总/清单/收据。driver唯一writer，不并发其他Stage训练、source或audit；所有原失败及八米模型保留，网络PENDING明确。fresh16未消费，无新7/8学习/实体部署。
