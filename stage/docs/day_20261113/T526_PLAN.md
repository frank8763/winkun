# T526 64.2秒早期反馈恢复资格验证

2026-11-13用户明确“下一步开始”；沿既确认15:30全部截止，14:54启动准备，余时不足完整新学习及32验收最坏，因此本轮只做原草案第一步零训练资格验证，不杜撰条件学习已安排。原T525完整结束审计/21原件远端逐SHA，原模型不改、不重启。

学生冻结接受T507field11+old8，任务同T525 sector0off2原起点/障碍/goal/时限。原学生轨迹77.8s最近端点.48335m，d<2.5m首次理想点目标曲率超实际最大转角上界为66.2s，提前2s固定64.2s第321决策接管，来源NEXT_AFTER_T525_GEOMETRY。该几何筛选非原生可达性证明，已见数据选点属探索性。

教师复用T514已核验strict平方油门/directsteer映射的点目标反馈，physical wheelbase.60，最大转角以实际nativeenv为准，vnom.08m/s、slowdistance.5m、mincosfraction.25、curvaturefloor.2m，不扫gain/时点/种子。相对T525神经教师+69.8s同时改变教师形式与时机，明确联合工程pilot，不能单因归因。训练期教师不使用路点/地图/规划器，未来部署仍singleActor，此教师混合不是纯自主策略成功。

原起点native完整推进学生前缀，不teleport、不重置clock/goal/history/执行器/GRU。到321决策后由反馈教师接管至终止，全序列与实际目标/姿态/参数/动作/ownership记录，前缀逐数组/初始积分匹配原T508。每教师动作做真实bound strict mapper roundtrip。学生hidden在教师实际动作下持续接收真实观测递推。完整教师safeGOAL才额外唯一4秒20决策反馈纠正后交回原学生。分别精确原生重生，全部full120/hidden/history/reward/动作/终态积分核验。教师/交回失败不加时点、不生成伪成功标签，不直接训练。即使两资格通过，也只保存合格候选数据，未自动注册学习。

0Actor/Value/Adam/PG/BC。最多2条原轨迹+2exact＝4*64112=256448物理，8main命令+6pref=14全部命令。main600/audit180/backup360，全部15:30stop，absolute native15:20更严；完整余时不足不启动。full2*1MiB+metadata.75MiB，own3MiB/global1GiB/free40GiB，CPU14/15总CPUQuota100%idle/nice19/GPUoff/Mesa见证。无新checkpoint拷贝、失败保留。旧8/field24/old275/six54/eight54/fresh16/78未验=null，原最佳11+8保持。

先零物理预检实际strict映射、完整model载入、worker取消以及321/20ownership合同，PASS且代码精确远端根树后唯一launch。唯一driver负责native全退/audit/本地manifest/RAM远端逐成员SHA/报告源码。网络失败REMOTE_PENDING，不称完成备份。科学不依赖网络LLM。Phase隔离。
