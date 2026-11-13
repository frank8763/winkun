# T534：原生状态持久化恢复资格验证

用户2026-11-13明确下一步开始，沿用17:30全部结束/17:15原生停止。T532/T533模型门槛失败，不接Actor，不重复旧目录。仅Stage，Phase完全隔离，单120因果Actor64真实carry，无地图规划路点教师路由。

固定T532原四窗口：旧成功/旧失败从4.8秒(24决策)；新父失败/新候选成功从64.2秒(321决策)。每窗真实从原起点以对应冻结策略重生全部前缀，逐决策states/actions/count/nextstates/pose/flags/reward/hidden/locations/history对原件精确匹配，原生初态integration也须一致。此处重生是历史重复，不是独立导航样本。

在窗口开始保存完整原生mjSTATE_INTEGRATION、Python环境属性(执行器/clock/prevaction/history/终止信息)、真实history witness、Actor64hidden、原策略噪声/动作tape和会话前缀。使用既有Stage捕获/恢复接口；gzip pickle持久化后重新读取本机可信SHA文件。先连续运行5决策(1秒)，再从磁盘snapshot恢复同Actor运行5决策，所有生成数组、逐hold见证、奖励、传感器历史、hidden和末integration须**逐位一致**。恢复接口重建deriveddata并恢复warmstart，是否足够靠真实数组比较，不事先宣称已通过。

全部4窗恢复通过才说明此有限样本的恢复合同有效。失败立即结束，不放宽误差/重试/切换snapshot实现，不做参数方向或学习。即使通过，本项也0Actor/Value/PG，0新模型，不是导航新增成功、原生方向梯度或全局可恢复证明。后续扰动Actor的hidden应另按其参数重建，不能套当前冻结hidden冒充参数差分。

全部前缀和两遍后缀计物理：(24+10)*100*2+(321+10)*100*2=73,000步。9条命令=1setup+4episode重建+4restore，不用命令名隐藏物理；审计0native，只读取原始snapshot/两完整后缀和源数组复核。每步执行仍原strict MuJoCo，无qpos运动覆写；snapshot恢复仅实验分支初始化。

own2.5MiB，4snapshot各≤384KiB/4双后缀各≤64KiB，协议metadata/日志在剩余内，global1GiB/free40GiB。main480秒/audit180/备份240/source120，完整准入保留1080秒；原生相对和17:15更早从严；17:30全停。CPU14/15合计1核/SCHED_IDLE/Nice19/GPUoff，实际Mesa llvmpipe须断言。不查看或修改Phase进程。源码精确远端同步后唯一launch，结束唯一audit/RAM全部原件远端逐SHA/源码同步。无自动下轮，无模型晋升。
