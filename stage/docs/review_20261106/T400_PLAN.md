# T400：同一策略的1/2/4环境后台吞吐与证据等价

2026-11-06用户明确提出多车并行并授权合理自行安排。T399已结束/审计备份/source全过之后才注册。Stage only，Phase完全隔离。此为独立新工程合同，不追加T399学习或物理预算。

同一个冻结T398warm Actor、同一已锁070/80任务/动力学，四个任务实例（LEFT/RIGHT各两个独立重置副本），按并发数1→2→4各完整执行一次相同清单。每实例从episode zero运行固定50hold=10秒=5000物理步，然后按事前工程诊断截断；原80秒任务时限与终止逻辑不改。这12条是吞吐/等价测试，全部属行政诊断删失，不统计导航成功率、碰撞率或RL样本效率；无随机探索/模型更新。每个实例独立history/hidden/previous/执行器/clock，同一权重版本，分进程CPU0–3固定单线程；不把四车放在一张相互碰撞的大图，不复制仓库或视频，关闭展示。

每档总工作量同为20000物理步，共60000；7个worker原生setup＋12轨迹＝19主命令，加预检2零物理命令共21≤24。主循环不重跑、不补seed或并发档位。每worker静态分配最多len(任务)*5000物理，聚合共享总60000而非各给60000。先READY后统一GO，分别计启动/初始化、采样循环（native_checked/Actor/其余）、保存、端到端墙钟、RSS。原生/逐hold数组SHA跨并发必须精确；输出完整120/history/hidden/ownership/奖励/nativeSHA，独立审计零native/零optimizer重算。只从已测试档中按端到端墙钟选择更快者；初始化和短片段工作负载限制必须说明，不把10秒基准速度保证用于任意长回合。

0Actor/Adam/backward/新训练；不安装Isaac/MJX/Warp或变更Torch环境。GPU已可用，但本pilot测原CPU路径，避免数值/物理变化混入。Gymnasium是接口而非物理引擎；本次先用现有等价进程隔离执行器验证，未来可封装VectorEnv，不能称已完整迁移Gymnasium或GPU仿真。资料见PARALLEL_SIMULATION_REVIEW。

own2.5MiB：12个完整＋每7worker最多1partial各64KiB＝1245184B，metadata1MiB＝1048576B，最坏2293760B＜2621440B；global1GiB/free40GiB完整admission，不删唯一证据。主300秒/audit60/backup120/source120/all600秒，真实launch锁绝对deadline；worker READY/GO及进程全退出均受本项截止，失败/数组不等停，保留所有partial与错误，禁止重启旧目录。正式0native预检独立前向/历史和50hold不可压缩上界、物理前取消/SIGTERM/native快照恢复通过后，干净源码/精确远端后唯一driver。

全部父/子原生退出才独立audit→RAM全原件远端逐成员SHA→源码根树。不自动导航学习、不晋升拒绝模型，不把吞吐试验当论文算法创新。old275/主54/fresh16/reload78未执行null，6m未过不7/8；部署仍单Actor无地图/规划器/路点/教师/路由。
