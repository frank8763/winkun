# T672：整套407任务的系统当前策略结果覆盖

T671四个已知旧反例保住但又失去三旧ID，完整407拒绝。继续只补少数已发现任务会偏移退化面，本项不再追加同批学习，而是冻结保留T501原权重sigma=.05，系统覆盖所有407开发任务原起点自主结果。无planner/waypoint/runtime teacher/router，所有teacherFalse/ownership0，不改几何/reset/目标/碰撞/20cm/时限。

383非场地任务按taskID排序后6组strided切分(每组63或64)，每任务4个固定独立声明随机单位；24场地按taskID排序后2组strided切分(每组12)，每任务8个固定单位。全部任务唯一覆盖一次分组，每个采样都full保存和exact重放，共1724原轨迹+1724exact，8批各自collect后立即独立audit，通过才自动接下一批。所有成功、碰撞、超时原样保留，不选成功、不改seed重试，不把exact或任务开发集当新独立泛化数据。采样次数不定义最终任务权重，学习仍须另行明确按唯一任务ID均衡。

每批max6000主管秒+600独立审计秒，≤110分钟；root16小时只是8个有限job接力总硬限，周末总11月23日07:30停学08:15停原生09:00全停仍从严。真实预计数小时、取决于长任务和资源让出；不承诺耗满。每批物理最坏为2×重复任务case.step_cap之和，命令2×采样数+worker6，先分别校验，root总额度为8批相加/48000science+4800audit，0Actor/Value/optimizer。存储按每条full4MiB/compactexact32KiB加每批16MiB及root32MiB算最坏，root≤7GiB且global10GiB。root当前约2GiB外部证据加root7GiB预留<10GiB，每批准入实际used+childworst；达8GiB触发归档须科学退出后处理，未核验不得删。

CPU8/10/12三native worker、14主管，400%Nice19/4GiB无swap/free40GiB，GPU关闭(无学习)。Phase只资源元数据，有物理核重叠暂停自己的科学组，硬时钟继续计。每批实际计数立即累计weekend LEDGER，不等八批全结束，root已完成批名列表防重计。工程失败保raw与已消费账，停止根队列；若需接未执行批，须独立注册且不重复已完成seed/批。

每批独立full history/atom密度/ownGRU/计数/exact数组与积分SHA验证；8批全部完成才报告全407采样覆盖，并按ID汇总成败、回报与稀缺支持/梯度。采样成功不是T501冻结性能改善，也不是新模型。下一有限全任务联合RL要独立代码/零步预检/具体预算/注册/GitHub精确源，不自动再用少数反例拟合或同旧batch无限训练。原384逐ID全保且新增、同候选完整407/audit仍唯一保留门。
