# T673：全407任务结果信号与实际位移的零更新诊断

T672已完成1724fresh完整因果历史+1724exact、8独立auditPASS、全部科学PID退出。按原collision-cost/safe边界口径，341任务样本全安全、50任务结果有变化、11全碰撞、5全零回报；旧T671新遗忘的dev6_13/25及dev7_17本次均有结果变化。这是有限样本支持，不是新策略提升；大多数失败field任务仍没有可区分终态结果，不能说全覆盖即解决探索。

本项仅计算：每ID等权、每轨迹终态结果LOO中心score梯度，原点全history自己的carry、原始动作固定、真实截断Gaussian原子密度；4/8采样次数不定义任务权重。逐轨迹score重组与直接surrogate梯度一致，删掉一轨迹后重新计算LOO baseline与方向余弦/范数。全同结果梯度零必须标记无信号，不能作为安全保护约束。

比较两个语义明确的解析首Adam位移：全407任务均值、仅24field均值，各加原10×387名义保持、同lr3e-5/eps1e-8/normclip1。仅按首Adam公式计算，不构造/调用优化器、不导出候选、不原生执行。此比较是诊断目标稀释/方向冲突，不是学习算法对照或超参搜索。对每ID报告位移点积、完整历史上的非线性样本surrogate、逐时刻最大/95分位与任务平均/整段累积条件KL。KL逐动作sigma从真实已核验metadata[.05,.05]读取，非名义动作保持的.2尺度。条件KL不构成闭环真实占用分布上的风险或保留保证。

六个预定mixed ID(旧13/25/7_17/21及field0_offset1/7_offset2)自身梯度单位方向，epsilon1e-6，中心差分门槛abs1e-8+rel1e-4。需全407完整density/carry/finite与六numeric全部过，至少3原384和1field有非零梯度，才能标记SIGNAL_DIAGNOSTIC_PASS；此门仅确认诊断可用，**不自动允许学习**。先读全任务冲突/无信号/单轨迹支配性与一手审查，决定一个主要变量的新有限学习；失败不换epsilon/阈值/样本seed救。

0新增物理/原生命令/Actor/Value/optimizer；1800main+1800独立重构、根4500秒硬停≤75min，64MiB/run、global10GiB/free40GiB，CPU14/400%Nice19/4GiB无swap、CUDA关闭。每次guard按Phase资源元数据严格probe，有重叠只等待自己，墙钟仍计。失败保存error/部分结果/账，独立新目录只接未执行工程，不重复旧科学更新。

源码/数据SHA锁定、真实入口零native零optimizer预检、独立ledger先reserve、GitHub精确全树核验后唯一launch。main与independent重算全梯度数组、方向、数值差分和KL结果逐值精确。保留模型仍T501，不拼接成功集合，不改T672或T671原源/协议/raw/audit。下一学习仍需原384逐ID全保+新增+同模型完整407/独立audit。
