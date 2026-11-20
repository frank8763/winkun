# T676：既有边界失效的连续代价与score方向支持

按NEXT_T676_BOUNDARY_SIGNAL_SPEC执行，独立零物理零Actor优化器诊断；不定义学习reward，不导出candidate，不使用已取消的切线投影。T675中径向变化最大的3个失败field全部越界，直接径向奖励不准入。现在只验证与既有场地约束一致的连续失败程度是否有估计支持。

从T672全部24field×8完整自主历史及对应exact，核对原`field_boundary=any(abs(poses.xy)>7)`。C定义为记录poses最大坐标越界量/7，只有job.fullfield_start_pose场景适用；非field原8m旧轨迹即使越过7m也不罚。报告bool、首记录越界、峰值/位置、终末nextpose额外越界(单列，不静默扩大旧规则)。记录峰值不是连续native峰值、障碍距离或安全/可达证明。全field ownGRU/density/模型sigma.05与exact数组积分SHA均复核。

至少2任务C范围>1e-12才有测量支持。负C的整回合score/固定scale512、task均值分母24、每轨迹LOO按真实截断Gaussian原子密度；对每任务前4/后4固定划分独立重算LOO，至少2个变化任务有half梯度点积>0且全梯度norm>1e-12，才满足有限稳定性门。分半和exact都不是新样本，不称置信保证。直接surrogate梯度须等于逐score合成，固定三任务sector0_offset2/1_offset1/7_offset1中心差分epsilon1e-6、abs1e-8+rel1e-4全部过。

解析首freshAdam(非优化器调用)针对负C field均值+原10×387名义回放，原lr3e-5/eps1e-8/normclip1，按实际表示位移评估负C固定数据非线性surrogate；线性和非线性均值必须>1e-12才可用。报告与T673原44有信号旧任务及field到达方向的冲突与非线性变化，不能将340无结果信号旧任务当已保护。不做投影/线搜索/权重/seed救门。所有检查过才BOUNDARY_SIGNAL_PASS，仍不自动启动学习：改变代价的目标权衡、数据支持和新有限学习合同须另登记，同candidate原384全保+新增+完整407/audit不变。

900main+900独立全部重算/root2400秒40min硬停，16MiB/global10GiB/free40GiB，CPU14/Nice19/400%quota/4GiB无swap/CUDA关闭，0新增物理/原生命令/Actor/Value/optimizer。来源T672至T675全部科学/driver退出后才运行；主/独立所有数组与结果逐值精确。先实际入口0步预检、weekend预算worstRESERVED，再GitHub完整精确树核验/唯一launch。健康只读不改源协议/外写runs/并行审计备份source。失败保raw/错误/计费，科学门失败不作为工程重试，不扫范围/epsilon/阈值。
