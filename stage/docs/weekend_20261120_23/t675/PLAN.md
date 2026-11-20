# T675：冻结完整历史中径向进展的测量合同

T674单次44旧任务投影已COMPLETE/auditPASS但科学FAIL：KKT/rank44/9active通过，field线性方向-5.73393e-7、非线性均值-5.70668e-7，且5旧任务非线性下降。停止该直接切线投影学习分支，不扫步长、阈值或权重；不证明所有约束优化无效。原保留T501不变，未做Actor更新。

下一问题是16个field任务在8次同任务样本中终态回报相同，LOO无区分。先核验完整历史是否有可重算、单位明确的连续进展量，不直接加入新奖励/Actor/Q。T675仅从T672的全部407任务1724原始full读取goal/poses/next_poses/native_counts/end_flags/终态remaining，核对状态连续、物理计数、全有限、原安全成功仍在20cm及最终XY距离对canonical remaining误差<=1e-9m。径向末端进展定义clip((d0-dT)/max(d0,.2),[-1,1])，另报记录端点的最小距离/索引/长度；这是直线径向距离，不是绕障可达距离、安全余量或连续native最小距离，不叫新导航能力。

报告每任务全部4/8结果和进展分布，尤其终态同回报的16个失败field任务。工程可用门：全407/1724几何及终态合同成立，且至少3个此类失败field在本批进展range>=.01。数值和1%门均预先锁定，失败不扫阈值/seed/样本数救。PASS只确认测量及有限样本差异，不确认可用学习梯度、奖励目标一致性或安全；下一梯度诊断/学习须独立协议，不自动采样或优化。

特别防止“势函数=新信号”的误用：固定初态、终态势归零、undiscounted完整return下，sum(Phi_next-Phi)=常数，不能凭加密时间奖励制造新的完整回报差异。若未来采用径向终态奖励会明确改变目标，可能鼓励贴障接近而非绕行；必须先分析原终态优先性、冲突和统计支持，不能称potential-based policy invariance。不据此改变GOAL/碰撞/20cm/时限验收或Actor输入，不加planner/teacher/router。

0新增物理/原生命令/Actor/Value/optimizer；600main+600独立全数重算/30min硬停，16MiB/global10GiB/free40GiB，CPU14/Nice19/4GiB无swap/CUDA关闭。单writer，资源重叠只等待自己、墙钟计费。源码/数据SHA冻结、真实测量函数fixture和望远镜恒等式预检、先worstRESERVE、GitHub精确树后唯一launch；T674及此前科学PID全退。main/audit结果逐值精确。所有失败raw/账保留，周一停止边界不变。
