# T323：两个人控可达40s状态的原剩余16s完成检验

正式固定协议；方法/近邻与反证见T323_RESET_COMPLETION_DRAFT。本任务不训练、不是部署策略，不晋升六米。原两条T317人控均超时，不伪称专家成功。原safeT255冻结，不使用T322失守候选。

各源只复现原200决策(20000native/40s)，全部12数组和真实传感器历史、边界120逐位核对后，后缀用safeT255对实际120输入canonical batch1直接输出原mode动作。前缀只重建训练初态；elapsed=20000、cap28000不重置、原goal/历史动作/hold/动力学原样，仅剩8000native/16s。末段不调用示范、不使用规划器/路点/地图/教师路由。与T320的全程固定高drive不同；原mode可到±.999999，按原生(-1,1)校验，不沿用人工UI的±.999输入限制或额外clipping。

2源各一次、每条完整12数组/积分精确原生重生一次，共4命令、最大112000物理（包括80000前缀重建）、300秒末15秒收尾，own3MiB/payload2MiB/metadata768KiB/buffer256KiB，2×512KiB原轨迹+512KiB中断/原子临时预留，global1GiB/free40GiB。0Actor/Adam/梯度/RL；0物理预检源prefix/历史/实际Actor输出域/完整容量。独立审计120秒0物理，核对人控前200/每个后缀Actor动作/奖励/时钟/history/全部12数组、终态积分与原始counts。

所有前缀动作明确HUMAN_PREFIX_RECONSTRUCTION，后缀FROZEN_SAFE_ACTOR_SUFFIX。整条属于诊断性混合行为，不是人控新成绩/从原起点自主Actor/onpolicy数据；若GOAL只能作为新的训练参考，原旧275/six54/fresh16/78全未执行。失败保留接触/超时/删失，不追加时刻/模型/噪声/幅度网格，不重启旧目录。

T322已完成全部92原件远端逐SHA/独立audit，所有进程已退；先本protocol/sourceinputSHA/0步预检/干净提交/精确完整远端根树后唯一launch；active不外写runs/改协议/source同步；结束父timeout/inhibitor退后独立audit/全部原件RAM远端读回/source同步。原模型/人控/失败/高清视频保留，Phase完全隔离。
