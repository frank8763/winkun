# T330：策略访问状态的真实人工恢复采集

2026-11-03用户要求加快六米，并明确回复现在方便再操作。T329已结束/独立审计/101原件全远端逐SHA通过；BC拟合下降而宽中心39.280秒接触余2.27249m，旧8/20，已拒绝保留。因此本项只把该失败BC候选当**研究采样器**，不晋升、不部署、不从T329继续优化。

## 新变量与一手近邻

从同一宽1.17m/80秒/原速度/最终六米goal/原起态开始，由真实组合Actor mode运行前30秒150决策，固定在其已知39.280秒接触前9.280秒交给用户。**这只是固定数据采集接管位置，不是部署时教师路由、运行时路点或目标切换。** 用户之后独占动作控制，真实state/history/执行器/clock/native state在同一env连续，不reset、不把成功轨迹动作错贴在新状态上。记录真实前缀和实际人工后缀，前缀绝不作人工标签，全部不伪造expert/on-policy logp。一次真实接管操作，成功/接触/超时/人工中止均保存；不保证用户能恢复。

检索日期2026-11-03。已读[DART CoRL2017 §4–5](https://proceedings.mlr.press/v78/laskey17a/laskey17a.pdf)及[作者代码README](https://github.com/BerkeleyAutomation/DART/blob/master/README.md)，用其恢复覆盖动机，不复现噪声优化/其MuJoCo旧任务，不复制作者算法代码。复核[DAgger正式摘要/算法](https://proceedings.mlr.press/v15/ross11a/)的学生状态纠正；本项不是完整迭代DAgger训练。

新读[HG-DAgger arXiv v2 2019-03-11 §I](https://arxiv.org/pdf/1810.02890)：人缺乏控制权、随机交替和延迟可能改变标签，故本项固定一次交接后人工独占，不随机混合每步。论文自动驾驶/后续不确定性风险阈值并未在此复现；未取得作者完整训练实现，不继承理论/安全保证。[ThriftyDAgger正式论文2021 §1](https://proceedings.mlr.press/v164/hoque22a/hoque22a.pdf)同时提示监督负担和接管时机代价，本项固定小预算，不引入其失败值头/运行时gating。当前人是少量非完美操作，不能叫最优专家；标准近邻不称原创。

## 采集接口与合同

来源safeT255冻结base+T329 BC residual，完整strict state_dict与组合policy hash锁定；前缀每步canonical batch1 q.step真实输出，不裁剪油门/转向，允许其原[-1,1]mode。实际人工输入仍限±.999，HTTP Origin/固定动作序号/唯一requestID拒绝重发合同不变。UI说明点击开始先运行30仿真秒、随后暂停；人工松开停止、单步.2仿真秒、真实速度/轮角/剩余80秒显示保持。若前缀提前终止则保留失败，不换接管时刻/重启/自动换人。

一条混合轨迹保存全120 states/nextstates（next链无损）、actual actions/counts/poses/endflags/原reward+3history/原始首末native积分；额外prefix_actions.json每决策动作/状态hash、handoff积分/状态hash，human_inputs与request_events只人工后缀。每条完整原始动作另一次真实原生重放全12数组/history/积分精确，不算第二条独立轨迹。部署成绩不使用此次混合结果。

## 预算与结束

1实际0物理几何/strict Actor首动作/复制渲染预检+1混合采集+1精确重生，共最多3命令/80000全部物理（含前缀）、1200墙钟末240重放收尾。0梯度/Adam/新RL；预检不推进实际Actor前缀，不能把dummy HTTP动作计人控。无第二次尝试/接管位置网格。own4MiB：单文件768KiB、checkpoint原子替换临时峰值再768KiB，payload1.5MiB；metadata2MiB、余.5MiB。所有newruns逻辑≤1GiB/free≥40GiB，源/模型/数据SHA、干净提交与精确远端根树后才唯一launch。active单writer，不外写runs/source同步。

所有父server/timeout/inhibitor/renderer退出后唯一零物理零优化器审计：实际prefix逐行组合动作、人工请求后缀、12原数组/history/clock/reward、初末积分、命令/物理/全部源SHA、原生精确重放收据。然后全部RAM远端逐成员SHA归档，最后source同步。失败不重启、不增加原尝试/物理/训练预算。

本项仅补纠正数据；不会自动声称学会六米/旧275或54通过，不进七八米。后续是否训练根据真实人工后缀质量另立有限协议与容量；不得无条件续T329100BC+8探索。原窄56秒与宽80秒任务、人工/混合/自主成绩明确分开；Phase完全隔离。
