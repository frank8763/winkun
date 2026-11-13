# T519：回到真实闭环全Actor强化学习

用户11月13日要求解释“前面11成功，现在为什么进不了实验”，并换训练思路。Stage only，Phase聊天/研究/模型/结果/进程完全隔离。接受T507/T508模型field11/24、old8/8字节不动；T516–518离线保持失败不等于该模型丢11个成功。旧locationRMSE.002约束是新实验的代理门槛，未证明是保持实际导航成功的必要条件。T515候选实际20项仅2成功也说明不能省略行为回归。

本轮暂停教师动作拟合、窄幅修正头和逐点硬保持。独立新机制：从接受模型同一函数开始，保留原8个drive/steer参数Adam1025/385矩，既有encoder/core/head新加第二Adam组0步，全部同lr3e-5。整套现有Actor可训，无新网络/新传感器/新reward/新动力学；独立Value有自己的encoder/GRU/head，与Actor不共享参数，冻结Value/Adam并且不用其估值。部署单120因果输入64真实GRUcarry，直接drive/steer，无地图/规划器/路点/运行教师/路由/任务ID。

固定最近失败task sector1_offset1，从原起点采集4条新iidGaussian sigma[.2,.2]完整自己策略轨迹（seed20261113519/i）+4全数组/积分原生exact。原cap34697/69.394s、.2m成功半径、几何、±7.5边界和原奖励不变。至少1实际qualifiedGOAL才最多16次真实clipped策略梯度更新及16shadow，否则0更新止步。这里每条轨迹都属于当前固定行为policy，没有教师动作、历史成功标签或条件prefix混成onpolicy数据。

策略梯度使用原物理步折扣reward-to-go/其余3条物理时间对齐基线，peer终止后baseline0，组内RMS缩放，clippedratio[.8,1.2]；这是critic-free PPO式有限pilot，不称标准PPO的全部组成、整体无偏或普遍降方差。每次梯度和分子logp都从真实全120history/h0重算当前GRU，完整BPTT不detach。行为logp/raw/float32边界atom永不替换，学习前mean≤1e-10/logp≤1e-9合同必须过。4条探索与原19完整成功history有soft输出保持loss系数1，但没有.002代理hardgate；step后经验KL>.01就止进一步更新，不承诺首步KL硬信赖区或安全保证。

一个候选实际执行旧8+全field24共32项原起点无教师自主检查，无论旧项中间失败也把32完整记录（除真实预算/异常）；32项再真实原生exact，全部full数组SHA和积分必须一致。结果接受只看：old8全安全GOAL、原field11逐ID全部保持、qualifiedfield>11、field接触≤9、0无效删失。满足才保留研究候选，否则整Actor/两组Adam/Value/ValueAdam回退，原最佳独立不可覆盖。old275/six54/eight54/fresh16/reload78本轮未执行null，不用old8保持冒充全旧能力保持。

磁盘有限，4条训练完整120/history/hidden/密度/奖励/动作全部保存。32项验收每次worker真实生成全数组并将完整hash/积分回传；第一次保存actions/nativecounts/endflags紧凑原件和全数组SHA，第二次同policy原生生成全数组逐SHA/积分精确，全部重复物理计账，不把replay当独立样本。验收完整120未存盘、离线audit不声称重建它们；其证据是两次实际原生完整hash一致。离线独立audit重建训练4的全部history/density/reward、所有真实ActorAdam更新、紧凑验收数组与接受/回退。临时全数组只驻RAM且不改变物理过程。这是公开的存储边界，不降低原生GOAL标准。

预算4+4训练/32+32验收，main76命令+pref6=82，物理8*34697+2*全部32原cap之和以正式BUDGET精确数为准；main1800/学习600/audit600/backup1200s，prepare学习1200/native2700/all5400硬停。own9MiB/global1GiB/free40，训练完整shape+模型/metadata/partial最坏预留实际核算，容量不足不启动不删唯一原件。CPU14/15合计1核quota idle/nice19/GPU禁用/Mesa，避免抢Phase GPU且不查看/调整其进程。

正式preflight0native、最多4synthetic丢弃Adam，真实完整density/BPTT、全Actor encoder/core确实更新、Value独立函数/Adam不变、整状态live-shadow/中断拒绝回退、保存读取及2worker取消通过。序列化scratch仅预检克隆无导航数据，可由固定fixture和源精确重建；保存SHA/recipe后移除，避免常驻磁盘，不删真实科研候选/失败。预检/代码提交/精确GitHub根树后唯一driver实际启动。运行中单writer不改科学源协议或外写其他runs/并发audit/backup/source。全部native链退出再audit/完整现有原始RAM远端逐成员SHA及报告源码同步。无自动续种子/轮数/网格。
