# T671：一次联合结果学习候选的可行性验证

T670四旧任务结果梯度有支持且首Adam公式位移同时改善4旧与newfield，投影完全不触发；不运行相同两臂或调权重制造冲突。本项仅检验一个真实联合学习候选能否在完整原起点验收中实现保旧+新增，不声称投影创新或算法比较。

初始T501权重sigma=.05，与T669四old各8真实自主轨迹、T663low首块24field各2轨迹的policy/hash/密度完全一致，所有来源均有exact审计。复用不计新独立样本，不使用随机成功为BC教师。28任务等权collision-cost终态结果(+1safe/-1collision/0其他)LOO基线，完整实际因果history的逐决策clipped ratio PPO替代目标、fixedscore_scale；叠加原replay10×387唯一名义参照。骨干冻结、11906适配参数、freshAdam原lr3e-5/eps1e-8、梯度norm1，0Value/Critic。

最多4次真实Actor更新+等量shadow+独立重建，各次BEFORE/AFTER账。数据在四passes中固定，不能称4fresh轮或无偏实际回报更新。每次先保存完整Actor/Adam；真实after-before位移首步需与T670解析firstAdam公式逐数核对<1e-12；四旧非线性样本surrogate各自不低于本次before(容差1e-12)，原任务latentKL≤.01/finite全过才接受，否则完整Actor/Adam及step回退并停本块。接受保留标准Adam产生的moments，无投影/缩放/seed网格。样本surrogate改善不保证实际确定性到达。

先live/shadow初始及最终state/trace精确，训练后克隆恢复，释放缓存再原生。先40项screen(原36+四新遗忘反例)，21指定旧screen全保且原384外新增才准完整407；24关键exact含四反例。候选全程单Actor、原canonical scene/reset/目标/时限/20cm/碰撞/边界判据不变。screen不是原384全保，随机GOAL不是确定性成功。

无新采样训练，最多407+24=431科学评估回合+worker6=437命令；物理硬上限实际sum(all407+exact24)，96MiB。5700主管秒含≤600学习与最多4500完整验收reserve，600独立审计，2小时job总停。CPU学习/audit14、原生8/10/12、400%Nice19/4GiB无swap/free40GiB，GPU关闭：该小适配器步骤当前完整CPU梯度审查约10秒，保持相同设备实现与独立审计一致。weekend总额度独立reserved/actual结算，不扩上限。

每个来源全history/density/Actor初始与T670真实loss梯度预检，0额外optimizer预检。科学结束独立逐case array/exact、live/shadow/history4更新state、回退和同候选407逐ID审计。只原384全部保持且至少一新增且auditPASS才可保留；联合387集合单独报告。原始模型/失败/raw全保留，远端pending不得删。若失败据真实新反例复盘，不以surrogate/样本改善写成突破，也不重复同data多轮。
