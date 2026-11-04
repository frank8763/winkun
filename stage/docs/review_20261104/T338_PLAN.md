# T338：固定恢复问题的动作空间校准PPO对照

2026-11-04用户明确“下一轮开始推进”。Stage only。原T336/T337已结束并核验备份，不重启、扩轮、换seed或复用失败机制预算。

## 假说与一手方法

T336首轮反向152个决策全部处于动作边界附近，实际变化约1e-7；T337同状态数值合同恢复了约.035/.030的执行标准差，尚无新的闭环成功。此轮检验该变化能否在T331真实48秒恢复起态增加真实GOAL与确定性恢复。

- CAPG ICML2018原文§3–4：https://proceedings.mlr.press/v80/fujita18a/fujita18a.pdf 。作者pfnet-research/capg的clipped_gaussian.py此前完整核对，blob b460fc4f34dc603977bcdeb04e7ecb9d61dcc34e，MIT。本轮未复制代码或安装依赖。
- PPO arXiv1707.06347v2，§2–4：https://arxiv.org/pdf/1707.06347 。真实新on-policy回报和正确新旧概率比；不是对人工示范伪造概率。截断目标不保证整体无偏或安全。
- P&C/SIL/GAC/ResMerge的采用/排除、实际阅读范围与反证见RESEARCH_DECISION.md。旧PPO、无旧保持、奖励罚项和投影均有负结果，不称此轮首次PPO或“取消保持即可成功”。

共同项：T333恢复研究父模型、同64隐层零初始化残差/参数量、同120因果输入/hidden每决策清零、同原奖励/速度/碰撞、同物理折扣、同48秒真实人控前缀、同4轮每轮4条新采集与8个完整batch Actor Adam。无新Q/V/BC/熵奖励/记忆模块。两臂均为获取阶段，旧指标原样测但不整块回退或因旧指标停止；safeT255不替换，不能据此宣称最终安全。

唯一实验因素是**分布及其位置参数化/尺度校准的联合包**：A保留tanh-Normal潜变量sigma=exp(-2.3)，B校准clip后的drive/steer std=.035/.030，并以M=1+sigma的有界location残差避开原父tanh局部小导数。两臂共同可训练logscale，倍率限制[.5,2]。因此不能拆开声称收益单因来自CAPG的方差定理；相对历史T336还共有PPO、更新次数和获取流程变化。标准方法/组合不称原创。

## 概率与状态合同

所有worker/learner使用相同batch1残差前向。两臂初始确定性动作严格同T336初始化控制；不同分布的随机动作均值不要求相等，CAPG内侧偏移显式保留。

真实接口执行float32。CAPG端点合并阈值是(1+nextafter(float32(1),0))/2；尾部使用该阈值的CDF质量，内部保留原连续latent draw及Gaussian密度。执行量化的内部动作虽然也是离散数值，但本方法没有把pdf谎称为量化bin概率：内部latent仍是策略记录变量。环境/history/上一动作只见float32执行动作，不能泄漏raw latent。tanh臂保留raw Gaussian draw并含tanh Jacobian；两臂PPO比值在各自相同的记录测度下计算。边界舍入/实际worker密度/初始ratio1/有限差分与live-shadow预检后才能启动。

每条恢复轨迹真实重放T331前240决策48秒；验证全12数组、history、初始120/完整203维积分，保留goal/clock/执行器/previous action。没有只set qpos的伪reset。每条再次独立原生精确重生；前缀、重放、失败均计命令/物理，重放不算独立样本。56秒起态仅正控制，课程prefix只在训练/诊断入口，部署没有前缀或路由。

完整MC reward-to-go，其他三条同物理时间回报为baseline，同行终止之后为0。每步真实native_count折扣；原80秒cap是真任务终止，行政中断不作为训练终止样本。四条完整新轨迹才更新；8live+8shadow全batch Adam，完全精确才保留。初始拒绝/中断整Actor+Adam回退只在合成预检检查，真实获取阶段不加旧保持回退。独立audit另重构全部真实live历史Adam，不计新导航训练。

## 任务与判读

先用共享初始确定性策略测**完整old275**，并测48/56秒两个控制及精确重生。各臂固定4轮、每轮4新训练+4精确重生，轮后48/56秒确定性检查。最终各测旧20、宽5开发、原窄左右2中心。所有碰撞/超时/GOAL/缺测/无效/删失分列。

只有探索动作变化且真实恢复/确定性恢复增加，才支持下一阶段恢复课程/成功复用。若只增加接触或确定性仍失败，应停止本假说的继续预算，不自动扫描噪声/步长/seed。16训练/臂是有限单seed机制pilot，没有统计显著性或通用优越性结论。

本pilot不执行最终完整old275/six54/fresh16/reload78，不将局部恢复或原窄两个中心当六米稳定；这四套完整发布门槛保持，后续巩固为单Actor之后另立完整验收。宽1.17m/80秒课程不能替代原.57m/56秒任务。原窄左六米可达性仍有不确定性。7/8/10米未进入。

## 硬预算与唯一收尾

最多32新训练+32原生精确重生；最多64live/64shadow Actor Adam，0Q/V更新；预检4丢弃Actor Adam另列。全部命令≤900/物理≤24M，注册任务最坏414命令/约12.8M物理（实际由protocol精确给出）；5400秒native，末1800秒不开始学习；外部timeout更早从严。audit≤1200秒0native/至多64丢弃历史Actor Adam；所有工作总≤7800秒含备份同步，不延长。

own14MiB/global1GiB/free≥40GiB。32完整×224KiB、4partial×224KiB、两个final Actor/Adam×512KiB、≤512compact×4KiB、metadata2MiB，合计严格小于14MiB。预检须核验完整数组未压缩最大字节与含Adam模型实测上界；不删除唯一资料/链接绕额度/静默扩上限。

协议输入/科学源SHA、零物理预检、干净本地提交与精确远端根树通过后，唯一driver启动。active单writer不改协议/科学源/其他runs/源同步。driver确认全部native父worker链退出后唯一audit、RAM全部原件远端逐成员SHA读回、报告及source同步；失败保留全部raw/error/模型，不重启或借号续预算。driver结束、失败或需人工时再通知，健康推进不重复状态。
