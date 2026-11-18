# T644：真实偏离状态的有限自主恢复探索

用户明确“好，下一步开始”承接T643结论与恢复课程建议；本项仅先完成真实成功覆盖采集，0更新，不自动进入RL或监督训练。T643原T501从T642真实第7决策/1.4秒状态接管碰撞、exact/audit/远端SHA已完成；名义成功不是可靠恢复教师。

固定同一T501模型、实际T642源prefix、1.4秒接管点、原80秒总时钟、原奖励及动作clip。唯一探索机制为原Actor各决策独立Gaussian raw_sigma=[.2,.2]，只在前7强制prefix之后采样；8个预先固定seed=[20261118644,i],i0..7，最多8条新恢复轨迹和8exact。没有seed/sigma/时窗网格或按结果追加，同一策略8rollout不算8训练重复。每个model从回合开始读取真实输入演化64carry，prefix不抽噪声、不计策略学习；之后实际sampled raw/atom/logp/action与RNG完整记录。actor始终冻结，不用探索成功冒充确定性学生成功。

3worker分批3/3/2收集，每批后做同seed逐数组/积分exact；8fresh全full唯一保存，8exact保存唯一compact动作/计数/终止+完整生成数组hash，并在主管线逐完整history/carry/RNG/密度/reward检查。独立审计8full真实重建、8exacthash与compact核对、源prefix物理边界逐位检查。只纳入原生安全GOAL且exact的实际执行后缀作为候选成功控制；原均值失败，不能把location当成功标签。失败原始全保留，任何工程异常停，不无限重启。

0Actor/Critic/optimizer/GPU，最坏640000物理步/22命令（3启动+16回合+3关闭）、26MiB（8full+3partial+1合同失败全数组各2MiB+2MiBmetadata，另compact余量由实际metadata保留），主管线600秒/审计300秒/备份120秒/源90秒。30分钟内全停且更早沿用T64221:25原生/22:05总截止；与T642/T643实际合计仍≤T64211575404步/420cmd/40MiB和全局1GiB/空闲40GiB。三核8/10/12+主管线14，400%低优先、4GiB无swap、Phase物理核资源隔离。

一手方向：[JSRL正式论文](https://proceedings.mlr.press/v202/uchendu23a.html)的课程状态及既有策略，[DAgger](https://proceedings.mlr.press/v15/ross11a.html)的实际访问分布。原实现沿自身T631/T643 audited stochastic-prefix/native/fullcarry合同，非完整JSRL/DAgger算法复现或原创算法。已有T631其他状态探索全失败、T643确定性恢复失败不外推本项可达性；8随机全失败也不证明物理无解。成功后才另注册恢复学习与旧新联合验收，不自动扩大优化器或原时窗。
