# T307：独立动作探索的目标条件候选工程检查

原T305因activegoal/历史goal混淆拒绝、T305R修复后因真实query-goal/旧previous-action的条件AR逆CDF退化再次拒绝；两原目录/源/协议/错误均保留并完整备份，不放宽阈值、不CDF clip、不补造logp，不重启。

本T307是**另一个分布明确的新候选**：独立bounded Gaussian采样（rho0、原两维sigma和条件边界不变），初始确定性native函数/120维hiddenreset Actor完全不变。旧AR相关性不进入新候选Actor目标；Critic前latent占位两列为0。旧AR的当前/前均值有限差分检查只作历史控制，不拿旧AR行为logp当iid onpolicy数据。

仍只0native/0Adam/0Actor/0Critic训练，复用相同八真实native目标历史证据。完整13目标字段/107其他字段、nativefirsthit/shortlast hold、原behaviorAR density、有限时域decision-endpoint占用/CONTACT优先/安全GOAL吸收/真cap与collector未知区分、每anchor bodyquery变换、随机Critic InfoNCE与未知mask、序列化和只求Actor梯度全部核验。新iid原生float32/float64动作必须在边界内，不沿用AR退化样本。

它通过也只是接口/梯度/数据合同，不是新分布导航有效或控制Q可用。仍有行为context续行和新query policy之间的offpolicy迁移问题，科学控制门槛false，不能从随机Critic梯度自动启动Actor训练。后续真正iid goal-value基线须另注册采集/critic训练/真实控制验证与条件Actor学习、完整验收和固定预算，不将其称标准SGCRL单因素复现/算法创新。

producer600秒/audit600秒/own1MiB/global1GiB/free40GiB，0新物理，不借号续旧科学训练。需T306独立audit先PASS、源码/全部输入SHA/零步预检/精确远端树后唯一launch。结束独立全合同/梯度数组精确重算、原始全远端备份。六米尚未通过，不训练7/8。
