# T685：新样本没有消除代理与确定性验收的背离

COMPLETE / PROXY_DIAGNOSTIC_COMPUTATION_PASS / 独立audit PASS。0物理0原生命令0Actor/Value/optimizer，无新candidate。原train48+freshIID32完整外部历史，两个固定T678候选与identitycontrol全部重新逐动作密度/owncarry核验，原train代理重构误差约1e-16。7.460076main+7.409010audit秒；driver810189/helper810200/science810201,810207与unit全退出。GitHub精确树931621717aa3614d571fe3dd3a9dd61496a25da3/commitbbffa6efd04370a7c06d2b4c61d87c1f39d4ebea。

|候选/已确定性失旧ID|train per-action代理|fresh per-action代理|fresh普通轨迹IS变化|
|---|---:|---:|---:|
|executed dev6_00|.000267825|.000252640|+.095708|
|executed dev6_02|.000383848|.000028345|+.009095|
|self dev6_00|.000265757|.000252774|+.095941|
|self dev6_13|.000105788|.000060847|+.040340|

全部16个candidate×freshID有限/集中度门过：ESS最小3.9441/4，最大归一权重.300955<.5；并非因长轨迹导致观测权重全部塌缩。KL是latentGaussian条件KL，fresh任务均值最大约5.11e-5，平均轨迹sum最大约.014706。可观察代理改善仍出现在独立创新新数据，而既有同candidate确定性screen却真实失败，不能简单归因只拟合T672旧batch。N4小样本、后验面板/拟合集适应性不允许据此说总体随机回报已提高或概率比估计可靠。

反例：field0_3新4条全collision，LOO代理严格0，但ordinaryIS变化约+.1109来自logmeanW约-.1175；SNIS变化0。这不是有成功信息的真实能力改善。dev6_20训练4条同回报，stored旧panel缺测null，重构代理0；fresh代理正但ordinaryIS变化约-.0167，需区别不同统计量，zero不是保护。所有8ID正负/LOO/IS/权重/lognormalizer/原子/长度与KL全表保留RESULTS，不挑好看的数。

结论仅：新样本上的有限正代理与确定性闭环失败可并存；不能把小KL、ESS门或proxyPASS当原384保留或恢复已拒绝候选。T684tau2失败路线仍停，不混入本IID分析，不救seed/时间结构。

下一不同验证问题：直接采集两个已经固定的T678候选在同8任务/同T684IID创新种子上的随机闭环，基准已消费32条仅引用不重跑；候选从未消费这些model×seed，fresh状态/动作将按自身策略演化。直接逐seed结果比概率重加权更能验证目前解释。见NEXT_T686_DIRECT_STOCHASTIC_VALIDATION_SPEC，尚未代码预检/登记/launch；0optimizer，不改变两候选、不挑最好、不据随机成功晋升。

工程账：原prepare误把T678 rejected(null numeric记录)当最终screen拒绝，按实际SCREEN.lost_ids和retained.initial校验修复；另主审中断导致最终输入manifest缺文件，均0native0optimizer，保守12秒计preflight。source仍活时早检查root被guard拒绝0science，待实际同步退出后精确核验才唯一launch；元数据2秒上界另计，SOURCE_GUARD_REJECTION保留。
