# 下一T684：固定时间相关性匹配新采样（准备规格，未登记启动）

前提T683语义合同PASS仅允许准备。不同于T679参数扰动，拟固定T501相同权重sigma每动作.05，两个噪声结构IID tau0 vs raw latent OU tau2s；不调幅度/时间/seed网格，无Actor/Value/optimizer、新Q或任务router。原120输入/自身GRU不加noise。每episodez0标准Gaussian，之后rho=exp(-实际上一hold物理秒/2)，真实native执行command仍clip/float32。只替换探索的时间相关性，不复制原数据当新样本。

固定任务olddev6_00/02/13/20 + field0_3/1_1/7_1/7_2（沿原探索面板，不按新结果换任务）。每臂每ID4条freshfull，各32，合计64fresh+64exact；共用预定innovations seed [68420261121,task_index,replicate]（不同任务/rep独立，两arm共享是配对commonrandomnumbers非独立重复）。每条须从新policy独立owncarry/reset，完整raw/epsilon/rho/dt/sigma/latent因子/conditionalclip因子具名保留并独立重算。rawpolicyhash要区分tau，现有T501权重hash单列；不把conditionalcliplogp标成marginalcommandjoint或当PPO。

工程准入先解决真实native Session.step时钟->sampler传递、payload/runtime inspection的密度语义；不能用旧audit假冒支持OU。可以把几何/native/history审计与概率字段分离，在审计适配视图明确标注被替换的诊断logp，原数组保持不改；更好是新明确概率审计逐字段重算。实际worker在不step模拟器的0物理预检中须验证真实Session/loader/全step函数绑定与clock、pool入口、每episode无carry/noise泄漏、IID完全等于原worker。若无法证明原入口合同则不launch。

独立科学门提前固定：tau2至少2field任务各有safe和nonsafe混合；至少2field任务tau2安全样本数严格大于tau0；old4合计tau2安全次数不低于tau0；所有full/exact/密度和独立audit通过。若FAIL保留有限否定、不增seed/样本数/tau/sigma救门；小N=4每ID，不能宣称总体统计优越/降低碰撞概率/确定性中心改善。PASS也只允许下一独立score稳定性诊断，不自动训练。单臂、总goal、碰撞/边界/超时全部报告，包括差异负向ID和真实raw/commandRMS、lag1、clip率/实际dt。

候选最坏预算固定64fresh64exact/4969728物理134cmd/3000main+300audit/root3600s1h/320MiB/global10GiB/free40GiB，须prepare按真实8任务step_cap重新精确计算并检查metadata/full4MiB上界、复合剩余reserve。原硬截止从正式预登记计算不延。CPU8/10/12native14独立audit/400%Nice19/4GiB无swap/CUDA关闭；Phase只资源元数据，重叠自身暂停。需先一手审查/真实入口0optimizer预检/ledgerRESERVED/GitHub精确源后唯一launch；目前只有此规格，无代码/预算预留/launch，不声称运行。

T220/T227/T287等旧Stage相关噪声探索不能抹掉或称新算法原创：新实验应在主审中明确旧场景/结构/密度/目标与当前T501的差别，并核查反证如TD3无OU增益。该实验检验当前已冻结模型上的时间相关探索，不替换原384+新增+同模型完整407/独立audit保留门，T501不变。
