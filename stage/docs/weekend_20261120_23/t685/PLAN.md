# T685：冻结候选在新IID样本上的代理有效性与轨迹概率支持

T684时间相关方案因old1/16 vsIID7/16原门FAIL，停固定tau2比较，不救tau/sigma/seed/sample。新IID32仍是相同T501sigma.05的未参与T678拟合样本；不丢数据也不把tau2混作IID。这一项无新采集/训练，固定两个既有T678executed/self拒绝候选，原0Actor模型T501为identitycontrol，不选模型/不更新/不晋升。

8ID固定沿T684olddev6_00/02/13/20+field0_3/1_1/7_1/7_2。原T672train每old4/field8共48完整，与T684新IID每ID4共32完整分别报告；核对无重复seed/hash、时间上candidate先冻结新样本后产生。新数据已见T684比较结果，这是后验诊断而非盲检/统计泛化。两候选同初始/同sigma，只头adapter更改，主干GRUhash相同；所有外部完整历史从h0重建各自carry，不从候选新state取baselinehidden。

严格复现旧代理 S_unclip=mean_i[adv_LOO_i sum_t(exp(delta_logp_t)-1)]/512 及S_clip.2=mean_i sum_t(min(ratio*adv,clip(ratio,.8,1.2)*adv)-adv)/512，不误称原来用tempered轨迹ratio。与T678已存nonlinear_panel同ID原train逐数值核对。另sum_t delta_logp是真正IID clippedcommand路径policy因子logratio，保每轨迹ell、未裁剪W=exp(ell)、ordinaryIS change=mean((W-1)*Y)、selfnormalizedIS/Y与delta、ESS、max normalized weight、logmeanW；SNIS有偏且普通IS有限样本可能极高方差，不凭N4重加权推出总体回报变化。log空间保溢出/下溢信息，不裁weight救门。

T684的logp具名RAWlatent，IID真正clipGaussian密度必须重构并核对conditional_clip_logp；不误读字段。原T672logp是clip，每动作atom真实EDGE元数据。终态Y=safeGOAL-collision，LOO只同task自身，sameY导致zero不是安全或保持。报告真实sigma.05任务均值KL、轨迹sumKL、长度、clipatoms和样本回报支持；KL明确是latentGaussian条件KL，不等于clippedcommand KL；小均值KL不保闭环/trajectoryIS。候选在train上拟合，因此trainIS不作固定策略无偏估计，新集候选在创新种子产生前固定也不消除N4有限性。固定支持判据：全部train/fresh指标finite，且新集每candidate×8ID ESS>=3、max weight<=.5；这只是工程集中度门，不是置信/安全定理。条件够才描述有限样本IS，失败则标不支持，不调阈值/weightclip/缩步/加样本救。

只对原先确有trainproxy正且candidate确定性screen已失败的oldIDs，列train/fresh符号；fresh同回报标无信号、符号反转标不确认，若fresh仍正而原确定性失败则标mismatch仍在。零点control逐episode/动作delta=0、ESS=N、ordinary/SNISdelta0须过；用不超过32动作手算proxy与logweights fixture验证定义及LOO分母。无任何科学结果授权学习；最终所有表/数组由独立过程精确重算。

0物理0cmd0Actor/Value/optimizer，无新candidate，600main+600audit/root1800s30min/32MiB/global10GiB/free40GiB/CPU14/400%Nice19/4GiB无swap/CUDA关闭；旧driver/helper/science先退出，一手审查/真实入口零步预检/独立RESERVED/GitHub精确源后唯一launch。当前T501保留和原384+新增+samecandidate407/audit门不变。
