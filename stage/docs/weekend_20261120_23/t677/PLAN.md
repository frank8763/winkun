# T677：相同参考历史/损失的目标语义与初始零点诊断

T676 COMPLETE/auditPASS但组合方向恶化；不接边界C奖励或权重网格。发现387参考=384T501自己的成功命令+3T653额外成功命令，初始保持项梯度非零。该事实不自动是bug或遗忘根因。先明确保持当前策略与拟合历史成功行为的差别，避免将两者混称。

固定相同387外部完整历史、冻结初始T501权重/sigma.05/11906适配参数、相同action_loss(内域平方误差、边界可行域hinge)/.2²、唯一任务等权和原10权重，仅比较目标：executed使用原成功命令；self使用初始T501在这些历史上自己的clip(location0,-1,1)命令并detach/冻结。不改loss为KL或latent平方，不加入T676失败C目标。按384旧来源与3新来源分解初始loss/梯度并按384/387和3/387重组，确认executed重现T673已存梯度。self只是保留当前函数，不凭空掌握那3任务；初始零loss/梯度不是闭环安全证书。

600main+600全数独立重构/root1800秒30min硬停、32MiB/global10GiB/free40GiB，0新增物理命令/Actor/Value/optimizer，CPU14/Nice19/4GiB无swap/CUDA关闭。两目标解析首freshAdam方向共用T673原all407终态collision-cost梯度，同lr3e-5/eps1e-8/normclip1，不调用优化器或导出candidate。核对真实表示位移与原executed公式一致；报告参考损失变化、原44有结果信号旧任务与field24的切线/固定数据非线性变化和实际sigma条件KL，不能把无信号340项当受保护。两类重叠任务按同一结果目标计算，不与代价混淆。

零点合同门：原387/provenance正确，self初始loss<1e-20和梯度norm<1e-12，原executed梯度重现最大误差<1e-12、来源分解重组误差<1e-12，两固定目标中心差分eps1e-6/abs1e-9+rel1e-4全部过、finite。门为REFERENCE_CONTRACT_PASS，仅确认语义和数值，不要求/暗示导航或梯度改善，不自动训练。真实入口预检先检全387self零点和饱和hinge固定目标，0优化器。失败保留，不调门槛救。

原T677文献子代理因usage limit失败，无科学消费；本次按修订诊断范围恢复，需一手审查完成/实际代码预检/独立ledger先reserve/GitHub完整精确树后唯一launch。main/audit数组结果逐值精确。所有T672至T676科学/driverPID先退出，锁定源协议不改。后续只选一个主要变量有限匹配学习，不能混入新reward再称目标作用，最终samecandidate原384全保+新增+完整407/audit才保留。
