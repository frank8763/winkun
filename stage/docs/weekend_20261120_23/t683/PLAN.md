# T683：时间相关原始动作的概率语义合同

T679/T681参数探索有真实新成功源但失旧；T680四pair估计不稳定；T682两个头因素均无法跨两来源保4旧，停止全维4pair训练和静态头修补，不救seed/幅度/网格。下一不同变量拟是动作噪声的时间相关性。先0native0Actor/Value/optimizer合同，不收集真实新轨迹、不权重旧回报、不导出candidate、不声称探索改进。

固定同T501实际权重/120自身因果GRU、raw_sigma每动作.05，IID tau0与唯一OU相关时间tau2秒。z0=epsilon0，rho_t=exp(-previous_native_count*.002/tau)，z_t=rho_t*zprev+sqrt(1-rho_t²)*epsilon_t；epsilon IID标准Gaussian，raw_t=mu_t+.05*z_t，原生command=clip(raw,-1,1)转float32。每episode resetnoise/GRU和RNG；录制sigma/实际dt/rho/条件sigma。要求全8既有T672实轨迹非终态native_counts[:-1]固定100(.2秒)才允许称固定时钟平稳sigma；若时钟适应噪声，条件Gaussian仍可写但不保证平稳边缘，当前准入FAIL，不改tau救。最后缩短终止count不用于下个不存在的动作。

学习候选的原始latent action路径密度定义：初项N(raw0;mu0,sigma)，后项N(raw_t;mu_t+rho_t*(rawprev-muprev),sigma*sqrt(1-rho_t²))。它是完整因果交互轨迹似然的policy因子，经确定性环境clipping生成执行动作；不是给定整个含未来外部状态序列后的raw条件概率。固定记录raw/外部完整历史/实际dt求score时须包含mu_t-rho*muprev两项，不把上一mu detach。此处只做数值合同，未来完整onpolicy/ratio/estimator仍需另审查，不以旧IID轨迹称AR onpolicy。

另报告给定previous raw内部状态的一步clipGaussian连续部分+边界原子，用既有EDGE浮点执行阈值；它的逐步乘积不是只观测clipped command历史的边缘联合概率。二维连续Gaussian原始路径联合验证、两个upperatoms的截断积分与代入单raw的naive条件乘积反例必须通过，禁止把条件原子logp直接当command路径PPO。latent与clipped度量明确分名，sigma.2名义loss不混入真实.05。

实际sampler先调用真实frozen Actor.step(sampleFalse)产生自有carry/mu，再用显式previous nativecount推进新noise；本合同只复用外部完整历史，不是原生worker接线已验证。tau0逐步raw/action/cliplogp须精确同真实原IID Actor.step；prefix/episode reset复现。8原hist来自T679锁定校准引用，已消费T672历史不算新环境样本；主/audit各重建所有数组，3固定方向epsilon1e-6数值梯度/abs1e-8+rel1e-4。actorstate逐位不变，无优化器。当前11906 adapter参数的导数使用完整自有GRU重构特征，固定外部历史时这些特征独立于adapter；不把可训练GRU hidden当外生固定状态推广。固定时钟的未截断噪声过程边缘平稳不等于给定观测或存活条件下仍平稳。所有条件且独立auditPASS才TEMPORAL_LATENT_CONTRACT_PASS，只准下一有界新采样准备，不自动采样/训练/宣称安全或泛化。

预算0physical/0nativecmd/0optimizer，600main+600audit/root1800秒30min/32MiB/global10GiB/free40GiB；CPU14/400%Nice19/4GiB无swap/CUDA关闭，资源重叠自身Stage cgroup暂停、硬截止不延。前置实际入口小前缀0步预检/一手审查/独立RESERVE/GitHub精确全树后唯一launch。
