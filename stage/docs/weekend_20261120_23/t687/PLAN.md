# T687：原网络全参数学习的因果梯度合同

T686直接随机验证未支持两T678候选总体更好，T685新样本正proxy/高ESS不能保闭环。停止旧候选/参数噪声/单头静态修补/时间相关性/seed网格。下一只改变可学习参数范围：从原11906command adapters改为现有整个64772参数Actor(encoder/GRU/head+两个adapter)。仍单120因果GRU64，零新增模块/teacher/router/Q/planner，不宣称容量不足或更多参数必然改善。

本项0native0optimizer，只验证未来scope对照必需的梯度与原数值准入，不导出新候选、不执行Adam、不学习。相同T501sigma.05初始、T672全407任务1724完整历史、每task等权/每trajectory等权LOO terminalY=safeGOAL-collision/peraction clippedratio.2/scale512、原387executed成功命令action_loss hinge/.2²/weight10。保lr3e-5/Adam eps1e-8/normclip1，不为fullscope重调学习率；原T678executed一次adapter提议作为历史同data/init/optimizer对照，已消费不重跑Adam。

全梯度必须q.sequence从h0完整反向GRU，不用既有M.features的no_grad缓存。逐完整轨迹streambackward，不在episode间串carry，不截断/沿用候选hidden；参考387亦完整q.sequence，固定目标actions，不将target或baseline oldlogp带梯度。原frozenGRU的adapter切片应与T677/T673已存全目标gradient逐元素误差<1e-10；这个恒等验证同时约束采样权重/终态/归一化/参考语义。

三个固定方向(全参数/仅GRU/仅encoder)在一个预定mixed oldtask4完整历史+2来源参考的联合probe loss中做中心差分eps1e-6、abs1e-8+rel1e-4，全episode实际carry前缀一致/末端到前GRU反传和非零新增范围gradient验证。数值probe不是环境导数或证明全数据估计可靠。

在全部407+387初始gradient有限/语义通过后，仅按freshAdam首步解析公式构造内存位移，不调用optimizer.step，真实浮点表示误差<1e-12、全state精确恢复。用同T678旧44+field完整历史每ID均值latentGaussianKL(sigma.05)，最大<=.01才fullscope数值pilot准入；全407均报告KLmean/sum与surrogate，不以它作旧保持证书。semanticPASS与pilot_admitted分开，KL拒绝不改lr/阈值/seed救；fullscope可能更大函数变化只是范围效应，不声称相同KL匹配/优化器优越性/创新。

全部梯度/位移/数组/hash/每task信息独立重新构造。0physics0nativecmd0Actor/Value/optimizer，无checkpoint导出，2400main+2400audit/root6000s100min/64MiB/global10GiB/free40GiB/CPU14/400%Nice19/4GiB无swap/CUDA关闭。只有语义和数值准入都通过，才另注册一次fullscope真实Adam+shadow+独立的有限pilot，原384逐ID全保+新增+samecandidate407/audit门不改。原T210大KL全参数失败须主审复盘，不当fullscope首次或恢复已失败数据/训练。先真实入口小probe0优化器预检/一手审查/ledgerRESERVED/GitHub精确源后唯一launch。
