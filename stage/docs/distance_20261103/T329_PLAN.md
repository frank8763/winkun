# T329：成功示范初始化与一次新鲜真实RL的冻结基础残差pilot

2026-11-03用户确认T328成功并要求继续六→七→八/全图、后续提速。T328独立audit及18原件完整远端SHA已过：1human真GOAL63.88秒、余.1999795m；全程drive=.999、实际速度均值.096518m/s。**仅1.17m宽缺口/80秒课程成功**，原窄场景56秒未过。此次先检验能否把完整成功经验变成自主闭环，速度参数保持；动力学速度范围升级后续单独协议，不直接把电机速度提高说算法收益。

## 方法/近邻与范围

完全冻结safeT255所有参数，新增已工程核验的120→64Tanh→2纠偏7874参数，单一计算图无任务/距离路由、无地图/规划器/路点/运行时教师/critic。原120/hiddenreset/原mode与boundedGaussian center保持。标准残差非原创；T31966线性头与T322全网不完美示范都失败，本项变量同时变为成功数据+非线性新增残差+参数隔离，只能联合工程pilot，不能单独归因或称完整DAPG复现。

一手资料本轮复核[ RRLfD v1](https://arxiv.org/pdf/2106.08050)III–IV固定base/残差/微调负结果（前轮已读作者项目页，未取得完整作者训练代码）；[DAPG v2](https://arxiv.org/pdf/1709.10087)IV示范与RL（前轮读mjrl/dapg.py），本项不用其自然梯度、不继承其操作任务保证；[Pardo2018](https://proceedings.mlr.press/v80/pardo18a.html)正式摘要区分真实任务cap/行政截断，80秒是真新有限任务，不能重标旧56秒成功。T308/T309值头控制负结果、T312R0保留更新、T322旧失守都纳入，当前不同时加critic/奖励/记忆/速度网格。

只用T328一条320决策成功human作监督，不伪造expertlogp或当当前policy数据。固定old211成功集每task32行共256用于保持；输入标准化只用此576行，不使用未来未见任务。100固定BC Adam，lr.001，8个转向变化行/其余保持各占一半，两个实际动作modeMSE；old256mode保持100，完整行64分块累积，clip1。参数全部基础冻结，输出层零初始化首隐藏梯度零如实报告。100live+100shadow逐步Actor/AdamSHA精确，保存唯一BC提案，old20全GOAL接受否则整纠偏+Adam回退；候选宽课程中心仍实际诊断，不因旧失败隐去。

随后从选定组合策略原起点采集8条新鲜iid boundedGaussian（8固定创新seed、同宽LEFT6/80秒原goal），并各全16数组/history/创新与终态积分精确原生重生。不是8独立训练种子。既存T308 rho0生成路径复制，仅task输出目录与80秒partial空间边界改到768KiB；真实primitive动作每100native/0.2s，原奖励，无AR历史密度冒用、无新critic。

只1次REINFORCE完整轨迹score更新：每条完整折扣G0减其它7条G0均值（独立episode的动作无关baseline），同批RMS最低.05缩放，loss=-mean[(G0-B0)/scale * sum(logp)] +原old256soft100。这是完整轨迹估计器，不是假称每时刻reward-to-go/无偏PPO整体；标准baseline已有先例，同批缩放与Adam不作整体无偏承诺。freshAdam lr.0001，完整canonical batch1均值/密度误差≤3e-6/.005，clip1。1live+1shadow逐位重构，同批sample k3≤.02/maxabslogratio≤1，再真实old20全GOAL才保留，否则整纠偏+Adam回退。无步长回溯/轮数/seed网格，8全同结局无有辨别信号照实报告，不能自动追加。

## 原生比较与预算

固定顺序，全部下列native结果每例另一次逐数组/积分精确重生（不增加独立样本）：

1. 零初始化父：宽LEFT中心+原窄六米左右中心，共3。
2. BC候选old20+宽中心，共21；是否拒绝分列。
3. 新策略8训练+8精确重生。
4. RL候选若数值合格则old20，数值不合格不执行old20（缺测不记失败）；宽中心无论如何诊断，最多21。
5. 最终选定候选old20+宽场景5固定平移开发(中心/±x3cm/±y3cm)+原窄六米左右中心，共27。5开发非未见留出。

全部最多2×(3+21+8+21+27)=160命令，按40k上界6.4M全部物理。main2700秒，学习截止main+2100秒、末600秒留最终验收；外部timeout更早从严，不延长。不能拿新编号续此机制预算。四CPU workers，真实复合state_dict strict加载/hash/实际首动作必须预检通过，0native4合成丢弃Adam+1真实BC完整反向，不算导航训练；数值/原生partial保留，不假称完成原计划。

own16MiB：8full×768KiB=6MiB、4partial×768KiB=3MiB、5patch×512KiB=2.5MiB、最多72compact×12KiB=.844MiB，rawpayload合计**12.344MiB**，metadata≤2.5MiB+buffer1MiB，总≤15.844MiB；实际payload上限设置13MiB、metadata2MiB、buffer1MiB。所有patch只新增纠偏/normalization/Adam并锁safe父SHA，不复制父模型；拒绝模型保留。global≤1GiB/free40GiB，完整预留后才启动。所有物理命令/回退/历史shadow/参数proposal如实核账。

训练完整120/history/创新/reward/密度保存，决策next_states链无损存储支持400行，不沿用T291的280行限制。评估仅存动作/counts/endflags/reward紧凑原件及全数组SHA/原生精确重生收据；独立audit不声称重建未存评估中间120。202主历史Adam（100live100shadow+1live1shadow），结束后独立再202丢弃重构，核对所有模型/Adam/回退/训练raw/物理/输入源SHA，native0。所有父workers/inhibitor/timeout停止后全raw RAM远端二进制读回逐成员SHA，再精确source根树。

本pilot**不执行old275/原six54/16fresh/78独立原生模型重载**，不晋升/不自动7或8。当前任务就是成功经验能否转化为自主课程能力与保持。真正原六米稳定仍需同候选完整原门槛，七米再八米按既有授权逐级注册。若本有限pilot失败，保留失败/原因，不追加100步或8seed。逆向课程/速度范围升级仍为下一独立机制，不当本轮已执行。
