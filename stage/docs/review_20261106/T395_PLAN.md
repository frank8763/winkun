# T395 一秒逐级恢复前沿训练

2026-11-06。用户已同意从56秒开始、56→55→54→53→52秒逐级前移，保持原80秒任务时限和原奖励；先完整磁盘admission。此为独立有界新机制pilot，非已启动结果。Stage only，与Phase聊天／模型／研究／结果完全隔离。

固定T393最终Actor SHA `f3a30d6a071124bb2998f167d9fcc4e3b3eb1c88b53a84e16db6e177e930f359`＋freshAdam。T394已记录56s随机4/4成功但mode未测，所以本轮首先做初始old8能力记录、56s mode及精确序列化parent读回，不把已有能力称新学会。底座safeT255冻结、120因果输入／6864可训练循环残差／真实history carry不变，T393原两个成功示范和权重不变，不新增Q或其他模块。

唯一新科学变量为已观察到成功覆盖附近的固定1秒恢复前沿，而非从50秒全失败点跳15秒。起级280hold，依次275／270／265／260；最多6轮，每轮4新独立iidGaussian后缀＋各一次全数组／积分精确原生重生。实际前缀仍从episode zero执行T388D真实成功动作，计入物理，保留clock／history／执行器／previous-action，当前参数重算hidden；前缀不作onpolicy likelihood标签，后缀BPTT可穿过前缀观测。部署不带强制前缀、教师／地图／规划器／路点／路由／动作掩码。

Gaussian执行域sigma[.2,.2]、float32边界尾概率、原奖励／成功半径／碰撞／折扣均不改。完整后缀四条LOO return、固定4×400尺度、PPOclip.1、Adam1e-4／8epoch／norm1、示范总MSE.1。初始／成熟序列重算真实概率须过数值门；live/shadow ActorAdam和trace逐位一致；max|logratio|≤.5才保留，否则整ActorAdam回退，2连续数值拒绝停止；中断回退并保留partial。

每轮仅在“保留更新＋本批≥2/4安全探索＋更新后同前缀mode安全”同时满足时前移1秒。52秒是最后级，不自动51秒／扩大轮数／seed／噪声或算法网格。若留级同样消耗六轮预算。实际PG梯度在加示范前记录，加入示范后的梯度增量范数与夹角单列；它们是clip/Adam前分项，不将loss小当无梯度，也不把raw梯度当Adam因果贡献。所有更新和拒绝原样保存。

最终候选测五个固定前缀mode及各一次独立序列化模型原生读回、final old8、原起点主任务两中心及重载、原057/56控制两中心。**本轮只验52–56秒恢复段，不安排完整54／old275／16fresh／78门槛，全部null，不能宣称全六米稳定／晋升或自动7/8**。初始只有56秒mode，其他点缺测不能算后续前后收益；最终跨点mode不等于多个独立地图。若原起点碰巧成功，也只记录固定中心，完整门槛须下一轮独立授权与预算。

先固定5旧冷检查点无损重编码、三条原字节流与原远端证据独立audit／metadata完整读回后再容量admission；原固定5预检释放8761172B，原12MiB门在0修改拒绝，依据本恢复段实际own12MiB将容量条件改8MiB，原源／失败保留；不增加冷集合或训练预算。旧模型／Adam全字节在本地新编码、原远端raw保留；safeT255、T299、人控和当前inputs不动。使用storage.py load_original显式恢复旧raw/gzip/旧XZ，不能读不存在裸路径。此整理为用户先前迁移清理授权及本次预算必要准备，非另开导航机制。

nav硬限110全部命令／4M全部物理，main5400秒、前3600学习／后1800验收，audit900／backup600／全部6960秒。24完整训练＋1partial各≤400KiB，6候选＋1final各≤224KiB、32compact各≤5KiB、metadata≤512KiB，总最坏12533760B＜own12MiB；global≤1GiB/free≥40GiB，runtime和归档都不越限。计数包含前缀、随机suffix、全部精确重生、baseline／mode／old8／完整原起点中心和独立模型读回；不把它们都算新学习样本。

先零原生预检4合成丢弃Adam：parent初始函数／hidden、执行探索、条件密度、fullhistory因果、成熟Adam隔离和拒绝／中断回退、PG/示范梯度记录不改变原fit更新、14零物理geometry＋1setup、取消guard／SIGTERMtoken、不可压缩输出上界。论文／作者代码核对在T395_FRONTIER_METHODS.md；standard RFCL/PPO/DAPG等非原创或安全证明。所有源码／输入SHA锁定、Git提交／远端精确根树后唯一launch，active单writer不改源／外写runs或重复finalizer。

main／native退出后唯一独立审计从parent＋raw重构全部实际ActorAdam／48以内历史丢弃更新、完整数组／hidden／噪声／密度／真实奖励和物理命令，0新增native；全原始RAM远端逐成员SHA后才称备份完成。失败保留全部证据和错误，不重启目录或借新编号续本机制预算。后续每项继续先向用户提一个关键问题、回答后另行注册。
