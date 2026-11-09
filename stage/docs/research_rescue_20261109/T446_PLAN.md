# T446：干净单Actor联合成功行为迁移，条件进入自身PPO

2026-11-09用户明确“开始下一步的实验”。T443宽1.17米/80秒六米54/54、旧267/275；T445两个代表例表明同起态T255成功、当前候选从首动作起驱动更大/转向幅度更小，随后偏离并碰撞；不将其当唯一因果。T444工程失败、T445几何重构FAIL保留，不据其净空作奖励或安全指标。Stage/Phase完全隔离。

## 新机制及范围

新Actor为120因果输入→128维Tanh编码→64维真实carry GRU→双动作头→1.25tanh；全部Actor参数训练，直接输出drive/steer确定性模式。不包含任何冻结legacy base、多层残差、任务ID、路点、地图、运行教师或路由。独立训练期V沿用120→64→GRU32→1，输入含原有剩余时间列；部署只导出/使用单Actor。输入规范化沿用已锁老211数据统计，clamp±10。固定单容量/单seed，不扫网格。这是结构与联合学习工程pilot，不单独归因容量、不称标准GRU/BC/PPO原创。

训练期两类既有成功源：T255的211个有效旧开发回合，T256保存的28629行完整120观测与原T255逐轨迹SHA完全对应，动作来自原成功执行记录；另固定T443困难臂最终模型的宽左右各00/07/14/21共8条确定性成功源。后者原完整验收只存紧凑数组，所以先8条原生重建完整观测＋8条精确原生重生，全数组/积分必须精确旧T443记录；重复物理计账，不算新增独立导航经验。不使用旧64回归hold作监督标签、16未消费导航hold或失败轨迹标签。

注册前另有1次真实长度工程丢弃Actor Adam（.330秒）单列，不作为导航训练/收益；模型未保存。正式有限联合迁移1500次Actor Adam，lr3e-4，每批旧8任务各1条完整history＋宽LEFT/RIGHT各1条，按固定轮转；每回合时序均值，旧合计1/2、宽合计1/2。全部history从学生h0真实BPTT，不继承教师hidden。内点执行动作平方误差、边界±1单侧hinge；目标是确定性执行模式，非教师Gaussian密度或凭空on-policy数据。每步与独立同初始/同序Adam shadow匹配，1500live＋1500shadow，checkpoints读回含Adam。常规因果policy distillation不是从零纯RL；数据拟合不等于闭环成功。

只有10任务组各自回合均衡RMSE≤.03，且旧左五米首10决策最大动作误差≤.10，才允许最多两轮自身闭环PPO。此门只筛掉明显未学到的动作函数，不是安全/导航通过证明。未过则0新RL，仍将固定BC末模型完成一次329，报告转移负结果，不自动补BC步数/换seed。

PPO每轮固定4条新学生探索（2旧/宽左右各1），2轮共8，另8全数组/积分精确重生；严格iid裁剪Gaussian sigma.2、float32边界概率、完整carry/history、物理gamma exp(-.002/60)、lambda exp(-.002/4)、原奖励/原生接触/GOAL/时限、独立V；lrActor3e-5/V.001、clip.2、每批最多4epochs/KL>.02停额外epoch。旧左5m00与06在两轮分开，另旧task0_00/task7_20；宽03/24左右；固定seed，不加恢复教师或保持损失。教师数据不进入PPO概率比；拒绝/中断/数值错整轮Actor/V/双Adam回退，不要求逐轮旧成功全集保持而无限回退。

## 验收与证据

固定最后模型完整旧275＋宽54，共329，逐项碰撞/超时/缺测分列。没有中间面板挑最佳，不以低loss或少数中心宣布稳定。本轮预算只覆盖开发集，不预留fresh16/78独立模型原生读回，six_gate固定False；即使329全过也只能准备下一独立冻结完整门，不能自动八米。原窄暂停，安全T255冻结、不晋升。

新student worker使用64维hidden，从自身零h0开始；旧Session隐状态首步16硬编码通过显式64零初始化适配，不将16padding当新网络carry。仅teacher重建用历史结构；PPO训练/部署始终plainActor。旧runtime/物理文件不修改。worker在送出完整payload前先保存full任务文件，保留批次返回前的已完成证据；compact最终329只保留动作/counts/flags与全数组SHA，独立audit不声称恢复不存在的完整观测。

独立零原生审计重做1500 BC＋至多16历史Actor/V Adam，逐步trace/最终Actor/V/双Adam/完整120/history/hidden64/密度/奖励/原生重生和SHA全部核验；不覆盖T422或T445原失败。模型不拷到普通Git；代码/协议/收据入Git，原始RAM分段全远端读回逐成员SHA才算备份完成。

## 有限预算与本地容量

独立新目标，不恢复旧目录/截止。最多14M全部物理、600全部pipe命令（含preflight24、READY/关闭），32固定重建/训练及exact＋329验收；实算最坏锁在BUDGET。main5400秒，其中前3600允许学习，audit1800、backup600、source及余量，all8400秒，启动时写绝对截止、更早从严；driver唯一finalizer，离线科学不依赖GitHub/LLM/网络。工程预检先测完整实际长度批次GPU前反与shadow时间，若1500live/shadow估计超2400秒则拒绝，不用短toy耗时冒充吞吐。

own18MiB：17full×448KiB（8teacher＋8fresh＋1partial）、模型3×2MiB、compact400×5KiB、其余metadata/protocol/preflight/logs2.5MiB，共17.89MiB。每文件超上限或总预算不足保留已存数据停止，不临时扩容。现有global1GiB/free40GiB不放宽；先对已远端逐成员核验且本轮不读的有限旧文本无损gzip，原字节本地可恢复/旧远端保留。所有实际数据/构造依赖/模型/NPZ/错误/审计/manifest/ledger/cases/replays/optimizer排除冷存，编码源码先精确GitHub树、独立字节audit后才预检/唯一launch。

本轮是第一份干净网络联合转移基线。成功后仍需公平对照、多seed/场景泛化、消融及真机感知/时延/控制接口和安全验证，不能凭开发329或常规组件宣称顶刊创新或真机已可用。

注册前两次150文件冷存选择上界拒绝均0native/0新Adam、未创建run目录，原源/error在registration_failed_001/002保留。随后仅将compact每文件6→5KiB及metadata3→2.5MiB作预算重分配、own19→18MiB，科学配置/训练/物理/墙钟不变；不扩大文件数或删除唯一数据。
