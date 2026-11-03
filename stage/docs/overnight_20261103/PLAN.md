# 2026-11-03 Stage 通宵计划

用户11月3日晚明确授权通宵，11月4日北京时间08:00查看。Phase完全隔离。新的T336是唯一拟运行导航：示范辅助、原生动态起态反向课程的在线SAC工程pilot。不是T329/T333监督预算续跑，不把所有运行编号当RL更新；原失败和safeT255保留。

依据[T335复核](../distance_20261103/T335_FINDINGS.md)，当前恢复候选能拟合人控起始动作，自己的状态在3.4秒出现>1cm记录偏差，7.184秒撞宽场景circle9。需要学习轨迹附近的闭环恢复。优先采用[RFCL §4](https://arxiv.org/html/2405.03379v1)：按能力推进的成功示范起态反向课程，再在原起点及邻域验证。作者curriculum.py blob69095f4cd90eb8967dc19fea503c4ca4c7e03994已读。与[RLPD §4](https://arxiv.org/pdf/2302.02948)的离线/在线各半采样、LayerNorm Critic及新鲜在线交互结合；作者train_finetuning.py blob80b1b9ca9a5073d98c3064452af709222a7704e1已读。两者标准做法不称原创。RFCL原文21 fully observable manipulation基准，不等同本部分观测导航。

## 本夜单一科学比较

相同T333恢复Actor冻结为基础、额外相同7874参数latent残差和2参数全局logstd、新鲜双Q/target/Adam；两个臂均相同SAC和示范回放，仅训练起点分布不同：fixed始终原起点；reverse从T331真实成功轨迹末段向原起点推进。不是相对旧实验单因素比较，也不是原版RFCL复现。使用新tanh-Normal定义，不沿用旧AR/截断高斯logp。

示范只用已核验宽1.17m/80秒T328与T331。T331前150策略动作允许作真实off-policy转移，绝不称真人标签/on-policy。训练恢复起点通过真实执行对应原生前缀得到，完整120、执行器、history、上一动作及80秒已耗时均继承，前缀全部计物理预算；只在回合开始初始化，部署无示范/教师/路点。最终goal不改，原速度不改。绝不将行政截止当终止bootstrap标签。

SAC双Q使用LayerNorm，2x128宽，全新；actor只训练新latent残差及logstd，frozen T333复合体保持不变。对原旧211的28629状态按8任务均衡抽样作soft100保持，真实old20全部安全才接受Actor+Adam提案，否则整Actor+Adam回退，Critic继续学习真实回放。两次连续拒绝停止该臂，不增锚点/步长/seed/噪声网格。标准Bellman soft-Q loss不冒充验证过的动作排序或原reward Q。

最多每臂8轮，每轮4真实训练+4原生逐数组/积分精确重生，总64训练+64重生，非128独立样本。每轮1000实际Critic+1000Actor Adam及同初始shadow精确重构；每臂首次1000Critic warmup另计。完整最坏主Actor16000、主Q18000，shadow对应另34000；独立audit再丢弃历史重构，不算新导航学习。每轮候选真实old20、课程frontier及原起点中心分列。课程推进由同前沿最近连续2次训练安全及当前中心确定性安全，最终仍必须回原起点。精确起点列表与所有阈值在protocol锁定，训练中不改。

## 硬预算和截止

own22MiB，训练64x224KiB=14MiB、两个最终Actor/Q/target/Adam包各≤1280KiB、紧凑验收最多832x4KiB、metadata2MiB、其余为buffer；global stage/runs≤1GiB/free≥40GiB。紧凑验收只存actions/native_counts/end_flags及全部原数组/模型/积分SHA，不宣称保存了完整验收state/reward数值；训练则完整120/history/reward/创新与积分，逐数组精确重生。注册时完整最坏预留必须通过，不删除唯一证据或扩大容量。全部主仿真≤2000命令/60M物理（包括每次前缀、拒绝候选和所有重放），所有样本真实计数。独立audit0新native，最多3600秒且不超过08:00。

学习最晚11月4日06:30停止；主仿真/验收最晚07:15停止；所有审计/归档与汇总08:00截止，外部timeout更早从严。07:00之后不开新训练。协议有限次数或安全条件先触发则提前停止，不为填满墙钟重复失败作业。未完成/无效/行政删失明确分列。

最终两臂先wide5+原窄左右中心。按wide5安全数、再原窄两中心安全数、精确平局优先reverse的预注册顺序选一个完整验收候选，不根据完整门槛后挑模型。该候选实际执行原old275+原six54。只有全部安全，才消费16fresh，再78独立原生模型重载逐数组/积分复验。课程/人控/中心/探索均不是六米稳定。64/78验收条件和原56秒碰撞门槛不放宽。全部完整门槛过才另注册7米，再8米；每次新级必须新完整最坏预算在本夜60M/2000和global余量内，否则不启动。当前协议不含7/8训练。

启动前真正主入口全局导入、真实worker strict动作、SAC live/shadow/拒绝回退的合成克隆、零物理环境/scene/clock合同、源/输入SHA、干净commit及精确远端根树。没有launch不能称已运行。active单writer不写其他stage/runs、改源协议或sync_source。所有物理worker/父/timeout/inhibitor停止后唯一独立audit，完整RAM Release每成员SHA读回，再代码/协议/汇总/收据源同步。夜间巡检stage-11-3只报真实变化，08:00一次结果汇总，停止后删除。

论文方向先复现强基线，检验“恢复邻域覆盖扩大，同时保持旧导航能力”的实证假设。RFCL/RLPD/CCIL/TaSIL近邻已存在，模块组合、日志合同、20例接受不能称原创或安全证书。长距和跨场景泛化未过之前，不承诺顶刊结论。
