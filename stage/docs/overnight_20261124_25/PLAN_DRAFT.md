# 2026-11-24至25通宵复盘与提案（用户决定保留方案、暂不启动）

用户最新明确选择“先保留方案，暂不启动”。以下时段和预算仅为草案，不构成授权；不启动准备、训练、验收或自动接力。

原白天授权19:30停学习/20:15停native/21:00全停不变。目前T720科学、driver、helper、workers、finalizer/source均退出。这里只保存复盘和可审阅提案，不创建通宵进程/定时，不替用户延长授权。新窗口建议次日07:30停学习/08:15停native/09:00全停；任何更早actiondeadline从严。

## 再审结论：科学对照可讨论，执行安排尚不足以承诺通宵自主完成

用户追问是否合理、是否会中断和浪费整夜后，重新审查如下。这是草案修订，用户“暂不启动”继续有效。

1. 当前只有文档，没有新实验driver、真实交接预检或故障分支演练。因此不得将“已安排”写成“已接上”，也不能保证无人值守完成。
2. 37,590秒是预留最坏路径，不是预计实际运行时长；程序可能提前结束，也可能遇到真实错误。原21:30示例到07:56:30，距08:15原生截止仅18分30秒，不能以总时长相加代替按顺序逐action准入。预算中已含准备，注册起算和准备起算须在正式协议中消除重复/歧义。
3. 2,276个science回合中864个用于训练与训练exact，另1,412个用于最终验收与exact。六段训练main上界14,400秒，其他science上界14,850秒；不能称整夜主要在训练。完整回归有价值，但每臂48次Actor更新是否足够尚无收敛或样本效率证据，四倍于T717本身不构成充分性证明。
4. lambda比较可以回答一个机制问题，但短trace不是已确定根因。继续冻结GRU意味着两臂共同的表示限制仍可能掩盖差异；两臂均失败不能推出信用分配无效或模型不可学。
5. 存储草案与8GiB准入余量仅约103,712,013字节（约99MiB），对未完成字节上界证明的实现过紧。正式容量准入须有可解释的额外余量；不能靠删证据、运行中降审计或抬高旧上限来容纳。

正式执行版应在睡前完成一次有界预检和真实首段交接确认：共同loader/worker/梯度/终止处理、独立审计时间、所有训练线程实际CPUmask、模型与Actor/Value/Adam跨段连续性、唯一driver与停止机制。实际消耗全部计账；不能把已训练首段再次当正式首段重跑。修订正式预算时先测当前新路径吞吐，固定足以解释学习曲线的训练量，再保留两组完整最终面板；现48步数字暂为提案，不许说已经足量。

接力语义须明确：差的奖励、碰撞率或中期能力门不触发停学或等待用户；健康的固定A/B两臂和最终面板按预定顺序自动完成。科学负结果仍产出完整比较、置信范围适用性说明和下一问题。数值/数据损坏、资源冲突、无法保留完整验收或absolute到期必须停受影响注册任务，保存原件；不得为“不中断”吞掉异常、重复消费、临时换算法或延时。可独立继续的块必须在启动前声明依赖和准入，不能事后挑选。

最终汇总、逐ID收益损失、成本曲线与备份应自动衔接，不因无好结果空等下一条聊天。有限计划提前完整做完即正常结束；不为填满整夜追加未注册实验。这里能设计的是“无须常规人工确认的有界连续执行”，不是“保证任何故障下永不停止”或“保证一夜突破”。在上述执行缺项解决且获得新授权前，本草案仍不具备launch资格。

## 已核实的结果与不足

T720完整174起点T719安全91，B0安全92；8个gained、9个lost，碰撞44vs41。T719原始18源8safe是局部能力，不能覆盖整体净退步。T719条件恢复4/18/1碰撞；T717同集合1/18，原始1/18，三轮72fresh只12ActorAdam/12ValueAdam，GRU冻结。T718同噪声随机对照35vs29/54safe，但7vs6碰撞，门失败。T716固定不同神经teacher原起点虽有成功库存，实际恢复仅5/18；不同policy成功union不能当单actor能力。T714成功来源监督旧403lost12、曝光44/78vsB053/78、fresh26/72vs28，门失败。T715末验收被用户暂停跨过原absolute，未完成不能叫完整算法negative。既有T698398core+14supp=412不同起点保留，有限扩容来源监督非固定容量/泛化证明。

问题不在于审计通过就等于方法有效。此前多次将小预算诊断、有限head更新和下一小修正串联，局部改善很快被更大回归抵消；这缺少预声明的单一主要假设与同预算对照。当前完整结果不支持“新能力已稳定、可以压缩整合”，也不支持“PPO或GRU完整训练已无效”。

T717原gamma_native=.999966667222216，lambda=.95/decision。典型100native步/.2秒决策时gamma^100*lambda=.9468386053，GAE未来TD误差权重半衰期约2.53776秒；40秒时lambda^200约3.505e-5，而gamma^20000约.5134。这只是trace算术，不是有效任务horizon或GAE错误证明：准确Value可传递长程信息；短Value拟合是否足够未知。因此需要对照，不直接改参数宣布根因。

历史git c4faa2e2的T719 SUPERVISION实际全部拟合OS线程在CPU14，虽然torch设8threads。此前配置检查不能替代实际thread affinity。无因果加速比可报，新预检必须查每个训练线程的CPU8..15可用集合。已有精确结果不因此作废。

## 今晚只回答一个主要问题

相同模型、采样预算、奖励、探索、优化器下，完整折扣反馈是否比短GAE trace更可靠地获取闭环能力？

A: Actor advantage GAE lambda.95。B: Actor advantage lambda1，即完整有限horizon折扣return减冻结Value。两组Value统一拟合完整MC return；否则直接改T717 lambda会同时改变Value targets，混淆Actor credit比较。两组不改变native gamma/奖励/终止/clipGaussian法则(.2,.2)/学习率/部署精度。新增共同Value target和训练mix使它们不是T717精确复现；A/B内只有Actor trace规则不同。lambda1不是无折扣，方差可能更大，不预设胜者。

同B0初值与同20198头/固定GRU、一个共同训练seed、共用预声明随机流。12trainingID与6sourcevalidationID不变，每arm12freshblocks，每block每ID原始起点1次+原固定B0prefix恢复1次，共24fresh；旧prefix score排除，自己的64carry完整重建。每block固定12exact（偶数block原始、奇数block恢复），全fresh保存fullpayload并独立审计，exact非新增独立样本。12block不是12个训练seed。各48Actor/48Value步骤/arm、live/shadow/independent完整重建；旧4096计数不加。比T717每arm新采样多4倍，但这个有界量本身也不是充分训练保证。不按开发分数换seed/增轮数或选最佳checkpoint，只用最后预定模型。

临时学习副本无旧403保持loss/拒绝guard，保留B0/T698不可变。新能力与后续整合拆开；不加传统导航、控制器、planner/waypoint、runtime teacher/router/TaskID。不同时解冻GRU或叠加SIL/AWR/新reward。全GRU反向会穿过密集径向读出，须先证明完整history梯度/内存满足8GiB；尚不能把unfreeze当即插即用方案。

## 一次完整结算，防止再次局部报喜

最终两模型均测完整旧403、完整曝光174、18条件恢复，记录逐ID gain/loss/collision/timeout/boundary重叠；原始18已包含174，不重复计数。旧403已有B0丢11，要同时报告相对T694403原成功和相对B0额外损失，不追“绝对零遗忘”。原始18恢复能力门沿用10/12训练+4/6validation、原始12/18、最终36fresh无碰撞，作为能力取得指标；整体实用门建议174安全至少比92多10、碰撞不多于41、旧403丢失不多于20。两门分开，门槛不是期刊录用标准。24个新前瞻起点仅按固定几何规则在训练前验证/锁定，B0及两个最终模型全部测，无效单列不换点；同一地图新位置不能叫未知地图泛化。

完整研究面板无论好坏都执行；只因真实工程错误、非有限数、资源冲突或预算/absolute停止，缺测如实报告。不在同夜失败后换算法/权重/seed救门。若新能力过门但保持差，才为下一独立窗口提出整合；若两臂均失败，明确这套模型/预算下仍未学会，下一候选需先检验可塑性实现，不再蒸馏“名义会做”的teacher。单pairedseed只是机制pilot，论文仍需独立训练重复及外部强基线。

## 有界准备和时间/空间准入

建议20:30–21:30准备实现与真实预检；21:30前后注册/精确source后唯一launch。新授权若较晚才到，不滚动延后次日截止，先按剩余完整路径准入。训练分六个连续且不重复的child（每arm3个，每child4blocks，原Actor/Value/Adam连续交接），每child2400main+600audit+60close<2h。最终old403每arm3000main+360audit，174每arm2700+300，18recovery每arm600+150，新24的B0+两model每个750+180，均各预留60close。每arm全12blocks后才选最后model，调度/固定顺序在协议里锁定，不按早期表现跳过比较臂。

提案上界92,794,924physics/2276science episodes、约2800管理含cmd；science29,250s/audit5,760s、准备900s、close900s、source/primary780s，共37,590s（10h26m30s）。从21:30注册的完整路径到次日07:56:30，仍须逐段核learning07:30/native08:15/all09:00及更早action，实测吞吐/冲突让出不能延时。余campaignphysics约930.66M、cmd20513、science75056s/audit9327s，从元数据上可容纳提案，尚非正式reserved。

现runs约5.51GB，至8GiBadmission约3.08GB。提案约2,974,810,112B，含所有fullraw、model/optimizer/hash、metadata和archive临时空间，仍须真实证明writer最大字节数。训练casecap≤64112，对应≤642decisions，原无损packed schema每decision1360B加固定/ZIPheader，训练NPZ拟上限1MiB；最终casecap≤83766对应≤838decisions，拟1.25MiB。必须用实际schema/dtype/最高熵压缩/ZIP开销fixture证明并锁定；不可仅凭旧实际文件较小就准入。没有删除旧原件，不增10GiBhard/free40GiB/8GiBtrigger。全模型原件/raw/失败0deleted。

CPU8..15八physical/800%Nice19/8GiBnoSwap、7nativeworkers8..14、fit8threads实际CPUmask验证，CPUonly不抢GPU。Phase仅资源进程metadata；只能暂停本Stage注册组，deadline照走。新independenttransport900，原transport与Actor4096冻结。完整source/preflight/fullworst/time/storage/atomicstatus/真正独立audit未通过，不launch；不能用“通宵不空转”替代准入。原白天21:00授权不因本提案改变。今晚目前没有新任务/REGISTERED/进程/automation。
