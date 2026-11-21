# T695：冻结T694与T501的固定一厘米起点邻域对照

T694已通过同模型原384全保+六项新增+完整407+独立恢复audit；该模型390成功冻结，T501原父模型保留。本项只检验这六个已公开开发任务周围的有限起点迁移，0参数/中心/系数/optimizer/Value更新，不作算法优越性、整个邻域安全或未知地图泛化结论。

在观察任何新结果前固定六项field0_1/0_2/1_1/2_1/5_1/7_1，每项四个初态位置：世界x正负0.01米、y正负0.01米，yaw和goal不变，原完整native快照/静态几何/终态碰撞/20cm到达/时间上限全部保持。只改初态位姿，120观测由真实物理及完整reset历史重建，不对120列独立加噪声。24起点全部预定，不按结果选择幅度、方向或子集；不加yaw/尺度/seed网格。每个newjob去掉只适用于原中心的initial_contract，保留其父起点身份并由真实canonical Session在首物理前校验scene、goal、实际pose、zeroowncarry/clock及无初始碰撞。正式collect先无Actor构造全部24初态，禁mj_step、计26管理/有效性命令，并保存首状态/积分hash；两policy首状态必须与该共同收据匹配，全部有效后才物理rollout。任一无效即停止并报告初始化有效性拒绝，不记导航失败、不缩幅替代。

两冻结policy逐点同任务同预算，每臂24full+24exact，共48fresh+48exact，6管理命令，128命令（96科学+6worker管理+24零步初态有效性检查+2有效性factory管理）；每次exact须所有原生数组/积分逐位一致。T694前向使用实际扩展loader，T501使用其原分支，同一canonicalfactory/GRU owncarry/动作clip-float32/控制周期；无外部老师、prefix、路由、规划器。顺序固定T501后T694，不按结果改顺序。匹配初态hash、scene、goal、首观测和零hidden在两policy间逐点相同。新旧起点身份与已消费protocol的静态pose/goal做元数据查重，若重合则明确历史暴露，不把重合叫新留出，不替换测试点。

逐点报告safe/collision/timeout/boundary、T694-only成功、T501-only成功、共同成败及每父任务四点情况，主比较是固定集合成功数和成对差，不只挑成功方向。exact是工程复现，不加独立样本数。没有再学习或新模型晋升；即使邻域失败也不抹T694原390个原始起点保持证据。邻域最多24点，不推出全空间概率/无碰撞保证，不在本测试后立即加中心。

最坏物理4*sum(24原cap)=3,801,552，128cmd，main2400/audit600/root3300秒，各child<2h，登记时锚定原deadline后不延。CPU8/10/12/14、400%Nice19、4GiB/noSwap/CUDAoff、root256MiB/global10GiB/free40GiB；48full*4MiB+48compact*32KiB+3partial*4MiB+24MiBmetadata<256MiB。所有失败/影子/独立消耗照计，模型不复制不改，原具名模型和raw保留。预检成本另入science，先worstRESERVED再精确GitHubsource且shell退出后唯一launch。独立audit保留完整历史/owncarry/density/reward/物理数组/ID/count检查；输入清单SHA首末检查，轻量资源/时间守卫在中间，不重复T694旧全source每row重读的工程错误。退出后才source/archive。

当前本PLAN只是固定规格，尚无新原生、登记或launch；正式入口预检与一手评测审查仍需完成。
