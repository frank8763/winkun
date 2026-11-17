# T626 多任务真实闭环Actor更新（已获明确授权，是否实际启动查launch）

目标是原起点新增成功及保持旧成功。本轮直接训练策略，不再继续单状态Critic补丁。当前保留T608固定开发34/49、场地13/24；部署仍单Actor120因果输入直接输出drive/steer，无规划器、路点、地图、运行教师或Critic选动作。标准组件不称原创或论文顶刊已具备。

固定八任务：旧dev_task0_13/dev_task3_13；宽六米LEFT00/RIGHT23；八米LEFT13/LEFT17；场地sector0offset2/sector1offset2。任务和4个探索seed在采样前锁定，32条从原起点全程学生探索，各一次exact，64物理回合全计；不是轨迹前缀课程/教师标签。固定初始T608，两sigma=.2，目标/时限/碰撞/到达.2m不改，CPU8/10/12三worker真实并行，parent可用14，GPU只作学习。

数据门：至少一条此前失败任务的新探索safeGOAL，且至少一条旧成功任务探索safeGOAL，全32完整实际历史/初始密度/奖励/终止/ownership与exact合格才学习；否则报告覆盖不足不优化同一批来掩盖。探索成功不当确定性验收成功，exact不当新增独立样本。无seed/噪声/时刻网格，单轮固定上限。

策略学习为无Critic的clipped-PG：每任务4条物理时间RTG/其他3条同任务时刻baseline，按任务均衡、再按回合均衡；裁剪ratio[.8,1.2]、每任务adv_RMS归一，完整实际因果历史BPTT。不是标准value/GAE PPO，也不声称无偏或方差已下降。仅T608独立GRU残差10274参数训练，冻结完整旧基体；初始Actor函数/策略hash精确保持，freshAdam lr1e-5、norm1，最多8次Actor尝试。没有新增Critic。

保持项在33条旧成功任务历史上锁定当前T608的unclipped location函数，按任务均衡平方误差/原sigma，weight1；历史源中未保存完整的第34成功八米LEFT17不伪造历史，已纳入新采样及最终闭环。保持只是软损失，不保证闭环保留。步后计算每任务在同实际历史上的latent Gaussian mean-KL，任一>.01或非有限则整Actor+Adam回退该步并停止；不缩步扫网格。clip策略KL受latent KL上界的说法只在同历史/固定sigma条件下适用，最终安全仍实测。

live与shadow最多各8，独立原始完整历史重构最多8，0额外优化器预检；首步/中断/拒绝全计。全部状态含成熟Adam回退逐摘要检验。若0实际应用则不重复初始49；有实际更新则完成全部49固定开发与4关键exact，不因头两个任务失败就停，只受硬预算/证据失败中止。原34逐ID不丢且新增>=1原起点成功才保留；否则整ActorAdam回到初值，候选失败证据留存。完整407/fresh16/reload78本轮未实现null，不称全局或新场地泛化。

存储改变是为遵守用户GitHub保存/不占磁盘要求：新采四条为一批，原worker先把完整NPZ写入本批缓存，再把原request/native result无损gzip保存，之后才派生指标；所有raw数组含完整120、hidden/extra/anchor/history、奖励、积分和动作。整批在RAM组tar.gz上传GitHub Release并完整读回逐成员SHA，证据receipt落盘后只释放本轮该批本地缓存；原始字节在远端不删，旧数据/模型不动。完整训练数组在进程RAM保留到学习结束，独立audit随后从远端原始成员重新读回，不能只核metadata。

这轮明确需要网络归档：断网则保留当前四条完整本地原件并停止，不宣称可无人断网持续训练。仅验证后释放本轮缓存，网络失败不删；新远端科学raw+root总上限64MiB另列，本地新增7.5MiB、日50MiB及global1GiB/free40GiB不扩。元数据用逐记录gzip保原字节语义，不重复写两份native原返回。工作RAM计划<=2GiB，非磁盘文件/无隐蔽tmpfs输出。

本地最坏5MiB用于失败批四full1.25MiB或三最长worker partial各1.625MiB，其他2.5MiB给49紧凑原始动作计数终止/候选.375MiB/无损元数据/日志/manifest。完整32原生训练和exact raw均远端；最终49完整数组在live检查，离线仅compact字段+原始返回+4keyexact，不称49后验全位姿重构。对无法落盘的partial缺失必须明文报告，不覆盖错误。

最多5177190新增物理/123原生命令/1800main秒，日仍22M/660/9000；独立audit900，尾360（backup180/source120+余量）。network单批90s，网络等待计主管线；按完整最坏余量准入。拟新窗口21:00停学21:15原生21:30全停，旧19:00学习截止已过；新窗口及新增8Actor、分批远端存储已获用户明确批准；准备预检不启动任何新科学。

GPU使用前/运行中检查其他compute，出现其它作业让出；CPU总400%低优先、实际绑核校验，Phase数据完全隔离。源码根树先远端精确核验；唯一driver顺序main→native全退出→独立Adam/数组/远端density/reward audit→manifest→root全原始RAM备份及报告源同步。错误保原件不重启旧root，不因有余时无限加轮。

一手来源：[PPO原论文](https://arxiv.org/abs/1707.06347)。本轮借用clip surrogate，价值模型/GAE未使用；多任务均衡、当前函数保持与完整闭环晋升是工程/研究协议，尚非已证新算法。
