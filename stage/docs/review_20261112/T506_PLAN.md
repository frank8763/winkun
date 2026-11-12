# T506：固定三个近目标错过任务的真实闭环RL

2026-11-12 用户要求继续推进。T505结束且审计/原始75文件全远端逐SHA/source根树均通过，所有进程退出。同一接受Actor已恢复旧八中心8/8并保住field7/24；当前剩余10接触+7超时，完整old275/six54/eight54本轮未重新验收。T505与历史T501/safeT255单独锁定，Stage/Phase完全隔离。

按T505原始diagnosis分型：5超时已接近目标后驶离、2超时全程慢且还远；10项在不同新方向接触障碍。不能把所有失败归因统一速度/时限，也不能凭观察宣布唯一本质原因。下一步先沿已有成功覆盖最接近的一类推进；依据已见开发失败的最近决策端点距离升序，固定三个未成功超时任务：sector7_offset1最近0.284388m、sector7_offset2 0.376048m、sector0_offset1 0.385563m。它们不是未见留出，也不是按新探索成绩挑选；连续native最小距离没有测到，端点最近值不冒充连续最小。实际距离约5.204/9.617/5.204m，原stepcap34697/64112/34697，时间不放宽。

文献：research_20261112_t506/PRIMARY_SOURCES.md核对原PPO、GoalGAN、ALP-GMM、RLOO以及反证。选择近目标任务是人工固定训练课程启发式，不是GoalGAN中间成功率估计、ALP-GMM历史学习进展或已训练任务生成器，不声称原创。PPO裁剪并不保证有限步信赖域/闭环保持；同行baseline在条件下独立不使整个RMS/PPO/成功门更新无偏。与T503不同起点/数据/保持集，不能单独因果归因课程收益。

从接受T505同Actor初始化，完整成熟Adam drive993/steer353保留，仅显式把监督1e-4学习率降至RL3e-5；新rl_initial仅optimizerlr变，Actorpolicyhash相同。11906个双神经残差参数可训；120因果输入/64真实GRUcarry/原骨干head冻结，Value和ValueAdam冻结且不进入本轮优势。部署单Actor直接drive/steer，无地图/规划器/路点/教师/路由。

固定三个原起终点任务，每个4条不同固定SeedSequence、原iid rawsigma[.2,.2]，共12fresh自身完整采集+12同策略/同创新流逐全部数组/积分原生exact。均从真正新回合原起点开始，hidden/time/history/执行器按T502已验证合同初始化，无中途goal切换/传送/计时清零。相同任务4条为不同探索轨迹，不是4个独立训练种子。所有成功/接触/超时/无效删失保留。至少一条原生GOAL且无障碍接触/未越±7.5m资格边界，才允许更新；否则0Actor/Value更新停止，不重跑/追加噪声/seed/预算。

学习完全用这12条当前策略采集，逐完整history重新核验真实mean<1e-10/logp<1e-9，沿float32边界atom密度和原奖励、gamma_native、物理持续时间构造有限RTG。同task其他3条在相同累计native时间的RTG均值作baseline，同行终止后续值0；系数gamma^t*(Gt-baseline)，各task四回合RMS，PPO ratio裁剪.8/1.2，三个task/各回合等权。最多16live+16shadow，KL>.005后止本次更新；保留采集history的初始unclipped location与当前已接受T505实际15成功history(old8/field7两families等权)，半mean/半最坏20%归一location风险、keep权重1。Value0，无成功筛选BC/额外示范；不是只监督回放的T504/T505。

唯一候选先真正old8+field7共15自主原生并15exact，逐ID全安全GOAL、0无效删失才做剩余17场地，再全24exact。field7结果从同一候选的guard复用，不能当独立新重复。最终接受必须old8全部8、最初7场地ID全保持、qualifiedfield>7、接触<=10且0无效/删失；否则整Actor+成熟Adam回退，原T505始终独立保留。若15守门失败，field24不执行null。当前旧275/six54/eight54/fresh16/独立78不安排，不能由15项保持推断379/383或全场地稳定。

全部最大物理与字节在T506_BUDGET.json实算：最多12+12+15+15+17+24=95个物理回合，主99启动退出命令+预检6零物理共105；44full文件，所有exact不存第二份数组但物理全部计账。own64MiB/global1GiB/free40GiB，rawshape+14MiB模型/metadata/partial最坏预留；不删除唯一证据、不静默扩额度。主3600秒学习相对1800，audit900、归档/source留1500、all6600，绝对学习注册now+3000/native+4500/all+6600，以更严截止从严。

预检实际15成功history与合成12样本完整history fixture，2clone各2次丢弃ActorAdam合计4、Value0/native0；真实密度/三组peer有限、保持初始数值零/固定骨干/成熟Adam/克隆一致/拒绝中断回退/序列化/新旧worker取消。合成fixture不算新导航数据。预检和本地提交/精确GitHub根树后唯一systemd启动，CPU14/15总1核idle/nice19/CPU-IO weight1/GPU禁用、实际Mesa llvmpipe，不能查看或调整Phase进程。

driver唯一writer；科学结束全部父worker退出后唯一零native独立历史Adam/数组/当前密度/peer系数/奖励/全部实际物理命令与门槛审计，RAM完整原始远端二进制逐成员SHA、报告及Git源码树同步。失败保留raw/partial，不恢复结束目录/不扫网格/不借号续预算。优先收集成功覆盖是否存在这一真实证据，不能保证3项或全部24本轮学会。
