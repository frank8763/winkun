# T600 同任务历史、同独立GRU的任务风险对照

两臂共同：完整T550冻结＋同32GRU差分残差，10274可训/10274冻结参考，原120输入和原64+双32真实carry。初始全函数精确T550，不续丢8项的T598候选。训练数据39旧T550成功history＋T598实际安全sector2_offset2恢复history；强制前缀只输入不当专家，teacher原location须匹配保存逐步输出和实际裁剪动作，两类都用rawlocation平方误差。40任务各先半时间均值/半时间worst10%误差；A任务均值，B半任务均值/半最差10任务均值。相同128fullbatch Adam lr3e-4/norm1，同初值/种子，不扫网格/不加步。两臂先都学完再做仿真；0新RL/critic/teacherquery。与旧T598数据/目标变化不作跨轮单模块因果；本轮A/B只比较任务层风险聚合。

各臂1完整自主恢复+1紧凑exact必测；恢复safe才15个固定sentinel（原八中心和T598失守项去重）；15全safe才补其余34项（15已有结果不冒充第二次独立物理重复）及4关键compactexact。原始结果在每一阶段和关键重放之前落盘，即使后续错误也不丢已测结果。全39历史成功ID保持且新增恢复safe才保留研究副本。两臂都合格按field成功数、49成功数、固定mean先于tail顺序选；不是全49通过/全场地/新增原起点任务/全局晋升。失败整ActorAdam回退。无407/fresh/reload/后续RL自动接力。

唯一pool全程共2worker，Base.Runtime初值2只记一次，关闭2；真实guard/request路径的零物理虚拟dispatch检查覆盖3×3全部9条件组合、完整114命令；113应拒绝，避免T598重复计数。所有虚拟物理量只是最坏调度预算，不是执行仿真。两臂完整最坏5246336物理/114命令，256live256shadow256audit及8丢弃pref另列。2full×2MiB、2model×1MiB、108compact×12KiB、2MiB metadata，own10MiB/global1GiB/free40；CPU14/15总200低优先GPUoff/Phase隔离。

提议独立新窗口：20:20停学、20:45停native、21:00全停，main2400/learn1200/audit900/backup240/source120秒及更早相对截止从严，完整3690秒余时准入；旧20:00窗口不改写。原生全退后唯一独立两臂256历史Adam重构/全部原始SHA/双额外carry和已有exact审计，RAM远端全成员读回，再源根树；driver健康禁止并发写runs/改源/重复收尾。

当前仅代码准备、零更新预检，未注册/未launch；需要新独立预算与截止确认。worst-task误差不是导航安全保证，组风险/循环残差不是已证明原创。
