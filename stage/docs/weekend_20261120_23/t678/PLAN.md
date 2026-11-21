# T678：仅参考动作目标不同的两候选单步实测

T677真实零点合同/独立auditPASS。原387动作损失初始梯度几乎全部来自3条T653成功目标；self目标初始近零。两首Adam实际方向差15.53%，但同10旧任务样本目标下降；不能认为更换参考就已解决遗忘。这次一次有限实测检验这个单变量，不能用诊断替代真实闭环。

两臂executed/self按固定顺序分别从同一T501sigma.05/freshAdam开始，均11906适配参数、原encoder/GRU/head冻结，原输入120与自身完整因果carry。共同数据为T672全部407任务1724已采完整自主轨迹，按唯一任务等权，每任务4/8样本等权，终态Y=原safeGOAL-collision，LOO完整episode baseline/固定score_scale512/ratio_clip.2。数据复用非新独立采样；不加入失败边界C奖励或径向奖励，不接Value/Q/teacher/router/planner，不改任务几何/原时限/GOAL20cm。

共同387参考外部历史、action_loss内域平方/边界hinge、除.2²、任务等权、weight10；唯一变量为目标：executed原成功命令，self为冻结初始T501在同历史的clip(location0,-1,1)自身命令。self不凭空获得3额外任务成功行为，也不保raw均值或随机分布；初始零点不是全状态安全证书。每臂仅1次真实Actor Adam提议、1次shadow、1次独立历史重构，0Value，无额外optimizer预检。lr3e-5/eps1e-8/normclip1，必须实际梯度与T677已登记目标对应数组误差<1e-10、表示位移与首Adam公式及对应存储位移<1e-12。真实BEFORE/AFTER计费，拒绝也记账，不因候选失败追加第二步。

真实step后仅finite和所有旧44/field完整历史的最大任务平均latentKL(实际sigma.05)<=.01作为数值准入；旧样本surrogate冲突完整报告，不冒充安全保证，不做已取消的投影或缩步。live/shadow全部参数/Adam/trace精确一致后导出提议模型并精确恢复初始状态。数值拒绝也保留原提议checkpoint/hash，不能把失败raw覆盖为父模型。

每臂先完整保存固定screen：原40面板+所有T501原384外23任务+历史遗忘ID+两方向所有负旧margin任务，去重(最多80)。screen每条full，全部独立原生exact(保存compact但全部数组积分SHA)；需要的原成功ID全保持且至少1个原384外任务成功才补剩余至完整407，完整评估预算单列<=4500秒。缺测null，不报0/407或样本成功为泛化。原384逐ID全保+新增+同candidate完整407及独立auditPASS才允许保留，两个候选成功集合不可拼接；联合387集合单独报。无保留时此两臂有限对照结束，不用权重/seed/step网格救。

注册前按真实screen计算物理上限：每臂sum(407 stepcap)+sum(screen stepcap)，命令407+screen数+6；两臂各自先准入，最坏相加记周末独立ledger。每臂5700science含600learning/4500完整407reserve+300audit，≤100min；root13200秒只是两个有限job接力容器≤3h40硬停，不是单个长job；周一边界更早从严。原始存储每screen full4MiB+exact32KiB、剩余compact32KiB、模型2MiB/metadata/partial槽有界，每臂384MiB、root800MiB/global10GiB/free40GiB。CPU8/10/12原生、14学习审计400%Nice19低优先4GiB无swap/CUDA关闭，helper只管理Stage cgroup，Phase仅资源元数据重叠暂停自己且墙钟继续计。

先真实入口0native0optimizer预检：两臂完整数据梯度同T677对齐、freshAdam空状态/初始hash一致与完整deepcopy回退、helper/childplan/全预算/密度/完整carry；先weekend worstRESERVED，代码GitHub完整精确树核验后唯一launch。各臂main全退后独立审计原生全部数组/积分、1次独立Actor重构/全state回退/Value原封。每臂完成立即累计账，工程失败保raw/error/消费且停root，仅新目录接未执行部分、不重跑已耗更新。科学全退后再归档source，健康期间只读监督不外写runs/改锁定源协议。
