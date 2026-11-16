# T582准备：固定当前策略数据，比较是否加入旧任务真实RL梯度

用户2026-11-16“下一轮开始”。现有窗口12:00全部停止；本方案完整最坏需约79分钟，**新13:00截止与独立额度待确认**。此文件不是启动；未出现AUTHORIZATION+protocol+launch前不能称已运行。T581已结束不重启。

共同初始为T550完整原函数＋已验证T581零残差/5986参数，seed581/scale.1/原120与64carry、噪声sigma.2/原奖励/物理时标/8步Adam上限/lr3e-5/保持100不变。0新教师/0BC/0Critic，不用被拒绝T581模型。标准多任务RL不称原创/完整CLEAR/零遗忘保证。

六任务固定：左六米00＋T581已失守的field_sector0off2/sector1off1/sector1off2/sector2off2/sector4off2。每项当前共同初始Actor4条seed[582,task,k]新鲜完整轨迹，各1原生exact；共24unique+24重复，两组共用同批，不计48独立采样。六任务同行物理时间优势分别计算，不能跨任务相减；行政删失不可作完整PPO数据。全部真实history重算body特征，不拼接不同轨迹hidden。

A NEW_ONLY：左六米PPO＋27历史保持。B NEW_PLUS_OLD：相同左六米PPO＋五个旧任务PPO均值＋相同27历史保持。**左六米系数两组均1，旧家族总权重B=1/A=0，五旧各.2**；相比旧草案的六任务整体均值，这是预注册前的尺度控制，避免B的新梯度自动缩成1/6。训练任务分组仅计算损失，不送进Actor。相同8次Adam上限/全24密度KL门，clip.8/1.2、grad1；优化器/数据一致，只改变旧任务梯度是否加入。两组各shadow、拒绝整ActorAdam回退；oldweight0时B须逐位等于A作数值反证。

每组49原起点自主＋四关键full/exact，原T55039成功ID全保且新增>=1才研究保留，不能新增抵销失守。完整407/fresh16/reload78=null；不称完整六八米/全场地或泛化。若均无新成功/互损，停止此机制，不扫轮数/seed/权重；下一需资格有效成功/恢复数据而非无限优化失败批。

最多7,521,044全部物理/160命令（24train+24exact+98自主+8exact+4生命周期=158）。两组最多16live、16shadow和16audit历史丢弃Adam，预检12丢弃Adam/0native。0新的源教师标签或规划器。main3600/learning2100/audit600/backup300/source180，加缓冲4720秒；拟12:10停学、12:40原生、13:00全部，更早相对deadline从严，待用户确认后锁定。

own49MiB：32full×1280KiB＋90compact×12KiB＋3模型×256KiB＋metadata4MiB＋2worker暂存full×1280KiB=50,651,136B≤51,380,224B。global预留1,072,863,099B<1GiB，余量很小，不扩大/删唯一证据。准入前重算实际容量；free40GiB，CPU14/15合计200%低优先、GPU关、Phase完全隔离。driver唯一writer，科学退后audit、manifest、全部RAM远端逐成员SHA和精确source；无网保本地并标PENDING。

零原生预检通过、有效新窗口/预算授权记录、源码根树远端核验后才允许唯一launch。active不改科学源/协议，不并发audit/backup或外写runs，不因几个中心/随机探索成功晋升。
