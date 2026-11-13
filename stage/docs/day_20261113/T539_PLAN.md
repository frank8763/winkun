# T539 整回合成功目标的一次策略梯度

用户要求下一步，T538已全结束/原始备份；不把1秒梯度当导航目标。当前窗口20:00全部停/19:45原生停，不扩大。Stage only，Phase完全隔离，单120因果Actor真实64carry，原MuJoCo动力学/场景/goal/时间阈值/噪声sigma不改，无地图规划路点或运行教师。

## 明确的新学习假设

T519等使用物理折扣塑形reward-to-go/同行timebaseline、16步PPO、旧history均值保持与成熟Adam。本项只做一次完整轨迹success-probability REINFORCE：每个episode的Z=整回合安全GOAL(1/0)，deterministic原Actor在相同任务实际超时，baseline=0。梯度损失为 `-mean_i((Z_i-b)*sum_t log pi_theta(a_it|full120history))`，不是每回合按长度取平均，不把近目标进度当成功，不使用同行RTG/PPO比率/Value/旧historyMSE。完整源数据失败轨迹仍纳入(其Z-b=0)，不把teacher动作充on-policy。

使用已有T519固定4条轨迹，行为policy hash精确当前初始化T507，3safeGOAL+1timeout；逐步全部density/真实64hidden/完整120与原数组哈希先核验。它们被其他实验用过，明确是历史样本复用与配对研究，不是新采样、独立重复或多seed结果。本任务为sector1_offset1(约5.20m新位置)，不是把所有全场地任务统称原窄六米。T527另一新目标0off2并非本批训练目标。

全原Actor64772参数可训练，一次SGD lr=.001/momentum0、全梯度normclip1，独立shadow一次，独立audit再重构一次。原T507的Adam与Value不改，保留在原完整checkpoint，通过显式SHA引用不复制；新candidate保存完整Actor及SGD完整空momentum state、原不可变Value/Adam引用。这样保留全部研究字节且不复制无变化optimizer。SGD相对成熟Adam、目标/估计器/步数均变化，是联合pilot，不能单因素归因或宣称原创新算法。

参考[Self-critical Sequence Training，CVPR2017](https://arxiv.org/abs/1612.00563)：用部署时确定性推理结果作为REINFORCE基线。原论文是图像描述，并未证明本车能学会；只借估计器，不照搬性能或称原创。CVF页面本次403，作者arXiv摘要可核验。

## 原生验收与证据

更新后，在4训练history与8旧成功history上的原始Gaussian均值KL上界mean≤.005/pointmax≤.05；clip动作KL由data processing不大于相应rawNormal KL，但固定history上的约束不是闭环安全保证。门槛不过就保留拒绝候选并回退，不改步长重试。

通过后，同一candidate从原起点完整old8+field24，并全部32exact逐数组/积分重生。接受必须old8=8/8且原field11逐ID不丢、field>11、contact≤9、0invalid/censor。否则整Actor/SGD回退原T507；失败候选保留，不再补步/换seed。old275/原six54/eight54/fresh16/reload78均未执行null，不拿本面板替代稳定全场地。

验证生成两遍完整数组并在RAM比较全部hash/积分；仅actions/count/flags紧凑文件落盘，独立offlineaudit不能重构未保存完整eval120，要如实写明。4训练完整120已经历史保存，独立audit可完全重算密度/成功梯度/新ActorSGD/回退。原先T519 exact不是本candidate的新物理重复，全部新eval都计本预算。

## 有界资源

只一次真实PG+一次shadow；预检2次丢弃syntheticSGD，audit1次丢弃历史SGD。0新训练采样，最多64原生eval/68主管线命令+5零native预检命令；所有物理最坏2*sum32caps锁protocol。新main1000秒(学习最多120秒)、audit240/backup180/source120；完整截止从严，余时不足不注册/不launch。19:30停学习/19:45停原生/20:00全停。

own1.375MiB，candidate≤640KiB，32compact各≤12KiB、metadata/日志与全部root硬额度保护；global1GiB/free40GiB，不删除唯一资料。CPU14/15合计1核，GPUoff/实际llvmpipe，Nice19/SCHED_IDLE/CPUWeight1/IOWeight1。注册源精确远端和零native预检后唯一launch；唯一driver收尾audit/全部RAM原始远端逐成员SHA/source。没有自动续训或新作业网格。
