# T656：明确复用合格恢复支线的单臂工程学习

2026-11-19。T655原成对采样结束，三个old任务original组1/8、0/8、3/8，recovery组2/8、1/8、2/8；80fresh80exact原生/密度/因果carry审计PASS、原始全远端SHA。原准入要求两支持每个old均有GOAL，original03为0因此正确0学习。LEARNING_NOT_RUN旧通用reason措辞“Both supports lack ...”不精确，实际是original03缺safe成功；原件不覆盖，此处明示纠正。

当前已有recovery三任务全部含成败的真实学生恢复信号，不再新增教师/探索/seed。申请把T655成对机制的未用16次RL中的8次用于合格支线单独学习；scope变更需确认，物理/命令/Actor/时间/磁盘累计硬上限均不增加。并不回写T655 qualified标记，也不伪称完成对照。

## 单臂输入与变量

从原T653 mean研究模型/freshAdam3e-5出发，精确复用T655 recovery旧3×8＋共享field8×4=56条/11任务；复用T501全384与mean8场地历史用于保持，数据SHA和完整carry/密度重新核验。不是56新采集，不充独立样本，所有前缀和exact物理已经计入T655。T655原起点组排除本轮学习，失败原始仍保留。

算法与已准备T655的recovery拟合函数同源逐SHA：11906adapter、固定encoder/GRU/head，终止safeGOAL clipped episodic PG、各任务等权、n8/n4独立peerLOO、fixed512 score尺度；真实prefix不计策略梯度但进入历史，原物理clock/remainingtime和GRU不重置。保持全384keep10和8fieldkeep10，每步全部历史缓存等价；8live8shadow8独立历史上限、无额外优化器预检/Critic/新teacher。KL每任务.01，拒绝整ActorAdam回退并停止，不调整lr/weight/轮数/噪声网格。

原文献和条件目标限制直接沿用../t655/PRIMARY_REVIEW，不增加新算法模块。它是单臂工程pilot，不再声称original-vs-recovery的因果优越性；Gaussian期望目标和条件suffix目标均不保证确定性原起点成功。

## 验收与停止

同36screen（24场地+6哨兵+6T651失效）/20full exact，原17screen成功全保且全部8mean场地成功ID全保才其余371补同候选407。全原384＋8mean成功并集（至少387）全保才保留，否则T501仍safe。缺测null，不把后缀恢复或总分进步充当保留。无独立布局/真机泛化声明。

15:30停学、16:10停原生、17:00全部停止不变。拟main2700/audit450/backup900/source240/余120，完整最坏4410秒，登记按绝对剩余完整准入；原生完整407需1700秒准入 reserve。T656仅最多12.1M物理433cmd/8Actor各类/128MiB。加T654+T655实际3707060物理188cmd1062.15375948main/0Actor，整组最坏≤15.80706M/621cmd3762.154main、Actor8<原16；仍≤30M1100cmd6300main/原64Actor组限/656MiB总新记录。科学结束前不备份/改源/并发别的run，0重复旧仿真，所有新验收物理仍计账。

CPU8/10/12/14物理核/400%低优先＋GPU仅学习/历史，4GiB无swap、Phase隔离、global10GiB/free40GiB、科学离线；所有raw保留，停止后唯一driver独立审计/RAM全原始逐成员SHA/报告与Git精确源。网络错pending不删本地。新代码0native/0optimizer真实56+384预检和精确远端源后，scope确认才唯一登记启动。
