# T520：修复T519唯一旧中心退化，保住12个场地成功

用户11月13日明确下一步开始。Stage only，Phase聊天/研究/模型/结果/进程完全隔离。T519完整结束，3/4探索成功，候选field12/24=50%且原11个全保持，新增sector0_offset2；old8=7/8，唯一dev_task6_13无碰撞超时终点.3157766m。16PG/16shadow/16独立历史Adam审计PASS，全部60原件完整远端逐成员SHA/source根树通过。此候选未通过当轮old8全保标准、未晋升；原T507/T508接受best11/24字节不变。

本轮研究初始化明确使用T519未晋升候选proposal.pt.gz与完整Actor/两组Adam/Value/ValueAdam，只读旧文件不复制初始模型。组一drive1041/steer401，组二encoder/core/head16步，实际逐参数检查；两组学习率3e-5，不重置moments/增第三组/只载权重。所有64772现有Actor参数仍可训，独立Value无参数共享且冻结、Adam不变/不参与估值。部署单120因果输入64真实GRUcarry输出动作，无地图、规划器、路点、运行时教师或任务路由。

固定只训练dev_task6_13原起点4条新iid原sigma[.2,.2]完整自己闭环（seed20261113520/i）+4真实全数组/积分exact，原cap23334/46.668s、原reward/几何/速度上限/.2m GOAL不改。至少1原生qualifiedGOAL才最多16次fullActor clipped策略梯度+shadow，否则0更新停止。物理时间三同行RTG基线、组内RMS、ratio[.8,1.2]、完整当前history从h0重算分子和BPTT；不将旧数据或教师动作冒充本轮onpolicy。行为logp/raw/float32边界atom固定不替换，初始mean≤1e-10/logp≤1e-9合同。step后经验KL>.01止学，不承诺首步硬KL/单调收益或普遍降方差。

辅助soft参考仍用原19个实际历史输入，但目标在新研究初始T519函数上刷新；这些输入的原始成功属于T507/T508，不声称T519在其全部实际状态上自主成功，更不是恢复专家。相同输入上保持当前函数是软正则，包含退化任务时也不能称提供成功修复动作。新增field成功sector0_offset2没有存完整120，因此不伪造其soft标签，直接由最终原生逐ID保留验收保护。无.002动作RMSE硬门槛、不新增示范拟合/残差/投影或Critic。

同一更新候选完整自主old8+field24共32原起点，无论中间已有失败也继续全部32（除硬预算/异常），再32完整原生数组SHA/积分exact。接受门槛：old8恢复8/8安全GOAL，T519全部12个field成功IDs保持，qualifiedfield≥12、field contact≤9、0无效删失。满足才形成“old8+field12”研究进步，不要求再次新增第13项；否则整Actor/所有Adam精确回退研究初始，原独立接受best11不动。fullold275/six54/eight54/fresh16/reload78未执行null，不能把20通过称全旧379/全场地稳定。

存储证据沿T519公开合同：训练4完整120/history/hidden/density/reward动作全部保存并离线逐项重构；验收全数组在两次真实原生中生成，完整SHA/积分逐项精确，磁盘仅actions/nativecounts/endflags及全hash/结果；离线audit不声称重建紧凑验收缺失full120。32replays是重复物理，非新增独立样本。输出source/模型/Adam/failure都保留，原候选不覆盖。

预算4+4train与32+32eval，76main+6pref=82全部命令，全部物理8*23334+2*全部32原cap之和，以正式BUDGET精确数为准。main1800s/学600/audit600/backup1200，prepare学1200/native2700/all5400硬停。own6MiB/global1GiB/free40；完整训练shape+32compact上界+候选模型2MiB+metadata/partial1.5MiB先核算，不复制初始模型，不能默扩或删唯一原件。CPU14/15合计1核quota/idle/nice19/GPU禁用Mesa，禁查改Phase进程。运行中唯一writer，不外写其他runs或改科学协议/源，不并发audit/backup/source。

预检0native最多4合成丢弃Adam，精确完整初始Adam/当前函数softtarget/全Actor梯度、Value不变、live-shadow/拒绝中断整状态回退/保存读回、真实worker取消通过后，源码本地提交/远端根树精确才唯一launch。预检序列化scratch无导航数据、可由固定fixture精确重构，记SHA/recipe后移除。全部native父worker退出后唯一独立实际历史Adam/全训练证据/32行为结果和exact验收、RAM全部原始远端二进制逐成员SHA、报告及源码同步。失败不换seed/阈值/网格或复活旧目录。
