# T382：成功动作先验与滚动原生训练教师

2026-11-05。用户明确要求下一步开始。新机制独立有界注册，不恢复T381目录、不扩其预算。当前文件不表示已启动；以实际launch/status为准。Stage与Phase完全隔离，safeT255、人控、所有失败原件不改。

## 问题、依据与限制

T381精确原生预测中33条全部接触、所有执行转向都为正，父8秒mode为[.800740,+1]；加权序列实际16.742秒接触。共同CONTACT罚未漏加，但没有可行候选给加权平均。因此本轮检验成功**动作序列**作为搜索先验，加上短周期重规划与可行性筛选，能否从同一个真实偏离状态给出可执行恢复。是三项联合工程pilot，不能单独归因或称算法原创。

2026-11-05重新查阅Williams等[作者PDF](https://nolanwagener.github.io/media/mppi/paper.pdf)有关滚动执行/移位初始化的机制，沿用固定作者AutoRally代码/GPS对照与反例说明，详见`T381_TEACHER_LITERATURE.md`。MPPI只执行首控制并移位，不等于一次12秒开环执行；短视野也可能陷入局部选择，均值匹配不保证非线性多峰问题中的安全。GPS具备训练期完整状态特权与师生分布匹配，本轮未实现其iLQG/BADMM，不声称复现GPS。

传感器与部署不变：120维因果局部输入、实际四帧history、真实GRU hidden carry的冻结T379 Actor；教师仅训练期使用MuJoCo完整state、plant和碰撞查询。不是整个训练无特权，也不是无地图Actor新成绩。人控同时时钟状态比learner差12.795厘米/8.188度，不将同时间人控动作直接当偏离状态专家标签；每个候选都要从当前实际native状态预测。人控动作不是人工中间路点，不增加地图/全局图搜索或教师运行时部署。

## 固定科学合同

同新0.70米/80秒中心、原起点、wheel5/force120/禁止倒车、所有轮胎和碰撞mask、.002秒native步、每hold100步=.2秒、.20米GOAL。不改原0.57米/56秒验收场景。

两次pass，各先完整从原起点重现T379历史失败，再精确重建40hold=8秒真实前缀。与T381相同，保存/恢复完整203slot mjSTATE_INTEGRATION、Python状态、四帧Witness、实际previous动作/计数与Actor hidden；以5hold原历史原生动作参考验证，再恢复。历史与第二pass都属于重复，不是新独立导航样本。

最多60次replan；每次预测20hold=4秒，一次只执行首1hold=.2秒。因此教师实际窗口最长8–20秒；完成60次后交回冻结Actor，继续到原80秒真终止。没有可行提案或加权验证拒绝时立即停此教师分支，不交给已知失败Actor冒充教师完成，也不增加重规划次数。

首nominal为固定成功T361实际actions[40:60]。每次真实接受一hold后，nominal取上一加权序列未执行19hold，再用T361 action[40+k+20]补尾；k为从0起的replan索引。所有索引最多到119，T361共295hold，因此不依赖未知末尾padding或到达后控制。始终保持实际clock/速度/转角/history，不移植人控pose或hidden。

每次固定5候选：nominal、两个固定Gaussian序列的±反向配对，共1+2×2。seed20261105382，replan使用SeedSequence([seed,k])；sigma[.20,.40]，每5hold一个噪声块，实际命令裁剪[-1,1]。不增加seed/候选/噪声/时域/迭代或控制器网格。减少候选数是完整预算选择，抗向配对也是本pilot设计差异，不能把联合结果归因于单因素。预检检查固定首批实际动作方向覆盖，但不重选seed。

所有预测经native.step_checked，不调用stock pose覆盖。每条到CONTACT/GOAL/真cap即停，否则4秒horizon。cost沿用1000*CONTACT+10*终端真实目标距离+.01*实际执行相邻动作变化平方和，温度1。只有CONTACT=0且完整horizon或真GOAL的候选进入加权。全拒绝则记录NO_FEASIBLE_TEACHER并停，不从碰撞样本选best。

对可行候选按softmax负cost加权完整命令序列；从完全相同snapshot额外native验证整个4秒加权提案。若其接触或非GOAL提前结束，记录WEIGHTED_NATIVE_REJECTED并停，不看结果后改选其他候选。验证通过后**再恢复actual snapshot**，只真实执行首hold，并逐位核对该hold的观测、原生积分与预测；不能把预测终态移植成实际进展。然后从真实新状态重规划。所有预测/验证/实际执行分别计费。

教师实际动作期间Actor仍在真实当前120输入上推进hidden；actual previous动作进入下一packet。保存ownership及单独命名的Actor proposal/logp，不把教师控制称为人工/on-policy数据。即使4秒可行也不是全程安全证书；只有实际完整混合轨迹无接触GOAL才是可进一步使用的教师数据。

## 证据合同

首pass完整混合actual轨迹保存full120/next、动作、counts、pose、flags、reward、history、hidden、ownership、Actor proposal与初末积分。预测紧凑原数据保存全部控制、每hold原pose/count/flags、完整分支trace与末积分SHA、length/cost/remaining、可行权重和每次actual snapshotSHA；预测完整120/history/积分由每hold完整record聚合SHA约束，不声称保存了预测每一步所有120原数值。

第二pass重新生成固定所有候选和实际控制，逐hold紧凑原数值/分支完整trace与nativeSHA精确、actual每hold完整record及最终全部数组积分精确。首差异立即停止并保存packet/history/积分/索引和hash，匹配前缀引用首pass，不生成无法保存的新后缀。工程/行政中断保留当前分支最多20hold完整数组与当前native积分/packet，标明partial，不写成任务CONTACT或成功率。

NO_FEASIBLE/WEIGHTED_REJECT属于明确诊断停止；实际状态可能仍未终止，不把未执行剩余时间算任务失败。actual CONTACT、GOAL、真timeout、预算删失、无效分别报告。指定wall–circle缺口穿越仍须另验，不将绕行称穿口。

## 完整资源与执行顺序

每pass完整最坏424命令：baseline1、prefix1、参考1、60×(5预测+1加权验证+1真实hold)、Actor续行1。两pass加1零物理geometry=849，硬上限860。每pass最坏800500物理：baseline40000+prefix4000+参考500+候选600000+验证120000+真实教师6000+Actor续行30000；两pass1601000，硬上限1610000。重规划被拒绝可提前结束，但不追加预算。

main2400秒，独立audit300秒（0新native/Actor/Adam）、RAM完整远端备份300秒、启动调度余60秒，全部3060秒。外层timeout及绝对deadline更早从严，不能续期。唯一driver只在所有native父worker链退出后串行执行audit与backup；driver健康时不再手工重复这两项。driver失败则保存error和raw，不重启训练或静默宣称完成。远端收据与精确源码树最终由汇总闭环核验。

own944KiB=966656字节/global1GiB/free40GiB。预留full或partial512KiB、predictions320KiB、prefix4KiB、metadata48KiB、log16KiB、首差异/中断32KiB，合932KiB留12KiB；注册前实算原global1072753805+966656=1073720461，低于1GiB的余量21363字节。预检以不可压缩内容模拟最大400hold full和60×6×20预测编码尺寸，0native、1geometry、0优化器。任何一项不能完整最坏保存则不启动，不扩global、不删唯一材料。

源码/协议/输入SHA（含T361、父T379、T381代码与结果）锁定、零物理合同PASS、代码提交、GitHub精确根树同步后唯一launch。运行中不外改协议/源码或同步其他源。结束独立audit重算噪声、nominal移位、可行性、加权、actual ownership/hidden/history/reward、所有核账和SHA；完整原始RAM远端逐成员SHA读回才称备份完成。Git不写大二进制。

## 学生学习与晋升

本轮0Actor/优化器/新RL。教师真安全GOAL且audit/backup全过后，才能另立纠偏学习/示范增强RL协议；不能直接用本轮失败控制当成功标签。教师成功也不是Actor自主六米通过。原old275/six54/fresh16/reload78本轮均未执行null，不进入7/8米，不提供冒充合格的新六米视频。研究创新仍是待证假设，标准MPPI、GPS与反向配对采样本身不称原创。
