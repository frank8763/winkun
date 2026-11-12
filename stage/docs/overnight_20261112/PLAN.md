# Stage 2026-11-12 晚离线无人值守实验

用户明确授权今晚后台实验，2026-11-13北京时间09:00全部停止。08:00停止学习，08:30停止原生仿真；相对本臂学习1200秒/main3600秒/audit600秒更早从严，不延长。Phase聊天/研究/模型/结果/进程设置完全隔离。CPU14/15合计quota1核、idle/nice19/CPU-IOweight1，GPU禁用、实际Mesa llvmpipe断言；不声称与其他任务绝对零竞争，不查看调整Phase资源。

## 当前真实结果

接受最佳仍是经T508完整冻结验收的同T507block0模型：旧八中心8/8、场地11/24（9接触4超时）。T509结束审计PASS、55原文件完整远端逐成员SHA：新两任务8/8探索安全GOAL，16live16shadow联合监督候选守门17/21，损失dev_task6_13、sector0_offset1、sector2_offset2，另新sector0_offset2仍超时；候选拒绝、ActorAdam精确回退。完整field24未执行null，不是0%成功率。所有T509进程退出，不重启旧目录或继续其失败预算。

今晚目标是定位drive/steer不同输出的学习干扰，并尝试保住已有19项的同时新增确定性成功。不是只加T509轮数，不是全24已经成功。原五米safeT255、联合旧271/275+六/八54历史最佳T501、当前11最佳均独立锁定，不覆盖失败/模型。

## 固定三分支与共同数据

唯一根T510，预先锁定三臂顺序：

1. drive_only：仅5953驱动残差参数学习，转向全部参数和对应Adam moments/step保持逐位不变。
2. steer_only：仅5953转向残差参数学习，驱动全部参数和Adam状态逐位不变。
3. joint_keep_priority：11906现有双残差参数，分别求repair/keep原始梯度，repair与keep内积负时仅将repair沿keep负向分量去除，再半repair半keep。明确是一侧保持优先PCGrad-inspired改写，非原PCGrad/CAGrad复现。

三臂都从同一个接受11模型+完整成熟Adam1025/385开始，lr1e-4，原骨干/head/Value/ValueAdam冻结。drive或steer冻结模块必须grad=None从而跳过Adam更新，不能零梯度后仍让动量移动；预检与每步均核验对应参数+Adam哈希。成熟Adam位移并不等于当前投影梯度，逐步记录rawdot与实际Adam位移dotkeep。初始keep为数值零，首步投影可能不活跃，不能声称从第一步就消除冲突。原论文/作者源码近邻与反证见PRIMARY_SOURCES.md，不称创新/无偏/全局安全证书。

只复用已完整验证的T509八条真实自身探索GOAL，源policy就是当前接受模型；它们现在是旧行为监督数据，不是新onpolicy。不新增teacher/planner/地图/路点/任务router/相机/容量或奖励/速度/时限/成功半径。部署仍单120因果输入真实64GRUcarry直接输出drive/steer。

repair按两任务等权、各4成功回合等权，执行float32动作沿既有裁剪边界可行集hinge，半全回合物理时间mean/半离最终goal<=1m物理时间mean。1m仅离线损失权重，不是waypoint/成功半径/部署控制。keep为当前接受模型实际old8+field11共19条成功unclipped locations，两个家族等权、回合等权，半物理mean/半upper20风险。总半repair半keep，sigma归一/双动作等权。完整当前真实carry特征缓存，逐数组/密度/source核验，不能每行hidden清零。三臂损失与步数匹配，仅固定可更新输出子集/投影机制不同；不是随机seed/噪声/学习率网格。

## 每臂唯一有限提案与验收

每臂只做一次固定32监督live+32独立shadow Adam（新PG0/Value0），0新训练采集；不重复拒绝提案、不从相同状态重试。之后实际21原起点自主无教师（old8+field11+2新）及21全数组/积分原生exact。原19成功ID必须全保、无无效/删失且repair代理下降。原19保住且>=1新任务qualifiedGOAL，才做余11场地+完整24exact；否则本臂停止，整ActorAdam回退，fullfield24=null而非失败率。最终old8安全、原field11全部ID保持、fieldqualified>11、接触<=9、0无效删失才接受研究候选。guard的13field结果同一policy复用，非又13次独立测量。

三臂全是同初始独立提案，不把拒绝/未验候选传下一臂，不暗中换seed或继承臂间预算。每臂main全部native父worker退出，独立零native历史Adam重构及full数组/history/密度/奖励/计账/回退auditPASS、本地完整SHA清单通过后，才允许启动下一固定臂。失败/数值合同不满足立刻停止整个链保raw/error，不自动修代码或无限替补。科学性失败（候选拒绝）不阻断下一个不同预注册机制。

若多臂通过，选择完整field成功数高者，再接触少者，再固定臂序；选择只使用已见开发集，不叫未见留出。原11最佳仍保留。未重新验old275/six54/eight54/fresh16/78，不能由old8与field改善声称全局或原379全部保持。

## 离线、本地容量与硬停止

本地driver是唯一执行器，所有科学代码/输入/依赖预落地，不调用LLM/API、不安装包、不依赖网络或定时聊天。模型/原始轨迹按SHA锁定。driver依本地协议确定接力，网络断开只使最终远端状态REMOTE_PENDING；科学/audit/下一臂不等待GitHub。全部科学完成后才尝试RAM分包远端读回逐成员SHA；网络失败保全部本地原始/模型/manifest/error，联网后统一补传。不把PENDING说已备份。

每臂最大77物理回合=21+21exact+11+24exact，81全部命令；三臂243+预检6零物理总249，最多96live96shadow和96独立丢弃历史重构，预检12丢弃历史clone另计。完整最大物理、rawshape在BUDGET.json实算，根main最多10800秒。各臂只一次proposal，已完成机制绝不重复耗时来填满钟点；若有限三臂提前结束，就完整归档报告并停止科学工作，不承诺09:00之前每秒都在训练。

global runs逻辑上限仍1GiB、free>=40GiB。根own52MiB；每臂必须完整最坏预留28MiB（32full shape+4MiB模型/metadata/partial上界）；按实际前臂占用+下一臂完整最坏核算，同时满足根52/global1GiB才准入，不保证所有臂一定有容量。预算/时限不足的臂明确SKIPPED，缺测不是失败，不删除唯一资料或扩大额度以强行启动。没有额外种子/轮数/噪声网格。

预检每臂两clone各2步共4丢弃Adam，三臂共12、0native/Value，检查实际8成功+19keep/边界/真实history/density、固定模块+成熟Adam冻结/原始投影与实际位移有限、克隆逐位/全回退/序列化、两真实worker取消；本地网络依赖静态检查也通过才启动。source提交并GitHub完整根树精确核验之后唯一systemd service，外层RuntimeMaxSec与control-group KillMode保证09:00上限，inhibitor防睡眠但不能防意外断电，不将断电后状态假装在跑。

科学完成driver唯一写FINAL_REPORT.json/md、NIGHT_STATUS、NIGHT_LEDGER，原始失败/候选全保，网络可用再全RAM远端备份/source。系统/协议失败写NIGHT_ERROR并停止，不启动已结束目录。没有新的无限“接力”或临时未注册机制。用户明早查看报告即可判断真实进步/缺测/预算跳过；不保证这三臂一定能解决全部24项。
