# T536 精简原生状态恢复：可执行方案，等待有效运行窗口

用户已确认新增73,000物理步、1MiB、18:30全部停止，但本次实际恢复执行时已18:55，超过该截止。此文件不延长授权，不代表protocol/launch或仿真已经运行。当前只完成代码、静态预算与零native工程检查。需确认新截止后记录T536_WINDOW_AUTHORIZATION，再正式注册/预检/源码同步/唯一launch。

## 假设与判据

T534因完整capture包含历史记录缓冲导致新目标单snapshot超过384KiB而止；旧两个窗口原生restore本已逐位一致。修复只移除rows/hist/hb/ha/proposal/owner/digests和Actor tape历史缓冲，保留完整mjSTATE_INTEGRATION、Python环境属性(真实history/随机数状态/执行器/clock/prevaction)、witness历史、steps/pn/previous、64hidden、previous_raw/location。不是修改控制策略或车体模型。

新快照压缩后32KiB硬上限，每个恢复前后必须字段比较确认全部非日志状态一致；故意损坏hidden/RNG的负例须被比较器拒绝。序列化检查不能替代真实native复验。

同T534四固定窗口：old_success/old_failure从4.8秒(24决策)；new_parent_failure/new_candidate_success从64.2秒(321决策)。每窗对应冻结T507/T527源Actor和原任务从原起点真实重生prefix，逐条完整120输入/输出/状态/pose/history/64hidden与源原件精确匹配。中途真实生成完整state而非由pose构造。存盘精简snapshot后，先同Actor连续5决策(1秒)，再从磁盘反序列化恢复并继续5决策。恢复后日志从空开始，全回合clock/history/steps/hidden不能归零，source匹配全回合索引。每个生成数组及末integration都逐位一致才过。

四窗全部通过仅证明这些固定条件下接口可恢复。任何窗口出现不一致/容量/截止问题停止，不额外补轮/换窗口/放宽数值，不自动参数方向或策略学习。0Actor/Value/PG，无导航增益或算法原创性声明。

## 预算与收尾

新物理(24+10)*100*2+(321+10)*100*2=73,000，9命令=1setup+4prefix episode+4restore，所有物理步均计。T534已用38,900步不抹除，若完整执行累计111,900步。main480秒、独立零nativeaudit180秒、全原始RAM逐成员SHA备份240秒、源码120秒及60秒余量，完整最坏1080秒准入。原生deadline=新确认总截止前900秒，相对main更早从严。

own1MiB：四snapshot≤128KiB、四双后缀≤256KiB、协议/状态/源哈希/错误/日志余640KiB，逐秒容量守卫。global stage/runs逻辑≤1GiB/free≥40GiB。CPU14/15合计1核/CPUWeight1/IOWeight1/Nice19/SCHED_IDLE，GPU关闭，native renderer实际Mesa llvmpipe断言。与Phase聊天/模型/研究/进程完全隔离。

独立audit在全部native退出后重新读取源数据、snapshot、两完整后缀，核对所有SHA/物理/命令/hidden/history/源匹配；它不新增native，不冒称第二次独立物理验收。唯一driver负责audit/全部原件远端二进制逐SHA/源码精确根树。结束不删除原失败证据或模型，Git不存大二进制。失败需如实partial收尾，不重启已结束目录。

实际准入还须main最坏480秒完整落在原生截止前，且原生截止保留900秒收尾，因此总余时需大于1380秒；1080秒总预算不是绕过更严原生截止的许可。
