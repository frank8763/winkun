# T480：旧任务真实经验强化学习回放，有界可行性试验

2026-11-11，用户明确要求“检查；然后下一个方向继续实验”。Stage限定，与Phase完全隔离。此文件不是启动记录。先等T479完整结束、独立审计/原始完整远端逐成员SHA、全部父worker/driver/timeout/inhibitor退出；若T479有合格保留的新模型，暂停本失败后设计，不拿拒绝候选初始化。

## 针对的问题与取舍

当前T477保留模型八米8/54、六米53/54、旧274/275，共335成功；八米成功左5右3。T478及T479 A已表明八米局部增加可伴随旧任务损伤，但未证明单一根因。T479 B仍在验收时，本方案只能准备，不能先判失败。

下一干预是让旧任务的真实动作和回报也参与学习，而不只约束旧历史上的均值。借鉴[CLEAR原文](https://papers.neurips.cc/paper_files/paper/2019/file/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Paper.pdf)与[V-trace/IMPALA原文](https://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf)，不是完整CLEAR复现或新算法原创证明。新鲜PPO保留；旧任务采用截断重要性比的离策略辅助，部署仍同120输入/GRU真实carry单Actor，无运行地图/规划器/路点/教师/路由。

为在global1GiB和完整383验收资源内推进，这次只注册一个候选，**不是匹配A/B消融**。与T479差别包括新批次、旧实际历史和旧RL辅助，不能把结果单独归因V-trace；正式论文消融需后续独立预算。

## 唯一预定流程

1. 复用T479已原生验证、精确重放且独立审计通过的8条八米成功完整历史及SHA；不用新物理复现已相同policy/hash数据冒充新独立经验。
2. 从当前保留模型原起点采集旧8类各固定`dev_task{0..7}_13`一个独立iid噪声回合，seed `[20261111480,0,i]`。成功、接触、真实时间上限都原样保留、纳入合法旧任务RL，禁止成功筛选造成行为分布误报。每条完整数组/hidden/history/位置/logp/边界原子/积分保存，并各做一次新原生精确重生。
3. 采集八米左13、左01、右13、右23各一个新鲜PPO回合，seed `[20261111480,1,i]`，完整保存并4次精确重生。总12采集+12exact；所有重复物理和命令计账。
4. 采集后所有native进程退出，独立数据审计0优化器：完整history、密度、奖励、积分/全部数组SHA；旧行为密度必须与真实初始Actor逐行精确。重要性比为1时V-trace目标须与同物理折扣的完整MC相符，误差≤1e-12。失败则保留原始并停链，0live学习。
5. 从同完整Actor/Value/成熟双Adam出发，最多16个epoch的单候选真实RL；新鲜八米PPO按侧池化完整回合优势/lambda1，原clip0.2/targetKL不变。旧8任务完整序列每任务等权1/8，V-trace辅助Actor系数1、Value系数1固定；保持项系数1。旧211轮转历史与新旧8实际历史在旧保持中各1/2，旧总保持与自身八米保持各1/2；自身八米左/右等权。无SIL、无新Critic架构、无CAGrad、无Adam重置或参数网格。
6. 相同数据、顺序、初值，独立shadow重算每次Actor/Value/双Adam，hash逐位一致。KL超过原上限则停止本候选后续epoch，不加步数。
7. 一个固定末模型完整383原生验收：八米54、六米54、旧275。须保住原335成功ID并严格增加八米成功才保留，否则整Actor/Value/两个Adam回退；中断同样整状态回退，不追加候选或扫seed。所有提案模型和失败结果保留。
8. 全部原生退后，独立0native历史更新重构/完整数据SHA/物理命令核账；全原始RAM分段上传并完整远端逐成员SHA读回，Git仅源/协议/汇总/清单/收据；最后精确根树同步。

## 概率、回报与V-trace合同

行为是已记录的真实iid bounded Gaussian，σ=[.2,.2]，边界原子与内部密度沿现有执行接口，完整GRU从h0重算，不使用旧hidden替换当前policy状态，不拿确定性验收轨迹编造高斯行为密度。

`rho_t = exp(logpi_current-logpi_behavior)`，`rho_bar=c_bar=rho_pg_bar=1`。`delta=clipped_rho*(r+discount*V_next-V)`，逆序构造`vs=V+sum discounted trace deltas`；Actor权重`clipped_rho*(r+discount*vs_next-V)`停止梯度，Value用停止梯度的vs目标。所有折扣按`gamma_native**native_counts`，真实有限任务终止含原时间cap的终端bootstrap为0，行政删失不准入。不是无限时域超时伪终止，也不替换原奖励。

已直接阅读IMPALA原文§4.1–4.2的目标、递推及截断固定点：有限rho上限会改变对应评估策略，不能把此辅助项称完全无偏的当前策略Value/策略梯度；tabular收敛也不直接覆盖本回合非线性GRU＋成熟Adam。这里只借用其有界离策略目标合同，不宣称完整IMPALA分布式系统。

同策略比率1时必须回到有限MC；非单位比例用独立枚举递推核验。`check_contract.py`已CPU通过：0native/0真实Adam、1合成backward，验证该恒等式、截断递推、目标detach和终端非零bootstrap拒绝。正式真实history/成熟Adam/shadow预检还须在前轮结束和容量审计后完成，不能用合成数学检查替代。

## 预算、容量与截止

新独立窗口，不扩大或重启T479：12M全部物理/500全部命令，实际最坏命令421=采集24+两阶段启停8+验收383+预检6。最多live16Actor+16Value、shadow对应32、独立丢弃历史重构32，预检最多8丢弃Adam单列。

capture主管线900秒/audit300；单候选主管线3600秒（学习至多1200）/audit900；末备份600/源码180。外层6900秒（115分钟），且全部最晚北京时间今天16:30、学习15:00、原生16:00，更早相对截止从严。完整最坏余时不够不启动，等待前轮最晚14:00结束；不为耗满时间无限重跑。

root20MiB，完整12文件每个≤800KiB、383compact≤8KiB、模型≤1792KiB、metadata5MiB，合计20045824B≤20971520B。满53334原生步对应最多534决策，现有packed26数组保留完整120及history/hidden，逐NPY数据和头部＋保守DEFLATE/ZIP上界757576B<819200B；正式预检/每条写入仍执行硬限，不靠平均压缩率。root协议和缓存计划分开SHA锁定，普通Git各文件<2MiB。

global newruns≤1GiB/free≥40GiB不放宽。先对已结束完整远端SHA备份日志做本地可逆gzip（FAILED子目录只有整个root SEALED、全部进程退出且完整原始远端核验后才可选择；FAILED摘要和错误仍保持原件）（原字节本地和远端保留），再视需要按已验证受控helper重新读回完整旧归档每成员SHA后移除未用NPZ缓存，保留恢复标记。当前输入、模型、summary/audit/protocol/error/ledger/canonical均排除。原T479未结束时不做任何清理；不足不删除唯一数据或扩大容量。存储两阶段均独立审计，完整root预留只在两阶段结束后准入。

## 诚实结论边界

fresh16/78独立模型原生重载本轮未预留，结果必须null；全383通过也不等于未见八米稳定、全图或实体机通过。旧8随机回放只覆盖固定中心各一条，不等于全部335历史；即便失败，不证明V-trace/CLEAR全家族无效。当前经验门槛是有限开发过滤，不是安全证书。研究依据及CPO/成熟Adam限制见research_20261111_wide8/POST_T479_UPDATE_CONSTRAINT_SOURCES.md。
