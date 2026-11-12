# 11月12日通宵续段 T511：冻结父策略与新神经残差

用户在T510结束后明确要求继续通宵并推进全场地24，沿用确认的11月13日09:00全部停止；08:00学停、08:30原生停，相对学习10800/main14400/audit1200及容量准入更早从严，不能延长。Stage与Phase聊天/研究/结果/模型/进程完全隔离。CPU14/15合计1核quota idle/nice19/CPU-IOweight1，GPU禁用/实际Mesa llvmpipe。离线本地driver不依赖网络/LLM/安装，网络只用于全部科学结束后的归档与源码，断网REMOTE_PENDING如实保raw。

T510实际三臂全部结束，审计/114原文件完整远端逐SHA通过：drive-only旧7/8且guard17/21，steer-only旧7/8且guard17/21，joint优先投影旧8/8但guard18/21、sector2_offset2接触；各32live32shadow、保留0。合计2805558物理/144命令/1683.007主管线秒。首两丢旧能力、第三保旧8但损一field，不以代理下降代成功；完整field24均未执行null。T508接受最佳仍旧8/8+field11/24，原safe5/T501/T505等独立保留。

新机制不是重复旧输出分配/梯度投影：冻结接受Actor全部参数（包含此前drive/steer残差），添加184->64tanh->2零输出末层初始化的新全局因果神经残差头，仅11970新参数训练。最终location=clamp(parent_location+0.1*tanh(new_head([真实hidden64,normalized120])),-1.25,1.25)。父策略原成熟Adam1025/385保存在原protected checkpoint不可变；新头独立freshAdam0/lr3e-4，不称继承父moments或从零端到端RL。Value/ValueAdam冻结0步，骨干全部冻结，初始27完整历史函数必须逐位等于父策略，新增architecturehash与语义函数相同分开注明。

部署仍单120因果观测/64真实GRUcarry Actor直接drive/steer；新头每时刻全局使用，不读任务ID/全图/路点/未来、不运行teacher/router。0.1仅raw location加性上限，不是全局稳定/碰撞安全证书。原成功半径0.2m、速度上限/几何/时限/reward/sigma不变。父权重冻结不等于组合输出或在线hidden轨迹不变，因此原生保护守门仍必需。

数据固定T509两任务8真实自身GOAL执行动作+当前接受19真实history保持，完整source/NPZ/全部数组/密度与积分核验。repair两任务/回合均衡，裁剪动作沿boundary feasible hinge，半完整物理mean/半最终goal<=1m物理mean；keep old8/field11家族等权，原unclippedlocations，半物理mean/半top20状态风险，总半repair半keep，双动作sigma归一。1m是离线权重不是waypoint/部署门槛。0新训练采集/0新PG，明确监督整合已有成功；不把旧行为数据叫新onpolicy。

最多6固定块各32live+32独立shadow新头Adam，独立audit最多192丢弃历史重构；预检4另计。每块导出同一组合Actor并做old8+field11+2新共21原起点自主无教师、21全数组/积分原生exact。保护最初19IDs，新成功加入；所有保护ID原生安全、无无效/删失且repair代理确实降低才临时保留，否则整块Actor+新头Adam回退并首次拒绝止学，不从相同状态重做。21全过提前止学。有新任务确定性成功的保留候选才余11field+完整24exact，guard13field同policy结果复用非又13独立样本。最终old8=8、原field11全部ID保持、qualifiedfield>11、contact<=9、0无效删失才接受研究候选；否则整实验回退到零新头初始，原接受11最佳独立不受影响。旧275/six54/eight54/fresh16/78不做null，不把19保持当旧379完整复验。

空间严守global1GiB/free40GiB，先对已远端完整备份的7个discarded预检clone scratch做cache清理，释放7447083B，哈希/远端manifest/收据在PREFLIGHT_CACHE_CLEANUP.json；原实际保留/拒绝科研候选和原始轨迹均未删，T507断电只读目录不动。原字节仍可由Release恢复，不能把已清理旧pref裸路径当仍存在。本新机制输入不使用这些scratch。

本轮own40MiB，每块开始前必须在当前实际占用之外预留完整下一guard21+最后11的fullshape+4MiB模型/metadata/partial（见CONTINUATION_BUDGET），不足则明确ADMISSION_STOP，不扩大额度、不删唯一数据或模型，不保证能执行全部六块。原生最多12*guard步上限+余11+field24；原命令6*42+11+24+4=291，预检6共297。与T510实际合计最大物理13986278<16M、命令441<600、主管线16083.007秒<21600，整夜合并上限明确。本新机制独立有界，非新编号续T510同方法预算。

每块前还需完整1800秒下一检查+收尾原生余时；上轮所有科学进程实际已退才能启动本轮。formal预检0native、2clone各2步4丢弃新头Adam：27历史零函数逐位、新头可训/父权重及原Adam不可变、密度/边界/有限梯度、新头Adam/完整回退与保存读回、真实新Actorworker加载取消、ID负例/容量准入。通过/代码提交/GitHub精确根树才唯一systemd后台。driver是唯一writer，运行中不外写runs/不改源协议/并发audit/backup/source。所有科学退出后独立历史全新头Adam重构/全原始数组/历史/奖励/密度/物理计账/接受回退审计，本地全SHAmanifest，RAM全原始远端逐成员SHA，写T511_FINAL_REPORT、RESEARCH_STATUS与源码同步。

文献近邻/反证见CONTINUATION_PRIMARY_SOURCES，Frozen-base residual/PolicyDistillation等不是原创；容量与优化同时变的工程pilot，不以父冻结、初始一致或loss下降冒称全局安全/泛化/因果创新。不保证新机制成功或整夜每秒训练。预注册有限机制/硬预算结束就保留报告停止，不重复失败提案耗钟；若未达24，记录真实差距供下一新机制而非宣布全场地跑通。
