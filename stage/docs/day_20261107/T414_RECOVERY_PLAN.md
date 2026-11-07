# T412工程缺项与T414残余预算恢复

2026-11-07，Stage与Phase完全隔离，窄场景保持暂停。

T412实际完成原八项8/8、宽LEFT五起点0/5接触、RIGHT中心1/1；18命令253273物理步87.620323秒，0live/0shadow/0轮学习。随后groups读取right_history_file时FileNotFoundError。该文件是T405 episodes/1.npz完整277×120成功轨迹，不是原compact episodes/3.npz。父协议继承消费路径而注册器只继承部分inputs，导致T413清理把未登记依赖移至已验证远端冷缓存。精确复现实际groups红灯；按原SHA bc43aaa9…恢复194581B后，同一个groups及八项support_losses前向绿灯。另一个只读检查曾把约1e-33旧KL舍入误差错要求严格0，原拒绝日志保留；修正仅测试断言，不改任何训练容差或目标。

原T412不重启、不覆盖FAILED。唯一原审计为FAILED_PRODUCER_PARTIAL_PRESERVED；独立partial_audit=PARTIAL_EVIDENCE_PASS仅表示14cases/18cmd原始证据账目与8full实际动作/hidden/reward/integration正确。57原件完整RAM归档、远端二进制逐成员SHA核验通过，原目录固定1711750B。

T414是相同科学配置的工程恢复：保留seed、两query起点对、每轮400BC/8PPO、sigma、学习率、奖励、参数、目标及门槛；Actor/学习/专家/原生worker等21个科学定义AST逐项完全一致。只补全所有实际消费历史／快照与旧模拟器间接依赖SHA，复制入口改新目录及共享预算；初始14条件会物理重复，全部policy/array/integration/result与T412逐位比较，不充新独立数据。原safeT255与LEFT专家和失败结果不修改。

恢复预算沿用原上限，不能新编号续失败预算：T412+T414合计20MiB，16M物理/900命令。扣原1711750B、253273物理和18命令、原预检8及新预检8后，剩19259770B、15746727物理、866主管线命令。已锁metadata预留从1.5MiB压到1MiB，科学槽数量不变，输出最坏19068928B，合计20780678B<20971520B。原8合成Adam结果复用，本恢复预检0Adam0物理，只实际4worker READY+4物理前取消。所有全局newruns仍≤1GiB/free≥40GiB（不沿用T412单轮39GiB例外）。

新相对学习2400／主管4200／全部6000秒均短于原协议；更严原launch绝对学习17:23:12、原生17:53:12、全部18:23:12固定不延长。诊断和原失败物理／重复物理／总今日预算另计。模型保留与完整54/275、16fresh、78独立模型原生精确及audit/备份门不变。未执行项null，不能称六米稳定或已进入7／8／全图。

启动前：实际完整groups＋八个supportloss、所有消费依赖登记、科学AST、实际入口global名、四worker物理前取消，必须PASS；提交精确远端源后唯一启动。运行active单writer不改源协议或并行audit／backup；driver全native退出后唯一审计和RAM远端完整逐SHA，结束原始备份后再清理冷载荷。部署单120因果Actor，无地图／规划／路点／运行教师／路由。
