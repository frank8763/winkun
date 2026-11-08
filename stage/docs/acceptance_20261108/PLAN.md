# T424冻结六米完整验收

2026-11-08用户明确要求“抓紧时间将六米验收”。本项为独立一次性冻结验收，不恢复T422、不训练、不改门槛或模型。Stage与Phase完全隔离，窄场景暂停，安全T255不覆盖。部署/验收只有单120因果循环Actor，真实carry，无地图/规划器/路点/运行时教师或路由。专用worker只加载一个Actor，不加载训练教师。

## 模型和原始证据

只用T422 learning/blocks/3/before.pt.gz，SHA256 73490bd3d6d7c73c9a351f30979010f90ef48be012c9508403068ae9d05dc38f，与blocks/2/rl_before逐字节同。它与BC2接受候选相同Actor（policy6ad474ec960d9f8c61a62abefe958b7b8e9bb057a2a23631916c53db06c31051），不同仅拒绝PPO后freshAdam。本项不使用Adam，不用仍保存拒绝RL的sampling.pt.gz。完整原始T422在关机前只完成开发14/14，未全量验收。

原T422历史优化器复现首BC0 AFTER0哈希差异仍未解释，明确FAIL保留。本项冻结载入模型的功能证据独立报告，不会将新验收PASS冒充旧训练审计修复，也不将模型晋升为研究证据完备。模型SHA、序列/逐步carry、场景/物理和原任务终止条件全部锁定。

## 唯一执行顺序与门槛

1. 四独立原生worker，从原起点重复14已知开发任务；每项policyhash、fullarraydigest、终态积分、物理数/GOAL/碰撞/超时须精确匹配T422 BC2真实记录，不误用初始B历史。该14为收费物理重复，不作新增独立覆盖。
2. 同一冻结模型完整54项宽六米开发、275项旧能力回归。两组均跑全，不因发现失败只选成功案例。宽场景版本仍1.17m/80s；旧任务保留原cap/场景。真实GOAL半径0.2m，碰撞优先、时限不改，无新的seed/扰动/阈值选择。
3. 只有329项全部有效安全GOAL，才消费16未见导航留出。旧freshledger逐SHA，加锁T412–T423增量14个cases/replays，确认这16job_id未出现；不把已知开发成功称留出成功。
4. 16同样全部有效安全GOAL后，才独立原生重新从磁盘载入同一模型，复验指定78项（8old+54wide+16fresh），全部120/history/hidden/动作/奖励/holdSHA/初末积分逐数组精确匹配先前完整参考。实际重新仿真计物理，但不称独立新泛化样本。

任何缺测null、无效与删失分开，不填0%失败率。完整导航门槛和原训练审计状态分开；失败不换模型/seed/阈值、重复旧目录或补训练，不启动7/8或全图。真通过也保留原训练证据缺口，不无条件晋升。

## 独立审计、预算和备份

0Actor/0Adam/backward。预检实际四worker首物理步前取消并全部退出，8零物理命令；序列/逐步carry误差2.22e-16，SHA/globals全部通过，无实际导航更新。先源码干净提交并GitHub精确根树核验，再唯一launch。后台systemd inhibitor与timeout，独立driver串行main→全部native退出→audit→RAM备份；运行中不改源码或并发finalizer。

最坏437回合＋4setup＋8预检=449全部命令，10,926,739全部物理（cap11M）；主管线2400秒、独立零native审计600秒、RAM备份300秒、source汇总180秒、全流程3600秒。新增48MiB，最多92full×384KiB＋≤275compact×5KiB及元数据预留，全局完整950882276B<1GiB/free≥40GiB，不下载全冷历史、不删唯一原件。

指定78参考及14precheck保存full；其余旧回归保存真实actions/counts/flags及全部原始数组SHA，不宣称compact无损重建缺失的所有120。独立审计重算full的部署动作、carry/history、原clock与reward，所有compact终止/SHA/账、完整条件门槛及78重载摘要，并保留证据范围。全部原始和失败RAM分段远端二进制读回逐成员SHA，Git仅必要源码/协议/报告/收据；网络失败明确PENDING，不假称完成。最后提交汇总并精确源根树同步。
