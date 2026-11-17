# T620 循环离策略学习先导对照（准备，未启动）

目的：用实际闭环状态、成功及失败转移学习价值，改善动作模仿后的状态漂移，比较固定成功前缀的作用。当前T608固定开发34/49、场地13/24，未全局通过。

两组同T608初始函数、全新Critic/Adam。Actor只训10274参数循环分支；双Critic各有独立归一化/GRU合计8610参数。每组48次Critic，每2次做1次Actor，tau=.005，目标动作噪声.2裁剪.5，物理时间discount，有限任务终止含超时不bootstrap。保持原clip与float32执行接口，不是标准SAC、完整RLPD或已证创新。

每组新采sector0offset2、sector4offset2、八米LEFT13三条随机闭环并逐条exact。A原起点全程学生；B只在第一项用成功轨迹前276决策，再学生续行，原目标总时钟不变。学生自身记忆在实际观察前缀连续计算。混合数据只作off-policy转移，不是自主成功/教师hidden/成功BC标签。

55条完整历史中54有PASS源审计可用，1条待审计排除。36条BC记忆缺next_state/reward/终止，不伪造转移。batch为2 prior+2本轮online episode，有放回采样，每条16时刻，从实际完整因果前缀重算当前/目标Actor及两Critic。历史不冒充新独立样本，不用未来或缓存hidden。

Actor loss=-Q1+10*原34成功ID当前T608函数MSE，不放宽碰撞/到达。每组等量影子和独立历史重构全部Actor/Critic/Adam/targets；成熟状态回退逐摘要检查，拒绝整learner回退。

先两个关键旧任务原起点+exact，全成功才三恢复+exact，全成功才其余47项及4关键重放。原34逐ID保持且新增至少一个原起点成功才保留。49为固定开发，407/fresh16/reload78未实现null，无全局晋升。

每组native全退+auditPASS+本地完整manifest才下一组；第二组仍独立相同初值。每组完整最坏时间/步/命令/盘准入，不足就不开始，不扫seed或续预算。17:30停学18:00停仿真18:30全停。断网科学本地完成，科学全退后RAM归档全成员SHA，网络失败REMOTE_PENDING。CPU14/15总200%低优先GPU关，Phase隔离。

小样本短先导的新Critic可能尚未学准；loss下降或动作噪声std不是导航改善证据。日物理/命令/主管线/存储总额不增，新增优化器额度待批准。
