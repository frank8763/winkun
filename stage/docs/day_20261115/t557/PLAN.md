# T557：冻结BC阶段归因，尚未注册或启动

目的：区分“32BC已破坏原任务”与“4PPO新增破坏”，不用保存状态动作误差替代真实闭环结论。所有任务均是已消费开发项，不能叫留出泛化。

固定两个已保存T555 behavior模型（project/scalar），其Actor/Adam权重均不更新。每个模型测以下6任务并精确重放：dev_task6_13、field sector0/offset2、sector1/offset1、sector1/offset2、sector2/offset2、sector4/offset2。它们是T555两候选原成功失守的并集，无结果后补选。每例保留完整120观测、真实GRUcarry、历史、动作、积分。12自主原生+12精确重生不是24独立样本；匹配基线和postRL结果直接复用，初始原生SHA必须相同。

判定分列：BC已经失败／BC成功而postRL失败／两阶段均成功；成功/接触/超时/无效/删失逐项。不能宣称首次精确分岔时刻或唯一算法因果，也不训练/晋升/自动接下一机制。若证据合同失败，保留partial和错误，不重启旧目录。

最坏1,257,916新物理步、32命令、1800主管线秒，独立audit300秒、备份300秒、source180秒；CPU14/15总一核idle Nice19 GPUoff，Phase隔离。原15:00all/14:30native从严不延长。计划从既有19M/600/12300日额度中单独重分配本scope，连T555累计≤5,392,152物理/216cmd/4353.581main，仍需用户确认scope后才注册启动，绝不复用T555失败训练目录或给其追加预算。

own17MiB：12full各1MiB上限+1原子暂存+2worker partial+2metadata；实际global最坏1,072,466,301B<1GiB、free>40GiB。全部模型只引用不复制。全原始结束后RAM归档逐成员读回SHA才称远端备份。

预检实际两个模型SHA/phash、真实因果sequence/cache一致（误差0）、未来输入不影响过去输出，0native/0Adam PASS；复用已审相同worker/动作/GRU/物理时标合同。预检不等于实际原生恢复或导航通过。T557尚无AUTHORIZATION/protocol/launch，当前没有后台Stage训练。

依据及后续分流见上级T555_FINDINGS_AND_NEXT.md；先区分BC与RL后再确定唯一学习变量，不预设更换任何方法必然有效。
