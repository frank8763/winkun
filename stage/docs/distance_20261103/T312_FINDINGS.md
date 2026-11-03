# T312检查：首批V筛选通过，Actor更新入口日志错误使producer停止

检查时间2026-11-03白天。原T312在北京时间**10:59:17.688383**结束为FAILED，实际main199.996650秒，早于12:25学习/12:55总截止。所有父/4worker/timeout/inhibitor均已退出。原source/protocol/status/summary/error/partial模型完整保留，没有重启或修改失败目录，没有新增原生训练。

## 当前真实进展

78项零物理几何检查通过；首批每臂16条实际多目标轨迹，共32采集+32原生精确重生。第一批两臂模型和创新流相同，16组轨迹数组/终态积分相同，物理成本计两次，但不是32独立训练样本。

全部142命令、913812真实2ms物理步。每臂首16探索均6安全GOAL/8接触/2超时，包含不同实际目标；实际六米部分每臂LEFT2接触、RIGHT2超时，属于随机训练数据，不是完整六米54验收，也不是新Actor学习结果。

每臂2fold新神经V各500live+500shadow Adam：合计2000live V+2000shadow V。新Actor Adam调用**0**，shadow Actor Adam **0**，保留Actor块0；两份partial checkpoint的Actor SHA均精确safeT255 `31b694d1ecf92dbcf7909f6ba6e73a808bc9725aeeacfb7584d0aab8fb162d54`，Actor Adam均fresh空态SHA `881489e608a38e54726d0751cd52ed648891a06e8603b81357fc9935f73c3d76`。失败前已算一次PPO loss.backward，但没有执行optimizer.step，不能将反向计算算成一次Actor更新。

## 值/优势筛选，尚不是导航效益

首批两臂数据相同，指标也相同，不能当两次独立复验：

- goal/episode等权留出零V回报MSE 0.604586211，新状态V MSE 0.002003694；各目标MSE门槛通过。
- 原六米LEFT完整参数样本梯度方差：零MC707.570271，MC−V1.906392，GAE1.497135；RIGHT分别90.380743、36.672144、32.705875。
- 各目标MC−V/GAE方差均在预注册零MC两倍门槛内；GAE与MC−V有限批均值梯度余弦0.837220。

这是每目标仅两条创新轨迹的有限pilot筛选，不是普遍降方差证明、无偏GAE证明或六米稳定。它支持检验下一Actor步骤，不能替代真实闭环安全到达。

## 根因及预检缺口

`actor_steps`发送事件payload含`arm`；主程序的实际日志适配器使用`dict(arm=arm, replica=..., kind=..., **payload)`，重复关键字导致TypeError。异常发生在第一Actor `opt.step()`之前，原程序将Actor/Adam整块回退，最终模型与初始完全相同。

原预检使用了一个简单追加列表的callback，未覆盖主程序这个实际日志合并路径。这是实现/预检覆盖错误，本次没有可用于比较PPO与GAE导航效果的Actor训练结果。原summary.stop_reason仍为STARTING，是旧摘要未更新字段；真正失败位置以failure.json的PPO_UPDATE_mc和原始traceback为准，不覆盖原summary。

候选修复在独立`stage/experiments/t312_event_contract/event_contract.py`：先复制payload，逐项验证已有arm/replica/kind与上下文一致，再合并；冲突明确拒绝，不静默覆盖。MC/GAE×live/shadow Actor及V实际payload六种形状、三种冲突、JSON往返与输入不变检查通过，0原生/0优化器。原主程序字节未改；修复主训练的独立协议尚未注册或启动，不冒称已经恢复运行。

## 独立复核与下一步边界

单独的failed-entry审计保留原producer FAILED，独立重做全部4000次已发生V live/shadow Adam、两臂完整score梯度/Gram/优势/门槛、Actor/V/Adam模型SHA与所有源输入SHA；32完整120/真实历史/创新/原奖励/密度、32重生哈希/积分和142命令/913812物理全部核对。结果是**完成部分原始和V重构PASS，原producer仍FAILED**，不能将本轮改称COMPLETE或Actor成功。

本轮旧275、原六米54、16fresh、78独立模型读回验收均未执行，缺测不填0%失败率。安全五米模型保留；六米历史最好完整研究候选仍T29910/54且旧274/275，不能晋升。没有七/八米训练，没有新的合格六米视频。

下一项应先落实真实callback适配器的主流程合同，并在独立有界协议中复用这批已核验的数据/状态V，再做真正Actor更新；不重新采集或重拟合首批来冒充新进展，不换seed/算法参数追结果，也不能借新编号延长旧预算。本次检查未启动修复主训练或新自动化。若旧绝对截止已过，不以恢复名义绕过它；任何新主任务须另锁实际容量/全部预算/源输入SHA后执行。

原始备份收据及归档SHA见`long_distance_20261101/release_receipt_T312.json`与T312_FINAL_LEDGER；以完整远端逐成员读回后状态为准。Phase完全隔离，allruns逻辑约988.36MiB≤1GiB，free≥40GiB，原模型/失败/HD视频保留。

备份现已完成：54原文件/15RAM分段全部完整远端读回逐成员SHA，归档c34e22033c76e294f108dd3e4fcb7eed5286fee7cff4da88e3276a16533450f5。第一次第7段读取TLS EOF中断证据在T312_TRANSPORT_FAILURE_001；在原600秒传输截止剩余内复用已上传段、重新逐字节读回后完成，没有重启科学作业或再次索要授权。
