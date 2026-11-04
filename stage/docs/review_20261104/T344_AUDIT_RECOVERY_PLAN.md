# T344独立审计失败的有界工程复核

原T344导航已结束：8新采集全接触、16live+16shadow、final宽0/5/旧2/20/窄0/2，不能晋升。原完整独立audit在第2块入口断言失败，已独立重构第1块8个历史Adam；原源码/blocks/summary/失败收据全部保持。

代码原因候选：Learner.load把before.optimizer直接传给Adam.load_state_dict，后者可能共享原快照tensor；shadow继续更新时污染before.optimizer。主程序在更新结束后才digest(before)，导致第二块输入Adam字段等于结束Adam。Actor.load_state_dict复制数据，actor输入字段仍为正确旧值。主live/shadow trace相等不等于快照回退安全；本轮实际没有中断训练块，但未来必须修复成熟Adam的所有快照引用。

仅做一次**零native、无新导航训练**工程复核：新的修复合同类load先deepcopy，构造两个共享输入的成熟Adam克隆，**不调用Adam.step**，用逐tensor受控写入/回退验证不污染源及另一个clone；同时对原类复现存储别名，不改变原模型文件。之后从原T340输入freshAdam完整重构T344两块16个历史Adam，逐事件/Actor/Adam/source/rawSHA全部核验。遇到第二块错误字段，明确保留observed与实际独立reconstructed输入，仅在它等于本块输出且actual input等于前块output时标记已解释，绝不改写原字段。

本补充限1200秒且不得超过原launch+6000秒全工作截止，0native/0新训练、仅16额外丢弃历史Adam。**独立审计历史Adam总计24（原失败8+本次16），超出原计划16的8步仅用于这一次审计故障复核，显式另计；不扩原导航物理/采集/RL预算或新编号续训。** 不额外做优化器预检步，不扫参数。own补充输出≤128KiB，仍原7MiB/global1GiB/free40GiB；输入SHA与程序锁定在audit_recovery_protocol.json后执行，失败原件保留且不得称完整审计通过。

本复核只判断实际16次导航更新是否可精确重构，以及下一实现的成熟快照隔离合同；即便通过也不证明导航有效/原回退实现安全/六米稳定。原audit仍FAILED；补充报告单独命名T344_AUDIT_RECOVERY，全部原始备份包含原失败、补充与新合同证据，之后源码精确同步。不得重跑原audit或重新启动任何导航。
