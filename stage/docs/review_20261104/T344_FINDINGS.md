# T344邻域在线恢复pilot：没有扩展成功覆盖

2026-11-04，Stage only，Phase聊天/自动化/研究/模型/结果完全隔离。导航主程序COMPLETE，全部父worker/timeout/inhibitor已退，不重启或追加种子/轮数。这轮没有突破六米，研究候选不晋升、不覆盖safeT255或T340成功模型。

## 实际结果

|项目|执行|安全GOAL|接触|超时|
|---|---:|---:|---:|---:|
|T342已知wide_dev_1成功原生恢复|1|1|0|0|
|初始T340候选宽开发|5|1|4|0|
|新训练round0，dx−.03m|4|0|4|0|
|新训练round1，dx+.03m|4|0|4|0|
|round0之后宽开发|5|0|5|0|
|round1/final宽开发|5|0|5|0|
|final旧20|20|2|12|6|
|final原窄六米中心|2|0|1|1|

已知T342成功在71.584秒完成，10个共同数组/终态积分精确历史；再次完整原生重放所有新保存数组也精确。这是已知数据的真实恢复，**不是新独立成功或本轮收益**。

initial宽5全部共同数组/积分精确复现T340：1/5成功，不是新泛化。新8训练全部为合法当前policy采集：dx−.03碰撞时间38.394、39.094、25.390、24.200秒；dx+.03为25.122、26.816、25.944、26.438秒。每条再原生完整重生，全部120/历史/动作/raw/边界/logp/积分精确，仍是8个新采集而非16个独立样本。

最终宽5全接触，分别24.034/37.064/25.310/36.630/29.064秒；原窄左25.326秒接触余3.480512372m，右56秒超时余2.392339354m。27项最终模型从磁盘读回后原生复跑全部逐数组/积分精确；其中final宽5与round1后检查同policy/起态，也是5次相同轨迹重复。全78个primary记录里32个是同policy/起态/积分重复（27读回+5末轮/终验），另有9条显式replay，不以重复放大样本量。

完整old275/原six54/16fresh/78发布重载均未执行/null，不把缺测填0%失败率，不把本轮旧20当完整五米保持。没有七米、八米或新的合格六米视频。

## 学习、计费与模型

从T340 final精确起步、freshAdam，保持原wheel5/force120、CAPG120/hiddenreset、奖励、学习率与PPO条件密度。新增两个偏移训练家族，同时加入已知T342偏移成功/T340中心336状态辅助支持，是联合工程pilot；没有在线全状态专家、规划器、路点或路由，不把演示设为on-policy资料。

两轮8新采集+8重生，另1已知源恢复+1重生；16live+16shadow真实Actor Adam，全部两块保留于研究副本；预检另4丢弃合成Adam。没有Q/V/新奖励/新动力学，T343提速结果不混入本轮。

92全部命令、1,474,370全部物理、462.5037910938秒；78个保存case、9个显式重生。模型`stage/runs/t344_neighbour_recovery/final.pt.gz` SHA `39122bd595a499e6c716b75711834044c7abfc62c0b1af4eef8ca4325ec942c1`；部分Actor digest `cde1fdc2bace65ddbf707dee73798e297c2dc30e44f6b99ac26664ec30d12917`；Adam `e7ae35c4d9d7d7cb9fe816ae565696f9ac195e0df928de6b482783b12b4ac3a1`。完整部署policy hash以原case为准，不混淆部分state digest。

## 独立审计原失败与补充复核

**原audit没有通过。** 第2块入口inputAdam哈希不符，原第1块8历史Adam已独立重构并核验trace/Actor/Adam/事件。原源码、协议、blocks、summary与`T344_ORIGINAL_AUDIT_FAILURE.json`保留。

原因已复现：Adam.load_state_dict持有传入快照中的优化器tensor引用；shadow更新时污染before.optimizer，而主程序在更新后才记录input_hash。第二块raw记录的input.optimizer为最终`e7ae...`，实际输入由前块输出和独立重构确认应为`973bd3d8...`；Actor输入字段正确。原字段不重写，本轮没有实际中断训练块，不能把潜在回退隐患说成已发生的回退事故，也不能将此元数据错误当导航失败的唯一原因。

经单独注册的有界工程补充，`T344_AUDIT_RECOVERY.json`为`ADDITIVE_RECONSTRUCTION_PASS_ORIGINAL_AUDIT_REMAINS_FAILED`：0新增native/新训练；从原输入完整重构16次实际更新、逐事件匹配live与shadow、全训练密度/120/history/奖励、原始SHA及最终Actor+Adam一致，27模型读回与9原生重生通过。原audit失败保持。

本次额外16历史丢弃Adam，加原失败8，**独立审计总计24**，不是原计划16，也不是24次新导航训练。成熟快照隔离修复在新`IsolatedLearner.load(deepcopy(...))`中用受控tensor写入/回退验证，0额外Adam.step；不能再只用空Adam预检来声称成熟状态回退安全。补充没有重跑导航、改动原producer或续其物理/采集预算。

原始完整备份收据见`../long_distance_20261101/release_receipt_T344.json`；只有该收据状态为FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED才称完成。补充源码/报告/原审计失败随Git精确源树保存，不将大二进制入Git。safeT255原SHA保持，所有失败模型和轨迹保留，newruns≤1GiB/free≥40GiB。

## 下一步的证据判断

本轮没有支持“从两个邻域增加采集并复用原成功监督即可扩展覆盖”这份有限配方的成功预期；不否定所有课程、PPO或示范RL。8条新探索没有成功，旧成功轨迹又未覆盖当前失败后的纠正状态，支持恢复覆盖仍不足这一假说；不能据此证明场景不可解、算法理论不可学或灾难性遗忘是唯一根因。

真实旧路线保持也未做到：首次更新后宽中心失守，辅助监督不是闭环安全约束。训练日志末epoch更新前max_abs_logratio分别2.60777/4.49543；它是实际概率变化的描述，不是KL、唯一因果证据或PPO clipping失效的代码证明。不能在没有匹配对照时将失败单独归因于这些数值。

下一机制应先用真实学生失败前缀验证能否得到成功纠正，再决定学习；不能将T340/T342在任意偏离状态的输出一律当正确教师。成熟快照隔离与已会中心的真实闭环检查应纳入下一合同。具体只草案见`NEXT_AFTER_T344_DRAFT.md`，不自动开启训练/噪声/步长网格。RFCL/DAgger/DAPG/CAPG等近邻与实现范围见T344_PLAN，联合工程pilot不宣称顶刊原创成立。
