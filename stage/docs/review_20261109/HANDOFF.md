# Stage交付入口

2026-11-09更新：用户在本次收尾之后明确要求不放弃、继续查资料并改进算法/网络。已完成[研究复核与新方案](../research_rescue_20261109/DECISION_AND_RESCUE_PLAN.md)，仅静态分析/文献，未启动新训练。下文“停止”记载T440结束时的历史决定，不是禁止新的已授权研究；旧目录仍不重启，新运行仍须独立有限协议/容量/截止。

仅Stage；Phase完全隔离。最后尝试T440离线前置计算超时，未产生候选，现按用户条件停止该线；0新native/0Adam。最终结论见T440_FINAL_REPORT.md/json，不把它视作算法否定或验收通过。原始14文件与存储账本已2分段全远端逐SHA核验。用户条件为最后尝试未通过就停止该线，不自动追加研究。

## 已验证结果

- 五米冻结安全T255历史旧275/275，五米高清视频/原件保留。它不等于当前宽六米模型。
- 宽六米T425 A 54/54但旧264/275；T437研究提案宽52/54旧271/275，联合门未过，不晋升。
- T439四提案修新丢旧，0保留、完整329未执行；不填写新完整六米率。
- 真正未消费16留出/78独立模型重载在上述候选均未执行，八米/全图/实体机没有通过证据。
- T422历史独立重构缺口仍保留，不能把后续audit通过当完整历史修复。

## 文件

- FULL_RETROSPECTIVE_AND_PAPER_GAPS.md：方法谱系、当前差距、论文待补。
- PRIMARY_LITERATURE_AND_DECISION.md：一手资料及已有近邻。
- PAPER_OUTLINE.md：诚实标注待完成的论文骨架，并非成稿。
- WHY_STILL_FAILS.md / FIRST_STEP_PRESERVATION_EVIDENCE.json：当前一阶保持局限的原始事件核查。
- T436_FINDINGS / T437_FINDINGS / T439_FINDINGS：最近冻结差分/真实RL/表示控制结果。
- T440_PLAN / BUDGET：唯一最后尝试、固定门槛与硬截止。

## 代码与恢复

仓库https://github.com/frank8763/winkun.git；源码Git，原始证据/模型在Release stage-distance-2026-11-01。各组release_receipt与source_manifest在stage/docs/long_distance_20261101，先核验收据再逐分段下载拼接、核归档SHA/每成员SHA。T437包含T438审计导入恢复，不覆盖原错误。T439包含driver路径错误与metadata-only收尾说明。

本地唯一仓库/home/linghu/Documents/Codex/2026-10-30/new-chat/winkun。只在stage/experiments及stage/runs工作，global1GiB/free40GiB。T435已把194份旧非输入JSON无损编码，原字节还在本地.cold.gz及既有已完整核验的远端原件；历史裸路径用stage/experiments/t435_relay_reserve/codec.py load_original或受控restore。因初始化嵌套依赖，5份已按原SHA恢复裸JSON（同时保留.gz），见T440_STORAGE_DEPENDENCY_RESTORE.json。其余模型/轨迹/失败没有删除。

停止只表示不再执行新实验，不删除模型/失败/原始证据，也不声称论文已完成。后续若重启研究须用户明确新授权、新预算和可复现初始化，禁止直接恢复已结束目录。
