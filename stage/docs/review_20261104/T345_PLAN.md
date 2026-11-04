# T345：固定真实失败前缀的恢复可用性

2026-11-04，用户明确继续推进。Stage only，与Phase完全隔离。新的有界零学习诊断，非T344续训；原审计失败及补充重构分别保留。

## 假设与固定选择

T344两邻域8条探索全接触、宽开发1/5→0/5，尚无学生实际偏离状态上的有效纠正。T340/T342各自在一个不同宽课程起态自主成功，不代表全状态专家。新问题：这些已经成功的策略，是否能从学生已发生的偏离状态恢复？

固定选已消费T344 `training_r0_n0`、`training_r0_n2`，不选其他轨迹。每条只一个时刻：`prefix_decisions=floor((historical_physics_steps-5000)/100)`，即距已知碰撞10至10.2秒前的完整决策边界；分别14100原生步/28.2秒、7600步/15.2秒。规则在新仿真前锁定，不扫描时间。

每条三种后缀：历史原动作失败对照、冻结T340 final、冻结T342 final，共6primary+6完整原生逐数组/积分精确复验。前缀从原初态真实执行全部原动作；不得中途重置qpos/qvel、goal、执行器、clock、history或上一动作。记录接手完整native integration、120维与真实历史；所有后缀接手状态逐位相同。

采用原速度wheel5/force120、宽缺口1.17m和80秒真实任务cap，不改变碰撞或成功半径。原动作后缀须与T344十二共同数组/最终积分精确。两个Actor按其真实确定性输出执行；不发明专家标签、不假造on-policy logp、无规划/路点/部署路由。此处接手只用于训练资料可用性诊断，不能称原起点自主成功。

## 文献与反证

2026-11-04再次检索一手资料：[DAgger，AISTATS 2011](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf) 的导言和算法强调在学习者访问状态取得纠正；[DART，CoRL 2017](https://proceedings.mlr.press/v78/laskey17a.html) 摘要及[作者README](https://github.com/BerkeleyAutomation/DART/blob/master/README.md)说明示范误差分布与恢复资料重要性。本轮不实现DAgger聚合训练或DART噪声优化、不复制作者代码、不调用其保证。当前任务传感120/0.2秒决策/80秒/车辆动力学与论文操控或运动任务不同。已有T334旧Actor接管失败及T344新在线全失败是直接负面证据，不能预设已有Actor可纠正。

不增加噪声、时间、动作、种子或模型网格。任何候选全失败都只否定这两个固定状态、时刻和候选的可用性，不证明不存在恢复或原六米不可达。安全GOAL且完整复验、独立审计通过才称有效恢复源；后续学习必须另立有界协议。无有效恢复则结束该诊断，准备真实人工纠正入口并确认用户方便，不伪造人控。

## 预算与合同

- 1预检零物理geometry；6primary+6replay；全部13命令/最多480000原生物理步，所有重复及前缀计费。
- 主任务900秒，最后120秒不允许无足够余时的新增批次；每个完整命令启动前至少余60秒。外部timeout900秒TERM/1秒KILL，更早从严。独立audit300秒、全工作（含备份）2400秒，按launch绝对时间不延长。
- 新目录own4.5MiB；6full各≤512KiB+1partial≤512KiB+metadata768KiB+logs256KiB。总newruns≤1GiB、free≥40GiB；预计完整预留1071936220B，约1.72MiB余量。不得删除唯一原件或用远端存在假设来扩大预算。
- Actor/Q/优化器/新on-policy训练全部0；两个模型及safeT255不改。没有加载/更新Adam状态，不触及T344原别名路径；后续学习仍必须成熟Adam隔离预检。
- 完整旧275/原six54/fresh16/reload78均本pilot未执行，保留null；原窄六米稳定门槛未过，不进7/8。
- 正式零物理预检：真实factory初始integration/120/原速度和force精确、两个模型strict load、部署mode与确定性draw一致、零native/Adam。
- 代码/输入SHA与协议本地提交、远端完整根树验证后唯一launch。运行单writer；不得改源协议、写其他runs、sync_source。完成先确认实际main与inhibitor/timeout退出，再唯一独立audit。重算所有prefix/full120/history/clock/原奖励/真实后缀动作/原对照及命令物理SHA。
- 全部原始文件停止写入后RAM分段归档、远端二进制完整读回逐成员SHA；Git仅代码/协议/报告/清单/收据，最后精确source同步。不覆盖T344或原失败。

当前只是即将注册的计划；是否真正启动以T345 launch/status为准。记忆对照及之后学习不在本协议，不得把计划写成已执行。
