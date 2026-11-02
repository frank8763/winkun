# T299：六米10/54，旧274/275；方向性进展，尚未稳定

2026-11-02独立有界实验T299结束，COMPLETE/stop SUPERVISED_OLD20_GUARD_FAILED。500 live+500 shadow自身轨迹条件NLL监督更新，新增导航RL交互/策略梯度更新0。模型从安全T255重新开始，没有续T298；修正辅助损失和canonical当前/前均值数值路径，初始梯度尺度一次固定，不扫seed/步数/噪声或损失权重。

| 完整原生验收 | 安全GOAL | CONTACT | TIMEOUT | 无效/删失/缺测 |
|---|---:|---:|---:|---:|
| 旧275回归 | 274/275 | 1 | 0 | 0 |
| 原6m LEFT开发 | 0/27 | 27 | 0 | 0 |
| 原6m RIGHT开发 | 10/27 | 0 | 17 | 0 |
| 原6m开发合计 | **10/54，18.52%** | 27 | 17 | 0 |

六米尚未过稳定门槛，不晋升/不替换五米基线；16fresh与78独立native重载**未执行**，不是失败0%。七八米未进入，无新合格六米视频。T299与T298仅各一次完整开发网格评估，不是独立训练seed或新地图泛化结果。

## 与上一轮的真实比较

- T298裁剪modeMSE候选：旧267/275、六米2/54；T299条件NLL候选：旧274/275、六米10/54。是已消费开发集上的描述性改善，尚不能把单pilot+数值路径差异称为随机独立因果证明或新算法贡献。
- 10个RIGHT成功id：03、09、12、13、15、18、19、21、22、24（前缀dev6_RIGHT_）。到达时间54.034–55.792s，接近原56s cap，不改时限或半径。17个RIGHT超时结束时剩0.208034–0.556274m，不能把最近一例差8mm算成功，也不能通过延长cap/扩大0.2m半径过关。
- 旧275唯一失败仍为LEFT5m dev_task6_24接触；其余274安全。它说明小动作误差保持仍不保证完整闭环保持，unsafe研究候选不能替换safeT255。
- 限定Stage T256–T298显式source_reset_id/原生JSONL历史扫描，03、09、12、15、19、22、24为此前扫描未见安全成功的7个开发点；13历史有原6m随机探索成功，18/21已知完整开发成功，不能将其都说成新可达点。详见T299_COVERAGE_INVENTORY.json。首次后验selector含course-id后缀匹配过宽，初版JSON和源码保留，纠正版排除course/auxiliary/local1m/remaining；本轮原始54结果不变。

## 修复了什么，还没解决什么

固定梯度尺度.0019520158667963997的full-parent 8轨迹范数比，在独立后验重算中逐位一致。原始modeMSE梯度norm6.6339496363，conditionalNLL norm3398.5121479；只匹配初始范数，不声称Adam/clip后位移相同。

全辅助数据conditionalNLL 239.1288342→62.0157378，下降74.06597%；modeMSE .2465345562→.2064574361，仅下降16.25619%。T298反而modeMSE更低(.1240)但原6m成功更少，说明代理动作误差和导航成功不等价。条件NLL没有被替换成oldlogp、没有截尾IS权重，当前及previous mean正确反向传播，原行为密度复核误差1.05e-13。

T299首次补了safeT255在同两个反事实辅助goal下的**自主原生对照**：父模型左GOAL、右CONTACT；T299候选也左GOAL、右CONTACT。初始120逐位相同。故不能再把T298/T299左辅助goal成功当作学到新技能。原T295八条记录动作重放GOAL则是数据合同，通过不等于任何父/学生自主能力。

后验latent-space AR恒等式说明：正确的相关动作联合似然可以较弱惩罚共同偏移，而强惩罚创新项差异，不能由jointNLL下降推出确定性部署mode靠近全部成功动作。具体rho=.98/128决策/latent constant shift2，在固定Jacobian示例中jointNLL增量4.5657，而独立边际增量256；是合成代数说明，不是本轮真实闭环失败的因果证明，也不是采样器/AR公式bug。当前结果给出的是概率学习有效、右侧覆盖扩张，LEFT6m仍缺安全通过。

下一重点分开：RIGHT超时末段效率，LEFT接触绕障技能，以及旧LEFT5m24保护。只给右侧追加轮数不能解决左侧。已经只读核验T263/T267中**五条互不完全相同的LEFT5.5m完整安全轨迹**，full120/全部数组SHA/firsthit终止均通过；这些是已消费自身历史行为，不是新独立导航数据，可供后续比“父模型本来就会的短辅助goal”更贴近困难区间的学习来源。尚未据此启动下一训练。

## 审计、计数与保留

353全部命令、5410948全部物理步、main1290.268357秒，均低于注册上限；4个预检丢弃Adam、500监督live+500shadow、独立审计再1000丢弃Adam重构，无新增RL调用。T299_AUDIT.json验证全输入/紧凑5数组派生reward/计数、五个监督检查点ActorAdam SHA及最终Actor+fresh emptyAdam精确。T299_POSTHOC_COMPARISON.json另复核原固定gradient scale及父/T298/T299代理指标，零native/optimizer。

最终checkpoint SHA `b15cddbbf5914ef5187ce221082c0e6cdbe3608b0f8a1ff1177f8fefa95f5751`；Actor `d104ea9dc4aaeef6fbdc65947392ce05f3334101c6393e18194ad9eb03ffebb0`；emptyAdam `881489e608a38e54726d0751cd52ed648891a06e8603b81357fc9935f73c3d76`。safeT255/两1080p用户视频SHA均原样保留；全部父/4worker/timeout/inhibitor已退出。own约4.41MiB<10MiB，global约1015.25MiB<1GiB/free约209GiB，唯一证据和模型未删。启动检查曾因源同步未完/证据未commit拒绝，均0native/optimizer、没有重复launch，见T299_LAUNCH_GUARD_REJECTION.json。

GitHub原始备份状态以long_distance_20261101/release_receipt_T299.json为准；首次上传HTTP408错误及远端检查证据保留，未备份核验前不删除唯一资料。源码最终完整root-tree同步后本项结束，不重启或扩旧预算。方法近邻和条件边界见T299_LITERATURE_NOTES.md；标准条件MLE/目标重标记与工程修复不冒称原创或安全证书。

最终备份已完成：378原文件、RAM归档4020264B，远端完整二进制及逐成员SHA均核验通过；[完整原始附件](https://github.com/frank8763/winkun/releases/download/stage-distance-2026-11-01/stage-distance-t299-04c3c3a36920.tar.gz)，archive SHA `04c3c3a36920894eba67e7b221628678893eecd6ec04759635bcbefbeda6e5e3`。首次HTTP408产生的starter状态附件不计完成备份，重试仅传输同一已停止数据，没有重启训练；错误及远端状态记录保留。
