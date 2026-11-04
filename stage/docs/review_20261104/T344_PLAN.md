# T344：两个偏移起点的真实在线恢复获取pilot

2026-11-04，用户明确“开始下一步实验”。Stage only，与Phase聊天/自动化/研究/模型/结果完全隔离。T340/342/343已审计与全原件远端读回，不重启或增加其预算。当前新runs剩余约7.60MiB，本次完整预留7MiB后才注册；不删唯一资料，不扩大1GiB。

## 假说与采用方法

T340宽1.17m/80秒原中心安全67.064s，但四偏移均接触；T342固定平衡资料拟合恢复旧20的17项，却失去原中心，只有已消费wide_dev_1安全71.584s。下一假说是：用真实新策略在相邻起态访问的状态做RL，并保留两个真实成功来源，可否扩大恢复覆盖。它不是再训练同一条中心监督轨迹，不声称静态MSE或增轮数即可解决。

从T340 final Actor精确开始、freshAdam，原CAPG动作空间/120因果输入/hidden每步清零、原奖励/PPOclip.2/8epochs/actor_lr/.035,.03实际动作标准差、网络容量不变。原速wheel5/force120，不混入T343新动力学。部署单Actor，无地图、规划器、路点、运行时教师、距离路由。

先从T342 final在已知wide_dev_1完整原生重建成功一次、再全数组/积分精确重生；它是历史已知成功的恢复，不是新独立成功。全部120/history/draw完整保留。与已完整保存的T340中心336状态、原T328真人320/T331真人纠正168/T338R自身76状态共同作辅助监督，真人策略前缀不冒充人工。T342任意学生偏离状态上的输出没有正确性保证，不取作DAgger专家标签。

固定两轮：round0=wide_dev_1（起点dx−.03m）、round1=wide_dev_2（dx+.03m），每轮4条当前策略独立噪声轨迹、每条再完整原生精确重生；共8新训练/8replay。每轮四条来自同一真实reset，用原已核验物理时间对齐的同行RTG baseline；终止同行后Gt0，实际native_count折扣，8PPO+衰减.05*.5^round示范项；失败也进入RL，成功轨迹可增监督组。不伪造示范on-policy logp，不接未验证Q/V。

这同时改变训练初态覆盖与成功支持来源，是联合工程pilot，不能单独归因任一模块或称原创。固定顺序是覆盖测试而非完整RFCL自适应score实现，也不称DAgger/DAPG原版复现。

## 基线、验收与停止

initial宽5全测，全部共同数组/积分须精确T340；每轮后宽5确定性检查，final同候选旧20/宽5/原窄左右2中心，保存读回后27项再原生复验。训练8与source1共9个full记录，9次重生都非新增独立样本；训练成功与确定性开发成功分别报告。

本次只回答有界恢复覆盖pilot，**完整old275/原six54/16fresh/78发布重载不在这份预算内**，未执行均为null。若筛查全过也仍需另立完整发布验收协议及容量，不自动7/8，不将新编号用作续训。原窄.57m/56秒与宽课程分列，safeT255原件保持。两轮后结束，无额外seed、噪声/奖励/权重/轮数网格。任一数值/源/预算/重生错误停止，训练块中断整个Actor+Adam回退，丢弃/保留计数分列，所有失败与partial保留。

## 预检、计费与磁盘

5零物理几何+2 source再生+5 initial+16训练/重生+10每轮检查+27final+27读回=92命令，注册最坏3.48M物理；硬128全部命令/4M全部物理。0人控前缀，起态只在回合开始原生snapshot/reset并施加原已定义偏移，不在rollout拼观测/重置clock/history/目标。native当步碰撞与GOAL语义保持。

学习上限16live+16shadow ActorAdam，预检4丢弃合成Adam另计，audit16历史丢弃重构，非新RL。主3600秒末1200验收、audit≤1200秒、全部6000秒，不延长；学习/发车留原协议admission余时。

9full×384KiB+4partial×384KiB+1模型Adam256KiB+69compact×4KiB+metadata1.5MiB=7,229,440B<7MiB（7,340,032B）。所有历史数据输入引用已有文件，无本地重复大包；RAM归档全部远端完整逐成员SHA后才称备份。global1GiB/free40GiB，运行中单writer不写其他runs/改源协议或源同步。

真实spawn worker完整draw/边界质量/logp逐位一致，四合成克隆Adam shadow/拒绝中断回退，5个reset真实history/完整积分/120输入几何预检；源与输入SHA、本地提交、远端精确root-tree后唯一启动。合成预检不是物理成功。结束所有父worker链退出后唯一audit与全部原始RAM远端读回/源树同步。

## 文献/作者实现与反证（2026-11-04检索）

- [RFCL ICLR2024 arXiv v1](https://arxiv.org/html/2405.03379v1) §4.2、§5.2及§6：已阅读从示范初态扩展到更广初态、只用forward的探索反例和仿真reset局限。只借邻域覆盖思想，不搬其SAC Q-ensemble、50%offpolicy buffer、动态timelimit或声称完整复现。作者[forward_curriculum.py](https://github.com/StoneT2000/rfcl/blob/main/rfcl/envs/wrappers/forward_curriculum.py)已读success_once_score、rank/staleness/reset段；其score优先偶发成功，1000seed方式未直接移植，不安装JAX。
- [DAgger AISTATS2011](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf) §3/Algorithm3.1已读：监督需要在访问状态查询有效专家；当前两个神经来源只在各自成功轨迹验证，不能假造全状态专家。本轮不用其保证，采用当前策略真实RL。实现库[HumanCompatibleAI/imitation的DAgger文档](https://github.com/HumanCompatibleAI/imitation/blob/master/docs/algorithms/dagger.rst)明确在线专家查询需求；它不是原论文作者实现，本项目不复制该库。
- [DAPG RSS2018](https://www.roboticsproceedings.org/rss14/p49.pdf) §IV.C和作者[mjrl/algos/dapg.py](https://github.com/aravindr93/mjrl/blob/master/mjrl/algos/dapg.py)衰减demo项/自然梯度段已复核。本项目沿用已核验PPO与location监督，无DAPG FIM自然梯度，非原版DAPG或新公式。原T333/T342静态拟合/闭环失败反证保留。

尚无多训练seed、未见场景或安全证书，不把组合称顶刊创新。若两偏移仍无可学习正例或final无新增确定性覆盖，应停止这份配方，按原始状态/动作/接触记录定位并写下一新机制草案，而非扩本轮预算。
