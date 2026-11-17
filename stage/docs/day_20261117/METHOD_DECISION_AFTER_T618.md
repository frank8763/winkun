# T618之后：文献与代码联合复盘、下一路线建议

2026-11-17。Stage only。用户要求全网查找与思考；本轮仅文献、源码和既有证据阅读，0新原生/0新优化器/0新模型更新。原15:00实验窗不延长，未注册或启动下一实验。没有下载论文/模型包、clone大型项目、安装依赖或读取Phase研究。检索不是穷尽全球，也不代表复现论文结果。

## 决策

优先把最近的“成功恢复标签资格→有限BC→静态动作保持”流程，替换成可利用失败反馈的、以真实闭环数据为主的循环离策略学习基线。采用RLPD为连续控制主参考，JSRL/RFCL仅提供训练期成功前缀课程；保留当前成功基线，持续采集旧任务与新任务自己的闭环数据。先核动作支持、Q目标与真实回报一致，再接Actor更新。首轮不要同时叠加HER、TaSIL、神经元重置和视觉。

这些方法此前已被讨论：[11月5日文献重设](../narrow_20261105/RESEARCH_RESET_20261105.md)、[11月16日方法决策](../day_20261116/t593_research/DECISION_AFTER_T594.md)（路径以仓库对应文件为准）。它们不能被说成今天新发现或已完整落实。今天变化是用T614–618明确的新反证，决定不要继续只围绕固定历史动作误差和失败末段教师转圈。

## 已有事实与仍待检验的因果

1. T608保留研究模型34/49，场地13/24；这是反复开发集合结果，不是全场地或独立泛化通过。T616B新增偏离接管与八米LEFT13原起点均成功，但sector4offset2失败，已回退。原34并未被新模型完整保持。
2. T614旧成功轨迹上执行动作峰值约0.001仍改变终局，T616B把40旧history峰值压至0.00025仍丢一项。说明此有限函数约束不足以保证当前闭环保持，不证明所有正则都无用。
3. T617两个教师的真实末段mixed结果都超时；T618三种固定干预也超时，最近记录距离0.220581/0.214522/0.262707m。完整数组可核验，但原native exact缺测，原producer工程FAILED不能改写成全协议完成。
4. 最近T613–616是BC监督更新，0新Critic；T608/609是基于同行MC回报的clipped策略梯度，不是标准带V/GAE的PPO，更不是RLPD。不能说“最新实验已否定SAC、PPO和所有RL”。
5. 过去T336确实试过SAC/反向课程。其152个前沿决策的采样实际动作与mode差中位约1.19e-7；10.55M计费物理中，真正新自主后缀400552步约3.79%。这些具体缺陷不能被“再用SAC+课程”掩盖；参见T336_POSTHOC_ACTION_SUPPORT与T337_FINDINGS。
6. 当前严格环境的_policy_to_commands源码把drive归一化经平方映射到非负wheel，steer映射为action[1]*max_steer。因此负drive不是主动倒车指令，输出已饱和时加raw steer不会增加可执行轮角。本文没有执行新环境来测真实最大曲率，不能据此称74秒状态物理不可达。原起点有成功旧策略，原任务本身并非已证不可解。
7. T617结果接口与T618审计mask两次工程适配失败由本次实现引入。对应完整轨迹虽可离线恢复，但原精确重放缺测；下一发布前必须让相同的完整执行→结果落盘→history审计路径通过正负回归，不能再只测孤立的数学公式和step wrapper。

证据：[当前状态](../RESEARCH_STATUS.md)、[T615](t615/OFFLINE_FAILURE_DETAIL.json)、[T616](t616/FINDINGS_AND_NEXT.md)、[T617恢复](t617r/FINDINGS.md)、[T618恢复](t618r/FINDINGS.md)。动作映射为仓库stage/runtime/mk-mini/training/e2e_phase10_lidar_v7_curriculum/phase10_strict_mujoco_env.py:39；它是已经核验的Stage历史环境依赖，名称含phase10不代表使用Phase研究。

## 按用途排序的一手方法

| 方法/来源 | 论文实际做什么 | 对本项目的取舍 |
|---|---|---|
| RLPD，ICML2023：[论文](https://proceedings.mlr.press/v202/ball23a.html)、[官方代码](https://github.com/ikostrikov/rlpd) | 结合先前和新交互经验，平衡取样、规范化价值网络与Q集成；数据可以次优。 | 最高优先。失败经验按真实回报训练价值，不用等每个新状态都有成功教师。它不保证旧任务保持，也不保证未覆盖动作的价值可靠。官方实现是前馈SAC，循环移植需新合同。 |
| JSRL，ICML2023：[论文](https://proceedings.mlr.press/v202/uchendu23a.html) | 先由guide执行前缀，再由学习者继续探索，逐步减少guide。 | 补短学习视野。成功前缀用于抵达可学习状态，不要求guide在每个偏离处恢复；guide覆盖和探索仍有前提。保留原clock/实际history，最终必须从原起点纯学生运行。 |
| RFCL，ICLR2024：[作者项目](https://reverseforward-cl.github.io/)、[代码](https://github.com/StoneT2000/rfcl) | 每条示范的反向课程后，再训练原初态分布的前向课程。 | 相比仅缩小接管步长，重点是后续回原初态分布；已有T336相近尝试，不能称首试。任意状态reset必须满足真实动态状态/观测历史/学生记忆合同。 |
| HER，NeurIPS2017：[论文](https://proceedings.neurips.cc/paper_files/paper/2017/file/453fadbd8a1a3af50a9df4df899537b5-Paper.pdf) | 把实际达到的其他目标用于额外训练。 | 第二阶段对照。重标整个目标相关120历史、奖励、终止和循环carry，避免碰撞改成功；原目标验收不改。失败回合变成其他目标的训练数据，不等于原目标成功。 |
| TaSIL，NeurIPS2022：[论文](https://proceedings.neurips.cc/paper_files/paper/2022/file/7f10c3d66c3b7863a9cda255dcac5bb7-Paper-Conference.pdf)、[官方代码](https://github.com/unstable-zeros/TaSIL) | 匹配策略局部Taylor项，并在专家稳定性假设下分析轨迹误差。 | 以后针对局部响应的独立消融，不用当前不可靠恢复教师直接套保证。固定观测的网络Jacobian不等同真实物理闭环稳定性。 |
| 循环RL基线，ICML2022：[Ni等](https://proceedings.mlr.press/v162/ni22a.html)、[代码](https://github.com/twni2016/pomdp-baselines)；[R2D2原文](https://openreview.net/pdf/387fb2fcee8f74c53cf707a9856f40c458f33933.pdf) | 重视Actor/Critic各自时序状态、序列回放和上下文。 | 这是实现前提。先用当前网络从完整实际前缀重算的结果作参考；截断/burn-in仅在误差实测可接受后用于加速，不拿旧hidden或零hidden替代。 |
| Policy Consolidation，ICML2019：[论文](https://proceedings.mlr.press/v97/kaplanis19a.html) | 不同时间尺度策略记忆辅助当前策略学习。 | 抗遗忘候选；先让旧任务持续真实采样并平衡回放，再与额外巩固项对比。任何蒸馏项都不能替代原起点旧成功测试。 |
| Continual Backprop，Nature2024：[论文](https://www.nature.com/articles/s41586-024-07711-7)、[代码](https://github.com/shibhansh/loss-of-plasticity) | 有选择地替换低效特征以减轻可塑性丧失。 | 可塑性丧失和遗忘不同。当前已有非零梯度和可应用更新，尚未证明死特征是主因；不能先随机重置成功Actor/GRU而期待保持。 |
| 训练期非对称Actor–Critic，RSS2018：[论文](https://www.roboticsproceedings.org/rss14/p08.pdf) | Critic可获得训练期完整状态，Actor只用可部署观察。 | 后备消融。可以保持单Actor部署，但需要明确新增训练信息，不能把仿真真值泄入部署，不能自动视作当前已获准。 |

DAgger本身并不形式化要求每个教师query完整成功；Stage成功后缀资格规则比原算法定义更严格。这里不撤销成功监督资格，只改变失败经验的用途：可作RL转移/风险反馈，不能冒充成功动作标签。更完整来源和官方代码核对见[METHOD_REVIEW_OFFPOLICY_PRIMARY](METHOD_REVIEW_OFFPOLICY_PRIMARY.md)。

## 更新工作也已核看，但暂不堆进首轮

- [Offline Goal-conditioned RL with Quasimetric Representations，2025论文](https://arxiv.org/abs/2509.20478)、[作者项目](https://tmd-website.github.io/)讨论用次优数据学习目标距离和片段拼接。
- [Scaling Goal-conditioned RL with Multistep Quasimetric Distances，ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/eddc0fab5a42f8d8de6eb5566cd9f1d3-Abstract-Conference.html)将多步回报与准度量距离结合，报告长时程仿真及真实机械臂结果。
- [FlashRFCL，作者2026-09页面](https://hichristensen.com/publication/2026-corl-flashrfcl/)提出GPU高吞吐仿真加课程；本轮只查到该作者页面摘要，没有据此核实完整实现或承诺本车单显卡加速倍数。

这些可以服务后续全场地长距离。当前车体/时序/终止合同尚未与其状态和动作模型对应；不能拿点迷宫、机械臂或其他平台的成功率推算本车结果。先把可比较的连续控制基线跑通，才有理由引入更复杂的距离表示。检索覆盖到2026论文不等于认定越新越适合。

## 下一实验按什么步骤推进

A. 先做完整软件合同回归，而不是再次只做局部预检。用已有T616/T617/T618真实数组覆盖纯学生、永久接管、短时干预交回、成功/接触/超时、两种记忆架构和无request原始返回；各自正确接受，篡改mask、动作、目标、history、终止必须拒绝。此项不需要新物理数据。

B. 明确一项学习目标：先提高当前失败场地原起点自主成功，并同时保住原34项。另设冻结当前成功模型作为对照，不用拒绝候选替代基线。开始先在当前单环境做正确且有用的数据学习，不先大迁移仿真引擎/相机。

C. 固定两组而非算法组合网格：两组采用同一循环离策略学习器、先前/在线平衡回放、同模型/同奖励/同交互和更新数；A原起点训练，B额外使用训练期成功guide前缀课程。前缀由已验证成功轨迹抵达；碰撞/timeout的当前学生自己的新经验也入RL buffer，成功示范可作prior。不能先用极晚且已偏离的74秒失败状态作为唯一训练分布。旧任务也要有当前策略真实交互，不能只有静态旧标签。B若从课程起点成功但原起点失败，结论仅局部课程学习，不能晋级。

D. Actor/Critic工程选择必须在注册前锁定。官方RLPD是标准tanh连续随机Actor，当前Stage使用裁剪Gaussian及历史残差；不能把该clip密度直接套入标准tanh-SAC entropy，也不能无声切换动作函数后宣称初始化完全相同。可以独立建标准循环SAC研究学习器，保留当前模型作为冻结参照；如果温启动改变了原动作/hidden或初始成功集合，必须先测并报告，不把初始化损失混成后续遗忘。Q集成和target networks仅训练使用，运行时只有一个Actor，不靠Q挑动作。

E. 训练前先确认实际执行动作有有效差异和反馈，避免T336饱和伪探索。Q需在按任务/整轨迹拆分的留出数据上验证回报和动作敏感性；Bellman loss变小或Q集成一致不等于真实可控。在已有分支无成功/回报无差异的支持之外，不能把Q当成功专家。若没有有效动作或回报区分，先处理数据/动作合同，不继续加更新次数。

F. 物理时间与终止目标核验：现gamma_native=0.999966667222216，每步0.002s；每个held action的gamma应按真实native_counts指数计算。原任务deadline timeout与工程中断censor不能混淆；训练截段又不是物理任务终结。[Time Limits in RL，ICML2018](https://proceedings.mlr.press/v80/pardo18a.html)明确区分有限任务时限和训练截断，不能机械把所有trunc都bootstrap或都置零。

G. 首轮有学习效应后，再单独加入HER或局部响应保持。训练期少量哨兵用于发现明显崩坏，完整候选后做34逐ID/49以及条件完整验收；冻结未采用候选不会破坏当前保留模型。新自主后缀、重复前缀、原生精确重放、更新/影子/独立审计/验收物理量分别计，不把总计费物理冒充新探索量。下一预算尚未核定，不做新原生/优化器或延长原窗。

## 端到端与可能论文贡献

上述课程、prior、失败回放和额外Critic都属于训练；部署可保持雷达/最终目标/车辆状态→循环Actor→drive/steer，0地图/路点/规划/运行教师或路由。该路线是示范辅助/离策略端到端RL，不称“完全不使用示范的纯RL”。加入任意新信息都要在论文训练/部署两列明确披露。

值得研究的命题是：在非完整观测和车辆运动约束下，按实际恢复困难组织循环经验回放及训练分布，能否同时提高恢复成功、旧任务保持和未见起终点泛化。与普通回放、课程、局部响应约束同预算对照后再判断新贡献；不能把RLPD+课程的组合命名为已验证原创，更不能许诺顶刊。论文需要多训练种子/未见布局/物理与传感扰动/真机验证。当前最急的是建立真实学习曲线和可靠执行链，不是继续增加算法名字。
