# GitHub与期刊方法复核：为什么24任务仍未推进

2026-11-16，Stage only。资料研究，不是T593训练注册；0新增物理步、0优化器、0模型修改。只在RAM阅读少量公开代码，没有克隆仓库、安装依赖或下载数据集。导航期刊专项见同目录NAVIGATION_PRIMARY_SOURCES.md（存在时）；代码片段行号与SHA见CODE_REVIEW.json。

## 项目证据先于方法选择

- T550同一保留策略field24=15/24（8碰撞1超时）；T591联合表示候选14/24（6碰撞4超时），拒绝晋升。
- T591是600步示范监督学习/臂，不是新RL；39旧成功history里只有15个field任务，3个新恢复suffix仍来自已成功ID：sector1_offset2/sector2_offset2/sector4_offset2。没有任何offset3任务的成功标签。它测试能力整合，不能解释为充分尝试攻克全部剩余困难任务。
- T5913handoff安全，相对其研究初始2新增1；原49却34成功，失去5个T550成功ID。T592五个固定同场景前后测量发现，其中4个在T562学生初始化时已失败，只有dev_task6_13从成功变碰撞；不能把5个全部叫本轮新遗忘。这个小panel不证明其他44项没有变化。
- sector1_offset1两模型都先接近目标后驶离；最新最近决策端点约.285m，未进.2m目标集，后约32s走远。不是到达后漏记、没有目标输入或单纯“时间不足”的证明。余下历史未过的9个field没有本次完整原因归因，碰撞/超时不能代替路径分析。

## 核读的一手方法与真正适用边界

### DAgger/DART：学生真实偏离与恢复数据覆盖

[DAgger，AISTATS2011](https://proceedings.mlr.press/v15/ross11a.html)及[论文Algorithm3.1](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)要求在当前策略实际访问状态取得专家动作、累计历轮数据。不是只反复拟合专家原轨迹。[DART，CoRL2017](https://proceedings.mlr.press/v78/laskey17a.html)优化示范期扰动以近似学生误差，迫使专家展示恢复；[作者GitHub](https://github.com/BerkeleyAutomation/DART)README确认公开仿真实验，依赖老MuJoCo版本，不宜直接污染当前环境。

最适合当前学生近目标驶离和实际偏离后的控制覆盖。Stage已有纠正尝试，不能把DAgger首次提出或说以前未做。真正待补是这次候选自身状态上的有效完整恢复，而不是同一旧prefix重复一次。教师从同一物理前缀、原时钟/执行器、自己的因果hidden接手；实际无接触GOAL才作成功恢复数据。无法恢复则该方案前提不满足，不给失败教师动作贴专家标签。

### CLEAR：旧经验也学习回报，不只静态动作保持

[CLEAR原论文，NeurIPS2019](https://proceedings.neurips.cc/paper_files/paper/2019/file/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Paper.pdf)§3将新经验on-policy学习、旧回放off-policy学习与行为/价值蒸馏结合。它提供缓解遗忘的基准证据，没有有限回放逐任务零遗忘保证。[CORA作者仓库AGI-Labs/continual_rl](https://github.com/AGI-Labs/continual_rl)包含CLEAR基线；它是后续基准作者实现，不是CLEAR2019原作者代码。已读clear_monobeast.py，包含replay和cloning队列。

Stage静态39history动作监督或新鲜多任务PPO不是完整CLEAR；旧教师/旧policy样本不能直接作为新PPO的on-policy梯度。此方向需合法行为密度和off-policy校正，或在仿真可访问任务上分别获取新鲜RL样本。不能再把“加100保持loss”叫CLEAR实现。

### RLPD：离线先验与真实在线RL联合

[Efficient Online Reinforcement Learning with Offline Data，ICML2023](https://proceedings.mlr.press/v202/ball23a.html)和[作者仓库ikostrikov/rlpd](https://github.com/ikostrikov/rlpd)提供可复核的SAC离线/在线结合基线。正文§4讨论对称采样、LayerNorm等设计。已核读[train_finetuning.py](https://github.com/ikostrikov/rlpd/blob/main/train_finetuning.py)：默认offline_ratio=.5，在线与离线batch拼接再agent.update，不是仅用示范BC。

它与当前“有成功先验、还需新状态探索”的问题相关，但不是汽车窄通道或24场景零遗忘算法。StageT336已尝试过RLPD启发的SAC/反向课程，未成功，动作接近饱和是相关负证据，不能换名重复同配置。若采用，须与旧pilot明确区分当前完整数据、动作分布支持、序列Critic/target语义、有效在线探索和旧任务覆盖；先冻结预算及强基线接口，不能只换算法名称。保持部署单Actor可行，但训练有Critic。

### 可塑性：有顶刊实证，当前只能是待验证机制

[Dohare等，Loss of plasticity in deep continual learning，Nature632:768–774(2024)](https://www.nature.com/articles/s41586-024-07711-7)研究长期学习中获取新能力的退化，并用continual backpropagation选择性更新低效单元。[作者主页](https://shibhansh.github.io/)指向[官方代码](https://github.com/shibhansh/loss-of-plasticity)；已读lop/algos/cbp.py，梯度更新后执行generate-and-test，含replacement_rate/maturity_threshold。

不能混淆“学不会新数据”和“学新后旧闭环失败”。T591联合表示能学会3个恢复，已有非零学习能力；T592还分离了初始化缺陷。目前没有与fresh-init匹配的长期学习速度、低利用单元或表示有效秩证据，不能断言可塑性丧失是根因。不直接在已成功Actor上随机重置；若后续验证需要，单独研究副本、保留测试和匹配对照。代码主要前馈网络和PPO基准，不能无审查套到现GRU/sine学生。

### IJRR的自适应on-policy正则

[Lee等，Dynamic regret convergence analysis and an adaptive regularization algorithm for on-policy robot imitation learning，IJRR40(10–11),2021](https://journals.sagepub.com/doi/10.1177/0278364920985879)讨论策略变化导致的访问分布非静态和AOR，实验cart-pole/locomotion。其条件化收敛分析不是每个导航任务保持保证。期刊条目和正文摘要可读；未核验其独立官方仓库，不虚构GitHub。它提供“为何反复纠正会抖动”的研究依据，但不因此建议再扫正则权重。

## 现在的推进选择（草案，无新训练）

1. 恢复保持15field成功和最近3handoff的能力整合目标，明确不能把T59114field候选晋升为基线；研究副本可以保留作数据生成来源。
2. 一条固定支线解决已证近目标驶离：采集当前候选实际近目标状态的合格恢复，学习后检验原起点，保持原.2m/期限/物理约束。成功不得仅由换阈值或加时间获得。
3. 另一条固定支线针对原未过field而不是又修成功ID。先选一个未消费新机制的困难任务，获取合格完整恢复或用合法新鲜RL探索；若没有正例，不把BC重复次数当有效困难任务训练。8个offset3全失败提示目标支撑缺口，但不证明同一几何/算法原因。
4. 训练的采样单位按任务和因果序列组织：旧成功原轨迹、当前学生偏离后的恢复、困难任务新在线数据均需实际覆盖。监督项与RL项语义分开；新旧RL回放如采用CLEAR/RLPD应落实其估计器，不能混入旧PPO。
5. 验收按ID报告保留15、恢复3与新增原失败任务，扩大到24/49；之后才完整407、未见layout/noise/机器人动力学和实机。局部恢复成功不能替代从原起点全程导航。

这是比换权重更具体的可证伪路线，不保证下一轮成功。文献组合不是原创；可研究命题是“在有限纠正预算下，有效恢复覆盖如何提高闭环鲁棒性并减少旧能力损失”。需与DAgger、示范辅助RL、joint replay基线在等物理预算/等数据量下对照，跨布局、多随机种子、实机验证后才谈论文贡献强度。
