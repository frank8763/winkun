# T517：逐更新输出硬约束，最多2048提案

用户11月13日授权“下一步开始，多跑几步”。Stage only，Phase完全隔离。T516离线512步后训练/同轨迹验证动作RMSE达标，但旧19和前缀保持不达标；全部停止/独立512重构PASS/18原件完整远端SHA。T517从同一接受父策略的新零输出修正开始，不继承失败T516候选/Adam，不重启旧目录。原最佳field11/24+old8保持不动。

与T516相同13986参数非线性logit修正、确定性seed20261113516、freshAdam0/lr.001、全父Actor冻结、相同20条完整history、30训练/29blocked验证标签、.5恢复/.1前缀/.4旧19均衡loss和所有阈值。唯一主要新机制为接受更新前的真实输出硬约束；提案上限2048，同时预算变化不能作单因素科学收益归因。无新原生/Value/PG。

每提案：从当前保留参数和完整Adam状态生成一次真实Adam位移，依序考察固定alpha=1,.5,.25,.125,.0625,.03125,.015625,.0078125。每次只缩放真实post-Adam参数位移，而不是原始梯度；所有旧19及同任务前缀单独各轴location RMSE≤.002且每episode maxabs≤.01，同时30训练标签的恢复loss下降>1e-12，才接受第一个满足者。验证29标签不参与梯度、线搜索、停止或候选选择。接受时保留该提案完整Adam矩与计数，但参数是缩步后值；这是显式带backtracking的Adam变体，非“重做缩步Adam”或TRPO。

若8个alpha全部不可接受，整提案Actor参数和Adam精确回退并停止，不清moments、不额外试seed/方向/更小alpha、不死循环耗时间。最多2048实际Adam提案/live，独立shadow同上，audit重构实际全部提案/每个trial/接受或回退；最大各2048+预检5丢弃Adam。保留的参数更新与实际Adam调用/拒绝数分开报告。上限不是强行运行完2048：没有可行下降步时必须停。

最后仍须训练+验证各轴executed-action RMSE≤.05，旧19/前缀分别location RMSE≤.002且maxabs≤.01才离线联合过。硬约束只覆盖固定存储输入，不能保证online状态、轨迹、碰撞或泛化；冻结父参数也不能保证组合策略在线不干扰。候选无论过否都保留证据，活动内存回零头，不晋升模型、不自动启动仿真。field24/old275/six54/eight54/fresh16/reload78均null，最佳11/24不变。联合过才另注册有界原生任务。

所有恢复/保持状态来源和原学生动作密度/teacher ownership合同继承T516。0物理/0原生命令，native setup/mj_step被禁止，main1800s/audit1800s/backup1200s，prepare science4800/all6600硬停；own10MiB/global1GiB/free40。CPU14/15总1核quota idle/nice19/GPUoff，不查看/改Phase进程。每个proposal最多8前向trial，最大16384trial/live；不是原生控制参数网格。每trial的alpha/旧worstRMSE/maxabs/训练repairloss/接受判定存事件，独立重构逐字比对。

预检5实际丢弃Adam=2live+2shadow+1强制拒绝，验证held标签梯度隔离、每个已接受step符合硬约束、强制拒绝Actor+完整Adam回退、保存重载/全序列特征一致/初始函数逐位。预检/源码提交/精确远端根树后唯一driver；active不改源协议或并发runs/audit/备份。结束独立Adam/全trial记录及指标/模型SHA重构、全部RAM原始远端二进制逐成员SHA和报告源码同步。
