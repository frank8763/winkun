# T625 固定目标Actor续行的回报学习与动作排序

用户已明确下一步开始、以后CPU/GPU加速；新窗口已批准19:00停学、19:30停native、20:00全停。现仅准备，新增Critic额度待确认；无run/launch，不复活T623/T624。日物理/命令/时间/磁盘总额不扩。

T623/624已验证同一状态固定T608续行的三分支真实回报，旧混合行为MC Critic首选+.1却真实最差。旧数据不包含目标策略一致续行时，MC不等于目标策略Q。本轮不从摘要得出唯一根因，不宣称传统Q已否定，不直接接Actor。

训练只用原T623三条完整轨迹：八米LEFT13，实际源随机轨迹前38决策作为因果输入，branch决策固定0/+.1/-.1转向、一个hold后全程同冻结T608确定性续行。监督仅branch起及其后实际物理时间MC回报，强制前缀不作Q标签。所有循环网络从完整真实120历史重算，不能截断/复制/清零接管hidden；branch是Q^T608(h,a)的一个实际样本，后段同策略状态动作值，有限任务终止含timeout不bootstrap。三轨迹不是三独立环境种子；仅一个训练任务/一个状态家族，明确小样本限制。

新采另一完整任务field_sector4_offset2的源T620A真实成功前38决策，三固定同幅动作/同T608续行，3原始+3exact/三CPU8、10、12并行，所有3完整保存/全memory reward ownership动作公式独立审计及数组/积分精确重放。该field4整个任务均留出，原T622 aware预训练也未训练此task ID，T625训练不读其标签或选择超参。冻结init为同T622 aware双Critic参数，重置Adam：标签策略已变化，非恢复旧Adam。两组同初值/顺序/数据/256更新。

A loss为每Critic .5后缀采样MSE+.5branch三动作MSE；每次全部3完整因果历史、每条16后缀时刻。B只加成对排序hinge：真实branch return差>=.02的动作对，margin=.01、weight1。后缀和branch各50%避免一个动作淹没在数百后缀点；非训练成功标签，不作Actor损失。torch.float64累计与GPU计算；输入保留原数组dtype后按原Critic normalization变双精度。每组256live+256shadow+256独立历史重构，两组共512/512/512，Actor/native优化器0、额外Adam预检0。

评价固定最终更新，不挑中间点、任务、噪声或seed。留出三动作真回报间隔>=.02时，两Critic都须预测同符号且gap>=.001；所有此类pair通过且有至少1有效pair，留出branch起后缀均值MSE<=.1且<=各自初始MSE的80%，才通过本有限校准门。没有有效pair记信息不足门不通过。3full共享一个源prefix不当独立泛化重复；新场地/扰动/多种子/Actor梯度稳定性均未测。门过也不自动更新Actor或报告导航改善，两个checkpoint都是价值研究结果。

native数据收集→native全部退出→独立原始审计+manifest→GPU两臂各live/shadow→退出→独立GPU Critic Adam/trace/最终预测逐位历史重构→manifest，才下一机制；driver独写。GPU使用前查其他compute，运行发现其他作业则让出；CPU总400%低优先，实际PID affinity必须8/10/12，parent恢复四核，Phase数据全隔离。

own10MiB最坏：3full+3worker partial各1.25MiB=7.5MiB，两研究Critic各.5MiB=1MiB，所有协议/trace/日志/元数据/manifest1.5MiB。全部原样保留、无二进制重复。物理3+3×64112=384672，命令6评估+3启动+3关闭=12；collection300main180audit，两学习各300main180audit，主管线总900/独立审计540，尾330，单块完整余额准入。日22M660cmd9000main50MiB、global1GiB/free40GiB不变。GPU等价预检0Adam，不用预检抵正式学习。

源码先精确GitHub根树核验，唯一systemd driver无需网络/LLM科学运行。全部writer退出后RAM原始归档逐成员SHA远端读回，网络故障保本地并标REMOTE_PENDING。没有可执行7/8全局晋升，不碰T608原34/49、场地13/24保留模型。

一手依据：[TD3函数估计误差论文](https://arxiv.org/abs/1802.09477)、[ICML2024 FQE分析](https://proceedings.mlr.press/v235/wang24be.html)。后者研究目标策略价值评估，不能将其理论保证套给本GRU/小数据先导；本轮用明确目标续行MC回归而非完整FQE复现。成对排序是受控消融，未证明创新。
