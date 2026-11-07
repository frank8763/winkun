# T408：先兼容左右真实成功支持，再双向闭环强化学习

2026-11-07，Stage only。用户授权继续宽六米→七米→八米→全图；最后窄实验T404失败，窄场景暂停。T406已审计和完整原件远端核验结束，不续其两轮预算。

T406的初始RIGHT已经0/27，不能说它将T404的RIGHT遗忘。下一问题是：**以T404保住的右侧/旧八为起点，同一循环Actor能否兼容实际左侧五条完整成功轨迹？** 先以小验收回答这个问题，避免再次花一小时前后重复完整失败基线。

初始化冻结T404 final learned参数、freshAdam；6864个可训练参数、真实hidden跨决策携带。保留120因果输入、原奖励、Gaussian rawsigma [.2,.2]、float32边界尾概率、物理/执行器和原生终止。任务固定宽1.17m、80秒、circle shift .6；这不是已暂停的0.57m/56秒或0.70m/80秒窄任务。轮速5/force120不改，以后提速须另立真机可实现的执行器合同。

先从当前模型原生重建旧八8/8并与T404历史full-payload digest匹配，RIGHT原起点完整history/hidden/积分须逐位同T405成功原件。记录LEFT五起点当前表现，不要求未学先会。旧八当前实际观测与raw location作为保持支持；补211历史T255访问状态的T255动作参考，后者不是恢复专家。

固定一次200步联合巩固，不扫epoch/权重/网络：每步一个旧八实际history（权重1）、一个按八任务均衡的旧211参考（权重1）、一个固定轮转LEFT成功history（权重.5）、RIGHT完整成功history（权重.5）。LEFT五条各40次，RIGHT一条200次；左右总侧权重相等，不因左轨迹数量多而淹没右侧。目标为真实原始unclipped location，完整当前history BPTT；不将老轨迹当新on-policy RL。独立shadow逐张量/Adam/轨迹相同；记录左右增量梯度范数/cosine，只作描述，不是遗忘的唯一因果证据。

巩固候选必须同一Actor原生通过旧八8/8、LEFT五例5/5、RIGHT中心1/1，任一失败整Actor+Adam回退T404并结束该有界机制。没有200/500/1000救援网格，不延长预算。共同14门通过才freshAdam开始最多两轮真实PPO，任务预定为新wide LEFT_00后RIGHT_00，各4独立创新探索+4全payload/积分精确重生、8live+8shadow。每提案仍须共同14全安全及maxlogratio≤.5，否整ActorAdam回退；保留/拒绝、随机/确定性、历史重合分别计。

训练后从指定candidate文件读回两中心并精确重生；只有joint门曾通过且最终仍通过，才对同一候选执行宽54和原旧275。全部安全GOAL才消费16未见宽任务，再78指定最终模型独立原生读回/全payload/积分精确，及独立审计/完整远端备份后才能称本任务六米稳定。尚未开发独立7/8代码，不能把草案称已执行；全图需要更远距离和未见布局验证。

own15MiB/global全部runs≤1GiB/free≥40GiB；T407仅将60个已完整远端核验的纯字节重复NPZ改显式可逆引用，唯一原始字节仍本地canonical和远端，模型/真人/视频不动。正式预算由prepare.py静态核算所有分支：21个400KiB full槽（8旧八+8新探索+4partial+1预检）、6模型≤224KiB、compact≤5KiB/条、metadata1.5MiB；完整最坏容量通过才launch。≤800全部命令/18M全部物理，main7200秒（前3600学习）、audit1200、backup600、source120，总9120秒从实际launch更严。四CPU原生环境并行、单writer，不依赖LLM推进；正常结束唯一driver串行audit/归档/source。

科学范围：联合工程pilot，不是单模块因果消融、原创算法、安全证书或真机已部署。当前实际成功轨迹支持与joint门解决的是已发现的单侧初始化/支持缺口；若失败，保留拒绝模型及左右拟合/闭环反例，再决定表示容量或当前分布纠正，不能再无界加BC步数。

## 本轮一手方法核对

- [Policy Distillation, Rusu et al.](https://arxiv.org/html/1511.06295)：读3.3/4.4关于多专家合单策略与任务采样，支持训练期整合的可行研究方向；原文Atari Q网络/分布蒸馏，不证明本项目连续循环location MSE足够，也不消除学生分布偏移。
- [Progress & Compress, Schwarz et al., ICML2018](https://proceedings.mlr.press/v80/schwarz18a/schwarz18a.pdf)：核对第2节progress/compress及EWC保护；说明学习/巩固分离与旧技能保护已有先例。本轮使用显式历史重放和14项经验门，无其双列/EWC，不称完整复现或原创。
- T342/T365/T391已有混合蒸馏负例，T402R分离巩固仅恢复局部能力；故本轮先要求双向原生共同门，不把离线loss下降当成功。DAgger/HG-DAgger等近邻及访问范围见RECOVERY_PRIMARY_SOURCES.md，未声称全文/源码全面复现。

部署始终单Actor直接drive/steer，无地图、规划器、路点、运行时教师/路由；Phase完全隔离。目标/速度当前仍含仿真真值，真机定位/雷达/执行器标定、延迟噪声与独立实体安全验证未完成。
