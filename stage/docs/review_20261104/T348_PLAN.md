# T348：用真实成功序列验证可学习循环记忆，而非只增加静态回归

2026-11-04 Stage only。用户要求利用新成功示范继续学习并解释速度。T347真人62.788秒安全到达、314真实决策、完整原生重放与独立审计/远端逐成员SHA已过；这是宽1.17m/80秒/wide_dev1(dx−.03m)示范，不是原窄六米自主通过。旧T344/345/346失败全部保留，不续原配方预算。T347车速慢因本次沿用wheel5/force120；该示范记录端点路径均速约0.097789m/s，全314动作drive=.999。T343力限干预支持驱动力限限制固定工况，但提速后的冻结导航仍碰撞。此轮不改plant。

## 新机制、对照与范围

在冻结安全T255（原old275/275、原six0/54）上建立**可训练**120→32编码器/GRU16/2动作纠正头。两臂同参数、同初始化、同数据顺序/权重、同600次Adam、同old与新序列，仅区别：memory跨每次真实0.2秒决策保留hidden；reset每次清零。头输出初始为零，初始动作共同精确安全基础。纠正在1.25*tanh(atanh(base/1.25)+head)坐标，执行clip到[-1,1]、float32；可学习头float64，冻结基础float32，以降低序列/逐步数值差异。此架构/数值路径/新数据均与历史不同，整体工程pilot；两臂差异仅支持该条件下的记忆比较，非原创方法或普遍因果结论。

**这轮是示范预训练，新增RL/Actor策略梯度/Q更新为0。** 部署仍单Actor局部120输入直接到真实动作，没有地图、规划器、路点、教师或路由。后续真实RL需独立注册，不能把本次监督更新当RL；失败不追加epoch/seed/头网格，不凭loss下降称学会。

旧资料为T256稳定五米开发211个真实回合，已核验每个offset的时间字段起于1，不拼接不同回合。旧安全Actor给出确定性目标。新5组为：T347新真人全程、T328真人全程、T331只标注第150决策后的真实人控（前150只作真实时序上下文）、T340自主中心、T342自主matched起态。人控失败T346不作成功专家。每步按固定seed选1旧回合（task按8循环均衡）+1新序列（5循环均衡），各.5损失，完整因果序列从回合零hidden重算并BPTT，不读取未来输入、不使用旧网络缓存hidden、不发明on-policy密度。固定lr=.001/600步每臂、梯度clip1；1200live监督Adam+1200shadow，audit另1200历史丢弃重构。预检8丢弃Adam另列。

## 一手资料核对与限制

2026-11-04重新检索[Ni等ICML2022](https://proceedings.mlr.press/v162/ni22a.html)、[作者解释](https://blog.ml.cmu.edu/2022/08/26/recurrent-model-free-rl-can-be-a-strong-baseline-for-many-pomdps-2/)和[作者policy_rnn.py](https://raw.githubusercontent.com/twni2016/pomdp-baselines/main/policies/models/policy_rnn.py)（本次实际阅读act的prev_internal_state传递、forward的T+1序列与mask片段）。作者使用分离循环Actor/Critic和off-policy RL，本轮只借正确时序基线要求，自己实现监督warmstart，不复制作者代码/声称完整复现。作者指出实现与上下文长度重要，并非任意GRU都有效。

[R2D2作者页面](https://willdabney.com/publication/r2d2/)及摘要强调旧参数表示漂移/hidden陈旧；本次OpenReview全文直链被浏览器验证拦截，不称已读全文。本轮完整序列每次从真实回合起点、当前参数重建，避免把旧hidden当当前状态；不实现分布式DQN/R2D2。此前DAgger/DART只提供覆盖动机，T345没有可用自动纠正、T346晚接管失败均作为反证保留。记忆与长期信用分配仍不同，本预训练不能证明长期RL信用问题解决。

## 零物理预检与工程合同

两臂初始动作与冻结基础一致；两步live/shadow各精确、source不可变；成熟Adam快照deepcopy加载、另克隆/拒绝/中断回退0额外Adam检查。真实新序列full versus逐步位置误差<1e-12、因果前缀一致、训练后改变hidden仅memory有作用；实际spawn worker逐步输出精确。7真实geometry（宽5/原窄2）0native。400决策无压缩NPZ最坏字节数须小于384KiB，不能靠零数组压缩率估预算。只能PASS、代码/输入SHA锁定/提交和精确GitHub根树后唯一launch。

## 验收和预算

初始共同冻结策略宽5只运行一次（初始两臂功能相同，不算两次独立样本）；每臂最终旧20/宽5/原窄2，再27独立原生模型读回，全部120/hidden/history/动作/积分精确。4个完整payload保留两臂宽中心与matched起态；其余55primary紧凑，54readback在RAM逐数组比较、所有hash与物理记录保留。逐个原生回合开始hidden=0，memory中途不清零；旧反应骨干仍冻结，并非全骨干RL。

7geometry+5initial+2*(27primary+27readback)=120全部命令，最坏4.52M所有物理；完整旧275/原six54/fresh16/reload78本pilot未执行/null，不能晋升六米/7或8。旧八中心与旧20、宽课程与原窄、两臂及已消费/重复分列。原窄严格完整门槛仍保持。

main3600秒末1200验收、audit1200、全工作7200秒，launch外部timeout3600秒TERM/1秒KILL更早从严。最多4worker，CPU0学习/CPU1–4原生，所有进程身份和心跳核验。own5MiB：4full各384KiB+4partial各384KiB+2模型各256KiB+55compact各4096B+metadata1MiB+logs256KiB，共5206016B<5242880B；全局newruns≤1GiB/free≥40GiB，完整最坏预留约1073607751B，仅约131KiB全局余量，不扩大或删除唯一证据。部分/删失原件分列，不作为导航失败率。

训练中单writer不改源协议/外写其他runs/source同步；整臂600live与600shadow必须精确一致，输入hash在更新前记录，所有load深拷贝；中断整ActorAdam回退。结束所有父worker/timeout/inhibitor退出后唯一独立重构1200历史Adam/完整raw/模型/源SHA与物理命令，全部原始RAM归档、完整远端二进制逐成员SHA，最后精确source树。原安全模型、失败和视频保持。
