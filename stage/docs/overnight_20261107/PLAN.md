# 2026-11-07 Stage离线通宵计划（T422）

用户明确授权通宵，并确认北京时间2026-11-08 08:30全部停止。Stage与Phase完全隔离；窄场景暂停，当前宽六米1.17m/80s/0.2m GOAL，安全T255不替换。今晚不是已结束T421的延长，也不无限重跑。根协议一次锁定，离线driver是唯一launcher/finalizer。

## 科学问题

T421临近目标60秒时已偏到车身侧向，24个六维开环提案均未GOAL。车辆负drive只是降速，不能倒车，旧日志轴距不等于原生FULLPLANT；不能直接断言物理无解。今晚改为在真实学生17hold=3.4秒偏离后，使用同任务完整成功参考的连续反馈控制。参考T366 wide_dev_3是既有成功历史原生精确重建，55.65秒GOAL，与当前任务初始120及203积分逐位相同，SHA/原独立audit/完整远端收据锁定。

## A：真实早期恢复反馈，零Actor更新

两臂time/progress：前者按真实物理clock对齐参考，后者只在上次进度至前方12索引的局部窗口以XY²+.04yaw²匹配、单调推进。进度臂不会因旧参考55.65秒到点而跳到目标。各臂完整原80秒时限，不resetgoal/history/执行器/hidden。每hold反馈横向、航向、纵向与速度误差；原控制feedforward加固定6gain，原生真实执行drive/steer限幅；接近目标降速，参考终点后转向真实goal，不沿切线继续追。反馈可用训练期特权位置、已有成功轨迹，部署绝不使用。

两臂各3代×12提案、elite4、同固定随机流，两遍完整原生精确重生，共72种控制增益、144提案执行；第二遍非独立新增样本。均值与方差仅用该代实际结果固定CEM更新，不追加seed/gain/轮数网格。真实GOAL无接触优先，按到达native时间排序；未GOAL按终点余距，碰撞最后。只选真实执行过的最优控制，再从原点完整重放全数组精确。搜索最坏6M全部物理/150命令/5400秒，独立零物理审计900秒。所有完整被拒120只在RAM双遍比较，持久化每候选实际control、pose、count、reward、flag、holdSHA、反馈progress/speed和最终积分及全数组SHA；不宣称compact可还原所有120。

没有真实恢复GOAL则不进入学习，写失败类型与原始证据，按有限协议结束。不能把“跑满”理解为重复无信息机制或杜撰持续训练。

## B：成功后才执行动作监督与闭环RL

从未晋升研究B初始化，完整14开发初始闭环（旧8、LEFT5、RIGHT1）精确核对历史。研究B原本旧7/8、LEFT4/5、RIGHT1/1，不把已有缺陷当本轮新遗忘。最多4轮；首轮复用刚真实验证的3.4秒恢复并标重复，后续从当前学生17/20hold交替的实际偏离状态生成反馈恢复。必须完整GOAL、无接触、全数组独立原生重生，才加入标签；按policyhash/first/arraydigest去重，所有物理重复照算。

教师没有神经latent，不能拿学生locations充教师标签。监督为实际有界执行控制：完整120序列和hidden carry后做suffix mask，内部动作平方误差、边界单侧hinge，只是确定性surrogate，不是teacher密度或onpolicy。保留全部初始自身成功history零KL、旧211任务平衡支持、既有左右完整成功及旧真实safe纠正；旧梯度优先投影不是安全证明。200BC/live＋200shadow逐步ActorAdam精确，训练encoder/core/extra共10722，base/head/adapter冻结。

BC接受须所有初始与已保留成功ID不丢、至少新修复一项；拒绝整ActorAdam回退。只有接受且旧8/8恢复才采4纯学生探索＋4全数组精确重生，8PPO/live＋8shadow，原奖励/原rawsigma/.2 iid真实截断概率/原PPO门槛；教师混合轨迹只aux，不进入onpolicy ratio。RL接受仍须所有保留成功ID不丢。连续2轮拒绝停止，不动态加轮、换seed、改门槛。

最后同一候选旧275＋完整宽54全安全GOAL后才16真正留出，再78独立原生加载模型、全部数组/积分精确；任何未执行是null，不填失败率。没有预实现7/8运行，真六米全过也只报告保留，不杜撰已进入全图。学习最坏25M全部物理/620命令/7200主管秒，其中5400学习，完整独立零物理Adam重构3600秒。

## 预算、离线及收尾

新增96MiB（search16、learning76、root4），全局完整预留991314409B<1GiB，free≥40GiB。搜索72compact各≤96KiB、选中full≤544KiB；学习最多14initial＋16train full各384KiB、3新反馈full各544KiB、至多22模型各512KiB、compact≤620×5KiB，另trace/日志/manifest预留均在96MiB内。历史数据按SHA引用，不复制全库，不下载全部冷归档。原失败与模型不删。

夜间≤31M物理、780全部命令（含10零native预检）、832live＋832shadow Adam上界、1664独立历史丢弃重构，预检8丢弃克隆另计。连当天既有17.399960M/1180cmd/6607.350091主管秒，完整最坏48.399960M/1960cmd/19207.350091主管秒，沿用50M/2600cmd/20000主管秒不扩。07:00停学习、07:30停原生、08:30全部停止，launch-relative10h与绝对截止更早从严；任一项启动须剩完整最坏窗口，合同异常停链并保partial，不恢复旧目录。

本机systemd inhibitor防睡眠、外部timeout和driver绝对闹钟，所有科学、条件分支、审计与本地SHA/manifest均不依赖网络/LLM/API/安装。结束后seal原始文件，RAM分包上传与完整远端读回可选：断网明确REMOTE_PENDING，不阻断本地科学或等待网络到超时。Git只必要源码/协议/报告/清单，二进制只Release。本地driver最后本地提交报告，在线才精确source根树同步；无网络明确SOURCE_PENDING。运行中唯一writer，不并发audit/backup/改源。

注册首次generic common.py冲突在0protocol/0native/0Adam前拒绝，源文本及错误保留registration_failed_001；唯一模块名修复后正式注册。零物理预检验证实际快照/真实初态/4worker取消退出、完整历史和8丢弃克隆成熟Adam隔离/回退、所有实际入口全局名。组件为经典反馈/CEM/示范辅助/梯度保护/PPO，联合pilot不能称原创算法、普遍保证或安全证书。
