# T312：真实多目标在线MC / GAE配对pilot

独立有界协议，非T287/T291/T309续训练。冻结五米safeT255保留，两新研究副本从同safe权重/freshAdam开始。Stage/Phase完全隔离；部署仍单120输入hidden-reset Actor，无地图/规划器/路点/运行时教师或路由。

## 比较和数据

两臂同原120容量、物理/奖励/时限、PPOclip.2、Actor学习率/soft100、全批k3/logratio及old20原生接受；仅优势为完整MC−V与物理时间GAE(lambda=.95)。共享改变是实际多目标数据、rho0 iid探索（沿用T309而不是旧AR）、新神经V、普通混合PPO不使用PCGrad，因此与历史MC/RLOO不是单因素因果对照，更不是标准PPO/GAE原创。

固定T309八个train目标（L/R×.75/1.5/3.5/6），从原base6物理起态、原56s cap开始，reset前史对齐当前实际最终目标；其后按真实历史滚动。每回合目标不变，不是将一次六米导航拆运行时路点。旧275/原六米54最终按原native历史协议，不能用新初始化替换旧验收。目标均已消费开发，不称未见泛化。

每臂两轮、每轮每目标2独立创新draw，最多32训练回合/臂，共64采集+64全数组/终态积分精确重生。第一轮两臂同policy/创新流16轨迹全数组须相同，物理计两次，不当32独立样本；之后各自真实反馈采集，不能重用另一臂离策略数据冒充on-policy。

## 价值与Actor

每批16轨迹按draw0/1两fold整轨迹交叉拟合：每fold8训练/8验证，全120输入64–64状态V、trainfold统计归一化、500Adam，live/shadow各完整拟合，预测/模型/Adam逐位相同。V不接动作梯度、不称Q/安全概率，不复用T292ridge或失败T309占用头。

进入Actor前两臂都须：留出goal/episode-equal MSE≤零V的90%，每goal≤110%+1e-12；每轨迹完整参数零MC/MC−V/GAE三套score梯度（16×3=48），真实均值/logp误差0，MC−V与GAE每goal两draw样本梯度方差均≤零MC的2倍+1e-12。报告GAE/MC均值方向，不以lambda<1估计器方向相同冒称无偏；两样本pilot筛选不产生95%统计或普遍降方差证明。任何筛选失败停止Actor阶段，不调V/lambda/seed/epoch。

通过后每轮8固定PPO minibatch（16轨迹各16行=256），两臂共同零MC RMS尺度≥.05，不做优势均值中心化。真iid boundedGaussian(.035,.03)，canonical batch1真实边际密度，不拿旧AR logp；每次完整数据k3≤原.02、max|logratio|≤1，失败停止整块并回退Actor+Adam，不扫stepfactor。数值通过后真实old20全安全才保留，否则整块回退；live/shadow参数与Adam逐步SHA一致，samepolicy guard缓存不当独立物理验收。

最大32live Actor+32shadow Actor；4000live V+4000shadow V。中间/拒绝候选保留完整训练数组、确定配方、逐步Actor/Adam SHA，独立审计精确重建；不为每步写大快照，也不删除既有失败模型。两臂最终Actor/Adam及最后V/Adam保存读回。预检6合成丢弃Adam/0原生是工程检查，不当导航更新。

## 预算与验收

- 78零物理geometry（8实际目标+54dev+16fresh）；128训练/重生；最多80旧guard、40轮中心、658最终old275+six54、32条件fresh、156条件独立native重载，共1172全部命令/32.816M全部物理。
- main7200秒，末1800保留验收；audit3600秒/0native、最多8064历史丢弃Adam重构。实际main落绝对截止，不延长/不重启目录/不加seed轮数。
- own43MiB：64full×232KiB=14.5MiB、4partial约1MiB、两最终Actor/V/Adam≤15MiB、四学习证据≤.5MiB、旧compact≤6KiB/新≤7KiB合计≤6MiB，payload37MiB；metadata5MiB+buffer1MiB。global1GiB/free40GiB，T311实际无损预留后才注册。完整训练保存120/十二数组与真实history3/创新，state-chain只消重复；评估compact保存动作/物理counts/终止/原环境reward及全数组/积分SHA，不能谎称所有评估完整120已落盘。
- 每轮旧八+原六米两中心分列。最终两臂都完整旧275+原六米54；同候选全安全才16fresh，全过才78独立读回逐数组/积分精确。几个中心/随机探索/更低Vloss不能宣称六米稳定。失败/无效/删失/未执行分开，未执行不填失败率。

完整新source/inputSHA、预检、干净本地提交和GitHub精确全根树后唯一启动。运行中只有此writer，不外写其他runs或sync_source/改训练协议。结束父/worker/inhibitor/timeout全退后独立audit所有raw/partial、计数、目标历史/密度/ActorVAdam配方与模型SHA，完整原始RAM远端逐成员SHA读回再source同步。六米若全门槛通过仍须audit/备份后另立7m协议，当前pilot不自动训练7/8。

这是标准在线Actor–Critic可行性pilot，32回合/臂和单训练seed不代表充分收敛或顶刊证据。现有PPO-style MC、V、无旧保持/RND负结果保留，不将成功希望当结论；引用依据见NEXT_ONLINE_LEARNING_DESIGN。
