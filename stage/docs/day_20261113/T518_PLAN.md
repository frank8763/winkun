# T518：活跃输出约束的局部切向更新方向

Stage only，Phase聊天/研究/模型/结果/进程完全隔离。用户11月13日要求下一步开始。T517已停止，实际3Adam/2接受1拒绝、19线搜索trial，live-shadow/独立全部历史重构PASS、18原件完整远端SHA，原最佳field11/24模型保持不变。不重启T517，不从其拒绝提案或已消耗2step候选续训。

新机制仍同T516零初始化13986参数非线性logit修正、seed20261113516、父Actor完整冻结、freshAdam0/lr.001；相同实际T514恢复59状态30训练29blocked验证、前166学生location保持、原19自身成功完整history、相同loss/阈值/8个alpha及2048提案上限。只改变“提案方向如何计算”，不扩已结束机制预算。不声称单因素导航收益，因为本轮没有原生。

每提案先在当前保留参数计算loss与真实Adam displacement及新moments；将参数恢复到该提案前位置（moments暂保留提案值），在此点对每个旧19/前缀episode检查各轴RMSE和maxabs。当误差≥各固定门槛的80%，成为活跃约束，计算该轴MSE梯度或该episode最大abs误差梯度，梯度L2归一；norm≤1e-15不加入。对叠成J的向量做float64 SVD，相对最大奇异值1e-8保留rank，d_projected=d-Vr.T@(Vr@d)。投影实际Adam位移而非raw梯度，所有梯度仅针对新13986参数；不读取验证teacher动作、地图/未来或taskID。

该等式切向投影仅限制活跃标量误差的一阶变化，不等于保所有旧逐点输出；非线性曲率/未活跃约束仍可能超限。必须继续T517原有实际输出硬检查：alpha=1到1/128八个固定值，旧19/前缀各轴locationRMSE≤.002且episode maxabs≤.01，并30训练teacher恢复loss下降>1e-12才接受首个。接受保该次Adamproposal moments与计数，参数为投影+缩步结果；不称重新算投影梯度Adam。全8失败则整Actor+Adam精确回退、停止，不新增方向/alpha/seed。每proposal全部basis labels/rank/方向保留比/正交残差、trial及回退写原始事件；独立全量重构。

最终门槛同原T516：训练和同轨迹held各轴teacher executed RMSE≤.05；旧19及前缀各轴locationRMSE≤.002且maxabs≤.01。所有都过仅允许另注册原生验证建议，不自动开仿真。本轮0physics/0nativecommands/0Value/0新PG，candidate保留但活动内存回零头，old275/six54/eight54/field24/fresh16/reload78全null。当前导航最佳仍11/24，不把有限stored输入约束叫闭环稳定、碰撞安全或创新。

最多2048live/2048shadow/2048audit实际Adamproposal，预检5丢弃Adam=2live2shadow1强制拒绝另计；SVD每提案最多60活跃scalar（每20episode2axisMSE+1maxabs），无SVD网格。main1800/audit1800/backup1200秒，prepare科学4800/all6600硬停，own10MiB/global1GiB/free40GiB；不足不启动、不删唯一数据。CPU14/15总1corequota/idle/nice19/GPU禁用，不查改Phase资源。原父模型/成熟Adam只读冻结保存。

预检0native，held-label梯度隔离、synthetic投影正交、actual两保留step/sameinitial/live-shadow、强制拒绝完整Adam回退、全部真实historycache与sequence一致、model保存重载。预检/本地提交/远端精确根树后唯一driver启动；active单writer不改协议科学源、不并发audit/source或写其他runs。全部科学PID退出后独立实际Adam/SVD/trial/指标/源模型SHA重构，RAM全原件远端逐成员SHA备份并同步报告。没有可行步就提前停止，不填钟点或谎报2048全完成。
