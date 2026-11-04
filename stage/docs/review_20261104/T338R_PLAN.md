# T338剩余预算恢复：逐条数值入口

2026-11-04用户明确“继续推进实验”。原T338为FAILED，原源、协议、299原件及原总结只读且已完整远端SHA读回；补充PARTIAL审计/数值修正3原件也完整远端读回。

本恢复只接入已核验CanonicalActor的逐条cache/stats/logp/draw，修复worker batch1与学习端batch160舍入路径不一致；不放宽原严格阈值、不替换保存logp、不改分布公式。CAPG/PPO已有方法及文献/作者代码匹配、负结果、研究假说见T338_PLAN及RESEARCH_DECISION；工程修复不是算法贡献。

两臂seed、初始化、4轮×4采集、每轮8live/8shadow、学习率、奖励/时限/碰撞、实际动作尺度、初始全量旧任务、终末开发和发布门槛均保持。原T338未发生任何Actor/Adam更新，首批4条tanh轨迹和精确原生重生、完整old275及48/56控制直接复用并按历史数据标记，不能称新增采集或独立重复；首次tanh更新输入必须逐位匹配这4条真实轨迹。CAPG预检在相同固定状态上的新分布采样仅为丢弃克隆工程合同，不伪称新on-policy原生轨迹。

正式预检须同时覆盖实际worker内部逐条运算和全部640个真实原采样状态，tanh原所有draw字段精确；两臂各1live+1shadow共4额外丢弃Actor Adam，原预检4另计、合计8。回退Actor/Adam、更新前中断与冻结基座须验证。预检0物理/1geometry；新增geometry与全部丢弃优化器如实另列，不增加导航训练更新预算。

保留原**学习绝对截止15:43:26.350992、原生16:13:26.350992、全部工作16:53:26.350992北京时间**。时间不够则标记未执行/删失，不扩轮数或改seed。原已花288命令/4178976物理，合并总硬上限仍900/24M；剩余最多28新训练+28精确重生、16轮后确定性检查、54终末开发与1零物理geometry=127新命令，最坏4512000新物理，原64live/64shadow真实导航Adam总量不变。独立audit≤1200秒重构已发生live历史，非新RL，零native。

两目录输出合计≤原14MiB；新目录≤12MiB（原目录实际约1.47MiB），global≤1GiB/free≥40GiB。新28full×224KiB/4partial×224KiB/2模型Adam×512KiB/128compact×4KiB/metadata2MiB严格小于12MiB。原数据只引用，不复制或删除，旧275不重跑。

新目录stage/runs/t338_remaining_recovery唯一启动，不重启结束目录。实际worker使用CanonicalActor；source/input SHA与预检、干净本地提交、精确远端根树之后才launch。active单writer；末由唯一driver独立审计、全部RAM附件远端逐成员SHA和汇总源码同步。新driver只stage明确JSON/MD，避免原ignored日志gitadd错误。若FAILED，保留全部raw/模型/error，不继续恢复。

原研究初始old275为274安全/1超时，未因本恢复抹去；safeT255只读。宽80秒固定48秒恢复和原窄56秒原起点分列。本pilot不执行最终完整old275/six54/fresh16/reload78，不称六米稳定/7或8，不晋升。部署单120因果hiddenreset Actor，无地图规划器路点运行时教师路由；Phase完全隔离。

正式预检首次在0geometry/0物理/0Adam因generic recovery_worker与历史模块同名拒绝；原source/protocol/error/ledger保存在preflight_failed_001。仅改唯一模块名t338r_core/t338r_worker，科学参数和所有绝对截止不改，重锁输入SHA。

入口收尾仅调度修正：CAPG先于tanh以防截止前仍未执行新机制；新轮发车余时下限30秒，但worker和学习器仍受原绝对学习截止硬中断，未完成按行政删失，不计导航失败。两臂可能因墙钟不齐，不据此声称算法优越。通过的实际worker/优化器预检source保持逐字节，调度修改前run/launch/protocol/PASS在entry_finalize_001保留；只做AST/输入hash/协议复锁，不新增物理或Adam。
