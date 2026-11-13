# T523 首次执行双任务成功/失败联合PG

2026-11-13用户明确“下一步开始”，疑似灾难性遗忘。T519研究12/24旧7/8、T520旧8/8但field11、T522BC旧8却field9，是共享参数更新后已知能力回退的见证；未因果排除闭环分布漂移或小动作敏感性，不定性为唯一根因/场景不可达。

T522全部科学及driver已退出、两审计PASS、77文件完整远端逐成员SHA补过；原pending错误/原报告不覆盖，见TRANSPORT_RECOVERY。T522第一joint_pg的8条原轨迹0 Actor更新，完整行为策略与本轮T519初始化的Actor/归一化/GRU/噪声密度完全相同，所以是本轮首次优化的同一行为策略batch。不是新增采样，不是8个独立复现实验，也不混入T522 BC策略的轨迹。锁定原8 full120/history/rawaction/atom/logp/native/reward SHA，逐行密度历史复核，全8保留（旧0GOAL3接触1超时，新4GOAL）。去掉“每任务需成功”资格门槛，合法失败轨迹同样进入同行RTG。无更换seed/补采样/网格。

两个任务各4同行，组内物理时间RTG和三同行基线/RMS、每回合等权，最终任务各1/2；完整当前history BPTT重算PPO numerator。全64772 Actor可训，完整两组Adam继承T519；sigma.2、lr3e-5、原奖励/时限/容量/soft保持/clip/.01KL保持，最大16live16shadow；独立audit再最多16历史丢弃重构。Value及其Adam冻结。记录第一实际更新前两任务PG梯度范数/余弦，只是描述证据，不当干扰因果证明。不新增EWC/Fisher/PCGrad等变量。

候选必须完整old8+field24原起点自主运行，全部32精确重生。old8全GOAL、T519原12field成功ID全保、field>=12/contact<=9/无无效删失才接受；否则整个ActorAdam回退T519研究初始，独立接受最佳T50711+old8不动。未执行old275/six54/eight54/fresh16/78=null，不称全场地全通过。

这是用户新授权的首次联合PG，不重启T522/不扩其已结束两分支。新预算68全部命令（64真实评估+4父worker会话），零新训练采集；物理精确上界BUDGET，main1500/学习600/audit300/备份600秒。沿用午间13:17全部截止，13:06原生/12:57学习更严，完整余时不足不启动。新own6MiB/global1GiB/free40GiB，CPU14/15总1核idle/nice19、GPU关闭，不查看Phase。预检零物理，原预检在4 synthetic discardedAdam后因历史载入器仍绑定旧Actor而拒绝，0原生/0真实批优化；旧源协议错误及clone保留preflight_failed_001。仅修复actual t501 Actor loader绑定后fresh4，共最多8 synthetic discardedAdam/wholeAdam回退和导出精确、同8原批密度/分组复核。科学本地离线独立，唯一driver负责结束/audit/manifest/网络备份与源码。所有原始失败保留。
