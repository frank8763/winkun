# T295结果：目标历史与first-hit合同成立，旧数值密度路径拒绝

T295 COMPLETE，16命令/307296全部物理步/179.749398秒、0 Actor/Adam/Critic更新。原8条T287行为轨迹完整物理重建，原全部物理数组/终态积分精确历史；原行为是3CONTACT+5timeout，未改写成原六米成功。另8次固定新目标记录动作前缀，8/8安全GOAL、0接触/超时/无效/删失/缺测。这些是自己的既有动作序列重放，既非冻结Actor新目标自主成功，也非新独立行为或六米通过。

所有13目标相关字段包括四帧history、107其他字段、真实2ms首次命中、动作hold内终止120及积分、环境reward/discount均通过原生与离线核验。每条首次命中都在hold内部，实际步数LEFT[12829,14759,11706,12123]、RIGHT[12122,12952,11527,12341]，不能用后来的整hold端点冒充终态。初始stale factory历史有真实生成位姿/目标/时刻见证；新目标初始历史明确属于由这些见证重建的反事实历史，后续由原生新目标实际生成。

原8条mean/logp全部逐位重算一致，原goal重要性比率1。新goal下8条均在现有AR terms的u转float64 CDF→ndtri路径被拒绝，未截尾/替换logp/改噪声。这不能证明数学概率为零：有限mean、sigma>0、action严格在(-1,1)的截断正态有正密度，CDF先取exp可能四舍五入为1。T295原代码与拒绝记录保留，筛查未过、不接Actor。

独立审计T295_AUDIT.json PASS（0新增native/optimizer），所有原父/inhibitor/timeout已退出；所有18原文件完整远端读回逐成员SHA核验，归档SHA 1162f3e3bef6811a5a0bc9ce58aa45c0e84ddbce0e7210fedb7eba88a01cbf22。[原始归档](https://github.com/frank8763/winkun/releases/download/stage-distance-2026-11-01/stage-distance-t295-1162f3e3bef6.tar.gz)，收据release_receipt_T295.json。首入口0步依赖路径错误亦保留，不重跑旧目录。safeT255/模型/高清视频保持。

接下来仅独立T296对数尾部复核相同8条/目标/阈值/AR分布，0native/optimizer。区分计算数值域与重要性统计支持，不借号补同机制导航预算。六米最新完整仍1/54，旧274/275；七八米未进入。
