# T567冻结研究学生自主三任务诊断

用户“思考怎么做，然后继续”后另立此诊断，不重启T562–T566，不再追加离线拟合。原17:30all、17:00native及日19M/600cmd/12300main不扩，CPU14/15总一核idle/Nice19/CPUWeight1/IOWeight1，GPUoff，Phase全隔离。

## 目的及门的解释

T566未超越前best，最近连续0自主native，离线门只是项目保守资格门，不是物理上禁止研究测试的定理。新协议明确允许一个**未晋升、未过离线动作门的冻结学生**做受控模拟诊断；不声称旧门已过，不绕门晋升、训练或进真机。此为独立研究问题：保存历史误差具体怎样变为自主偏离/碰撞/超时？结果无论好坏完整报告，不替代32自主/407/留出资格。

模型在新测试前固定T562：相对T564/T565/T566均值locationMSE最小、RMS余量最好，不根据新测试挑模型；SHA与policyhash锁定，无权重/优化器变化。原T550保留Actor不变。120因果obs/64真实GRUcarry、同原车辆/传感/geometry/goal/history/clock/速度/动作clip/奖励，无地图/规划/路点/运行教师/router。

## 固定3个任务与完整证据

sector0_offset2、sector1_offset2、sector4_offset2原起点/目标/原时限不变，全部已消费开发；对应T550原成功历史已SHA核验，仅作保存基线，不新跑教师。每个冻结学生自主1次＋同模型独立原生精确重生1次，共3unique+3exact，0新样本噪声/seed/训练。顺序锁0/1/4。一个worker串行，实际原生全计预算。

保存3条完整states120/nextstates/动作/native_counts/位姿/history/hidden/密度/innovation/ownership/初终积分/逐holdSHA及原模型SHA。每条实际worker记录全数组，独立audit重算每决策动作、真实hiddencarry、密度、奖励/history，并逐数组hash与积分核对3exact。初始native积分与对应SHAchecked T550成功trace一致。

描述量固定：首次按决策对齐偏离T550超过3cm或3deg；终态与最近记录端点距目标；按记录位移/物理时长的平均速度、<.03m/s时长比例；drive均值、|steer|>.99时长比例、末决策朝向目标角误差。不是连续native最小间距/真实轮速，不把记录端点当全步精确最小。

结局按GOAL/CONTACT/TIMEOUT报告；若最近记录端点<.5m且最终比最近远>.3m，仅标TIMEOUT_AFTER_NEAR_APPROACH描述，不认定因果。首次偏离对齐同初态/同决策时间，须原始history核验；不将后续不同状态当同状态对照。

## 计数/存储/时窗

最大2×3×64112=384672全部物理，6请求+4保守进程命令=10，main1800/audit600/backup300/source180（总2880+缓冲），学习0/Adam0/RL0。与原日实际累计以BUDGET机器核验，不扩19M/600/12300。主线截止min(start+1800,17:00)，all≤17:30。完整余时不足不启动。

全642决策的packed数组真实schema对n和n+1history均外推，加65536ZIP开销，938000B≤每full928KiB(950272B)，预检逐3source核验。原T555旧schema推导没有正确外推n+1，不能直接套其859664值，本轮正确938000。自己own5.25MiB：3saved full+1atomic full+1workerpartial=5×928KiB，metadata/log/manifest512KiB，共5275648B<5505024B，余229376B。只一个worker，不按双worker部分文件超额；模型仅引用不另存。不删唯一历史证据，global1GiB/free40GiB，完整own+131072guard准入。

首次预检因keep_rows精简记录无array_hashes拒绝，0native0optimizer，旧source/error保留preflight_failed_001；改为SHAchecked原数组计算初态digest，原成功自测row需由数组构造timeout/remaining等派生字段。只是诊断metadata适配，不改传感/科学任务/模型，恢复PASS。0物理sequence/step/因果/成功自测指标一致。

唯一driver本机断网可main/audit/manifest，全部native父worker退出后RAM原始全成员SHA远端读回/source精确根树；断网明确PENDING不阻塞科学。不并发写runs或改源/协议/重复audit/backup。诊断结束即停，不自动训练、教师搜索、晋升或进八米/全场地。下一机制须依据真实结果另立预算，不拿新编号续失败局部拟合。

## T567R入口恢复差异与合并计账

原T567 FAILED，0物理/0优化器/2进程命令/6.531445秒，尚未任何navigationrequest；worker数改1但READY等待仍2，IndexError。构造异常路径pool.close已join worker，原service已inactive，全部原件和远端收据保留，不重启。T567R新目录只将READY按len(children)等待；科学模型/T562三目标/所有描述量/原时钟/步数/精确重放不变，不续任何成功或失败回合。

原10命令计划未执行，新恢复max10命令加原2，合并12从原日600额度余额重分配，不扩大日总。384672物理不变；主管线预算1800再加原6.531445秒，原始每次真实计账。新own5.125MiB=5373952B，3full+atomic+singlepartial＋512KiBmetadata的完整最坏5275648B低于上限。原失败目录约60356B已计global，不删除。原零原生schema/selfdescriptor/时序检查恢复PASS，新增读取源等待遍历数为实际children的回归检查；不是实际native成功预检。

原离线动作门仍未过，只做冻结研究诊断，不作为晋升/真实车资格、不开新训练，不改17:00native/17:30all。
