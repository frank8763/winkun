# T694：完整旧历史的有限函数兼容性复现（准备，未登记/启动）

用户在状态汇报后明确要求继续思考。依据 EXACT_COMMAND_RETENTION_PRIMARY_REVIEW 的范围澄清与 T546/T547 正证据，选择一个有限监督整合可行性问题，不另造强化学习算法、不恢复 T693。

固定11906输入零空间方案未准入；本项明确**扩展静态神经读出**，原120输入/64GRU和T501参数全部冻结。一个前向图同时计算原网络与紧支撑径向神经读出，不读取TaskID、世界路径/路点、教师或外部控制器；1405条固定源决策最多1405中心，新增至多2810监督系数和固定中心/半径。这个表示具有显式记忆/插值性质，不借“神经网络”措辞掩盖此限制，不称原创、端到端RL进步、鲁棒或未知地图泛化。

## 唯一可行性问题与既有对照

从真正保留的 T501 出发，能否在同一个Actor内严格保住原384个成功起点，并兼容全部六条已审计的新增原始起点成功历史？这修复 T550所暴露的保护覆盖不足和继承未晋升父模型问题；不是把 T546 小面板当全407通过，也不取消 T668 失败恢复门。六源是 T681 direction00_minus 的原始自主成功，不是任意学生失败状态的成功恢复标签。

这是**有限兼容性复现**，不是容量/算法/样本效率优越性对照。保留T501及其已验证完整任务身份作为历史基准，不重跑已消费的相同模型；明确无新鲜匹配Adam对照，因此不得归因“径向法优于SGD/PG/BC”。只有完整同候选384全保+新增+407+独立audit通过才研究保留。本项若无源/保护分离或精确命令合同则在登记前停止，不通过另外一轮诊断、改变中心/精度/支持规则救。

## 数据、函数与精确条件

保护集合：T691/reference_contract中全部384 origin=T501_original_success 的完整确定性历史，逐sourceSHA和原始起点ID核对。新集合：原T691已经共同锁定的全部六项 field0_1/0_2/1_1/2_1/5_1/7_1，242/422/193/199/187/162决策，共1405；相对T501六项新增，相对原387仅三项新增，不称六个未见任务。原初态/scene/goal/clock/cap/contact/0.2m门均不变。

每条历史在当前T501中从零hidden按**真实batch1 step**重算自己的carry和归一化输入，f=concat(h64,n120)。不从来源直接复制hidden，也不把batch近似均值当实际命令。源模型只变adapter，因此固定body特征坐标一致，仍须逐step核验。

中心取全部源决策特征的精确去重。仅在输入逐坐标相同、目标float32命令逐位相同、原T501均值逐位相同才合并；任何冲突拒绝，不丢样本。每半径固定 r=.49*min(距所有384保护点最近距离,距其他不同中心最近距离)，沿用T549已执行支持规则。phi=relu(1-||f-c||²/r²)²。固定最小距离>1e-10，所有保护输入phi必须**恰为0**，中心矩阵必须**逐位I**（condition=1）；否则停止，不调r/中心/阈值/solver/ridge。

恒等中心矩阵只需一次监督系数计算 weights=源实际float32命令的float64表示−原T501 batch1均值；无Adam/PG/Value/Q。径向分支phi全0时使用原网络原算术结果（torch.where），避免加零/改变求和顺序。所有原384命令与六源命令必须在新Actor真实batch1执行下逐位相同、carry一致；小MSE/数值距离不等于通过。系数实际提交、shadow、独立重建各计一次Actor学习，即使没有optimizer.step。

## 后续执行门（尚未实现/登记）

先在docs保留几何/源重构准备，不写runs、不实际提交系数/导出候选。随后完成真实loader/canonicalSession/worker/审计/退出/中断入口0native0optimizer预检；合成基函数fixture和零权重扩展初始函数应与T501逐位一致。再独立worstRESERVED、GitHub精确树且source shell退出后唯一launch，不把准备当学习。

拟两个顺序child：fit→shadow→独立重建/命令验证（main1200audit900），evaluation（main5700audit600），各<2h，root10200秒有限调度，原campaign stop从严。实际拟合前记录共同几何与源SHA；拟合拒绝模型保留，不能提前宣称准入。fit完成即冻结模型，原55full55exact全32旧保且新增才补余352同candidate407；独立原生审计/关键exact和完整384逐ID包含门不变。所有new6/olds生成原生physical arrays都与各自源比较，ownership/Actor密度不同的hold摘要不与跨模型源强求相等。新source/旧source全积分、命令、终态必须核对；余352可依现有compact+fullhash合同保存。

最坏评估沿用13,625,321physics/468cmd；fit0物理/管理命令另按实际入口6核定；Actor live/shadow/independent各1、Value0、额外optimizer预检0。CPU14准备/拟合审计、8/10/12 native，400%Nice19/4GiB/no swap/CUDAoff，global10GiB/free40GiB，拟root512MiB。全部预算、schema、源依赖和有限deadline仍须完整真实入口后重核；目前无RESERVED/run/protocol/launch，不引入任何已锁定旧deadline延长。

准备实测：完整384共77589个batch1保护点、六源1405点，最近old/source距离1.8424388913、最近不同source距离.06939471270；全部保护phi恰0、K逐位I/condition1，0个精确重复，尚未提交系数。准备mainbody351.832s/峰值1103687680B；故在首次登记前把拟合/独立审计工程上限固定1200/900秒，评估5700/600，各child<2h，root10200秒含1800调度余量；无旧deadline更改或科学阈值放宽。

完整零更新入口包含实际1405单元零读出RAM序列化/真实loader、batch1初始函数、worker/canonicalfactory绑定、保存真实历史的密度/奖励/carry/源物理hash审计及损坏hash/ownership拒绝，全部PASS。首次最后metadata漂亮排版2132668B超2MiB而拒绝，0拟合/0native/0optimizer，原error及源码快照保留PRE_REGISTRATION_NOTES。仅改为紧凑JSON1876550B（parsed配置相同），不重跑已消费几何/入口检查；正式登记两份protocol逐字相同紧凑序列化。
