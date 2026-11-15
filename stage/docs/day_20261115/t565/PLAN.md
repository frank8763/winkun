# T565 多动作联合阻尼校正

用户要求“这样一步一步修太慢了，加大力度；继续”。在原已确认17:30全部停止和日19M物理/600命令/12300主管线秒余额内落实联合机制，不扩CPU/GPU或磁盘，不重启T564。学习16:30/native17:00/all17:30与相对更早从严。完整最坏main2400/audit900/backup600/source180/缓冲60余时不足不启动。

## 为什么改为联合求解

T564保留18次参数更新后全局peak .182749→.139496（−23.67%），24fit仍过，27RMS全过，但三条max仍>.1。相邻steer217/218误差梯度cos−.998879；修217会抬218，原单峰方向五档均无整体下降。不是严格输入标签冲突/不可表示证明，也不是车碰撞原因（0native）。只追单峰的局部方向效率已见局限。

同T564最终未晋升研究副本的权重初始化（原T550保留不变），同120因果观测/64真实GRUcarry/93186可训encoder/core/student，normalization/head/adapters冻结。27原T550成功history/6779决策不增加数据；完整history当前权重BPTT，forcedpulse监督排除但保留history。原clip/车辆速度/执行器/奖励/终止不变，0新RL/Value/Adam/教师采集。部署单Actor，无地图/规划/路点/运行教师/router/中心库。

## 固定联合残差方向

每次完整27历史前向，构造全部13558有效命令坐标的有符号误差e=clip(location)−保存实际float32动作。选择所有abs(e)>.1的坐标，若少于32则补最高误差到32；最多128，超出即合同停止，不无声漏掉难例或加容量。top排序由固定CPU确定计算，selectedindices记录；包含217/218这类近峰，不仅top2。针对所选e计算93186变量的有符号Jacobian J，保留完整当前history梯度。

内部目标残差b=sign(e)*relu(abs(e)−.075)，.075只是训练余量，正式max≤.1/RMS≤.02门不改。H=JJᵀ，lambda=max(1e−12,1e−4*mean(diagH))，一次求解(H+lambdaI)alpha=b，delta=−Jᵀalpha；这是固定残差正则化最小二乘的双形式，不是完整LM、不是所有坐标共同下降保证。lambda公式固定不扫阻尼网格，正则化防近反向梯度造成不稳定；不能证明已解决条件数/泛化。

||delta||上限.005，超出按单一比例缩放；用无momentum/无decay的SGDlr1仅作该计算位移的应用，不称SGD目标训练或真实RL。最多12联合方向/12线性求解。每方向固定1/.5/.25/.125/.0625/.03125/.015625依次缩放同一delta，最多7实际全history复查、不额外求解或optimizer.step。

接受必须原24history仍逐条RMS≤.02/max≤.1、全27最大historyRMS≤.02、全27最大误差比当前下降≥1e−6或正式27门全过。不要求每项数值逐项不变，非全状态保持。无因素过则整参数/SGD状态回退并立即停；27正式门全过即停。不重试同一方向、不换seed/阈值/阻尼/步数网格、不为填钟点无限重跑。

每次保存selectedindices、J/Gram/RHS/direction哈希、lambda/rawnorm、所有因素指标，J只RAM临时不占磁盘；完整源/输入/trace可独立重建。freshshadow逐位重复全部solve/参数/接受/trace，非独立数据，audit再完整重建。预检2联合方向+2shadow（4应用/4solve）已通过且两步经验接受；是丢弃工程预检，0native/0新保留导航。模型raw上界974768B<1.5MiB，真实sequence/step/密度/因果/参数回退/RAM重载PASS。

## 原门全过才自主验收

27保存history离线门全过，才2short500物理+2exact（完整history核对新worker），再同候选old8+field24的32自主+32精确native全数组hash/积分。原23自主GOAL ID必须保持；24拟合门不是24导航成功率。离线失败0native记录，缺测null。275/six54/eight54/fresh16/reload78/yaw不在本轮，不自动进新RL或晋升。即使32保持，也只是开发支持迁移资格，跨布局/实体车仍缺。

## 预算/运行

新增最坏3,203,204物理/72命令/2400main（相对1500学习）/audit900/backup600/source180；最多12live联合应用+12shadow+12audit历史丢弃，各对应一次solve，预检4另计，0Adam/0RL。日合并T555/T557/T559/T561/T562/T563/T564实际由BUDGET核账，不超19M/600/12300。主管线2400秒预计包含拟合/条件原生，超时保留partial，不延长。

own6MiB：Actor及stateless应用模型1.5MiB、双workerpartial2MiB、两short保存128KiB/原子128KiB、32compact384KiB、trace/元数据/log/manifest1MiB，共5.125MiB＋.875MiB余量。J最大128×93186double约91MiB仅RAM临时，完整history图/梯度RAM≤约4GiB，非磁盘缓存；不与Phase抢GPU，CPU14/15聚合一核idle/Nice19/CPUWeight1/IOWeight1、GPUoff。global1GiB/free40GiB，完整own+131072guard准入。

唯一driver本机断网可执行科学/独立audit/本地manifest；全native退出后RAM全原始远端逐成员SHA，网络不通明确PENDING；最后source精确根树。active单writer禁外改源协议/写其他runs/并发audit备份。有限方案完成/失败提前停，不承诺跑满17:30或必成功。

## 论文与反证

多残差联合方向可能缓解相邻梯度竞争，也可能正则化后无法下降、越过局部近似或损伤未选坐标；必须实测全history接受和自主门。残差选择/目标/方向/步长联合变更是工程pilot，非单模块因果/原创算法。Tikhonov/GaussNewton/LM已有近邻，来源SOURCES；不借其理论宣称安全/高维全局收敛或真实车有效。
