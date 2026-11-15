# T566 第一步：历史保持约束进入联合方向求解

用户在NEXT_EXPERIMENT_PLAN_AFTER_T565后明确“第一步开始”。新机制不重启T565、不追加其方向，仍日19M物理/600命令/12300主管线秒；16:30停学、17:00停原生、17:30全停，相对更早从严。完整准备/预检/源/容量后才能启动。

## 起点与工程范围

从T562未晋升研究权重和freshstatelessSGD初始化（同policyhash核验）；其RMS最大.018813，优于T565 .019995的余量。此初始化改变与约束算法联合，非单因素因果证据。T550原保留模型不变。120因果obs/64carry/93186encoderGRUstudent可训，原head/adapters/normalization冻结；完整27T550成功history当前权重BPTT、forcedpulse不作自主标签但保留history。执行器/clip/速度/目标/时限/奖励保持，0Adam/RL/Value/教师采集。

## 联合方向及保持约束

1. 构造全部13558有效命令坐标e=实际float32clip(location)−保存原动作；该cast由本机autograd按cast语义回传，非直通替代clip。选所有abs(e)>.1，补top至少32，最多128；超容量停止不漏选。
2. 有符号残差Jacobian J、b=sign(e)*relu(abs(e)−.075)、lambda=max1e−12,1e−4mean diagJJᵀ，d0=−Jᵀ(JJᵀ+lambdaI)^{-1}b，实际用solve不显式逆。一次残差线性solve计1。
3. 构造方向约束 Aδ≤c：每条27history的mean执行命令平方误差梯度，c=.5*(.02²−当前MSE)；原24合格history中当前abs命令误差≥.07的全部坐标，其abs误差梯度，c=.5*(.1−当前abs误差)。固定protected坐标上限128，超出停止。负浮点极小margin clamp0，必须原非线性资格仍成立。不声称保护所有未选坐标。
4. 将d0欧氏投影到这些线性不等式，min .5||δ−d0||²；行归一化（零梯度行仅c≥0可忽略），双变量λ≥0，固定256次循环坐标下降，无额外参数网格。每次λi=max(0,λi+(Ai*d0−ci−(AAᵀλ)i)/||Ai||²)。c≥0保证δ0可行，但有限求解未达残差不宣称不可行。
5. 计算投影实际primal violation、dualKKT残差、dual非负、complementarity和stationarity；全部有限且归一化单位容差1e−8，未达则不应用参数、记录停止。不可行合成问题须被拒，不能拿未收敛方向训练。
6. δnorm上限.005，正比例缩小在c≥0下保线性可行并再次数值检查；投影KKT只指缩放前，不称缩放后仍是d0最近点；但不保非线性。SGDlr1/无momentum/无weightdecay仅用于施加δ，无Adam。固定七因素1/.5/.25/.125/.0625/.03125/.015625，仍原24fit保持、全27RMS≤.02、全局max实际降≥1e−6或原27门全过才接受。不合格整参数/SGD回退，全部因素拒绝立即停，不重复同方向。

最多8联合方向，最多8残差solve+8投影solve/256sweeps each；保留原27门全过即停。freshshadow逐位重构全方向/约束/投影/trace/状态，audit再独立历史重构，不当独立样本。原Jacobian/A/Gram/duals只RAM，保存每项SHA及selected/protected坐标/残差证书；不落大矩阵盘。

## 预检与条件自主验收

合成已知投影、冲突不等式拒绝、真实history两次丢弃方向+两次shadow（首版4应用/4残差solve保留；注册前补全KKT/缩放可行残差再fresh4应用/4solve，累计8预检，原件保留preflight_before_certificate_completion）、参数拒绝回退、RAM模型读回、因果sequence/step/密度和输入SHA必须全过。真实方向若QP残差导致不应用，也仍需shadow输出精确、且不得把0步当成熟状态已验证。所有实际计数由PREFLIGHT记录。

只有27逐history RMS≤.02/max≤.1全过，才2short500native+2exact（fullhistory审计新worker），再同一学生old8+field24共32自主+32全数组hash/积分exact，原23自主GOAL ID必须保持。此为前述计划B的条件资格测量，不新RL或完整407。未执行275/宽six54/eight54/fresh16/reload78/yaw均null；拟合24/27不是自主成功率。

若拟合全过但32自主失守，说明固定历史迁移不够，不能回到无限追loss；下一另立学生实际偏离的已验证恢复覆盖。当前部署仍单Actor无地图/规划/路点/运行教师/router/中心库，不加相机或新信息。

## 全部最坏预算

新3,203,204物理/72命令/2400main（相对1500学习）/900audit/600backup/180source，完整4080秒＋缓冲。主line/审计物理不会重复未记录，解算/应用次数分别计；max8live应用+8shadow+8audit历史丢弃及各16solve，预检另列。前日实际4,848,104物理/216命令/3539.455209main，最坏合计8,051,308/288/5939.455209，不超原19M/600/12300。正式BUDGET机器重算，不以本文数值取代准入。

own6MiB：模型1.5MiB，workerpartial2MiB，short128KiB＋原子128KiB，32compact384KiB，trace/metadata/log/manifest1MiB，余.875MiB；global1GiB/free40GiB，完整own加131072guard预留。CPU14/15总一核idle/Nice19/CPUWeight1/IOWeight1，GPU关闭，Phase全隔离。新残差J128×93186double及约束A155×93186等RAM临时，不磁盘缓存；容量及时间预检不足不启动。

唯一driver本机断网科学、退出native后audit/manifest/RAM远端逐成员SHA/source根树；无网保留REMOTE_PENDING，不能称上传完成。active单writer禁源/协议外改、写其他runs、重复audit/backup或新机制并发。

## 限制与失败解释

投影是有限保存历史的线性化保持，不是精确非线性约束、全状态安全或跨布局泛化。约束源和目标反复消费，均为训练开发；带约束方向已有近邻，不称原创或借完整CPO/LM理论。若线性残差未达/零方向/全部实测拒绝，只否定当前固定线性模型/迭代预算，不推出严格无可行网络或场景无解。不能为填满17:30追加网格/轮数或放宽门。
