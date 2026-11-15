# T564 局部尺度的最大动作误差下降

在用户明确授权的17:30下午窗口及原19M物理/600命令/12300主管线秒余额中，另立不同优化机制，T563结束不重启/不加其提案。学习16:30、native17:00、all17:30，更早相对限时从严；源码/协议/容量锁定、全根树远端核验后唯一启动。

## 证据与改变

T563八次全回退提案完全相同：0保留/0native，17.057935s；不能算8独立方向。三个因素的peak分别1.146774/1.280873/.638388（初始.182749），RMS .241605/.080932/.030396。4+6预检及8live/8shadow/8audit历史丢弃各自分列。零优化器DIRECTION_DIAGNOSTIC：freshAdam首参数位移L2=.09074231；globalpeak对该方向一阶导数−.968228，本地点梯度范数166.2153。支持测试局部步幅/非线性，非已证唯一因果/保证下降。

同T562完整history单Actor（120obs/64carry/93186trainable），初始化相同T562未晋升模型，head/adapters/normalization冻结，部署无规划/地图/路点/教师/router/紧支撑中心库。相同27T550原成功history，ownership强制帧不监督但保留真实前缀。当前权重全historyBPTT；不是缓存旧hidden，仍0新RL/Value/教师采集，执行器/clip/奖励/速度/clock不改。

## 不再用Adam大位移

每次前向完整27 paddedhistory，计算doubleclip命令与保存实际float32命令的全局最大绝对误差E；沿其自动微分次梯度g下降。SGD无momentum/weightdecay，lr=1，预先把grad缩放成位移：radius=min(1e−4, .25*max(E−.08,.001)/||g||)，delta=−radius*g/||g||。.08为训练内部裕量，不改变原.1验收门；若||g||≤1e−12或非有限则记录停止/错误，不伪造裁剪导数。SGD实际步数如实记录，0Adam。这个半径是局部一阶启发式，不是真Lipschitz/安全证书，不把它称为原Polyak算法。

每步仅一次SGD提案，固定1/.5/.25/.125/.0625缩放同一参数位移（不额外optimizer.step）。候选必须：原24合格history逐条仍RMS≤.02/max≤.1；全27最大historyRMS≤.02；全27最大误差比当前保留状态下降至少1e−6，或已经原27门全部通过。只保持门资格，不要求每项误差逐数值不增。全拒绝就整模型/SGD状态回退并立即停，不重复同一状态。最多128SGD提案；原27门全过即停。

max有多个活跃坐标时负次梯度不保证全局max下降；实测全history接受规则仍必要。clip区外梯度可为0，这一机制若因此停止就保留反例，不改STgradient。不同当前最坏项会切换，不将旧20决策固定视作完整误差集合。该机制同时换目标/优化器/局部尺度，是联合工程pilot，不能单归因SGD或范数限制，更不称创新已证。

同初始化freshShadow精确重复全部梯度/位移/接受轨迹，非独立新数据；独立audit再重构全部SGD和Actor状态。4预检丢弃SGD/0Adam：真实因果step/sequence/密度/模型RAM重载/整参数回退PASS，模型raw上界974768B<1.5MiB。继承报表中Adam等旧标签及未使用旧fit_lr/rejectstop字段已改为与实际SGDlr1/firstrejectstop一致，原source/PREFLIGHT/progress及0重算元数据纠错保留metadata_before_correction，不改已测试学习计算。

## 条件原生与边界

仍27每条RMS≤.02/max≤.1全过才2short500物理+2exact（fullhistory审计），再old8+field24的32自主+32全数组hash/积分exact。原23自主GOAL ID须保持；24拟合历史不是24自主成功。未过0native，所有自主/275/six54/eight54/fresh/reload/yaw分列null；不做新RL或晋升，不用低误差充成功。后续实际偏离恢复覆盖/跨布局/实机仍需要独立证据。

## 全最坏预算

3,203,204物理/72命令/3600main，学习相对1800；audit1200/transport600/source180，必须完整预留5580秒及缓冲。SGD≤128live＋128shadow＋128audit历史丢弃，4预检另列；0Adam/0新RL。机器BUDGET计T555/T557/T559/T561/T562/T563实际，不超原日19M/600/12300，零梯度诊断另记不伪装训练。

own6MiB：Actor＋无momentumSGD模型1.5MiB，双worker partial2MiB，两short128KiB/short原子128KiB，32compact×12KiB384KiB，metadata/trace/log/manifest1MiB，共5.125MiB，余.875MiB。global1GiB/free40GiB，完整own+131072guard准入。CPU14/15总一核idle/Nice19/CPUWeight1/IOWeight1，GPU关闭，Phase全隔离。

唯一driver自动学习/条件原生、退出后独立audit/本地全manifest/原始RAM远端逐成员SHA/source根树；科学不依赖网/LLM，断网待备份明确。活跃单writer不外部改源/协议/写其他runs或重复audit/backup。有限失败提前停，不承诺填满17:30或必然成功。已有标准最大误差/梯度缩放/信赖步近邻，来源SOURCES，不称安全或泛化证明。
