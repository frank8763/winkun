# T473：旧15固定偏离状态的完整余时域动作优化教师

用户继续授权；T471同状态固定T255reset教师恢复旧03成功，旧15仍接触，原327carry也已失败。T471已结束审计/完整原始远端逐SHA；原T470/T472学习门未过，仍未运行。新机制直接优化完整剩余物理控制时序，不能称续跑神经恢复教师、扩原轮数或扫接管时刻。

## 固定算法与原生合同

只做old dev_task6_15；A325作为冻结反例前缀/影子Actor，不晋升或初始化未来学生。每pass从原起点自主20决策=4秒/2000物理步，和T469实际源的full120/动作/history/hidden/203积分精确。原cap23334，剩21334物理步=214hold，最后34步；clock/goal/真实速度/执行器/previous控制/history/学生carry全保留。

prior是已有safeT255从原起点旧15安全GOAL的193个实际控制，原字节SHA锁定；它不是新状态的合格控制证明。相位CEM借鉴既有T391纯控制参数化：16节点×drive残差/单调相位，phase=累计最大值(20+axis+5*clip(phase_delta,-4,4))，再clip到prior全序列范围并线性插值，drive加残差，最终命令clip[-1,1]。零theta准确按原时间推进，prior结束后夹到最后控制，继续原cap；不新增时限。

固定seed20261110473，初始sd[.12,1]、最小sd[.03,.2]、最多3代每代12（mean/zero或best加5对反向高斯样本），elite3、均值/标准差更新alpha.5；不是超参数网格。评分预声明按真正无接触GOAL/到达物理时间/动作平滑优先，其次无接触末距，最后接触末距；同分候选ID。首个实际safeGOAL立即停止搜索，否则最多36。选择已完整测试的整序列，不执行未经验证的加权均值。

两pass独立从原起点重生相同A前缀并捕获完整快照；每个候选从同snapshot恢复，真实native预测都计物理。学生Actor每个teacherhold仍actor_active=True递推hidden/记录密度，强制动作ownership=1，绝不是onpolicy样本。第二pass相同seed/自适应/停止点，逐候选full数组/积分/控制/成本精确；各pass再执行实际选中完整序列并对比，前缀不算新独立数据。不能将任一预测成功叫学生自主通过。

## 预算与停止

32MiB own（36×448KiB原始候选full上界+两selected full+partial448KiB+协议/快照/预测/控制/manifest余量），global1GiB/free40GiB不扩。最坏全部物理=2×(2000+37×21334)=1,582,716，守卫1.6M。main最多77命令（1setup+两pass各1prefix+36candidate+1selected）；零原生预检2命令另计，79≤80守卫。控制CEM不是Actor或Adam更新；本轮0Actor/Value/optimizer，禁止backward。

main3000秒/audit900/backup900/source180/all5100，更早绝对明早11月11日07:45原生/08:30全停从严，属于原夜40M物理/2000命令/14400main秒剩余，不加新额度。T468+T469+T471已11,244,734物理/824命令/1254.073802main秒；本轮完整最坏累计12,827,450物理/903命令/4254.073802main秒。首次native以前预检snapshot捕获恢复、完整prefix与zero-theta/候选随机/分段时间、容量，以0物理/0Adam执行；不提前做搜索。

全部分支原始保存；无safe控制是正常负结果、0BC不追加同机制预算。源、恢复、数值/数组/记录合同失败保留partial停止，不重启目录。main active单writer，不改源码协议/外写runs/并发审计/source。结束全部进程退，再独立重算所有控制、CEM自适应、每原始history/密度/奖励/计数/选择和两pass逐位证据，唯一RAM归档/全远端每成员SHA，精确source树。未备份明确REMOTE_PENDING，不删唯一数据。

## 条件后续与论文边界

仅教师实际safe+exact、audit及完整备份通过后才另注册联合神经学生学习；所有已有03/00/21成功恢复与wide真实保持数据可在新合同中明确使用。未来学生仍从原327初始化，单120因果真实GRUcarry Actor直接输出动作，部署不加入地图/规划/路点/运行教师/路由。CEM/相位配准/GPS式教师训练已有研究，非原创/安全证书，控制优化成功不保证学生能学会。完整329、16fresh、78独立原生重载/audit/备份门均本轮未执行，用户看过六米后再8，本程序无8启动。Phase完全隔离。

相关项目内已核一手证据与负结果：narrow_20261105/T381_TEACHER_LITERATURE.md、T382_MODE_PRESERVATION_RESEARCH.md、NEXT_AFTER_T382.md；本轮不复用T381 cap40000/hidden16/fixed070 Session，而用当前E.L.Session(cap23334/hidden64)。T391曾找到教师恢复但学生BC失败，保留其限制，不把换CEM名字当解决端到端。
