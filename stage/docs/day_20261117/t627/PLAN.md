# T627 CAGrad任务梯度组合对照（用户已批准，实际启动查launch）

要解决的可检验问题是多任务更新方向冲突，而不是再放大旧动作MSE。T626完整49为新增2/丢旧2，旧field4未参加本轮PG、旧八米LEFT13参加PG仍丢。旧函数MSE在初值梯度为零，不能替代旧任务回报方向。真实8任务梯度零optimizer检查已有18?具体负cos数量以PREFLIGHT为准；平均单位方向的最差任务dot=-.0369500，CAGrad组合后+.1812489，方向cos=.92924，固定零更新门已过。不能据此声称闭环保持保证。

共9任务、每组4完整轨迹。原T6268×4全部从完整远端原始成员SHA重新读回，仅初始T608行为对齐、旧数据不算新onpolicy采样，后续8步是历史clipped surrogate。新增遗漏旧任务field_sector4_offset2从原起点4条随机学生T608轨迹及4exact，两臂共用。新采是Stage单Actor真实经验，无teacher/no planning。所有真实120/前动作/GRU64+extra32+anchor32/history/奖励/积分保留，完整因果历史重算。所有成功/碰撞/超时照记。

两臂从同一T608与空SGD初始化、同9task数据/old33函数软保持/每任务latentKL.01/梯度norm1/8步及完整49+4keyexact；A逐任务梯度平均，B只改变组合为固定c=.4的CAGrad。每任务目标是该任务4条真实RTG/clipped-PG+共同旧函数保持loss。保持梯度已纳入每个任务梯度，再组合，不在求解后另加破坏方向；全向量norm裁剪仅正比例缩放，记录实际SGD参数位移和每任务梯度dot实际位移。两臂SGD无momentum/lr=.001，不用Adam再改变组合几何；这是匹配几何对照的选择，不声称CAGrad官方必须SGD。

CAGrad对偶：g0为9梯度平均，G为列梯度矩阵，A=G'G/trace(G'G)，b均匀，min_{w>=0,sumw=1} w'Ab + c*sqrt(mean(A)+1e-8)*sqrt(w'Aw+1e-8)。无SciPy、不安装依赖；用确定性simplex投影梯度/回溯解同一凸目标，FW gap<=1e-8且最多4000迭代，40回溯上限，失败停而不临时调容差。最终d=(g0+lambda*Gw)/(1+c²)，epsilon按Gram scale同比。借鉴官方mtrl默认实现rescale1，eps/求解器为本已明文实现变体，不冒称源码逐行复现。c0恢复均值、任务排列不变、逐任务均值梯度等于原标量目标梯度的零步检查已过。

CAGrad(c<1)不保证每任务dot都非负，后续步可能仍局部伤害任务。所有task-dot、负cos计数/权重/GramSHA/实际步长/latentKL记录，不能用代理loss替代导航。与T626的差异还包括field4覆盖和Adam→SGD，不把跨轮效果全归因CAGrad。两臂的唯一变量是组合方法，匹配最终原34保持+新增原起点才保留研究副本；拒绝整Actor/SGD回退，原T608与拒绝候选都保留。0应用不重复初始49。最终任务全49一次，不头两个失败就截断；纯预算/证据失败才停。407/fresh16/reload78本轮null、不全局晋升。

每臂最多8live/8shadow/8独立历史SGD重构，共新增16/16/16，0Adam/Critic/额外optimizer预检。优化器类别明文算Actor更新，今日Actor92→108须确认；旧Critic1120不扩。同型完整state/trace/任务梯度/对偶权重以及导出policyhash独立重建，不只核loss；旧32资料从远端再读，新增4亦完整SHA。新样本与重放物理全计、原样失败保留，无重启旧run/无参数或seed网格。

最大5381004物理/120命令：4新+4exact及两个49+4exact=114评估、3启动+3关闭=6；原日累计最坏16263683/581，低于22M/660，主管线1800，独立audit900，尾360，日main最大7902.31<9000。有效21:00学21:15native21:30all，不延长；启动完整学习准入1200秒，main1800+audit900+尾360总51分。

本地own6MiB：新采四full每1.25MiB短缓存，只在完整上传+远端RAM全成员SHA后释放，断网保当批停；原32不落盘。原生采集3worker，其最坏当前第四未完整partial+前三full与metadata预留<=6MiB。模型各229376B安全上界203055；双49compact各318273B全无损安全上界，runtime每file12KiB仍守；评估只2worker同时（第三空闲），最多2×1.625MiB partial，再加模型、紧凑资料、协议/原返回gzip/日志保持6MiB内。日50MiB/global1GiB/free40GiB不扩，远端新增raw+root<=16MiB，原32已备份不重复上传。只新批缓存已授权释放，唯一失败数据/旧模型不删。

CPU8/10/12原生三核收集、评估两核按存储限制、parent/audit14可用；GPU float64学/独立重构，其他compute出现让出，Phase研究数据不读取。唯一driver顺序main全native退出后audit及最终原始归档；native接收先无损保存，两个候选eval独立路径不覆盖。源码根树GitHub核验后唯一launch。当前仅准备，确认新增16Actor额度已批准，源码精确同步后唯一启动，不拿旧额度继续。

一手来源：[CAGrad论文](https://arxiv.org/abs/2110.14048)、[作者仓库](https://github.com/Cranial-XIX/CAGrad)、[mtrl实现](https://raw.githubusercontent.com/Cranial-XIX/CAGrad/main/mtrl/mtrl_files/cagrad.py)。见PRIMARY_RETENTION_REVIEW的CLEAR/DAgger反证与限制。PCGrad/GEM/函数投影历史已有失败，本次是新batch真实任务回报梯度的匹配CAGrad消融，不称首次想到梯度冲突或原创创新已成立。
