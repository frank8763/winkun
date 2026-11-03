# T322：全网络、转向分量、变化点均衡的示范初始化工程pilot

2026-11-03，Stage only。T319仅66参数头，old20保持而六米两中心失败；T320固定高drive的人控派生两行为都接触。原T317两条无碰撞超时(.752/.775m)不是成功专家轨迹。现在检验原网络表示能否吸收转向经验，不继续固定油门/输出头/噪声网格。

## 一手文献与限制

- [DAPG论文v2](https://arxiv.org/pdf/1709.10087)读取方法第IV节/式6及示范项衰减讨论；[作者dapg.py](https://github.com/aravindr93/mjrl/blob/master/mjrl/algos/dapg.py)读取BC初始化、示范梯度权重和自然梯度更新。借鉴先初始化再真实RL的分阶段思想；本文专家前提与我方两条超时不同，本任务不是完整DAPG复现，也不下载/复制作者代码或新增依赖。
- [Fighting Copycat Agents, NeurIPS2020](https://papers.nips.cc/paper_files/paper/2020/file/1b113258af3968aaf3969ca67e744ff8-Paper.pdf)读取3–5节：历史动作相关可造成复制捷径，作者用目标条件对抗和信息瓶颈。T319已有直接复制上一动作的低误差反例；采用变化/保持分层报告和固定均衡监督，**不是作者TCA方法复现，均衡也不保证消除捷径**，不删除原120历史动作。
- [POfD](https://proceedings.mlr.press/v80/kang18a.html)方法先前已读，允许少量不完美示范但引入占用判别器/改变学习信号。当前不同时新增判别器/Q/奖励，防止将未验证价值再接控制。
- [NAC](https://arxiv.org/abs/1802.05313)仅本次核对摘要/前轮记录；不是据其结论证明本项目有效。标准强化学习可超过不完美示范，但不能从距离接近就虚构正确优势。
- [AGAIL IJCAI2019](https://www.ijcai.org/Proceedings/2019/0487.pdf)本次检索摘要摘录，PDF读取重定向失败，未声称读全文。其“partial action sequences”是缺少时刻标签，不应偷换成只监督转向维度的理论保证。

多项联合改变：冻结特征→全网络；两动作监督→转向监督；均匀行权重→转向变化/保持各半。因此只能工程pilot，不能把增益归因某一因素或宣称原创。保持原safeT255、原120/hidden每步清零、原native奖励/56s/.20m/碰撞优先，不加入地图/路点/规划器/运行时教师或路由。

## 固定方法

从safeT255+freshAdam开始；原网络所有原可训练参数开放，仅原固定logstd保持。隐藏每步清零导致某些GRU循环权重可为零梯度，报告实际改动，不假称每个参数都更新。两条人控按整条分：attempt01的280行只训练，attempt02只最终报告，禁止用它选步数/权重/模型。

仅转向raw输出拟合真实转向，`a_steer != observed_previous_steer`为精确变化点，变化点与保持点各占总权重0.5；这不是动作差分等价MSE包装创新。人工drive不作为目标，原父Actor在人控状态的raw drive输出以权重1保持；原211成功任务固定8×32=256旧行mode动作保持权重100。原base lr3e-6、encoder/GRU/head lr1e-4，clip_grad_norm1，**固定200Adam**，不网格/早停挑模型/额外种子。完整行梯度按固定64分块累积，非采样16行；这是监督梯度而非同策略密度。canonical batch1用于报告和native，两数值路径误差预检，不替换真实logp。

本任务专门检验示范初始化，**RL更新0**；尚不把它叫示范增强RL的最终算法。随后若old20保持且表现支持，再另立完整预算的真实onpolicy RL，不借本编号续200步。人控不造expert logp，不进入PPO同策略样本。即使示范误差很低也不能绕过native。

## 有界执行/验收

只有一个最终200步候选；old20+原six54，不论old20结果都保留six54作为研究诊断，每74条各独立native精确重生，148命令/3,854,686物理上限（以protocol实际cap求和为准）。不是74独立seed；74重生不增加样本量。原275完整/fresh16/78保存重载不在本初始化试验内，**不晋升、不进7/8**；需同候选全部门槛后另行推进。

总1800秒，末60秒收尾；训练最多600秒；独立audit900秒、0新增物理、200丢弃历史Adam精确复构；preflight仅1完整真实loss反向但0实际ActorAdam，4小合成丢弃Adam验证同步/全状态回退。最终真实Actor+Adam保存<=8MiB；compact74×8KiB、至多4partial、metadata；own14MiB=11MiB payload+2MiB metadata+1MiB buffer，global1GiB/free40GiB，结束不删除原模型/人控/失败。完整12数组和积分在实际重生逐位核验，仅compact4在磁盘，独立零物理audit不假称重建未保存中间120。

T321已无损释放34256322B、独立审计及7原件全远端读回通过；当前空间可完整预留。source/inputSHA/预检/干净提交/精确远端根树后唯一启动，active单writer不外写runs/改协议/sync。结束所有父4worker/timeout/inhibitor消失→独立重算200ActorAdam、原始/物理/命令与SHA→全部原件RAM远端读回→source同步。若失败保留partial和缺测，未执行不是失败率，不能借改编号重复200步。
