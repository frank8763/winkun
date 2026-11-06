# T395之后：当前学生状态纠正，还是保策略势函数？

2026-11-06，Stage定向一手检索。本文只作研究建议，未注册下一实验；本次0仿真、0backward、0优化器、未写runs。下一轮仍须先向用户提出一个关键问题，等回答后再注册、容量admission和启动。部署保持一个120维因果循环Actor，无运行时教师、地图、规划器、路点、路由或混合导航。

**建议优先检验有限次“当前学生到访状态→自动教师验证恢复→独立辅助纠正＋新鲜学生PG”，暂不以严格保策略的势函数修复完整回报LOO退化。** 这不是新算法主张，也不是再做T391同一固定示范的1000步BC。新的可检验变量是更新后重新采集学生状态、重新取得恢复标签；监督与真实策略梯度分开记账。

## 1. 本项目的卡点和反证

T395六轮24条新随机后缀＋24条精确重复，48live/48shadow Adam。56秒4/4、55秒2/4、54秒4/4安全；随后53秒三轮12/12真超时、0接触。三轮53秒实际PG范数约`1.36e-17–8.83e-17`，加权示范增量约`.143–.145`；数值更新保留不等于任务PG提供了有用排序。范数在clip/Adam前，不据此精确分配Adam的因果贡献；未执行PG-only或BC-only更新反事实。[T395只读原始算术](T395_POSTHOC_READONLY.json)、[原协议](T395_PLAN.md)

正式完成记录另明确：最终56/55/54秒mode安全、53/52秒mode超时；old8由初始8/8变7/8，丢失left5的`dev_task6_13`；新070/80原起点LEFT接触、RIGHT到达。完整54/275/16fresh/78未执行，不能称六米稳定。原driver科学结束后的TLS失败保留，另行有界零native传输收尾已完成91原件远端逐成员SHA；不把证据传输当新的训练或科学结果。[正式结果及收尾](T395_FINDINGS.md)、[old8及原模型哈希证据](T395_POSTHOC_READONLY.json)

直接反证是T391：自动教师确实从B实际2.4秒偏离态恢复到GOAL，但一次固定恢复后缀＋原成功轨迹、1000次BC仍使学生21.440秒接触。低MSE、一次真实恢复、早期跟踪改善都未产生自主成功。不能把“学生更新后的新状态也覆盖了”当已证事实。[T391结果](T391_FINDINGS.md)、[教师/学生合同](T391_PLAN.md)

T394冻结数据已经表明56/60秒存在成功覆盖、52秒全超时；T395终验所测mode可成功的最早接管点为54秒，尚未解开53秒；初始54秒mode缺测，不能据此宣称本轮把能力边界前移。参考前缀、当前网络hidden、物理状态和clock必须一起保留；原参考54秒能成功，不代表学生从53秒自己驶到的54秒状态等价。[T394结果](T394_FINDINGS.md)、[T395已有方法核查](T395_FRONTIER_METHODS.md)

## 2. 一手方法：采用什么，排除什么

| 方法及实际查看范围 | 与当前问题相关的内容 | 采用/限制与Stage重叠 |
|---|---|---|
| HG-DAgger，arXiv **1810.02890v2**，§II、Algorithm1 | 学生行驶，专家接管恢复；只在恢复期间收集专家标签，逐轮聚合并重训。专家可见完整state，学生只见observation。论文中的不确定性阈值主要用于分析，运行时回交保守控制器是未来工作。 | 采用“当前学生分布上的恢复＋多轮再收集”。本项目教师是有限仿真搜索，不是人类全状态专家；不新增ensemble、risk gate或部署切换，也不继承DAgger保证。T391已有一次这种数据聚合，故一次查询后固定拟合不构成新的机制检验。[论文§II/Alg1](https://arxiv.org/html/1810.02890v2#S2) |
| IWR，**2012.06733v1**，§III、§IV-B/C、§V-B | 以干预与非干预样本分别归一，等比例采样，防止稀少纠正被大量普通数据稀释；非干预动作同时约束策略漂移。其BC/加权回归解释不把专家动作变成学生on-policy PG。 | 采用数据来源和权重分开记账的原则，不机械复制50/50。失败前的学生动作可能正是需要纠正的动作，不能全部当正确标签强保留；旧能力优先用已真实成功的父策略history。不同权重或额外保持项若同时引入，只能报告工程组合pilot。[论文§III/IV-C](https://arxiv.org/html/2012.06733v1#S4.SS3)、[作者项目](https://sites.google.com/stanford.edu/iwr) |
| DAPG，**1709.10087v2**，§IV-C，Eq5–6；作者`mjrl` | 新策略轨迹的PG加示范score辅助；实际使用NPG和衰减示范权重。论文明确BC存在自身状态分布偏移，单BC往往不成功。 | 支持“PG＋独立示范辅助”的结构；现有PPO＋固定`.1*MSE`已经是工程改写，不是DAPG复现。教师分支不能伪装成学生策略轨迹，也不能用其成功回报替学生失败轨迹赋正优势。[论文§IV-C](https://arxiv.org/pdf/1709.10087v2)、[作者dapg.py L54–107](https://github.com/aravindr93/mjrl/blob/3871d93763d3b49c4741e6daeaebbc605fe140dc/mjrl/algos/dapg.py#L54-L107) |
| RFCL，**2405.03379v1**，§4.1–4.2；作者curriculum wrapper | 按示范分别维护逆课程，SAC/Q ensemble和50%离线replay；随后前向课程扩展初态分布。代码reset直接设置示范state、`step_count=0`。 | T395已借用了小步前沿动机，继续减小步长或加轮数不增加机制。暂不引入新Q、reset计时或其完整系统；原80秒clock和真实历史规则照旧。单参考恢复不能替代RFCL前向泛化。[论文§4](https://arxiv.org/html/2405.03379v1#S4)、[wrapper L121–144](https://github.com/StoneT2000/rfcl/blob/7bd660f05dbda1ec8881bcfb250a8d1881532dbb/rfcl/envs/wrappers/curriculum.py#L121-L144) |
| AWR，**1910.00177v3**，§3、Alg1、§3.3；作者`awr_agent.py` | 值函数回归后，以`exp((return−V)/temperature)`给回归动作加权，并裁剪大权重；代码确实更新critic后才更新actor。 | 暂缓。它是off-policy方法，需要可用的值估计，不能把任意教师成功标签叫作优势；相同回报不会经指数自动产生正确排序。新增值误差与权重变量超出最小改动。它不等同于无Critic的IWR权重。[论文§3](https://arxiv.org/pdf/1910.00177v3)、[作者更新L213–235](https://github.com/xbpeng/awr/blob/831442fb8d4c24bd200667cbc5e458c7657effc2/learning/awr_agent.py#L213-L235)、[权重L396–415](https://github.com/xbpeng/awr/blob/831442fb8d4c24bd200667cbc5e458c7657effc2/learning/awr_agent.py#L396-L415) |
| AWAC，**2006.09359v2**，§4、Alg1 | 使用off-policy Bellman Q评估和优势加权最大似然更新，可从离线数据再转在线。 | 暂缓完整AWAC；教师辅助不要求立即引入Q。现有Stage失败Q不因换算法名自动合格。未完成其作者实现核验，不能称已具备可复制AWAC基线。[论文§4/Alg1](https://arxiv.org/pdf/2006.09359v2)、[作者项目及代码入口](https://awacrl.github.io/) |

Sirius作者实现可核对IWR基线的工程语义：`bc_iwr.json`启用`use_iwr_sampling`，`IWRDataset`按`action_modes==1`给干预样本权重`N_nonint/N_int`；这是采样权重。循环GMM训练另有逐样本加权负log-likelihood；二者均不等于PPO优势或合法on-policy比例。[Sirius README的IWR基线](https://github.com/UT-Austin-RPL/sirius/blob/2af35b48bec553afeabf5a1d5c3c2796a597e6c2/README.md)、[config L40–43](https://github.com/UT-Austin-RPL/sirius/blob/2af35b48bec553afeabf5a1d5c3c2796a597e6c2/robomimic/exps/sirius/bc_iwr.json#L40-L43)、[dataset L767–789](https://github.com/UT-Austin-RPL/sirius/blob/2af35b48bec553afeabf5a1d5c3c2796a597e6c2/robomimic/utils/dataset.py#L767-L789)、[BC_RNN_GMM L662–665](https://github.com/UT-Austin-RPL/sirius/blob/2af35b48bec553afeabf5a1d5c3c2796a597e6c2/robomimic/algo/bc.py#L662-L665)

此处Sirius是其作者对IWR的**基线实现**，不冒充IWR作者公开的原始训练代码。其部署有人类监督的整体方案不满足本项目单Actor部署合同；本次只参考训练数据标识和回归权重。[Sirius正式论文](https://www.roboticsproceedings.org/rss19/p005.pdf)、[作者项目](https://ut-austin-rpl.github.io/sirius/)

## 3. 严格保策略势函数为何不是本轮完整回报LOO的最小修复

Ng/Harada/Russell给出的保策略形式是`F(s,s')=gamma*Phi(s')−Phi(s)`；有限终止要处理末态势。Grześ §3–4、Eq3解释了终止项为什么能改变偏好，明确将有限时间停止也当作终止。[原始ICML1999论文](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)、[Grześ AAMAS2017 §3–4](https://aamas.csc.liv.ac.uk/Proceedings/aamas2017/pdfs/p565.pdf)

下面是对**本项目当前估计器**的代数推导，不是新定理。把时间、可恢复native状态或完整因果历史写进`z_t`，令每个实际转移的折扣为`d_t`，`D_0=1, D_{t+1}=D_t*d_t`。定义

```text
F_t = d_t * Phi(z_{t+1}) - Phi(z_t)
G'  = sum_t D_t*(r_t+F_t)
    = G - Phi(z_0) + D_T*Phi(z_T).
```

若GOAL、CONTACT及真实80秒TIMEOUT的末态势全置0，同一53秒起态的完整回报都只平移`−Phi(z_0)`，**完整G的差值及四轨迹LOO精确不变**。所以它不能把T395的`1e-16`级完整回报spread变成可靠任务排序。Cipollone等2022论文PDF第3页“Reward Shaping for Episodic RL”、Eq4已明确讨论类似限制：return-invariant shaping不能靠有限未到达轨迹的完整回报差持续提供探索偏置；其后采用非return-invariant的biased MDP及额外Q估计，不能直接借来作为本项目PPO保策略证明。[一手2022论文该段/Eq4](https://prl-theworkshop.github.io/prl2022-ijcai/papers/PRL2022_paper_24.pdf)

保留非零`Phi(z_TIMEOUT)`，用末距制造完整G差值，则留下`D_T*Phi(z_T)`，改变这个80秒有限任务的训练偏好；不能同时承诺严格保原目标。T197/T280的距离代价负结果必须保留，不以“潜势/进度”换名重做。额外对F裁剪、每步另加距离成本、不同折扣/hold时标、按动作选择变化的势都不能直接套上述证明。[本项目历史排除规则](T395_FRONTIER_METHODS.md#2-固定真实历史奖励和部署语义)

这**不否定PBRS通常有用**：TD/bootstrapping、估值初始化或reward-to-go的有限样本方差可以改变。完整return-to-go的变化是状态/历史相关的基线项，而不是新增真实任务信息；若要换估计器/学习V，须另立协议并保留其误差反证。采集中行政删失也不能伪作80秒真终止来置零；不能通过偷偷刷新clock取得正回报。

## 4. 可提交用户选择的最小有界草案

这段只给科学范围建议，不是执行协议或资源准入。可问：**“下一轮是否只针对53→54秒衔接，做两次当前学生真实状态的自动恢复纠正，再让单Actor用新鲜自身轨迹学习，同时要求保住原old8？”** 等答复后才锁精确父模型、唯一snapshot规则、教师参数化、源/输入SHA及全部硬预算。

1. **父策略和比较起点。** 倾向T393最终Actor＋freshAdam，以已记录old8=8/8作为旧能力起点；所有比较臂同一父/optimizer状态。它在53/54秒的初始mode此前未测，必须先测，不能把T395结果代填。若改用T395最终父，其old8只有7/8；冻结该父在失败13上的动作不能叫恢复旧能力，须另有成功父标签和恢复目标。两选择不能训练后按结果切换。[T393状态与初始化依据](T393_INITIALIZATION_METHODS.md)、[T395 old8](T395_POSTHOC_READONLY.json)
2. **限定两次collect/update。** 每次只从同一53秒真实参考前缀启动，先让当前学生自己执行到预注册的一个53→54秒内snapshot；例如固定两hold后53.4秒只是待选例，不临时挑最好时刻。保留真实clock、物理/执行器/previous-action/120历史；当前参数从episode zero重算hidden。第二次必须用第一次更新后的学生新到访状态重新查询，不能复用第一次snapshot冒称新覆盖。
3. **每次最多一个教师cohort。** 建议每次至多12个预注册候选、首个实际safeGOAL即停，无成功就停止该pilot并留负结果；两次最多24候选，不扩seed/幅度/搜索代数。只能复用T391已有有限仿真教师能力作为工程起点，不能保证更小预算会找到恢复。每个cohort选定序列须从该真实snapshot在原80秒内原生执行GOAL并精确重生，才进入辅助标签集；无效、接触、超时全部保留且不标为正确标签。若要只教短桥动作，应先验证“短纠正后回当前学生自主续行”仍成功，而不是用整段教师成功代填这一条件。
4. **PG轨迹和教师分支分开。** 原学生随机suffix必须完整无教师执行，保存行为概率、原回报与终止；教师从已存snapshot另起研究分支，不覆盖原学生轨迹。每次每臂4条新学生suffix＋各一次精确重复、固定8epoch：单臂两轮最多8新/8重复、16live/16shadow及独立16历史Adam。teacher动作/被强制参考前缀PG-mask=0；真实学生suffix才进入原执行域Gaussian/PPO。辅助项仍总系数`.1`，内部旧成功/累计纠正的固定配比须事前锁定且按轨迹归一，不能因新样本条数暗中改变总强度。不加Q/新奖励/新网络或runtime gate。[T395既有训练合同](T395_PLAN.md)
5. **对照是主张的前提。** 最小因果比较是同父、相同53秒起点/噪声配对/更新数/epoch/总辅助权重的“仅原固定示范＋真实PG”控制；实验臂唯一增加重新查询/聚合。两臂合计16新/16重复、32live/32shadow，教师查询成本另报，不能宣称计算或样本效率公平。T395的不同课程/父状态不是该匹配控制。若容量只容单臂，只能报告有限可行性，不归因纠正机制；另要证明PG贡献还需teacher-aux-only等消融，本小pilot不自动安排。
6. **旧能力和最终自主验收。** 至少初/末同old8逐ID全安全，尤其`dev_task6_13`，加独立保存模型读回；任一丢失则研究候选拒绝。可用T393真实成功history作辅助保护，而不把失败frontier动作强保留。有限old8不是275保证。最终53秒mode、54秒保留、原起点LEFT/RIGHT分别报告；参考前缀后的恢复不叫全程自主。完整54/275/16/78若本pilot不安排仍为null，不能晋升、进7/8或用中心成功宣布六米稳定。

正式注册还必须逐项预留所有前缀、完整学生/教师分支、精确重复、baseline、old8、mode、原起点和模型读回的物理/命令/分段墙钟；两臂模型/Adam、raw/partial、审计、RAM远端逐成员校验的字节也全部计入。T395收尾剩`5030926B`只是容量快照；**尚未证明匹配控制和实验臂能在该空间内共同有界执行，admission pending**，不得把本文示例计数当容量通过。global1GiB/free40GiB不变，未知容量不据历史剩余额度承诺启动。不续T395六轮或重启T391；固定终点，不选中间最优，不因全失败再无限扩预算。[项目实验流程](../research_20261031_navigation/experiment_workflow.md)、[实际剩余额度](T395_FINDINGS.md)

应分别报告：教师能否从**这次**学生状态恢复；纠正误差/旧成功保持；原始PG与加权辅助梯度；数值保留；新鲜学生safe/contact/timeout；无教师mode及全程结果。若PG继续近零，只能说辅助在驱动更新，不能称真正RL已突破。第二轮再失败可以否定该有界组合的目标，不否定全部DAgger/DAPG/循环策略、教师可达性或更广泛PBRS。

## 5. 作者实现版本、许可证和访问限制

| 读过的实现 | 固定版本/许可证核验 | 阅读边界 |
|---|---|---|
| `aravindr93/mjrl` DAPG | `3871d93763d3b49c4741e6daeaebbc605fe140dc`；[LICENSE Apache-2.0](https://github.com/aravindr93/mjrl/blob/3871d93763d3b49c4741e6daeaebbc605fe140dc/LICENSE) | 本次读取dapg.py L45–120，含优势归一、衰减demo_adv、NPG和KL步骤；未安装/执行。|
| `UT-Austin-RPL/sirius` | 本次`git ls-remote`固定master为`2af35b48bec553afeabf5a1d5c3c2796a597e6c2`；[LICENSE MIT](https://github.com/UT-Austin-RPL/sirius/blob/2af35b48bec553afeabf5a1d5c3c2796a597e6c2/LICENSE) | 读README、IWR config、dataset干预采样和权重相关片段、BC循环加权相关片段；不是完整工程审计，不复制/运行。|
| `xbpeng/awr` | 本次master固定`831442fb8d4c24bd200667cbc5e458c7657effc2`；[LICENSE MIT](https://github.com/xbpeng/awr/blob/831442fb8d4c24bd200667cbc5e458c7657effc2/LICENSE) | 读更新L213–235、TD-lambda回报及优势/指数裁剪L379–415；未执行。|
| `StoneT2000/rfcl` | `7bd660f05dbda1ec8881bcfb250a8d1881532dbb`；[pyproject L13–17声明MIT](https://github.com/StoneT2000/rfcl/blob/7bd660f05dbda1ec8881bcfb250a8d1881532dbb/pyproject.toml#L13-L17) | 本次重读wrapper L108–151、299–335。该commit根目录未见独立LICENSE，raw `/LICENSE`为404；只报告元数据声明，未复制代码。|

HG-DAgger论文为CC-BY-4.0；本次在论文和定向搜索中未找到可核验的作者实现入口，不宣称全球不存在代码。IWR v1论文为arXiv non-exclusive license；作者项目未给出可直接核验的原始训练仓库，故用清楚标识的Sirius基线实现交叉核查，不假设其代码许可证就是IWR原实现许可证。AWAC作者项目指向`vitchyr/rlkit/examples/awac`，web tree访问受限，本次未核其commit/许可证，不将其实现细节用于建议。[HG版本/许可证](https://arxiv.org/html/1810.02890v2)、[IWR版本/许可证](https://arxiv.org/html/2012.06733v1)、[AWAC官方入口](https://awacrl.github.io/)

GitHub连接器本次fetch/fetch_file返回`Unknown tool`；随后只读GitHub网页/raw内容和`git ls-remote`，未clone、安装或运行作者工程。GitHub公共API一次403 rate-limit，改用已固定commit的raw，不反复索权。若干arXiv HTML、PDF续读和GitHub tree请求超时/受限，以上只报告已读章节。ICML1999 PDF文字提取乱码，故终止结论主要核对可读的Grześ原论文与2022原论文，并给出显式代数；未假称原始PDF全篇逐字可读。一次错误arXiv编号返回无关数学论文，已排除，未作为任何结论依据。

本地阅读为AGENTS相关约束、当前RESEARCH_STATUS首条及T391–T395相关计划/方法/结果/只读算术、迁移context/progress/cleanup与实验流程；未重新审计原始runs。未读Phase、全部Stage聊天或所有原始ChatGPT历史；未完成全面新颖性调查。上述组合全有明确近邻，只能称针对当前失败证据的工程研究假设。
