# T299：修正辅助目标的条件动作似然，再检验完整六米

2026-11-02，用户继续推进六米授权；Stage/Phase完全隔离，safeT255五米和原失败/用户HD视频保留。T298已结束不重跑、不追加500步，不从其退化模型续训。重新从冻结safeT255开始，固定同8条T295完整120/目标历史/first-hit辅助成功前缀、2辅助goal、相同seed/500步/批次索引/旧8任务保持/learning rates/gradclip1/Actor容量/hidden reset。六米最新完整T298 2/54且旧267/275，不晋升。

## 科学差异与条件概率

T298辅助loss经硬clip后的mode回归，在LEFT约99%drive样本处有零局部导数。T299改为同一连续截断高斯copula AR策略的**条件动作负对数似然**：完整当前新goal120及实际前一decision120分别用真实batch1路径重算mean，前动作和前mean均正确进入条件AR，保留前mean反向梯度，first decision使用reset边际。使用已验证T296对数尾部公式，不改变rho=.98/sigma(.035,.03)、Actor输出/执行读取、采样器或物理时长，不用IS/PPO或截平重要性权重。历史动作是自身goal重标记的监督目标，不是新goal on-policy RL。

NLL与MSE单位不同，预注册一次固定scale：safeT255对8条整轨迹等权的full gradient norm(modeMSE)/norm(conditionalNLL)。零物理/零优化器所得6.6339496363/3398.5121479=**.0019520158667963997**。只固定一次，训练不自适应调尺度/权重，旧modeMSE保持权100、baseLR3e-6/residualLR1e-4、同每轨迹8样本/旧每任务4anchor、500步live+500shadow不变。它仅匹配初始辅助梯度范数，Adam/clip后有效位移未必相同，不作随机独立因果证明。条件NLL的canonical numerical path与原MSE路径不同，属于明确的loss+数值工程适配。

本协议只隔离验证辅助学习修正，**新增RL交互/策略梯度更新为0**。监督后保存Actor+fresh empty RLAdam用于精确哈希参考，监督Adam用中期/最终SHA及注册独立500live+500shadow重构保留，不在本目录续训。后续RL必须在旧任务保护和完整容量核算下另立有限协议。本线整体仍是自身经验整合+真实DRL的端到端单Actor谱系，不把本项监督当RL或标准GCSL完整复现。

## 原生验收和预算

修正训练后做两safeT255父模型辅助goal自主原生控制、两candidate辅助goal自主控制及真实old20；初始历史和T295的真实factory生成位姿反事实goal历史完全一致，不切换运行时目标/路点。原父控制此前未执行，必须如实报告，辅助目标成功不等于新6m成功。无论old20/辅助/动作误差如何，候选完整old275+原6m54均执行；缺测/无效/接触/超时/删失分列。全部安全才16fresh+78独立native读回，不能用若干中心或2/54叫稳定，七八米不在本协议启动。

460全部命令/12.88M全部物理步、3600秒main（1800学习、1800验收），外部timeout3600秒TERM/1秒KILL更早从严。own10MiB/global1GiB/free40GiB完整预留通过：451×10KiB紧凑5数组（reward精确导出env-eta*cost）、2.5MiB finalActor+emptyAdam、1.5MiBmetadata，no new full training arrays；源8条完整120保持原SHA。451容量包括2父+20保护+2辅助+275旧+54新+16fresh+78reload+4partial。全部0新增导航RL。500live500shadow自监督Adam和4预检丢弃Adam分开；结束独立1000丢弃监督Adam精确重构，audit墙钟1800秒/0native/0新增导航训练。

预检已PASS：原8mean逐位和logp误差1.0481e-13，90位14极端标量/梯度合同通过，真实LEFT/RIGHT当前及前均值有限差分、LEFT裁剪MSE零导数/NLL非零导数验证；1live1shadow更新ActorAdam精确，拒绝/中断各一次完全还原，共4丢弃Adam、0native。scale零优化器计算后纳入源协议及输入SHA，不按原生结果调。未出现预检失败，不重复预检。源码本地提交/GitHub完整root-tree精确同步后唯一launch；运行中native-exclusive不外写runs/源同步。

结束全父/4worker/timeout/inhibitor消失后独立全部输入、模型、监督过程、紧凑派生数据、命令物理预算核账；所有原始/失败保存，RAM归档远端全成员SHA读回后源同步。没有新算法/安全证书/顶刊已达成声明。[GCSL原文](https://arxiv.org/abs/1912.06088)及[作者训练源码](https://raw.githubusercontent.com/dibyaghosh/gcsl/master/gcsl/algo/gcsl.py)是先行方法，当前有界连续AR适配与闭环收益需实证，不凭梯度存在宣称能学会。
