# T563 峰值动作误差学习＋经验门保持

用户明确允许新窗口今天17:30全部停止，沿用日19M物理/600命令/12300主管线秒余额，不扩大总额，不重启T562。学习16:30/native17:00/all17:30，与相对限时更早从严。

## 已知证据与目标

T562原T550成功27history仅24条同时过RMS≤.02/max≤.1，但27条RMS全过。剩sector0off2/sector1off2/sector4off2共20个决策至少一坐标>.1；实际有13558个命令坐标，不把20决策混同20坐标。最大.108855/.182749/.162611，仍未自主native。新任务只处理剩余拟合峰值，不能称导航恢复或跨场地泛化。

同T562架构、120因果obs/64真实carry、93186encoder/core/student变量、T562最终未晋升研究模型初始化＋freshAdam；旧normalization/head/drive与steer adapters冻结。policy architecture保持T562以验证初始policyhash精确相同，不用改名装作新网络。仅训练端同27原T550成功完整history，forcedpulse帧不监督但保留真实历史；所有hidden当前权重重新BPTT，不复用旧hidden。原奖励/时限/执行器/最高速度/clip不改。部署单Actor，无中心库/运行教师/地图/规划/路点/router。

## 新损失与固定提案

每步完整27 paddedbatch，损失为半27history等权locationMSE＋半全13558有效坐标中最大的32个**clipped-command坐标平方误差均值**。这借鉴top-k难例，目的从平均/20%尾部转到有限峰值；不是标准CVaR导航风险。command项用可微doubleclip，门用实际float32clip；饱和区clip导数为0，locationMSE项保留对原location标签的梯度。不对专家数据构造on-policy密度，不新RL/Value/教师采集。

最多240次Adam提案，lr3e−4/clipgrad1，freshAdam。每次从上一个保留ActorAdam状态算一次Adam候选，然后固定因素1/.25/.0625依次缩放其真实参数位移；所有尝试复用同一次Adam moments，无新增Adam调用。这是固定有界backtracking，不是学习率/seed/噪声网格。接受必须：

1. 初始24已达标history逐条仍满足RMS≤.02/max≤.1；是保持门内，不是每一误差逐数值不增。
2. 全27最大history RMS≤.02。
3. 全27最大命令误差比上一个保留候选下降至少1e−6，或已经全部通过原27门。

有一个因素通过才保留该参数与本次Adam状态；全拒绝整ActorAdam回退，拒绝计算仍如实算真实Adam调用，不伪作0更新。连续8拒绝停学；27门全过立即停学，不追满240/不挑中期checkpoint。所有提案trace含loss/梯度/保留factor/累计接受/连续拒绝/三因素门指标，未执行因素用−1标志，不算失败。相同freshshadow精确重复路径，非独立数据；audit从头重建全部保留/回退的ActorAdam。

## 只在原门通过后原生验收

仍是27每条RMS≤.02/max≤.1。失败0native，不能将未执行填0%成功率；通过才2短500物理＋2exact（保存完整history验证实际worker新Actor），再同候选old8+field24的32自主＋32精确原生全数组hash/积分。必须保原23自主成功ID；24拟合history不是24导航成功，不能混淆。若原生失守不保留新模型，冻结T550安全原件不变。32保持只记开发支持迁移资格，275/six54/eight54/fresh16/reload78/新yaw均不执行null，不自动进入RL/全局晋升。

## 工程合同与预检原件

首次预检4丢弃Adam/0native后拒绝：原检查误要求被真实保护门拒绝后的freshAdam必有成熟状态，科学拒绝行为正常。原source/error/progress保留preflight_failed_001。恢复预检4对应克隆提案＋2独立合成成熟Adam回退，共6丢弃；累计10。保护门、损失、步数、科学阈值均未改变。另一次旧次数metadata把prior4误记10，原PREFLIGHT_BEFORE_METADATA_CORRECTION及纠错记录保留；仅改计数字段为prior4+current6=total10，0重算/0native/0Adam。live-shadow、真实因果step/sequence、密度、模型RAM重载、拒绝整状态和成熟状态回退PASS。预检不算导航成功。

## 全部最坏预算

新增3,203,204物理/72命令/3600main，其中相对1800学习；audit1200、transport600、source180。live≤240/shadow≤240/audit历史丢弃≤240，10预检丢弃另列；没有新RL。与T555/T557/T559/T561/T562实际累计由BUDGET核账，必须仍≤19M/600/12300；T560只读诊断8.792秒另记。启动前必须完整余时容纳全部最坏5580秒及缓冲。

own8MiB：完整ActorAdam3MiB＋双worker partial2MiB＋两short保存128KiB＋short原子128KiB＋32compact×12KiB384KiB＋metadata/trace/log/manifest1MiB，共6.625MiB，余1.375MiB。不缓存大history特征/归档包；模型raw上界2465744B<3MiB。global1GiB/free40GiB，完整own+131072守卫预留；CPU14/15总一核idle/Nice19/CPUWeight1/IOWeight1，GPU禁用，Phase完全隔离。

唯一driver自动学习/条件原生，全部native退出后独立audit/本地manifest/RAM全部原始远端成员SHA/source树。科学无网络/LLM依赖；断网原始保留remotePENDING，不能杜撰继续在线分析。driver健康期间不外写runs/改科学源协议/重复audit备份，不启动另一机制。有限方案失败或完成提前收尾，不强行填满截止。

## 可证伪边界

本机制将损失与经验接受规则成套改变，是有界工程pilot，不能单归因top-k。有限样本门只能保这些历史的门资格，不能保证参数附近、学生访问分布、新布局或实体车。失败若连续拒绝意味着此固定Adam方向/门组合未找到保留更新，不证明不存在其他方向/车辆无法通过。标准top-k/GEM/约束接受均已有近邻，来源见SOURCES，不声称原创或安全证书。
