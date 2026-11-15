# T568实际学生状态的一次恢复资格验证

用户要求继续并询问全场地。当前16:30学习截止已过，不训练新模型；原17:00native/17:30all、日19M/600cmd/12300main不变。新T568不是延续T566离线拟合，且不是T567R自主GOAL的晋升测试。

## 固定起点与教师

同T562冻结Actor、同sector0_offset2原reset/goal/128.224秒cap、同history/执行器/clock，全程从原起点真实步进。学生前195决策（39.0秒）严格精确复现T567R已审失败轨迹，剩余时钟继续，不中途teleport/hidden/history重置。接管步的全部当前输入/pose/hidden/history须同原学生逐数组精确。源模型SHA及原失败fullSHA锁定，prefix物理全计预算。

接管点由保存学生轨迹唯一规则预先确定：用T549既有响应gain.7112683393及nominalcurvature.7771794303，预测到.2m目标圈所需命令曲率首次超限的前一真实决策；得到195/39.0s，d5.155748m，朝向误差17.7658deg，cos>0，代理.772984<.777179。这个常数gain公式是工程代理，不是安全/可达证明，不扫接管时窗/幅度/速度。

195起只在本次训练数据资格采集使用T544既有goal feedback：物理wheelbase.6、nominalSpeed.14m/s、slowdistance.5、minheadingSpeedfraction.25、curvaturefloor.2。原steer/drive映射由实际native env严格核对，运行prefix后从真实位姿/原goal算控制，actor仍每步走真实hidden，但实际动作owner=1，全部teacher_actions/车辆参数/ownership保留；不把教师action/logp称on-policy。无地图/路径规划/路点，教师不会用于学生部署。

只1条实际恢复＋1精确原生重生。原时限内safeGOAL且0collision、全prefix/动作/hidden/reward/history及exact积分通过，才记teacher_qualified；contact/timeout均原样保留并不作为成功监督。教师成功也不算学生自主成功，不新建/晋升学生模型、不自动启动学习或改目标阈值。

## 数据与独立审计

保存完整120states/nextstates、动作/位姿/nativecounts/hidden/history/初终积分/每holdSHA、teacher动作/参数与原模型SHA；审计独立重建原前缀各数组和接管边界、每步教师公式、Actor真实carry/密度/环境奖励以及1exact全部数组/积分。0优化器/0Adam/0RL，originalmodel不变。

后续若资格过，才另立学习窗口/完整最坏预算，联合真实恢复数据＋两条T562自主成功history＋旧能力保护；训练后必须无教师32自主复核。此处不把下一方案当已执行。

## 全部预算

1回合＋1exact max128224物理，2请求＋4保守进程命令=6；main900/audit300/backup300/source180，完整1680秒＋缓冲。日累计实际与最坏由BUDGET机器核验，不扩大19M/600/12300。开始前必须main900早于17:00且完整收尾早于17:30；学习永远0。

own3.5MiB：onefull960KiB＋atomic960KiB＋singleworkerpartial960KiB＋metadata512KiB=3473408B<3670016B。真实schema max642决策（包括n+1history及teacher字段）上界943176B<960KiB，预检PASS。global1GiB/free40GiB、完整own+131072guard，旧唯一证据不删；CPU14/15总一核idle/Nice19/CPUWeight1/IOWeight1、GPUoff/Phase全隔离。

唯一driver本机断网执行main/audit/manifest，原生全退后全部原始RAM远端逐成员SHA及source根树；网络不通明确pending，raw原地保留。active单writer不外改源/协议/写其他runs/并发audit备份；有限资格失败即结束，不扫描或无限重试。
