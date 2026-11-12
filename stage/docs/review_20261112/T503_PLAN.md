# T503：全场地新起点的真实闭环策略学习

用户2026-11-12最新要求思考分析并下一步；本轮是独立新有界自身采集RL，不继续T502冻结探测预算。原T501联合最好模型（旧271/275、宽六54/54、八54/54）以及safeT255原件独立锁定，不因新场地研究副本的结果覆盖。

T502新24起终点迁移仅5GOAL、11接触、8超时；6个超时错过目标后驶离、2个全程偏慢，不能只改统一速度/时限。本轮固定选择已见失败中最近决策余距最小的两项：sector5_offset1（最近.200593m）和sector2_offset1（最近.201323m，随后接触）。它们为新起点下约5.204m真实全回合任务，无中途传送/倒计时重置。没有按训练结果再挑任务或扩noise网格。

固定8新自身采集：两任务各4条独立iid噪声（原rawsigma[.2,.2]，固定8个SeedSequence），同一初始Actor，完整原生GOAL/碰撞/超时和完整120/hidden/history/innovations/raw/atom/logp/奖励保留，再同策略/同创新流8exact。噪声样本不是8个独立训练seed。没有新教师示范/规划器或人工动作数据混入onpolicy。

先决门：八条至少一个无碰撞原生GOAL，才允许一次策略更新。若全无成功，本轮记录数据并零Actor/Value更新停止，不重跑/追加seed/轮数。该数据依赖更新门是工程选择，不声称整体无偏。

更新：保持T501同一双动作神经残差Actor/64真实GRUcarry/11906可训参数，固定encoder/core/head。成熟Adam drive896/steer256 moments完整保留，只显式把学习率从监督.001改为RL3e-5；保存新rl_initial.pt.gz，与原policyhash/参数相同，joint仅optimizerlr不同，后续全状态精确锁定。Value及ValueAdam完全冻结、不用于本轮优势，不接入未经验证新Critic。

真实采样密度在完整当前history上重新计算，均值误差<1e-10/logp<1e-9，float32边界atom沿原合同，不能换旧logp或放宽阈值。每回合原奖励按原native gamma构建有限reward-to-go；同行同task其他3回合在相同原生累计时间处回报的均值作基线，同行结束后续回报0；系数gamma^实际时间*(Gt-同行基线)，每task等回合RMS，不减每条时间均值。固定PPO式ratio剪裁.8/1.2、最多16ActorAdam、KL>.005后停本次更新；相同数据独立shadow及最终独立历史重构，ValueAdam0。不是新颖公式、普遍降方差或无偏证明。

保持：379个T501自主成功history+5个T502新场地自主成功，按old271/six54/eight54/field5四组等权、组内等回合，未裁剪初始locations（初始保持为数值零），半均值/半最差20%归一动作风险；自己8条采集history也做当前location保持。keep_weight固定1，不调参、不扫网格。只是软保持，不保证未重测的379闭环无损。

一个最终候选先old8中心，再完整24场地自主开发+24exact。研究接受要求原场地5成功ID全部保持、fieldqualified成功>5、接触<=11、old8全安全、0无效/删失；不满足整Actor/成熟Adam回退，保留候选/原始失败。新的24任务仍是已见开发，不称未见留出。完整old275/six54/eight54本轮未执行，不能把软保持/old8当原379全部复验；fresh16未消费。初始最好模型仍单独保留。

环境完全复用T502已锁几何/8方位reset/history/cap/±7.5m字段资格，原0.2m成功半径、约69.394秒训练时限、动作/奖励/传感器不变。部署仍单Actor120因果输入真实carry直接输出两动作，无地图/规划/路点/运行教师/任务路由。

预算：最多3616354全部原生、82命令（主76+预检6），主管线3600秒含前8+8采集与16live/shadow，学习相对1200秒、独立audit600、all6000且备份/source留1500秒。学习绝对2026-11-12T18:46:45.628434+08:00、原生2026-11-12T19:31:45.628434+08:00、全部2026-11-12T19:56:45.628434+08:00，相对更早从严。own48MiB，rawshape28210128B+14MiB模型/metadata/partial，global完整预留985697046B<1GiB/free40GiB。CPU14/15总quota100%=1核idle/nice19/weight1，GPU禁用/CPU学习、实际Mesa llvmpipe初始丢弃渲染断言；Phase完全隔离。

预检4丢弃synthetic ActorAdam（两clone各2步），完整真实起点成功states驱动但合成终止/奖励/随机动作fixture不是原生导航数据。密度/peer系数有限、keep初始一致、成熟Adam/克隆/回退/序列化、新旧场景worker取消均通过；0native。旧T502零更新与注册计数字段在launch前metadata-only纠正，原协议另存，学习函数/预算不变，4clone不重复。

正式源码提交/精确远端根树后唯一launch。driver唯一writer，active禁止并行source/audit/backup/额外native；结束所有科学进程退后独立audit完整8fresh、replay、数据密度/奖励、最多16历史Adam/回退及新24结果，再RAM全raw（含sealed预检）远端逐成员SHA读回、Git报告源同步。任何失败保partial，不恢复结束目录、不借新编号续预算。无训练成功/验收改善保证。
