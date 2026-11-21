# T693：动作测度导数与匹配 score 对照的有限 Actor pilot（工程准备中）

主审见 `../T693_METHOD_SELECTION_PRIMARY_REVIEW.md`。已有有限三child runtime与完整零更新入口预检PASS（PREFLIGHT.json），尚未登记/启动时不称已学习。PREPARATION只为早期合同；完整入口见下文最新补充，T501不变。

## 科学目标与固定采样

只比较 Gaussian 均值信用估计器。两臂同 T501/sigma .05、11906 adapters/frozenbody、executed387 参考及原 weight10/.2²、freshAdam lr3e-5/eps1e-8/clip1；每臂至多一次 live/shadow/independent，Value0。新 RL 目标是已暴露的八 ID 局部随机目标 mean_ID E[Y]/512，**不是全407目标或确定性部署梯度**。主审所列 T623/T625/T659 条件动作不预测完整候选的反证继续有效，无拟合 Q 或运行时干预。

固定 ID 为旧 dev_task6_00/02/03/14 与 field0_offset1/2_offset1/5_offset1/7_offset2。每 ID 原 T672 按采集顺序前四条，不筛结果。`t693_estimator.slots()` 先用 default_rng([69320261121,0]) 一次产生8×4个 uniform integer [0,1024)，之后才读取长度；当前9个 active（old3/field6），其余23项贡献0、不重采。全部32个 slot/length/sourceSHA/mask/command 保存在 PREPARATION。主审最小 active old+field 支持门过只说明可执行，不证明信息量或方差。

每 active 前缀两臂各四 suffix，合计72fresh+72exact；每条包含完整原初态再生和前缀物理成本，不把复制历史算实际执行。MVD 两 coordinate 各一个 ±Rayleigh pair，另一坐标独立 normal 且pair共享；score 两pair每pair两个独立 normal 2D command，估计 (Y1-Y2)(z1-z2)/(2sigma) 再平均。MVD 的 D=(Yplus-Yminus)/(sigma sqrt(2pi))。二者乘 fixed1024/512=2，再按8×4均值；current branch raw latent另存，命令依现有clip→float32。重复量化命令不重采。

action RNG namespace [seed,1,task,rep,coordinate]（MVD）、[seed,2,task,rep]（score）；suffix [seed,3,task,rep,pair]，每pair共享、两臂匹配且独立于所有 current draws。强制原 prefix 和单 branch 时 ownership1/stochastic0，随后 T501/sigma.05 自身反馈续行 ownership0/stochastic1。prefix不消费suffix RNG、不复制hidden、不reset horizon/clock。相同创新不要求未来状态或动作相同。所有physical完整 raw/integral/exact 仍须真实审计。

完整有限采集后计算两个梯度；RL gradient 为零/非有限则该臂0更新终止，不靠参考梯度单独制造新候选。非零臂先仅解析表示固定首 Adam，再检查原68任务 mean latent KL≤.01与finite/state恢复；失败0实际更新，不缩步。通过才实际1live/1shadow/1independent，正式模型来自实际 Adam，不把解析末位hash冒充导出。必须拒绝与历史已消费候选完全相同的模型。

各候选原55full55exact/32旧保护+23outside；所有32旧全保且有新增才补剩352同candidate407。完整384全保+新增+407/audit才留，联合387另报，两臂不能拼。全部负差、collision、timeout、无变化都保留。不追加 slot/seed/幅度/sigma/step/权重或换 critic 救本提议。

## 已做的准备与尚欠证据

实际 preparation 用确定性积分核验15个 constant/linear/quadratic/clipped/threshold Gaussian导数，R density、signed coordinates、zero constant credit、fixedslot scaling。32来源全数组SHA、实际创新流、float32命令、T501自己的fullhistory mean重构均过；真实11906参数 rawmean synthetic detached credit方向差分误差1.11890e-11、整Actor/Adam精确恢复；0native/optimizer，峰值763822080B，elapsed3.540985584秒。这个 numeric 是合成 credit 链式合同，不是新环境PG或闭环提升。

仍须实现并验证真实 canonical Session/worker，prefix初态/scene/goal/zeroGRU、时间余额、单branch ownership、suffix RNG与自有carry、物理积分及故意损坏字段拒绝、no-native管理6、真实loader、退出和中断回退。然后完整真实0native0optimizer preflight、独立worst登记、精确GitHub树和source shell退出，才唯一launch。不能把这次准备PASS标为上述所有入口已通过。

## 时间容量纠正：不得按原两小时 root 启动

锁定 masks 后 collection worst 5,392,224physics/150cmd；两候选沿用既有完整验收上界27,250,642physics/936cmd；总32,642,866physics/1,086cmd。既有 admission2500physics/s 给仅物理13,057.1464秒，已超过主审初拟5700main。因此**原单root≤2h分配未准入，不登记、不launch**；不是MVD科学失败，也不能只采分支后自动无限诊断。

仓库已有 T678/T691/T672 有限多child执行形式。注册前允许的工程布局是同一科学设计、三个顺序child：collection3000main+300audit，mvd5700main+600audit，score5700main+600audit。各child≤2h，root总14400main+1500audit+300监督收尾=16200秒（4.5h），与周末学习/native/allstop从严。不存在旧锁定deadline延长，因为T693尚未注册。若实际storage、完整入口或剩余ledger仍不满足，注册前停止；不删数据/降验收来凑时钟。两候选在collection前共同锁定，第一臂结果不得改变第二臂。

拟root2304MiB：collection1280MiB、每candidate384MiB、finalmetadata16MiB，余量240MiB；global10GiB/8GiB archive trigger/free40GiB不变。fixedactive collection72full×4MiB+72exact×32KiB远低collection上限，但全部schema/header/model/partial/log/metadata/独立audit峰值还须实际代码证明。现约2.61GiB加2.25GiB仍低8GiB；不需要以未经核验删除来准入。所有数字均尚未RESERVED。

下一接续应完成上述真实入口与有限runtime，不重复已完成preparation、不再另造回归/OPE诊断或重跑T692。准备耗时已单列ledger，正式登记只能计尚未结算的预检新增耗时，防止重复记账。所有Stage资源隔离和原周一停止定时保持。

## 完整入口与中断补充（登记前）

72个实际Session.step边界/worker/loader绑定见证，真实保存轨迹的完整carry/密度/奖励审计和损坏carry拒绝；ownership/command/prefix/nativecount/latent/integration损坏拒绝；suffix从首个创新开始/独立namespace、no-native6管理计数均通过。Session constructor的新场景零GRU/scene/goal/initialintegration断言在正式首物理前运行；预检superstep与数组见证不是实际新物理。

两学习臂使用合成detached credit检验真实11906参数/387参考/68KL入口：梯度独立合成误差4.44e-16和1.11e-16，numeric误差1.40e-9和1.18e-10（原atol1e-8+rtol1e-4门未改），零RL梯度拒绝、全部state/Adam恢复；没有把合成标签当新环境PG。PREFLIGHT峰值1328107520B、mainbody20.388s，0native0optimizer。初次preflight在F.K不存在时已拒绝，尚未梯度/原生/更新，原error/源码SHA与全文保留PREFLIGHT_FAILURE_001，保守20秒计费。修为已存在的E.N.K.R integration，不动科学变量。中断补充真实保存边界+故意Actor状态扰动后全state恢复通过，无真实optimizer/native，10秒上界。进口/元数据/finalize再10秒上界；T693_PREPARATION已结10秒不重复。

初始注册锚定root原absolute deadline，此前未注册草稿不构成已锁定job；登记后不延。启动前锁定所有代码与上述文件SHA，registration与child实际deadline逐一记录。全过程三个child共同预先锁定，不因首候选结果改变第二臂。
