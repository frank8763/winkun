# T692具体下一规格：同一行为策略的动作价值信息与真实分支

状态：只是一手审查和具体规格，尚无experiments/t692代码、实际入口预检、预算登记或launch。不能称新Critic已学习，更无Actor更新。T691/R两固定64步臂均KL拒绝0native已停止，不缩lr/steps救。主审见T692_ONPOLICY_VALUE_PRIMARY_REVIEW.md。

## 固定问题与数据

全T672407ID/1724fresh完整历史，同T501sigma.05单行为；exact不作新样本。按既有每ID采集顺序非field4前3train末1heldout、field8前6train后2heldout，共1293train/431heldout。该数据已参与开发，不称新地图/未见任务泛化。输入只含冻结T501完整owncarry所构建184features与当前实际执行float32 command2；禁future/end flags/TaskID/全局几何特权输入。预测Y=safeGOAL-collision，逐executeddecision同一终态Y，无bootstrap/shaping/额外reward。

这至多拟合feature条件投影E[Y|z,a]。184features未证明是完整H的充分统计；随机当前创新不能消除压缩后隐含混淆。pi_sigma续行不是deterministic中心Q，密度/量化语义不偷换。T622混合后续行为的MC和T625单prefix三branch/旧pretrained失败不作初始化。单行为数据区别实际成立，但不证明Critic有可用动作梯度。

## 两模型与loss

新初始化两同seed/同state_dict的186→64→64→1 tanh Critic，各16193allocated参数，最后输出层zero-init保证初始函数相同；blind仅mask当前action2输入，effective类不同照报。输入归一化仅train数据或原Actor固定normalized定义，明确选择一种并锁定，不由heldout调整。均500freshAdam lr1e-3 eps1e-8、同batch序列/固定seed69220261121、1024行/步，无中间模型选择。每row task均匀→该task训练episode均匀→固定slot0..1023均匀；unexecuted slot lossmask0且不重采，maskedMSE乘2。于是期望loss=(1/407)Σ_task [1/(n_task*512)Σ_episodeΣ_{t<T}MSE]。512只是常数尺度，不是horizon截断；真实原caps最大64112、记录decision≤1024应预检。不能除以未来实际长度T或把任务/trajectory总loss都叫等权。所有同Y组保留，不选择只mixed拟合。

每模型live500/shadow500/independent500，合计Value各1000/Actor0；Critic自始不指导Actor。两个固定模型训练完再算所有heldout指标。报告逐ID、all/old/field/mixed分层、initial与末模型fixeddecisionloss和每trajectory首决策MSE，负差全报。aware的两个globalheldout指标均至少比blind低10%，old/field各fixedloss不得高于blind2%，全部finite/复算/hash过才准入已预留nativephase；否则本job报告停止，不另跑回归诊断或换seed/模型/标签。

## 同一job的条件真实因果检验

四ID在新拟合前固定：dev_task6_00、dev_task6_02、field0_offset1、field7_offset2。选择其预定firstheldout轨迹（非fieldrep3/fieldrep6）decision20。prefix存活/原起态/goal/scene、全部外部状态、前20执行动作与nativecounts/物理时钟、自己的GRU须精确重构；不复制别人的hidden，不重置时限。已有原数组只读核查显示decision20均存活且四动作distinct，不能按新预测或分支结果更换时间、幅度、任务。

第20动作四种为clip(mu_T501(H20)+[+.05,0])/[-.05,0]/[0,+.05]/[0,-.05]，float32执行。prefix和这一动作都明确intervention ownership1，未执行提议/raw/logp按强制标签核验，不称普通onpolicyPG/成功BC数据；之后恢复T501sigma.05自主随机续行，owncarry按真正被干预的外部历史延续。四suffix创新seed固定[69220261121,task_index,rep0..3]，各action共享同rep创新是CRN，每分支观测/状态/动作可分离，不能要求后续相同；每次从原起点重生prefix并计全部物理。四action全部预先预测并保存hash，再做64full+64exact，exact不增加独立样本。所有成功/碰撞/超时/边界/终态Y和全数组/积分保留并独立审计。

每prefix预定driveplus-minus和steerplus-minus，共8 contrasts。报告4×4实际Y矩阵与4配对mean gap/预测gap/误差/ties；|empiricalgap|≥.5为finiteinformative，要求≥2项跨≥2prefix且含field、aware对全部informative符号正确、无informative错号，才FINITE_ACTION_CALIBRATION_PASS；缺informative为inconclusive/停。blind同prefix不同action必tie，不人为分胜负。N4/CRN只是有限描述，不能给高置信真实Q排序或gradient认定。无论结果本job不更新Actor；之后若要guidance必须新的原理/证据/注册，而非自动队列。

## 具体上界与准备

四case stepcaps实际23334/23334/34697/64112，64full+64exact最坏4,655,264物理，128science+6管理=134cmd。模型训练/heldout条件不满足则0native，不为验证而改gate。拟单job5700main(学习≤900)+600audit/root7200s≤2h，Value live/shadow/independent各1000、Actor0、额外optimizer预检0，新root512MiB/global10GiB/free40GiB/8GiB归档触发；CPU8/10/12 native14学习audit/400%Nice19/4GiB无swap/CUDAoff同设备。释放训练大features缓存后再起nativeworkers避免内存叠加。

先实现真正fixedslotloss一致/实际teacherowncarry184提取和动作列blind/realMC标签/wholetrajectorysplit零泄漏、zero-init/500step影子模型整态独立存储/finite梯度与density/quantization、真实worker-Session强制prefix/一branch/随机suffix/reset/RNG/nativecount/ownership/审计无native路径全入口预检，特别覆盖有/无native管理命令均6及完整postexit审计字段，不重犯T691错误。新场景初态零GRU/canonicalfactory guard在正式首物理前核验，预检不冒称已有新物理。

再按剩余ledger和真实字节/资源独立worstRESERVED、精确GitHub源shell已退/树相同后唯一launch。运行健康只读、无并行science/audit/backup/source，原绝对deadline不延。失败不增加branchrep/seed/幅度/时刻、模型/更新数/奖励或强行Actor。T501及全原384+new+same407/audit最终保留门不变。
