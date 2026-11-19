# T651：在已修复的原宽场景上续做一次有界策略梯度

用户最新要求“检查，然后继续下一步”。T650已冻结原T501，原五米/宽六米/八米6fresh+6exact全部安全到达、全数组和积分逐位同T636、独立审计PASS，0学习。T649两臂各1提议均KL超限回退，原宽任务初始化被绕过导致旧场景几何/八米目标不一致，不能作为原wide算法性能结论。T650修复与勘误均保留，不重启或覆盖旧任务。

本轮是已批准T649窗口及剩余额度内的一次修复续行，非新无限实验。不延12:08:57学习/12:38:57native/13:08:57all。原T649实耗6,780,012步258cmd2124.778s、Actor live/shadow/history各2（不是账本的16预留上界）；T650实耗373,288步18cmd92.951s/0学习。本轮最多14M物理465cmd3000main、8Actor实际/影子/历史各次、0Critic/额外optimizer预检、root192MiB。三run合计最坏21,153,300步741cmd5217.729main、Actor各10，均≤原31M/1020/6000/各16；三run物理文件合计预留≤原512MiB，global10GiB/free40GiB不扩。全部准入再运行。

## 修复后的任务与数据

调用T650已复现的canonical wide/radial工厂。所有任务首步前按实际历史task族锁定scene；19训练任务和14关键评估另锁初始120、203积分、goal、h0。所有八米radial目标必须等于T636的左右目标。原field24历史实际上是narrow geometry，保留legacy原ID，不以修复名义静默变宽；未来真正wide全场地需新几何和ID。

修正另一项判定边界：±7米限制仅适用于带fullfield_start_pose的field24。原wide8 RIGHT目标x=8.1787合法，不能把原wide8到达判为全场地越界。原始field_boundary测量仍保存，但safe/outcome仅对field族应用。零步正负回归同时检查该规则，防止越界场地被误放行。

只替换原T649四个实际错误的wide6/wide8任务组：各4新随机T501轨迹+4exact，16fresh16exact，最多1,493,344步38cmd/600main240audit。原T649其余15任务60条数据的scene/initial与canonical reference一致、权重完全相同，重新逐SHA/fullhistory/density核验后复用；明确60不是新独立数据，不重采浪费。统一19×4=76轨迹用于本轮，原错误16不再混入。至少2新任务有成败差异才学。

## 单轮学习而非重新声称公平对照

本轮只做pg单臂，最多8提议，不重复已回退的BC、不宣称PG优于BC。原T501初始化、全64772参数、真120因果GRUcarry、原T649终止safeGOAL概率/固定512/任务LOO/clipped ratio目标、全384每提议保持/keep10均不变。唯一一次明确步长修正：Adam lr由1e-4降为3e-5，原因是T649第一提议最高latentKL .0162已越.01，尚未提交任何更新。不是按新验收扫lr；不再自动调lr/阈值，不引入QP/教师/规划。

每提议后真实19task latentKL均值任一>.01即整Actor+Adam回退并停止；拒绝仍计一次。8live+8shadow+8独立完整历史重构为硬上界，0额外optimizer预检，初始密度/梯度只backward不step。不再将T649单次回退后的测试视为新策略改善。

完整field24+6旧哨兵和14full exact，旧6+原field5全保且新增原起点安全成功才补377组成同候选407。原384逐ID全保且新增才研究保留，否则T501不动。初始几何合同不通过立即0步拒绝；随机成功/代理loss/小KL/六个案例重现均不宣称全场地通过。缺测null。

CPU实核8/10/12及主管14，400%低优先，GPU学习让出其他计算，4GiB无swap、亲和性守护、Phase全隔离。先shared独立审计与原生全退，才PG；训练释放后才验收native。每main2400、audit600、学习600相对上限且更早绝对截止从严。科学完全离线；结束后备份300/source180，断网保原始pending，不运行旧T646补传阻塞本轮。资料和原错误记录都保留，必要结论更正回GitHub。
