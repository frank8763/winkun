# T562完整当前历史的表示学习接力

用户允许下午继续至15:00。T561正弦读出未过离线门，40.400939s，0native，1500live+1500shadow/1500audit丢弃，已完整备份、退出。不得称仍在运行。T562是预注册后唯一新表示学习机制，不续训旧目录或扫epoch/seed/omega网格。

## 决策

普通tanh和sine冻结特征读出均未过门；T560没有发现精确相同特征/冲突标签，但这不证明信息充分。下一检验联合学习内部表示是否改善成功动作迁移。从T561最终**被拒绝研究副本**初始化（并非晋升策略），freshAdam。冻结safeT550不变。

同120因果观测、64真实GRUcarry、相同原动作clip/native执行器/速度/时限/奖励。仅解冻encoder(15488)+GRU(37248)+student(40450)，共93186参数，mean/scale/原head/drive_adapter/steer_adapter冻结。原head的输入h会变化，所以不是旧公共函数严格保持。单Actor直接输出动作，无紧支撑中心库/路径检索/地图/规划/路点/运行教师/router/任务ID。

27原T550成功历史，23主+4已成功扰动，所有原完整120输入、前序实际动作/clock/GRU逐当前权重从h0重新运行；ownership强制帧不监督但保留history影响。不是缓存旧feature/hidden做伪端到端。所有27按完整长度padding并行成batch（CPU总额仍一核，不新增车物理），每history损失半mean/半top20% locationMSE等权，500完整batchAdam固定lr3e−4/clipgrad1；不改奖励、不新RL。与T561相比改变了初始化/可训表示/完整batch目标分配，联合pilot不可单独归因。

500live+500独立freshShadow完整ActorAdam/trace精确，独立audit再500丢弃历史重构，4合成预检Adam另计。固定500步不选中期checkpoint、不追加步数。相对学习1800秒且绝对13:00更早；超时保留partial、不恢复旧目录。

## 门槛与条件原生

同27每条实际clip命令RMS≤.02/max≤.1；失败0native记录，不把未执行当失败率。通过才2短500物理+2exact，完整保存history核验真实worker密度/step与sequence/hidden/reward；再32自主旧8+field24以及32精确native全数组hash/积分。必须保住原23成功ID；净分增加不抵消丢失。未过不保留新学生，T550不替换。过门只记有限开发支持迁移，old275/six54/eight54/fresh16/reload78/新扰动均null，不宣称全局稳定或泛化。

## 全部预算

新增最坏3,203,204物理/72命令/3600main（相对1800学习）/1200audit/600backup/180source，原日19M/600/12300不扩；T555/T557/T559/T561实际消耗由注册BUDGET合并，T5608.792秒零native诊断另列，未冒充训练。13:00停学习、14:30停原生、15:00全停，完整最坏余时不足不启动。预检估计live+shadow500步约368秒，非保证；主管线最晚main+3600独立停，早于绝对截止从严。

own7.5MiB：完整模型含ActorAdam3MiB、双worker partial2MiB、两short保存128KiB/原子128KiB、32compact×12KiB384KiB、其余metadata/trace/log/manifest768KiB，共6.375MiB，另1.125MiB预留。实际模型原始上界2465744B<3MiB。global1GiB/free40GiB硬守卫、CPU14/15总一核idle/Nice19/CPUWeight1/IOWeight1，GPU禁用，Phase全隔离。

唯一driver离线训练/条件原生退出后自动独立audit/本地manifest/全部原始RAM远端每成员SHA/source精确根树。断网科学不依赖LLM/GitHub，原始本地保存并明确REMOTE_PENDING；不能承诺断网后继续由LLM临时改算法。driver活跃期间不外写runs/改源协议/并发audit备份。无后续未实现接力，失败不杜撰仍训练；15:00是硬上限，不为了填钟点无限重跑。
