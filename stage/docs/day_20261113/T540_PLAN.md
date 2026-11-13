# T540 仅补同一冻结候选剩余22项初验

T539实际1SGD+1shadow已完成，235,017物理/16命令后发现输出cfg继承了原T531的out字段，主动TERM停止。旧54个已归档原件SHA全部未变，新建本轮文件18个已逐SHA归位T539/misdirected；原原件/FAILED/协议/错误/partial保持。旧8中心初验8/8，场地只2项(1GOAL1TIMEOUT)，另外2项中断和20项未开始。独立重算1SGD/完整训练密度与候选hash通过；全field24仍null，不晋升。

本项不重新训练、不换模型、窗口/奖励/门槛/任务/seed不改、不重跑原10项。固定candidate SHA5d2e1771fdc2c53f2583a3385594ba37555578838b64724aee74a023161a252d，仅补原顺序剩余22项的从原点自主初验。包括重新执行两中断项，它们旧43,623步仍计入T539，不充新独立训练或抹账。保留旧10与新22合并同policy first-pass old8/field24；即使全过也不晋升，因为32exact/完整old275/原six-eight/fresh/reload均未执行。

工程修复是明确 `cfg['out']=canonical_current_OUT`，并要求zero-native预检对比继承旧out确保不同。所有worker请求/compact/rows均当前OUT，禁止写任何旧runs。旧源码及原protocol不改，新的输出路由规则独立注册；不借此增加训练或搜索。

紧凑证据无压缩NPZ保存actions/最后native_count/最终flags，前面全部hold=100、flags0必须先断言；独立audit按此精确重建counts/flags并比原生成SHA。固定shape未压缩最大字节在预检锁定≤10KiB，22文件最多220KiB；其余148KiB供小协议/metadata/日志，own368KiB/global1GiB/free40GiB。完整120在RAM产生后只保留hash，离线不能重构，明确不声称full120独立验收。terminalcontacts保留hash，碰撞标志和场地qualifier保留。

新原生上限sum22caps=1,361,791，连T539235,017最多1,596,808<原3,201,204；新26cmd连旧16=42<原68，不扩大科学物理/命令包络。0Actor/Value/PG，candidate引用不另复制。main480/audit120/backup180/source120，当前user20:00全部停仍不改，新的只读补验原生截止19:50以留10分钟收尾(旧已结束T53919:45不修改)。CPU14/15合计1核/GPUoff/Mesa/SCHED_IDLE/Nice19/低IO，Phase隔离。源码精确远端/零native预检后唯一launch，失败保partial不重启；独立compact/初验合并核验及全部原件RAM远端SHA/source。此项可能因墙钟未完成，缺测/null和行政中断分列，不承诺全过或执行32exact。
