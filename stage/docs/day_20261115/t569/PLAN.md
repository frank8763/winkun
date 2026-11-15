# T569补齐冻结T562完整32开发基线

用户要求下一步继续并问全场地还差什么。T568实际学生39s前缀＋训练反馈safeGOAL72.412s、exact及audit/备份均过，但0模型学习，不能冒充学生成功。为下一学习保护真实能力，当前16:30学习截止后只做冻结测量；原17:00native17:30all和日19M/600cmd/12300main不扩。

固定同一T562 model/policy SHA。复用T567R已审三条实际自主full（sector0timeout、sector1/4GOAL）及其3exact；不复用教师T568、其他模型或无物理结果。只新增缺29项（old8+其余21field）从原起点/原goal/clock/geometry/速度/动作，单worker串行纯Actor，0teacher/优化器/RL。合并成同策略old8+field24完整32，所有任务开发已消费。按任务ID报告成功/接触/超时及相对T550原23成功的损失/新增，不能净总分掩盖失守。

完整32不是正式迁移/稳定资格：T562原27离线门仍未过，新增29只有1次自主，不另29次精确原生重放。新29实际完整120/history/hidden/动作/密度/奖励/积分在RAM生成并由主线逐决策核对，紧凑保存actions/native_counts/end_flags及所有完整数组hash/积分SHA，离线audit只能重核保存紧凑字段与计账，不能独立还原完整120。原3full另重新读回全部数组/模型/时序核对。报告明确新29无独立物理重复、不把mainRAM检查写成独立全history审计。

没有留出消费、checkpoint选择、噪声/seed改变、新学习或新模型，未过资格不晋升、不进真机。无地图/规划/路点/运行教师/router；T550原保留Actor不替换。

## 预算与截止

新增max1408266物理/33命令/900main，audit180/backup180/source120共1380＋缓冲，必须主line900全在17:00前，完整收尾17:30前。所有原始记录/部分/失败保留；若wall时间不足导致未完成，未测null而非失败率，不启动续跑旧目录。

own3MiB：singleworkerpartial1280KiB（838决策packed真实schema外推1202992B<1310720），29compact×12KiB＋原子1compact12KiB，metadata/log/manifest1.25MiB，合计2990080B<3145728。已实际生成838决策不可压缩紧凑测试10275B<12KiB。模型只引用，原3full只引用不复制。global完整own+131072guard<1GiB/free≥40GiB，CPU14/15总一核idle/Nice19/CPUWeight1/IOWeight1、GPUoff/Phase隔离。

唯一driver科学结束且native退出后独立紧凑/原3full audit、本地manifest、所有新raw RAM远端逐成员SHA/source根树；断网pending明确。活跃单writer不改源/协议/写其他runs/并发audit备份，不自动训练sector0；有资格恢复数据留待新学习窗口。
