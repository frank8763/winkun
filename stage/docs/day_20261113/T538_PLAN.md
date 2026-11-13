# T538 持久化修复，等待新增5,000步替换授权

T537在首窗10分支完成后，拼接300个ZIP字段产生137,994字节，超过锁定的131,072字节文件cap，实际5,000物理/12命令停止。10个case保留指标与数组哈希，两个repeat的哈希一致，但完整数组包未落盘。原FAILED/协议/error全部保留，metadata审计PASS不等于原始数组完整验收；后3窗未执行。

修复只改变持久化，不改科学参数：26类数组沿branch轴stack成26字段，使用不压缩NPZ。按四个已保存的同shape真实suffix测量完整10branch文件字节；不压缩使固定dtype/shape的字节上界不依赖扰动数值。每完成一个分支，立即将目前已完成的所有branch原始数组写入临时文件、fsync、原子替换窗口文件，再写case元数据。因此不能再先跑完10个分支只存hash而没有原数组。

预先核算4个完整窗口+1份最大原子partial+320KiB元数据总量≤1MiB，byte上界及截断负例在T538_STORAGE_CHECK锁定。运行中仍全局1GiB/free40GiB，不删除唯一证据；每窗输出文件上限来自未压缩固定schema真实测量，不事后扩大。

所有科学设置精确T537：4snapshot，D=T527-T507，alpha0/±.01/±.005，perturbed hidden从同prefix重新计算，原生未来120闭环1秒，每个exactrepeat，共40branch/20,000新物理/45命令。首窗10条全部必须精确匹配T537已记录policy/arrayhash/指标，明确历史重复不作独立新增。T537旧5,000步不抹除，合计25,000，比原20,000预算增加5,000，**需用户明确替换预算授权后才正式注册/启动**；不借新编号偷偷扩大。总截止仍20:00，原生19:45，main480/audit240/backup240/source120，更严余时准入不变。Actor/Value/PG更新仍0，无新机制网格/阈值更改/模型晋升。

独立audit重读全部新保存完整数组、逐条重算策略、扰动hidden、指标与两尺度有限差分，并校验原首窗历史hash。0native audit，不把同seed/同source原生重复算独立数据。全部原件RAM远端逐成员SHA/source后报告，无自动下一学习。当前文档与storage_check只是准备，授权文件/protocol/launch不存在前不宣称启动。
