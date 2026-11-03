# T328：宽缺口80秒人工采集，时间合同显式改变

用户反馈全速仍无法到达。原T327已结束（2无碰撞TIMEOUT，完整重生），两条全部280行drive=.999，实际观测速度平均.103914/.096308m/s，56秒时余.904028/1.012562m。采样位姿弦长5.850239/5.426357m，径向接近5.095972/4.987438m；弦长不是连续真路长。最大drive是规范化命令，不等于1m/s；原配置speed_gain=.03/max_wheel=5，对应命令速度上限约.15m/s，实际原生速度还受物理响应影响。不能怪用户未全速或据两条宣称原任务全局不可达。

本轮是独立课程数据采集，保留T327同一1.17m宽缺口，只把cap由28000原生/56秒改为40000/80秒；clock字段按新cap计算。动态速度增益、车体、障碍、目标半径.20m/碰撞优先/2ms/hold100保持。原56秒结果与原地图验收不覆盖，80秒课程成功不当56秒原六米通过；不把放宽时限包装算法提升。课程80秒足够与否仍看实测。

只1次新真实human尝试+1全12数组/history/终态积分原生精确重生，不扫时限/速度/seed或追加次数。1零物理真实geometry预检+1record初始化+1replay=3全部命令，≤80000全部物理，0Actor/Adam/梯度/RL。main1200墙钟末240留重放/收尾，外部timeout更早从严，不延时。等待不推进物理。原128来源和规范化输入域±.999保持，慢速播放只影响墙钟。取真成功参考而非让AI代驾驶。

使用T327同一宽场景构造与mj_setConst/BVH、ray/独立MjData接触/渲染预检；新初始120/hash/时钟单独核验。复制的session_base.py明确把所有硬编码cap改40000，完整packet/hold/replay使用同cap，旧原件只读。预检核对实际max_steps=40000，初态scene/积分与T327相同、clock初始1，并测试临近cap的零原生collector替身，防止页面写80而物理仍56。界面必须显式80。

own4MiB：单raw≤768KiB+同大小原子partial=1.5MiBpayload，metadata1.5MiB，buffer1MiB，global≤1GiB/free40GiB；不生成模型/磁盘视频。所有成功/碰撞/超时/删失及动作请求先持久化再step，0伪造expertlogp。记录真实全120、history、积分、命令与奖励。结束父server/render全退后独立300秒raw/history/reward/80秒clock/序号/终止与精确重生审计，RAM全部远端逐成员SHA/source同步。active单writer禁外写runs/source更新，Phase隔离。

T327已真实动作/精确重放并完成审计与备份后才本protocol/sourceSHA/预检/干净提交/精确远端根树/唯一launch。当前只是准备，不宣称已采集新成功或训练。后续若课程成功，可用于原goal下的学习，但缩时/原窄缺口迁移必须独立验证，不自动七八米。

时间与删失语义依据[Time Limits in Reinforcement Learning, Pardo等](https://proceedings.mlr.press/v80/pardo18a.html)，本次核对正式摘要，区分任务真实有限时域与行政截断；本课程80秒是真实新任务上限，56秒旧任务仍独立。不是偷偷把旧TIMEOUT重标成未终止。课程采集与原科学结果分列。
