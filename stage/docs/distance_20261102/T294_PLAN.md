# T294：第一动作的同状态、固定续行回报对照

用户11月2日继续推进授权。T293 matched full/blind预测失败，不接Actor/不扫模型；本次新问题是从操作上取得第一primitive动作下的固定续行价值，而不是从混合行为预测外推控制梯度。safeT255/所有失败/高清视频保留，Stage/Phase隔离。

## 固定两起态与四个分支

只取T278已冻结的实际六米终点余1m LEFT/RIGHT完整原生reset快照；环境/目标/朝向/积分warmstart/执行器/历史/时钟/上一动作保持。初始120用T289已保存同snapshot首行作零物理核验，其源Actor不是本次Actor，本次加载并锁定safeT255。两种来源只共享初态，不混模型。原两起态都完整56秒超时，不宣称已会后段。

每起态四分支：原冻结Actor全程parent控制；以及以当前parent分布为参考的三个固定copula边际分位latent(0,-1)/(0,0)/(0,+1)，仅强制第一个100 native步hold（0.2s）。三个分位drive逐位完全一致，只有steer变：LEFT约-.03956/-.00956/+.02044，RIGHT约-.04760/-.01760/+.01240。不是幅度/seed/时机网格，不选择结果好的参数。parent确定性drive与截断分布中位drive有区别，所以parent与quantile0不当作纯steer因果比较；steer对照只能在三个同drive分位之间。

第一个hold后，每一动作都由**同一个冻结safeT255 Actor**在分支各自真实状态上输出；hidden仍每决策清零，前动作/hold输入按实际执行更新，不回放旧未来动作、不保留后续随机噪声、不改物理位置。这样定义的值是Q_(固定确定性续行)(初态,第一动作)的单次确定性原生回报，不等于T293混合行为条件预测/AR噪声续行的Q。

每分支一次unique+一次独立精确重生，共8+8/16命令，≤448000全部物理步，900秒/own3MiB/global1GiB/free40GiB。两个parent的全部12数组/终态积分必须与已消费T287局部控制精确一致；另外重复所有分支逐数组/积分重生。新物理全部计费，重复不算新独立seed。full120/12数组通过state-chain无损保存≤192KiB每unique，partial replay≤12KiB、metadata768KiB，整体<3MiB。

## 报告的价值及解释

保留原环境reward和所有原目标结果；另只读计算G_eta=.1（原T280有界endpoint接近代价）作为已明确目标的固定续行MC值，主要safeGOAL仍只有原生无碰撞GOAL。三个quantile的驱动同值，可计算Gplus-Gminus和除以实际steer差的局部有限差分；这不是总体无偏PG、校准Critic、跨起态通用动作排序或六米成功。contact永远失败，不以更近代替GOAL。

first-action外部强制是诊断干预，不是部署教师/路由；不训练Actor/optimizer/新Critic。数组中的logp=0是显式deterministic占位，禁止当on-policy AR/PPO训练样本；若以后利用这些数值，需新的固定续行MC策略改进合同，不暗中当当前探索分布Q。未通过两个固定原目标也不自动加动作幅度、持续时间、seed或起态格点。

## 预检与结束

已零物理零优化器核验源/模型SHA、初始120、三个quantile同drive且steer严格有序、全函数名/容量/预算。正式源码和协议提交/远端完整根树核验后唯一launch，尚无launch/status时不可称运行。持单写入锁，不写其他runs/源同步、不改训练中的协议。

结束所有父/inhibitor/timeout退出后独立审计全部payload/原始物理命令/12数组/精确重生/模型SHA；对8条每个第1hold后动作逐状态重算，必须等于同冻结Actor；parent控制精确历史；独立计算两起态4分支回报与分位差。0新物理/优化器audit，然后RAM完整远端逐成员SHA读回，最后源同步。失败/partial保留不重启。

完整275旧/54六米/fresh16/78读回本轮不安排，未执行不得记失败率。无论分支结果如何，均非新学策略，不晋升、不进7/8；已有T255原安全模型不改变。与T269不同：其source是T267候选原轨迹24/36s的左侧状态，本次是safeT255在两原六米终点附近fresh-history reset的首次控制值，不能把旧结果复制当新对照。
