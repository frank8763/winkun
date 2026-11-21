# T692：同一行为策略的动作输入未通过留出门

COMPLETE / 独立audit PASS / HELDOUT_ACTION_INPUT_GATE_FAILED。两freshCritic各500live、500shadow、500independent，Value live/shadow/independent各1000，Actor0。0物理，6管理cmd；38.742456main+33.348723audit秒。driver838672/helper838678/science838679,838813及stage-t692 unit全退。精确GitHub树1e27d8655218df2b32dbfd329aa3ddb3d493e80f/commitaba6d31a5a053071adf744d10aaa85777ae68b67后唯一launch，source300已按commit去重结算。全模型/轨迹源/失败证据本地保留，T501仍保留。

|最后固定模型|train fixedloss|heldout fixedloss|heldout初决策MSE|heldout原384 loss|heldout field24 loss|
|---|---:|---:|---:|---:|---:|
|aware|.03768948514|.05987071151|.2859139637|.05387420248|.07550743214|
|blind|.03670338733|.05879567476|.2882113587|.05305773202|.07080399579|

aware fixedloss比blind高约1.83%，初决策MSE只低约.80%而非至少10%，field误差高约6.64%超2%。四预定门只原384不高2%成立，global两项与field门失败。两arm同初始化/同500batches、全407/1724真实因果历史和1293train431heldout重构、fixedslot1024mask×2、无future输入、MC terminalY、实际command量化、Value/Adam/trace逐位独立复算均过。不因实现PASS称学习有效；feature条件投影不当完整历史Q，不保证当前动作信息可辨识或最优policy。

条件native64full64exact全部未执行，不能报branch无效/零成功率/真实actiongap或Qgradient已验证。PRE_NATIVE_PREDICTIONS只有固定模型对预定4command的前向值，不是新原生证据或planner。无论未执行原因，Actor未更新、full407=null，无晋升。停止这次固定valueclass/500step/数据split，不加head/epoch/seed/label/branchrep挽救。更不能从同policy排除混合标签后仍失败推断所有Critic无效；σ.05有限覆盖、冻结feature混叠/有限样本/模型拟合等仍是兼容解释。

第一prepare错误把branch须存活20决策的检查施加全部历史，合法短terminal轨迹被拒，0Value更新0native；源snapshot保留144秒上界。修改为全data0<T<=1024、branch四源仍T>20，没有删除短回合或改科学门。成功preflight20.701s+补充metadata92s，保守累计257.158s全计。未发生科学重启。

下一需脱离同batch拟合/盲目附加价值网络或参数/权重救门，先复盘终态反馈信用与真正确定性旧能力保持的可检验变量。没有新的科学run/登记/launch，不将此负结果包装成已解决导航。原384全保+新增+same407/audit的保留门不变。
