# Stage 迁移与恢复

目标仓库：frank8763/winkun。仓库可见性为 public。用户授权迁移 Stage 并清理冗余；完成后等待用户检查，不继续实验。

## 归档组织

- `stage/current/outputs/`：T148–T164 当前工作区的代码、协议、结果及研究说明。
- `stage/legacy/`：Stage TD3 历史代码和证据、实际运行依赖，以及清单列出的保留模型（模型在 Release）。
- `stage/runtime/mk-mini/`：实际 Stage 环境加载的模拟器文件和场景资产（大资产在 Release）。
- `stage/manifests/`：原始路径、归档路径、字节数、SHA256、运行环境与暂停记录。

本归档不是所有早期 Stage 分支的完整镜像。未覆盖的历史目录和 T145/T146 原始目录仍留本地；不删除 Phase。旧目录中未被引用且不属于最终/具名模型的逐回合中间断点，将在完成远端验证后按明确清单清理，代码、日志和结果保留。

## 下载与校验

1. 克隆本仓库。
2. 从 `stage-archive-2026-10-30` Release 下载全部 `stage-artifacts-*.tar.zst` 和 `SHA256SUMS`。
3. 在附件目录执行 `sha256sum -c SHA256SUMS`。
4. 在仓库根目录逐个执行 `tar --zstd -xf /附件目录/stage-artifacts-NN.tar.zst`。
5. 执行 `python stage/tools/verify_snapshot.py --root .`，核对归档清单中每一文件。

附件保存全部选定原始文件的字节内容；不把重复跑出的实验当作独立样本。旧绝对路径保留在原始程序及协议中，避免改变源码哈希。因此此包是可核验的原始快照，**不是已完成路径重构的一键跨机器训练工程**。

## 恢复原始路径

`archive_manifest.json` 将每个归档文件映射到原始绝对路径。保留源文件哈希，需要在隔离环境中恢复原目录结构并安装匹配的 Python 包。现有文件若不同，必须先调查，不能直接覆盖。`python_packages.json` 记录本机包版本；MuJoCo/驱动与图形环境还需目标机器核验。

T164 不支持直接重启现有入口；后续需要明确实现并验证状态恢复，包括网络、优化器、replay、RNG 和预算计数。保存的是回合边界检查点，不含停止时完整实时模拟器状态。此次整理后等待用户检查，不进行训练、评估或自动恢复。

## 核验状态

远端内容、附件读回哈希和本地清理结果以本目录最终 `CLEANUP_REPORT.md` 及对应 JSON 清单为准。迁移过程中不将“已准备”写成“已远端备份”。

## 迁移后的日常运行

本机主工作目录将设为原工作区下的 `winkun/`，以 `LOCAL_WORKSPACE.json` 的绝对路径为准。通过根目录 `run_stage.sh` 使用已经安装的 mk_mujoco Python 环境，并将仓库内已验证的模拟器源码加入 PYTHONPATH。保留模型/数据在本地仓库路径可用，大文件由Git忽略。旧 `outputs/stage_t*` 路径通过兼容链接指向仓库中的同一份文件，保持历史协议的绝对路径可用。

新实验只在 `stage/experiments/` 开发、在 `stage/runs/` 保存结果；这次仅执行只读依赖/哈希检查，不继续训练。跨机器仍需依赖安装与旧绝对路径适配，不声称已实现一键移植。
