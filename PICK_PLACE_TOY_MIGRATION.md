# pick_place_toy：迁移结果与审计依据

真实数据迁移和下游加载验收已于 **2026-09-27 完成**。正式路径为 immutable Raw v30 → 独立迁移工具 → Raw v34 → 正常 Zarr v15 exporter。正常 reader 仅接受 v34，`dexmani_policy` 未修改。

## 已发布数据

| 数据 | 本机路径 | Episode 数 | 行数 |
| --- | --- | ---: | ---: |
| 原始 v30，保持只读 | `episodes/pick_place_toy` | 61 | 14,309 |
| 已迁移 Raw v34 | `episodes/raw_v34/pick_place_toy` | 61 | 14,309 |
| 已导出 Zarr v15，13 种模态 | `datasets/pick_place_toy.zarr` | 51 | 11,710 |
| 未进入 Zarr，仍完整保留在 Raw | 见下表 | 10 | 2,599 |

183 个源文件的迁移前后及最终复核 SHA256 全部一致；122 个 RGB/depth 文件按字节复制，没有重编码或重采样。九个直接映射数组及行顺序保持不变，包括失败行和 NaN。输出经 staging 校验后原子发布，工具拒绝覆盖已有目标。

## 语义证据与映射

[A01–A19 操作者证词](outputs/data_refactor/pick_place_toy_operator_attestation.md)、[历史源码语义审计](outputs/data_refactor/pick_place_toy_source_semantics_audit.md)及[原始采集日志核对](outputs/data_refactor/pick_place_toy_original_log_evidence.json)共同支持此次人工语义审计。[迁移证据文件](outputs/data_refactor/pick_place_toy_migration_evidence.json)将依据绑定到各段三个源文件的 SHA256。

操作者确认了采集方式、设备及关节语义、实际安装、相机标定、触觉归零和已完成的缩放、最终动作目标、观测先于动作、没有删行/重采样，以及暂停后保存退出和录制连续性。**这些是明确归属于操作者记忆的证词；没有恢复逐次 converter 执行日志、原始配置快照或动作发布时间戳。** 文件哈希、默认配置和历史源码本身不能单独证明这些采集事实。

| 项目 | 本批次迁移决定 |
| --- | --- |
| 最终动作 | `action_arm_joint_sent` → `action_arm_joint_target`；`action_hand_joint` → `action_hand_joint_target`；不使用旧 `action_arm_ee` Cartesian intent |
| Effort | `arm_tau` 原值 → `arm_effort`；SDK-native，SI 单位未验证，不标为 Nm |
| 触觉 | 两个数组均 scale=1；保持原值和 NaN，不再乘 10，不重新归零 |
| 相机 | 使用各段已有的 aligned RGB-D 内参、畸变、base-from-color 和 depth scale；不加载当前相机标定替代历史数据 |
| 物理手部安装 | 位置 `[-0.015, 0, 0]` m、wxyz 四元数 `[0.707107, 0, 0.707107, 0]`；依据操作者确认及一致的历史配置，写入各段 Raw |
| 离线几何 | mount 只读 Raw；URDF、指尖 link 顺序和点云处理参数来自显式 exporter 配置 |
| 桌面平面 | 历史平面未获证明，显式 `remove_table=false` |

旧 status 按 v30 解释：0 正常、2 IK 失败、4 retarget 失败。`frame_valid` 还要求 action queued、旧观测/相机标志有效、arm/hand connected 且 hand qpos 不 stale。保留了全部 31 个 false rows，不用证词覆盖已记录异常。触觉 validity 与数值一致性独立检查：有效必须全有限，无效必须含 NaN；矛盾时拒绝迁移，不填补或修复。

`episode_valid` 同时受源文件事实和生命周期证据约束，不能用 requested true 覆盖失败事实。相邻有效行间隔超过两个控制周期（16 Hz 下为 125 ms）的四段标为 false，其余 57 段依据证词、日志及源事实标为 true。Raw 不保存 synthetic timestamps 或旧诊断字段；这些历史证据保留在原始 v30 和报告中。

## 整段拒绝的 10 个 episode

下表均为 `episode_20260827_` 前缀，行索引从 **0** 开始。“观测/相机无效”表示旧 `observation_valid` 和 `flag_camera_fresh` 同时为 false；具体物理原因尚未建立，不能断言相机硬件故障。

| Episode 后缀 | 完整行数 | 源文件记录的异常 | 额外生命周期判定 |
| --- | ---: | --- | --- |
| `172305` | 304 | 行 264、265 观测/相机无效 | 有效行间隔 187.5 ms，episode_valid=false |
| `175240` | 301 | 行 27、28 观测/相机无效 | 有效行间隔 187.5 ms，episode_valid=false |
| `194525` | 286 | 行 238、239 IK 失败 | 有效行间隔 187.5 ms，episode_valid=false |
| `195951` | 318 | 行 157 聚合和密集触觉含 NaN，旧 validity 均为 false | 触觉独立于 frame_valid，仍整段拒绝 |
| `220112` | 223 | 行 220 观测/相机无效 | — |
| `220747` | 313 | 行 51 观测/相机无效 | — |
| `220919` | 225 | 行 94 观测/相机无效 | — |
| `223607` | 240 | 行 18 观测/相机无效 | — |
| `223729` | 192 | 行 111 观测/相机无效 | — |
| `224527` | 197 | 行 133 观测/相机无效；行 140–158 IK 失败 | 有效行间隔 1250 ms，episode_valid=false |

所有坏段均完整保留在 Raw，Zarr 中为 0 行；没有拆段、删除坏行、拼接、插值、平滑或重新采样。

## 实际执行与验收

实现起始代码基线为 `753d7ec36085c84f479da4855042c158582484a8`。完整结果保存在本机 `outputs/data_refactor/`（该目录不提交 Git）：

| 记录 | 实际结果 |
| --- | --- |
| [有证据 dry-run](outputs/data_refactor/pick_place_toy_migration_attested_dry_run.json)及[迁移完成报告](outputs/data_refactor/pick_place_toy_migration_completed.json) | 实际迁移退出码 0；61 migrated、0 blocked/failed；逐段 v34 reader 重开、数组/媒体/源哈希验证通过 |
| [FK 预检](outputs/data_refactor/pick_place_toy_preflight_fk.json) | 全部 measured/target arm 及 fingertip FK finite 检查通过 |
| [显式导出配置](outputs/data_refactor/pick_place_toy_export.yaml) | 完整点云参数，remove_table=false，无第二份 hand mount |
| [Zarr dry-run](outputs/data_refactor/pick_place_toy_zarr_dry_run.json)及[实际导出报告](outputs/data_refactor/pick_place_toy_zarr_export.json) | 全量逐帧转换；51 段通过、10 段拒绝，无额外派生失败；两次边界、拒绝集合和逐段模态哈希一致 |
| [最终全量验收](outputs/data_refactor/pick_place_toy_final_validation.json) | 源及目标全部文件哈希、Raw 数组/标定/有效性、Zarr 全量读回及 Policy 加载通过 |

最终验收检查了全部 13 种 Zarr 模态的逐段哈希和浮点有限性；每段关节/动作拼接、effort/current/触觉映射、EEF/指尖 FK 的 float32 数值逐元素一致，rot6d 合法。`episode_ends` 精确对应完整 Raw 段，终点为 11,710，无被拒绝段的残留行。

未经修改的 `dexmani_policy` 公共 contract 和 CPU `BaseDataset` 分别通过 action(19) 与 action_ee(21)。两种模式各有 11,659 个 horizon=2 窗口，实际读取首/中/末窗口，覆盖 RGB、点云、触觉及几何等七种模型输入模态。

独立审查曾发现 legacy 连续性检查遗漏相邻有效行的间隔，已修复并验证 3dt 拒绝、2dt 接受及源失败事实不能被 true 覆盖。Synthetic v30→v34→v15 全链检查通过：8 段/24 行完整保留 Raw，仅唯一 clean 段的 3 行进入 Zarr；另覆盖触觉乘 10 公式、validity 矛盾、缺媒体报告、无证据不发布、拒绝覆盖、staging 清理和源哈希不变。Native v34 smoke 覆盖结构拒绝、失败/NaN 行、camera START fail-closed、Raw mount、连续性/暂停 latch、ring cutoff/drain/overflow 及整段 export rollback。

以下离线检查全部通过（当时 121 个 Python 文件）：

```bash
python -m compileall -q dexmani_real examples
ruff format --check dexmani_real examples
ruff check --select F401,F821,F822,F823,I dexmani_real examples
git diff --check
```

没有执行训练、硬件连接、相机采集、物理回放或 rollout。离线验收不等于重新测量历史标定或真机安全验证。

## 重现入口

保留 [独立迁移工具](examples/migrate_raw_v30_to_v34.py)用于审计和重现；正常 reader 不承担历史版本兼容。现有结果不允许覆盖，以下命令须使用**尚不存在的新目标和新报告路径**：

```bash
python examples/migrate_raw_v30_to_v34.py episodes/pick_place_toy \
  --output episodes/raw_v34/pick_place_toy_new \
  --evidence outputs/data_refactor/pick_place_toy_migration_evidence.json \
  --dry-run --report outputs/data_refactor/pick_place_toy_new_dry_run.json
```

审核报告后，去掉 `--dry-run` 并使用新的 `--report` 路径实际迁移，再执行正常导出：

```bash
python examples/export_policy_zarr.py episodes/raw_v34/pick_place_toy_new \
  --config outputs/data_refactor/pick_place_toy_export.yaml \
  --output datasets/pick_place_toy_new.zarr --chunk-frames 16
```

证据只适用于哈希绑定的这批源文件。源文件或历史事实变化时必须重新审计，不能复制 assertion 来绕过语义 gate。

## 历史审计摘要

以下是此前取得的历史事实，保留用于解释证据来源；旧 v33 迁移和旧 salvage 准入规则已不再使用。整理前的完整记录另存于本机[历史记录副本](outputs/data_refactor/pick_place_toy_migration_history.md)。

- 2026-09-26 在代码基线 `436f8ba7dd1e5b70762e6ca0789bb226d9557138` 执行只读审计：61 段、每段 165–331 行，14,309 帧 RGB 全部解码、深度全部可读；640×480、uint16 深度无全零帧，RGB/深度/控制行数与 num_frames 一致。相机 metadata 一致且通过数学检查，深度尺度约 0.00025 m/unit；这不能单独证明物理标定正确。旧 v33 reader/export dry-run 在版本检查处拒绝全部数据，不能算后续 FK/点云验收。
- 旧 control_hz=16，timestamp 和观测锚点严格递增且间隔 62.5 ms；源时间戳非零且不晚于锚点，最大年龄 arm 33.18、hand 34.74、camera 58.95、VR 20.60 ms。整齐网格本身不能证明历史上没有暂停或重采样。
- 全部旧 task_label=pick_place_toy、success=true、truncated=false、stop_reason=manual、min_frames_met=true，camera writer 错误为空；没有 technical_status、had_pause、provenance_workflow、termination_reason。这些旧标志不等于当前有效性证明。
- 旧 status 0 共 14,288 行、status 2 共 21 行；8 段共 10 行观测/相机 false，其中一段与 IK 失败重合。异常相机行的源时间戳和彩色/深度帧号并未重复，不能按“只是重复帧”放行。除上述触觉异常行，扫描到的其他浮点数组无 NaN/Inf。
- 历史源码 `4bba54d` 提供 v30 producer/status/action 语义；`aac23d2` 之前曾在触觉 bias 捕获/减除前施加 0.1 scale。`data.h5` 的 mtime 晚于媒体；历史 v29→v30 converter 为 copy-only，但未找到对当前文件的实际调用记录。60 段含历史模型/标定哈希，`episode_20260908_212811` 缺少五项哈希；这些哈希未独立恢复全部采集工件。
- 原始日志在 `~/.dexmani/logs` 匹配全部 61 段名称及行数。Aug27 的 60 段有五个无接触样本完成 tactile software bias、camera ready、16 Hz grid 的记录；Sep8 对应材料分布于 recorder/hand/camera/teleop 日志。三个 session 的 shutdown fault 出现在 recorder 停止之后，不证明录制行受影响。
- 从提交 `23a789b` 恢复的[历史 salvage 清单](outputs/data_refactor/pick_place_toy_historical_salvage_manifest.json)记录 v25→独立 v26 副本、触觉乘 10、源不变。它只有集合级指纹，不能证明当前 v30 的逐文件变换；当前集合指纹与旧源和旧规范化集合均不同，metadata 变化也会影响哈希，不能据此再次缩放。
- 补充证词前的[无证据 dry-run](outputs/data_refactor/pick_place_toy_migration.json)完成审计但 0 eligible；[首次实际尝试](outputs/data_refactor/pick_place_toy_migration_attempt.json)退出码 1，61 blocked、0 published、183 个源哈希不变，未创建目标。这些是历史失败记录，后来在取得并绑定操作者证词后才完成实际迁移。
