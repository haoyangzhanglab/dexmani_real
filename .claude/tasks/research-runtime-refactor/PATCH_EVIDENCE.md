# PR #17 有边界修复交付证据

日期：2026-09-20。仅 F1–F9；TASKBOOK 的 KEEP / REMOVE / DEFER 未修改。

## 版本、环境与授权边界

- 仓库：`haoyangzhanglab/dexmani_real`，分支 `refactor/research-runtime-v2`。
- PATCH_BASE：`464fde7c30ec7a1187da7ecee6bcb1b53223d64b`，等于本次开始的实际 HEAD / review tip。
- IMPLEMENTATION_BASE 保留为 `5b9f8968a10c845d695de418b632e6ce64de3be6`；原 main 为 `156b7b4a8b0aa5f42ba12eb2f1abe335d6c5dd1d`。
- 最终被测源码及测试 SHA：`4ee497d50e30303d75fb7a6dfd751a260e475601`。后续交付提交只新增本证据文档。
- 原 `9913e25f3dc65e6a3b631a3ee83a7b8474d6b7aa` 的 129-test、性能和 raw 数字仍是历史声明，不作为新版本结果。
- 开始时 tracked/index/untracked 均干净；无用户未提交修改。阅读根 AGENTS、相关源码、TASKBOOK、REVIEW_PROMPT、历史 HTML 和本地 STATUS；未发现更深的 AGENTS 覆盖。
- Python：`/home/zhanghaoyang/miniconda3/envs/real_robot/bin/python`，3.10.20；`dexmani_real` 来自当前 worktree。未安装或升级依赖。
- 导入及测试构造路径经检查；SDK 提交用 fake，真实消费/录制/监督 owner 仍执行。无设备连接、发现、home、servo、teleop、物理 replay、policy rollout、标定写入或相邻 policy 仓库修改。

## F1–F9 fact-check 与修复

开始矩阵全部为 **CONFIRMED**；没有 ALREADY_FIXED / NOT_REPRODUCIBLE / BLOCKED_BY_ENV 项。PATCH_BASE 未包含这些修复。

| Finding | 本地事实与实际 owner | 修复与回归证据 |
|---|---|---|
| F1 | `CommandStreamConsumer.resync_if_stale_generation` 可读到比 worker permit 更新的代；arm/hand mismatch 分支会 advance | 两个真实 consumption 函数在重基超越 permit 时结束 tick；`next_record` 检查稳定身份损坏；`advance` 锁内检查代。真实 arm/hand 三代交错、SDK 前撤销、SDK 内撤销晚返、下一 tick 顺序测试；旧 ACK 等待不得返回成功。 |
| F2 | `RecorderIO` finalizer / shutdown 异常写 motion fault | `io_worker` 仅记录 evidence failure，异常退出使该 channel 不可重用；活跃非 daemon finalizer、writer、staging 不伪装释放。真实 loop + Event 控制 finalizer timeout、未释放 writer、shutdown exception；verified cleanup 未确认退出仍禁止 IPC close。 |
| F3 | `PolicyRunner._poll_recorder` 的 max_frames 分支结束 trial | 删除容量到 `_invalidate_rollout` 的控制路径。实际 client.add_frame → STOP → RecorderIO → recorder 保存两行 prefix；RUNNING / trial count 不变，saved 只计一次；12 次交替 IK/workspace miss 不结束 trial、不扩容量、不继续录行。 |
| F4 | arm/hand history reader 优先 payload publish，未检查真实 ring commit | 两个 reader 检查 `source <= payload_publish <= ring_commit <= anchor`（无有效 payload 时间时按原兼容规则使用 commit）；不改变持久化字段含义。arm/hand commit>anchor、payload>commit 被排除，old-but-causal 通过。**这是 main 已有合同缺口，不是本重构首次引入。** |
| F5 | pending finalizer 阻止 B；optional service failure 在 ARMED 结束 run plan；START 失败分支复查不完整 | 唯一 recording context 保留原 trial 身份；新 trial 可不录制；所有 START 结果共同复查 B/S/Q、代和物理起点。optional startup / READY / heartbeat / exit 独立于 critical roles；保留所有已启动 handles。两 trial、START 成败中 S/Q/epoch/physical change、真实 supervisor 及 READY helper 回归。 |
| F6 | blocking predict 迟返后才计运行结束时间；父预算原因丢失 | motion 撤销 owner 记录一次软件 RUNNING end；runner 以对应 run 的真实 start/end 计分母。t=0 开始、t=2 S、t=10 predict 返回得到 2 秒；父预算 2 秒、实际 poll 在 2.1 秒撤销得到 2.1 秒和 timeout。重复 S/Q/FAULT/cleanup、ARMED/home 不覆盖；纯 command rebase 不结束 trial。 |
| F7 | 未发布 deque 清除缺失/重复 DROP；迟到 prediction 计为 0；generation 历史累计量误作 pending | `_drop_unpublished` 一个 owner 计后缀，pending head 不另加一；迟到 prediction 使用其真实长度；wait tracker 可静默关闭；consumer pending 分别报告且说明可能已入 SDK。8 动作已发 1 后 STOP 只报 remaining=7；已消费 FIFO 不报非零 pending；teleop/keyboard 同一 candidate 不重复打印 DROP。 |
| F8 | cleanup / recording 摘要复用总体成功或 session_failed | `_session_result_facts` 分开计算 evidence、verified IPC cleanup 和总体 0/1；`_report_session_end` 也覆盖异常 cleanup，另列进程 exit/escalation。policy failure 不自动标 recording failed；evidence failed + clean cleanup 可表达；未确认 cleanup 明确 incomplete。 |
| F9 | HTML 的 run_policy `--dry-run` 不受 parser 支持 | 删除错误命令；历史单 episode 命令无法核实，标 unverified。新 fixture 用真实 process_episodes dry-run/output-root，源 hash 不变且无输出目录。没有给 run_policy 增加 dry-run，更没有执行它的 main。 |

未改范围有源码依据：ring timestamp 写入语义、schema v29、RGB/pointcloud 身份链、raw reader、processed loader 没有为本修复改义；年龄/偏差质量 gate 不恢复。Teleop 原有 recording capacity 静默暂停仍保留，F3 的 trial 容量解耦作用于 PolicyRunner；teleop camera_stall 保存前缀和 live-input pause/re-anchor 保留。

## 新增字段及辅助函数的唯一用途

| 新增/扩展 | 唯一用途及 owner |
|---|---|
| `RuntimeChannels.evidence_failed` | evidence owner / supervisor 单向置位，表示会话 evidence 失败；不授权 motion 撤销或结束 run plan。原 `session_failed` 继续表示 terminal policy/session failure。 |
| `run_started_generation` | safety owner 保存原 RUNNING 身份，使 command-only rebase 不改变 trial 身份。 |
| `run_ended_generation`, `run_ended_started_monotonic_ns`, `run_ended_monotonic_ns`, `run_ended_reason` | safety 撤销 owner 的单个最新终止快照；runner 按身份消费软件时长与原因。不是事件历史或 epoch registry。 |
| `RunEndReason`、`RunStateSnapshot` 的 ended 字段 | 为上述单快照传递原因及时间；不宣称物理停止。 |
| runner `_trial_generation` / `_recording_trial_id` | 分别绑定唯一 active trial 与唯一录制事务；新 trial 不覆盖旧 recording context。 |
| runner `_drop_unpublished` | 未提交后缀计数/日志及 wait span 关闭的唯一 owner。 |
| runner `_stop_recording_capture` | 把 STOP 通道异常限制在 evidence，并保留未确认事务占用。 |
| `start_evidence_services` | 可选服务一次有界启动/READY，保留 process handles 并检查 critical workers。无异步 START 缓存。 |
| `_session_result_facts`, `_report_session_end` | 从 evidence 与 verified shutdown 事实计算并显示分项摘要。 |
| `ExitReason.EVIDENCE_FAILURE` | supervisor 的固定 evidence-only 分支；不增加可配置策略系统。 |
| `PublishWaitTracker.note_dropped(report=...)` | owner 已打印真实 DROP 时只关闭 span；未另外创建日志服务。 |
| teleop `_note_recorder_transport_failure` | 替代原 fault helper，仅报告 evidence 失败。 |

未新增 ACK registry、STOP 事务框架、每动作 barrier、heartbeat 线程、failure-policy registry、source archive、FULL/miss/latency 终止阈值。CommandStreamConsumer 的 generation 检查使用现有短锁，锁不跨 SDK。

## 实际验证与退出码

最终 SHA `4ee497d50e30303d75fb7a6dfd751a260e475601`：

| 实际命令 | 结果 / 范围 |
|---|---|
| `python -m unittest discover -s tests -p 'test_*.py' -v` | **160 tests，OK，0 fail，0 skip，exit 0**；`/tmp/pr17-full-final.log`。包括现有 provenance、projection、timing、FIFO、recording、interface 回归。 |
| `python -m compileall -q dexmani_real examples tests` | exit 0；仅编译。 |
| `git diff --check` | exit 0。 |
| `git diff 464fde7c30ec7a1187da7ecee6bcb1b53223d64b HEAD --check` | exit 0。 |
| `git diff 5b9f8968a10c845d695de418b632e6ce64de3be6 HEAD --check` | exit 0；额外源码 diff 复核未恢复删除的质量 gates。 |
| `PYTHONPATH="$PWD" python /tmp/pr17_readonly_data.py` | exit 0；只读当前已知 raw/processed 根；结果见下一节。脚本内容列于下文，日志 `/tmp/pr17-data-final.log`。 |

执行过程中还运行了 targeted discover（`-q`）：command_stream 31、observation_admission 14、deployment_evidence 31、recording_preservation 12、examples_interfaces 21、teleop 通配 5、sync_policy_timing 7、policy_projection 13；各最近一次均 OK / exit 0。最终 full suite 已重新覆盖全部当前用例，不将旧批次的数量相加作为当前总数。

缺陷先失败的证据：A 三代/稳定损坏/旧 SDK 游标共 6 个失败子例；B 两 reader 共 4 个失败子例；C 初始 4 个失败；D 实际分母和 suffix count 2 个失败；复核另有旧 ACK 2 个失败子例、STOP exception 1 error、teleop 双 DROP 1 failure。修复后对应用例通过。`/tmp/pr17-c-before.log` 的初次 shell 包装因尾随 tail 返回 0，但日志内 unittest 为 FAILED；从未计作 passing。接口测试编写中曾因 fixture task/output 名称及 no-hand 配置假设失败，修正用例假设，未改运行合同/阈值/skip 来通过。

所有完成的测试命令保留 Python 本身退出码，未用 tee 的退出码冒充测试结果。临时原始日志仍在 `/tmp`；中断进度账本为 `.git/research-runtime-refactor/STATUS.md`。测试使用 fake clock、Event 或同步受控 hook，不靠随机 sleep 复现并发。

## 数据保护与本次只读边界

已知根：`episodes/pick_place_toy`、`episodes_processed/pick_place_toy`。本次没有重算、迁移、写元数据、覆盖 derived、导出实际 Zarr、清理目录或修改 checkpoint。

本次新读数：**61 个 raw data.h5，schema 29，共 14,309 行**通过当前 EpisodeReader 的 `require_valid`；**60 个 processed H5**通过 `load_processed_trajectory`，`action_arm_joint.dtype == float64`。此为 reader/loader 兼容检查，不是物理 replay、完整每字节内容证明或 task-success 验证。真实树没有做内容 hash 前后全量比对；mtime/status 和目录存在不能替代该证明。没有已知真实 legacy 根，未制造 converter，也未宣称验证了未知数据。

首次只读脚本误把 `data.h5` 文件而非 episode 目录传给 EpisodeReader，exit 1；修正调用参数后重新完成上述检查，未改数据或 reader 来通过。

实际临时只读脚本如下（imports 和 loader 路径先检查，无设备打开）：

```python
from pathlib import Path
from dexmani_real.recording.storage.reader import EpisodeReader
from dexmani_real.replay.trajectory import load_processed_trajectory
raw = Path('episodes/pick_place_toy')
processed = Path('episodes_processed/pick_place_toy')
paths = sorted(raw.glob('*/data.h5')) if raw.is_dir() else []
rows = 0
versions = set()
for path in paths:
    with EpisodeReader(path.parent) as reader:
        reader.require_valid(purpose='PR17 read-only compatibility')
        rows += int(reader.h5f['meta'].attrs['num_frames'])
        versions.add(reader.schema_version)
print('raw_reader', {'files': len(paths), 'rows': rows,
                     'schemas': sorted(versions), 'available': raw.is_dir()})
files = sorted(processed.glob('*.h5')) if processed.is_dir() else []
for path in files:
    trajectory = load_processed_trajectory(str(path))
    assert trajectory.action_arm_joint.dtype.name == 'float64'
print('processed_replay_loader', {'files': len(files),
      'available': processed.is_dir(), 'arm_dtype': 'float64'})
```

新单 episode 预检由 `AffectedParserSmokeTest.test_single_episode_processing_dry_run_is_read_only` 执行：仅 `tests/raw_episode_fixture.py` 创建的临时 fixture；真实 subprocess 调用 `examples/process_episodes.py <临时episode> --dry-run --output-root <临时processed/provenance_fixture>`，exit 0、1 accepted；源完整文件集合及各文件 SHA256 不变，输出目录未创建。这不是历史未核实命令的补录。

## V01–V22 独立收尾复核

“满足（离线）”仅限列出的源码/fixture/mock 证据；不等于真机通过。没有另开写入 agent；实现完成后另行按 owner/dataflow 反查。

| 项 | 结论 | 证据 / 边界 |
|---|---|---|
| V01 | 满足（离线） | raw reader 实际只读检查；provenance/mask fixture；schema/layout 未修改。 |
| V02 | 满足（离线） | recording_preservation：短 camera_stall prefix、partial/corruption、active writer 保留。 |
| V03 | 满足（离线） | processing_admission / processing_provenance；多 IK_FAIL 保留，annotations 和 policy_eval 边界保持。 |
| V04 | 满足（离线） | real arm/hand consumption + fake SDK，absent、单/双 consumer 和跨进程 FIFO。 |
| V05 | 满足（离线） | FULL 同 candidate/targets/id；timing/projection 不提前推进。producer 调用链静态复核。 |
| V06 | 满足（离线） | 稳定 sequence / generation corruption 报错；EMPTY 等待。 |
| V07 | 满足（离线）；实手条件暂缓 | hand 中间 setpoint、CRC_UNCONFIRMED、exact endpoint；旧代 ACK 不成功，实测容差未知。 |
| V08 | 满足（离线） | SDK 前撤销和 SDK 中晚返、下一代首命令与后续 tick，无逐动作 barrier。 |
| V09 | 满足（离线） | projection/hard-limit 回归；SDK error/e-stop owner 保留。 |
| V10 | 满足（离线） | recoverable miss 保留 committed reference；容量耗尽后连续 miss 仍 RUNNING。 |
| V11 | 满足（离线） | arm/hand commit 因果、old sample reuse/warm-up、同源 tactile、RGB/cloud identity。 |
| V12 | 满足（静态+离线） | producer stall 与 payload truthfulness 保留；删除的 pointcloud age 前后 gate 未恢复；真实源断流未试。 |
| V13 | 满足（离线） | obs_t/action_t、infer 后首动作、实际发布 dt、慢推理与 FULL 无 catch-up。 |
| V14 | 满足（离线） | 真实 supervisor fake clock 撤销 blocking run，旧 prediction 不发；policy loop 无 heartbeat deadline。 |
| V15 | 满足（离线） | START 两结果重查；optional READY/death/heartbeat 继续两 trial；required failure/Q/e-stop 优先级。 |
| V16 | 满足（离线） | 唯一 recording identity、重复 terminal 只计一次；未确认进程退出不释放共享资源。 |
| V17 | 部分满足；legacy 条件暂缓 | 当前 raw/processed 只读兼容与 float64；未知/真实 legacy 根没有提供，未验证 converter。 |
| V18 | 满足（离线范围） | 下列 EX01–EX14，硬件/GUI 入口只做明确静态或 mock 检查。 |
| V19 | 满足（离线） | 后缀7、late prediction、消费完 pending0、FULL span 单条 DROP；不同动作同原因不被合并。 |
| V20 | 满足（离线） | mean/p95 来源真实完成 predict 样本；无样本 unavailable；publication/软件 RUNNING 分母不推断物理完成。 |
| V21 | 满足（静态） | README/help/docstrings 更新，F9 历史声明隔离；TASKBOOK 未修改。 |
| V22 | 满足（本地） | 原重构分支，显式 staging，后续 commit，无 amend/push/merge；完整 diff、index/worktree/untracked 检查。 |

## EX01–EX14 参数/结果链核对

| 项 | 结论与实际验证 |
|---|---|
| EX01 run_policy | 受影响：real parser → run/record config 静态 → PolicyRunner trial/result 与 lifecycle summary 回归；未运行 run_policy.main，无真实 artifact。 |
| EX02 collect_teleop | 受影响：real CLI → resolved config → mocked session，no-record、no-hand+no-record、返回值；evidence service wiring 与 camera_stall prefix 测试。保留 no-hand 禁录制约束，help 明示需同时关 recording。 |
| EX03 keyboard_teleop | 受影响：配置/返回码 mock，共用 FIFO/generation/ACK；keyboard 重复 DROP 改为 owner 唯一报告；未运行键盘控制主体。 |
| EX04 replay_episode | 共用传输受影响；parser/provenance fixture、真实 processed loader float64。原 replay 操作期限不变；未做物理 replay。 |
| EX05 calibrate_camera | 共用传输受影响；真实参数传递到 mocked calibration session；motion/home/collision/save guard 静态核对，无标定写入。 |
| EX06 calibrate_vr_heading | NOT_AFFECTED：独立定位/保存路径未改，共用 robot FIFO 不适用；AST 与静态 import 检查，未接 HTS。 |
| EX07 process_episodes | 实际 fixture dry-run、annotations/output-root parser 与 processing/provenance 回归；源只读，无全量转换。 |
| EX08 export_policy_zarr | NOT_AFFECTED：本补丁未改 export；现有 output/no-overwrite/protected path/preflight guard 回归通过。未重导出现有 Zarr。 |
| EX09 visualize_episode | NOT_AFFECTED：raw reader 字段未改；当前数据 reader 兼容与 AST，未启动 GUI。 |
| EX10 visualize_episode_processed | NOT_AFFECTED：processed schema/provenance 未改；loader/validator 回归与 AST，未启动 GUI。 |
| EX11 visualize_policy_rollout | 无 trace 的 raw 实际 `main(..., --info)` fixture 返回基本信息，坏 trace 仍失败；不加载模型。README 同步校正。 |
| EX12 pointcloud_process_example | NOT_AFFECTED：交互 table calibration/config 未改；AST/静态，未运行相机或 GUI 主体。 |
| EX13 realsense_record_example | NOT_AFFECTED：独立 driver 诊断入口未改；AST/静态，未运行相机或 GUI 主体。 |
| EX14 xhand_control_example | NOT_AFFECTED：native SDK 诊断定位、CRC 语义不改；仅 AST/静态，未尝试不存在的离线 --help。 |

## 本地提交与最终边界

| SHA | 标题 |
|---|---|
| `85484380e5b2918ced7f2326938fc873e828d290` | fix(control): preserve command identity across epoch resync |
| `68e3dd191f85e39098c7273149931ce8b7562fb8` | fix(inference): enforce causal ring commit admission |
| `18901cdaeffd23bdb17cb7f3bb7ade13bcd99c3d` | fix(recording): isolate evidence failures across trials |
| `d9f55aa55e45844ed58d872a2ff379a0e06b9e08` | fix(runtime): preserve run termination facts and accurate drops |
| `f6116f87b1b0a389baa7fbd20052fc9e10a435af` | fix(runtime): close late acceptance and evidence exception paths |
| `aadf358eca540ebc717f6bc6515bacbad668152f` | test(docs): verify offline interfaces and correct historical evidence |
| `4ee497d50e30303d75fb7a6dfd751a260e475601` | fix(teleop): report each pending cancellation once |

本文件由后续 `docs(tasks): record PR17 bounded fix validation` 提交；该提交 SHA 以 Git log / 最终回复为准，避免自引用。被测代码提交时 index/worktree/untracked 全空；文档提交后再检查三者及 PATCH_BASE diff。

反向复核重点：较大 ACK 只在同代无跳序的消费前提下证明前驱；旧 SDK ACK 不穿过 generation wait 检查。Evidence latch 没有退出 run plan 的权限，Q/critical failure 不被持续 evidence 事件吞掉。旧 recording 的身份、reason、terminal consumed 不被无录制 trial 重置。软件首次终止只在离开 RUNNING 时写入；pending 日志不把累计历史或双 consumer 数相加冒充 distinct actions。

未验证/条件暂缓：真实机器人与相机运行、CUDA 推理、真实 policy artifact、实手 exact endpoint 容差、真实设备断流/SDK 延时分布、性能基准、未知 raw/legacy 根。离线通过不等于真机安全或物理动作完成已验证；timeout / 系统停止不推断 task success。
