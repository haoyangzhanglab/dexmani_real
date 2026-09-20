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

---

## R1–R4 后续有边界修复（2026-09-20）

本节是新的修复批次；上文 F1–F9、旧 PATCH_BASE、160-test 和真实数据读数保留为历史记录，不能用来代替本节的验证。

### 本批基线与结果

- PATCH_BASE：`11911a1cba4812175e6bf19eb0695932f6f33bd2`，实际开始 HEAD，与指定 review tip 一致。
- 分支：`refactor/research-runtime-v2`；未在 main 修改。开始时 index/worktree/untracked 全空。
- 原 IMPLEMENTATION_BASE 仍为 `5b9f8968a10c845d695de418b632e6ce64de3be6`；旧批次 PATCH_BASE 未覆盖。
- Python：现有 `real_robot` 环境 3.10.20；没有安装升级依赖或改 Git 身份。
- 最终被测源码/测试 SHA：`671a67cac2440bb94ff0821152e2652d54cfd874`。
- 本 SHA 完整离线回归：**179 tests，OK，0 fail，0 skip，Python exit 0**。后续仅提交本交付证据。
- TASKBOOK、AGENTS、CLAUDE、agent 配置未改；无 push/merge/amend/no-verify。

### Fact-check、修复与具体证据

| 项 | 实际开始结论 | 最终修改位置与行为 | 本批回归 |
|---|---|---|---|
| R1 | CONFIRMED；两行真实 STOP/drain/writer 探针只剩 aborted JSON，assertion exit 1 | `recording/client.py:31,246` 增加默认 False 的 retain_partial；`teleop/episode_samples.py:25` 传递；`teleop/loop.py:454,622,763,807,948,1014` 标记自动中断；`recording/io_worker.py:404` 合并既有 corruption 保留条件；`recording/recorder.py:785` 明确 incomplete 提示；`deployment/executor.py:660,739` 自动非保存 STOP 保留、结果提示未发布 | `TeleopInterruptedRecordingTest` 实际 controller→client→STOP→RecorderIO→临时 EpisodeRecorder；shutdown/fault/ESC/arm reject、Q/ESTOP、Q/SHUTDOWN、明确 SAVE/D、Q/TIMEOUT；两行 HDF5 时间戳逐值核对。`InterruptedRetentionBoundaryTest` 证明 2 行低于 flush interval=32，writer 未释放时 staging 不动，释放后才保存数据；旧三参数消息、位置参数 stop_episode 及 STOP 幂等；Policy helper→真实 client 消息；实际 terminal 只打印一次未发布并由 recorder 报告真实保留路径。 |
| R2 | CONFIRMED；FULL 的 continue 先于 release | `teleop/keyboard_session.py:673,844,903` 小局部 drop_pending helper；先处理当前 moving/release，再重试原 candidate；确认释放后取消未提交目标，原 ACK/timeout 仅针对已 publication action | `KeyboardReleaseLoopTest` 真实 `_run_control_loop` 和 `_publish_keyboard_target`；受控 LoopRate/fake clock，无随机 sleep。持续 FULL 后释放、容量恢复、已 ACK/无 predecessor、未 ACK 原 0.15s timeout、持续按住同 candidate 单次成功、短暂释放后重按、Q/ESC/home/epoch、重新 anchor 和新按键均覆盖。 |
| R3 | CONFIRMED；teleop_active 令旧 max_frames 获得新 B 的暂停权限 | `teleop/loop.py:570` 仅用现有 recording_active 门控；当前顺序仍在 terminal 清除前判断首次容量，不需要额外快照或 persistent ID | `TeleopCapacityOwnershipTest` 真实 poll→B→poll；Event 控制的真实 finalizer 在 B 期间保持活跃，旧 pending/done 均不改新 RUNNING/generation；新正常 capture 可开始并保存；首次 terminal 直接到达仍暂停一次；STOP/保存结果各消费一次。 |
| R4 | CONFIRMED；hand 已 clip，返回链和日志仅包含 arm | `deployment/executor.py:219,260,1216,1320` 内部 `_PolicyClipReport` 从第一次投影携带 hand correction；每 action 合成一行 CLIP | `PolicyEndpointClipReportingTest` 真实 decode/projection/prepare/dispatch，实际 SafetyGate；仅 hand、仅 arm、两者、无 clip；nextafter 级微小 correction；3 次 FULL 后成功，projection/log 各一次；action/action_ee 数值与原投影函数及既有 clip 结果精确一致，input 不变；NaN/shape 仍走合同失败。 |

没有 ALREADY_FIXED 或为清单制造的修改。R2/R3/R4 的反例也实际在各自修复前执行失败；不是仅静态推测后标通过。

新增生产接口只有 `StopRecording.retain_partial=False`、两个 STOP helper 的同名 keyword-only 参数，以及 executor 内部的 `_PolicyClipReport`。Keyboard 的 `drop_pending` 只取消一个未提交候选并关闭原 wait span。没有 ACK/STOP/trial/report registry、失败策略框架、heartbeat 线程、source archive、每动作 barrier 或新的终止阈值。

`hand_max_correction_rad` 是 decoded hand 与 projected hand 的最大绝对差，单位 rad；`hand_joint` 零起始。没有重新 clip，不使用 measured hand 计算，不用 allclose/deadband，使用 17 位有效数字。原 `arm_max_delta_rad` 仍表示裁剪前 command step。投影函数、bounds/roundoff、SafetyGate、worker hard limit、SDK slew/CRC/exact endpoint 没有改动。

### 实际验证命令、退出码与失败记录

命令均使用本环境 Python；重定向后立即保存 `$?` 再显示日志并以该值退出，没有以 tail/tee 状态代替 Python 退出码。

| 命令/执行阶段 | 实际结果 |
|---|---|
| 初始只读 inline R1 probe（真实 stop_recording/client/RecorderIO/EpisodeRecorder） | 2 行被删除、仅 aborted JSON；期望保留的断言失败，exit 1；未驱动完整 Teleop loop，不冒充 R1 最终集成测试。 |
| `python -m unittest discover -s tests -p 'test_teleop_command_span.py' -v`，R1 修复前 | 5 methods，9 个失败子例，exit 1；`/tmp/pr17-r1-before.log`。 |
| `PYTHONPATH=tests python -m unittest test_teleop_command_span.KeyboardReleaseLoopTest -v`，R2 修复前 | 初次夹具错误传递 PreparedCommand 构造参数，10 errors，exit 1；修正夹具后旧生产代码实际 7 failures，exit 1，`/tmp/pr17-r2-before.log`。前者不作为缺陷复现。 |
| `PYTHONPATH=tests python -m unittest test_teleop_command_span.TeleopCapacityOwnershipTest -q`，R3 修复前 | 3 methods，旧 pending 与迟到 terminal 两项失败，exit 1；`/tmp/pr17-r3-before.log`。后续增加 Event 控制真实 finalizer 的验证。 |
| `PYTHONPATH=tests python -m unittest test_policy_projection.PolicyEndpointClipReportingTest -v`，R4 修复前 | 4 methods，6 failures/subcases，exit 1；`/tmp/pr17-r4-before.log`。 |
| `python -m unittest discover -s tests -p 'test_recording_preservation.py' -q` | R1 时 14、最终补证后 15 tests，OK，exit 0。 |
| `python -m unittest discover -s tests -p 'test_teleop_command_span.py' -q` | R1/R2/R3 分别 5/11/14；最终补证后 15 tests，OK，exit 0。 |
| `python -m unittest discover -s tests -p 'test_deployment_evidence.py' -q` | 31 tests，OK，exit 0。 |
| `python -m unittest discover -s tests -p 'test_policy_projection.py' -q` | 17 tests，OK，exit 0。 |
| `python -m unittest discover -s tests -p 'test_sync_policy_timing.py' -q` | 7 tests，OK，exit 0；CUDA OOM 是既有 mock 异常用例，不运行 CUDA。 |
| `python -m unittest discover -s tests -p 'test_examples_interfaces.py' -q` | 21 tests，OK，exit 0；parser/mock/AST/临时 fixture 路径。 |
| `python -m unittest discover -s tests -p 'test_*.py' -v`，`8d439db` | 177 tests，OK，exit 0；`/tmp/pr17-r1-r4-full-final.log`。自审后增加两条测试，以下最终版本重新全量验证。 |
| `python -m unittest discover -s tests -p 'test_*.py' -v`，`671a67c` | **179 tests，OK，0 fail，0 skip，exit 0**；`/tmp/pr17-r1-r4-full-tested.log`。 |
| `python -m compileall -q dexmani_real examples tests` | 最终被测版本 exit 0。 |
| `git diff --check` | exit 0。 |
| `git diff 11911a1cba4812175e6bf19eb0695932f6f33bd2 HEAD --check` | exit 0；另在证据提交后复核。 |
| `git status --short --untracked-files=all`、staged/unstaged name-status | 最终被测代码提交时均空；证据提交后再复核，不用单个干净 diff 冒充全部状态。 |

所有 focused 和完整 suite 均没有新增 skip、删除失败断言或放宽生产阈值。旧 160-test 只属于旧 `4ee497d`，不属于本次被测版本。

### V01–V22 本批矩阵

“回归执行”表示本批最终 full suite 的实际离线覆盖，不是重新完成真机/真实数据验收。

| 项 | 本批状态 / 实际证据 |
|---|---|
| V01 | 本批 raw/schema/reader 未改；provenance、masked NaN、source hash 的临时 fixture 回归执行；未重查真实 raw 全库。 |
| V02 | R1 新增完整 controller→真实文件证据；2 行低于 flush interval、writer release Event、零行/显式 discard/camera_stall 原用例全部执行。 |
| V03 | 本批 processing 未改；processing admission/provenance 回归执行；无批量转换。 |
| V04 | 本批 FIFO/worker 未改；command_stream 的单/双/absent、真实 spawn IPC、fake SDK 有序消费回归执行。 |
| V05 | R2 真实 keyboard loop 的释放优先、原 candidate/ID/FULL 恢复与单 tick 提交；R4 实际 dispatch 的重复 FULL 一次投影/CLIP；既有 transport/timing 回归执行。 |
| V06 | 本批 corruption/EMPTY 合同未改；command_stream 稳定身份/sequence 损坏回归执行。 |
| V07 | 本批手部物理合同未改；中间 setpoint、CRC_UNCONFIRMED、exact endpoint、旧 ACK 回归执行；真实手部 measured tolerance 仍条件暂缓。 |
| V08 | 本批 generation/SDK fence 未改；SDK 前撤销、调用内晚返与三代交错回归执行；R2 增加控制器 epoch/home 取消验证。 |
| V09 | R4 数值精确保持；projection/roundoff、SafetyGate 与 hard limit 回归执行；没有扩大界限。 |
| V10 | 本批 recoverable miss 策略未改；projection/deployment evidence 回归执行，包括保留原 trial/reference。 |
| V11 | 本批 causal ring/tactile/RGB 合同未改；observation admission/provenance 回归执行。 |
| V12 | 本批 source stall/quality gate 未改；已有离线 admission 与 evidence-role 回归执行；未重新验证真实设备断流。 |
| V13 | 本批 cadence 未改；7 timing tests 与新增 FULL dispatch 回归执行，无 catch-up。 |
| V14 | 本批父预算/心跳合同未改；真实 supervisor 配 fake clock、blocking predict 迟返回归执行；无真实 CUDA。 |
| V15 | 31 deployment evidence tests 与全量回归覆盖 START、optional service、S/Q 优先；R1 的 STOP retention 不授予 evidence 运动权限；R3 旧容量事件不干扰新 B。 |
| V16 | R1 writer 释放前 staging 不动、实际 terminal 未发布与真实路径；R3 旧 pending/done 单次消费；原 verified cleanup/Policy trial 计数回归执行。 |
| V17 | 本批 raw/replay/derived 合同未改；fixture/provenance loader 回归执行；真实 legacy/真实全库未重新验证。 |
| V18 | 下表 EX01–EX14 全部有本批结论；21 interface tests 与相关真实 owner 离线回归执行；硬件/GUI 主体未运行。 |
| V19 | R2 每次 pending 取消一条 DROP；R4 hand-only/combined/tiny 一条 CLIP，FULL 不重打；原 F7 suffix/WAIT span 回归执行。 |
| V20 | 本批统计定义未改；真实 RUNNING 终止分母及迟返预测回归执行；未重新做性能测量。 |
| V21 | README 同步自动保留、Q TIMEOUT、旧容量事件、keyboard release 与 CLIP 字段；相关源码 docstrings/comments 更新；任务书和稳定 agent 文档未改。 |
| V22 | 原目标分支，显式路径 stage、后续本地 commits；无用户修改混入；检查完整 PATCH_BASE diff 和 staged/unstaged/untracked。提交清单见下。 |

### EX01–EX14 本批矩阵

| 项 | 本批变更/验证结论 |
|---|---|
| EX01 run_policy | 受影响的是 executor 报告与自动非保存 STOP；README/docstrings 已说明 arm pre-clip step、hand endpoint correction/rad/joint 和 FULL 一次报告。真实 decode/projection/prepare/dispatch 用例及 parser/trial/evidence 回归执行；未运行 run_policy 主体、真实模型或新增 dry-run。 |
| EX02 collect_teleop | 自动中断与明确 discard 分开；TIMEOUT 原默认丢弃，ESTOP/SHUTDOWN 保留；终结显示未发布，recorder 日志才是 incomplete 实际位置，RecordingFinished.path 仍可能是 reserved final path。实际 Teleop 消息/结果/文件与旧容量事件测试执行；CLI→配置→mock session 回归执行；原 no-hand 禁录制约束不变。 |
| EX03 keyboard_teleop | release debounce 先于 pending retry；释放确认取消候选，等待已提交 predecessor 原 ACK/timeout 并 measured anchor；短 gap 保留候选，持续按住同 ID/targets。真实 loop 回归与 CLI→mock session 返回码测试执行；未连键盘/机器人。 |
| EX04 replay_episode | 本批未改，未直接调用 StopRecording；共用命令传输 API/原 deadline/abort/保存语义不变。processed parser、replay provenance/loader fixture 回归执行；未物理 replay，未重新读真实 processed 全库。 |
| EX05 calibrate_camera | 本批未改，未直接调用 StopRecording；共用 motion API/原 deadline/abort/save guard 不变。参数到 mock calibration session 回归执行；无标定写入。 |
| EX06 calibrate_vr_heading | NOT_AFFECTED：独立 heading 定位/保存接口，本批未改；AST 与受影响符号扫描执行；硬件/保存路径未重新验证。 |
| EX07 process_episodes | 本批未改；现有 parser、annotations、provenance、单临时 episode dry-run/source hash 回归执行；没有批量处理真实数据。 |
| EX08 export_policy_zarr | 本批未改；output/preflight/no-overwrite/protected-path 回归执行；没有写现有 Zarr。 |
| EX09 visualize_episode | NOT_AFFECTED：raw reader/schema 未改；AST/符号检查执行；GUI/真实全库未重新验证。 |
| EX10 visualize_episode_processed | NOT_AFFECTED：processed schema/loader 未改；AST 和相关 provenance 回归执行；GUI 未重新验证。 |
| EX11 visualize_policy_rollout | 本批未改；trace-less raw fixture 信息及坏 trace 拒绝回归执行；未加载模型/GUI。 |
| EX12 pointcloud_process_example | NOT_AFFECTED：camera/pointcloud/交互 table calibration 本批未改；AST 执行；主体未运行。 |
| EX13 realsense_record_example | NOT_AFFECTED：独立 camera driver 诊断，本批未改；AST 执行；主体未运行。 |
| EX14 xhand_control_example | NOT_AFFECTED：native SDK/CRC 诊断合同未改；仅 AST；没有运行主体或 --help。 |

### 本批提交、反向自审与未验证边界

| SHA | 标题 |
|---|---|
| `4720bea272c5b36a6ce472a2fe2bd991f44365a5` | fix(recording): preserve controller-interrupted captures |
| `ce4e3128e76fa64809c33e4be1cc6f65edb17554` | fix(keyboard): process release before FIFO retry |
| `77819377dc0e41d5c869eefa59c71029ae3c5aa2` | fix(teleop): scope capacity handling to the active capture |
| `8d439db14463e7c5ceb5affa88c219bb4dc77d48` | fix(deployment): report hand endpoint clipping |
| `671a67cac2440bb94ff0821152e2652d54cfd874` | test(recording): verify terminal reporting and policy retention |

本节证据另作后续 documentation commit，SHA 由最终 Git log/交付回复给出，避免自引用。恢复记录追加到既有 `.git/research-runtime-refactor/STATUS.md`；未覆盖历史基线。

反向自审逐项结论：自动 retain 已经到实际 HDF5 源行，不是只到 mock；明确 D 只有一次 STOP，finally 不能改写；确认释放后取消 pending，home/reject/epoch 也取消，不会 re-anchor 后复活；旧 max_frames 的 pending 和 terminal 分别有失败前/通过后证据；hand-only 一行可见、FULL 不重复投影和报告。完整套件执行已有 F1–F9 回归，未发现本批破坏其离线合同。

本任务未写入现有 raw/derived/processed/Zarr/checkpoint；实际数据输出仅测试临时目录，compileall 仅生成编译缓存。没有对真实数据做前后全量 hash，因此不把 Git status、mtime 或历史盘点当作全库内容不变证明。本批未重跑真实 raw/processed 全库 reader，也未借用旧数据读数声称本批通过。

条件暂缓/未执行：真实机器人/相机/手和 SDK 运行、运动/home/replay/rollout、实手 measured tolerance、真实 CUDA/model artifact、真实设备断流/延时与性能基准、未知 legacy。没有修改相邻 dexmani_policy。离线通过不等于硬件验证。
