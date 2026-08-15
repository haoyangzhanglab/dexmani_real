# Phase A 审计证据（audit-pass 项，2026-08-15）

Phase A 目标：软件正确性 + 硬件 SDK contract + IPC/lifecycle + perception/recording。
对 HEAD `34ddc70` 逐项审计后，A1–A20 大部分已在 main 落地；需实际改动的 5 项
（A0/A2/A9/A12/A15）已各自独立 commit，其余为 **audit-pass**（不改代码，仅记录
file:line 证据）。本条目即 audit-pass 证据汇总。

执行边界：全部为离线审计 + 离线回归（`checks/offline/`），**未做任何真机运动 /
采集 / 标定 / replay live**。

---

## A1 — C24 恢复已含 `motion_enable(True)` + Mode 6 ready，且 SDK 失败不发送 servo

- `dexmani_real/robot/arm_loop.py:192` `_require_sdk_ok(operation, code)` — 非零返回码即抛异常，使后续 servo 不发送。
- `dexmani_real/robot/arm_loop.py:202-215` C24 恢复序列：`clean_error → clean_warn → motion_enable(True) → _enter_mode6_ready → 新鲜 get_joint_states`，每步经 `_require_sdk_ok` 校验，`motion_enable != 0` 时不会进入 servo。
- `dexmani_real/robot/arm_loop.py:245-248` `_enter_mode6_ready` = `set_mode(6)` + `set_state(0)`（各经 `_require_sdk_ok`）。
- 可执行证据：`checks/offline/check_arm_c24_recovery.py`（断言调用顺序 `clean_error < clean_warn < motion_enable < mode/state ready < get_joint_states < servo hold`，且 `motion_enable != 0` 不发 servo）。

## A2 — （已改动，非 audit-pass）

`fix(xarm): live-confirm cached error and fail-closed connect recovery`（commit）。
主循环仅在缓存 `error_code != 0` 时 `_read_live_error_code` 同步确认（读失败 → latch fault）；
connect-recovery postcondition 改为 `_read_live_error_or_fail`（读失败 → 1，fail-closed）。
可执行证据：`checks/offline/check_arm_live_error.py`。

## A3 — homing 后恢复 Mode 6 已 fail-closed

- `dexmani_real/robot/arm_loop.py:1482-1487`：`_controller_error_after_home = _read_live_error_code(arm)`，
  异常时 `_controller_error_after_home = -1`（fail-closed，不恢复）；`_restore_mode6 = abort is None and _controller_error_after_home == 0`。

## A4 — 已 `accepts_motion_commands`（Mode 6 就绪门控）

- `dexmani_real/robot/arm_loop.py:634` 初始 `False`。
- `dexmani_real/robot/arm_loop.py:718-743` 仅在 `ARMED/RUNNING` 且 Mode 6 ready 时置 `True`；`FAULT/DISARMED` 复位。
- `dexmani_real/robot/arm_loop.py:846` 非 `accepts_motion_commands` 不发 servo。

## A5 — TCP load 已配置化

- `dexmani_real/config/defaults.py:277-278` `tcp_load_mass_kg` / `tcp_load_cog_mm`。
- `dexmani_real/config/defaults.py:328-332` 有限性校验（mass 有限且 >0，CoG 为 (3,) 向量）。
- `dexmani_real/robot/arm_loop.py:99` `tcp_load_mass_kg` 经 `default_factory` 取自配置。

## A6 — 已 `clear_local_error()`

- `dexmani_real/robot/xhand.py:876` `def clear_local_error()`。
- `dexmani_real/robot/hand_process.py:317-327` 连续发送失败看门狗自动 `clear_local_error()`。
- `dexmani_real/robot/hand_process.py:417-424` 手部 `error_state` 时的恢复路径。

## A7 — 无 `XHand.stop()`

- `dexmani_real/robot/xhand.py` 仅有 `disconnect()`（`:799`），无 `stop()` 语义残留。
- 全库唯一 `.stop()` 为 RealSense `pipeline.stop()`（`sensor/realsense.py:400,418`），与手部无关。

## A8 — 两 worker 均 `try/finally`，hand finally 仅 disconnect

- `dexmani_real/robot/arm_loop.py:1077` `finally:`（清理 `set_state(4)` 物理停机）。
- `dexmani_real/robot/hand_process.py:472,480` `finally:` → 仅 `hand.disconnect()`。
- 可执行证据：`checks/offline/check_worker_cleanup.py`（异常安全清理）。

## A9 — （已改动，非 audit-pass）

`fix(shm): stamp ring publish timestamp after payload commit`（commit）。
`ring_buffer.py` `write()` 改为 `begin_write(seq, 0)` → payload → `stamp_timestamp(monotonic_ns())` → `end_write(seq)`，镜像 `camera_ring`。可执行证据：`checks/offline/check_ring_commit.py`。

## A10 — `get_last_k` 覆盖槽 / oldest-first / `k>maxlen` 抛错

- `dexmani_real/shm/ring_buffer.py:277` `def get_last_k(self, k)`。
- 覆盖写槽 `continue`、`count<k` 返回更短、`k>maxlen` 抛错。
- 可执行证据：`checks/offline/check_ring_history.py`。

## A11 — coupled ACK 含 `hand_seq > action_id` 立即失败 + hand health gate

- `dexmani_real/policy/safety.py:655-672`：`hand_seq > action_id`（被更新代次取代）立即 `return None`；
  `hand_seq == action_id` 时校验 `connected`/`state_valid`（health gate）才确认。
- 可执行证据：`checks/offline/check_coupled_ack.py`（arm-only / arm+hand / hand-supersede 三态）。

## A12 — （已改动，非 audit-pass）

`fix(policy): preflight hand mechanical+delta bound on coupled paths`（commit）。
抽 `validate_hand_command_delta`，接入 teleop / replay / `publish_joint_targets` / return-home。
可执行证据：`checks/offline/check_hand_delta.py`。

## A13 — baseline 用 raw `wrist_rot`，非 gated

- `dexmani_real/teleop/arm_mapper.py:152-155`：`_last_wrist_rot` 追踪 **raw** 腕部姿态作 delta 基准，
  注释明确“用 clamp 后输出作基准会导致 spike 后 baseline 漂移”。
- 可执行证据：`checks/offline/check_vr_rotation_recovery.py`。

## A14 — desk-plane 单源 + align mode 支持

**desk-plane 单源：**
- `dexmani_real/sensor/camera_process.py:64-70`：`desk_plane = tuple(table.plane_abcd)`（`table.enabled` 为 False 时 → `None`），
  作为唯一生产来源。
- `dexmani_real/config/runtime.py:260`：`table_config["plane_abcd"]` 从 resolved 环境写入。
- `dexmani_real/sensor/camera_process.py:271-274`：`cfg.desk_plane is None` 时 `desk_plane_path=""`，禁用旧 `desk_plane.json` 自动加载。

**align mode（color_to_depth 亦受支持）：**
- `dexmani_real/sensor/realsense.py:442,444,466`：按 `align_mode` 选 intrinsics 目标流
  （`depth_to_color` → color 目标，`color_to_depth` → depth 目标），故「依赖 depth_to_color」不成立；
  生产点云路径拒绝真正危险值 `"none"`（`camera_process.py:194-197`）。
- 可执行证据：`checks/offline/check_perception_contract.py`。

## A15 — has_pointcloud 基于有效点云帧计数

- `dexmani_real/recording/recorder_client.py:165` `frame["pointcloud_valid"]`。
- `dexmani_real/recording/io_process.py:235` / `camera_stream_writer.py:128` 消费有效点云帧。
- （改动项 `camera_pointcloud_config` 入 `/meta` 见 commit `fix(recording): persist pointcloud filter config`，
  可执行证据 `checks/offline/check_pointcloud_metadata.py`。）

## A16 — `atomic_json_dump` 已用 + 备份 copy

- `dexmani_real/recording/transaction.py:51-76` `atomic_json_dump`：`mkstemp → dump → flush → fsync → os.replace → fsync(parent)`，崩溃不会留下截断/缺失目标。
- `dexmani_real/sensor/pointcloud_processor.py:574`（`save_desk_plane`）使用 `atomic_json_dump`。
- 备份 copy（覆盖前 `.bak` 带时间戳）：
  - `examples/calibrate_camera.py:521-525`
  - `examples/calibrate_vr_heading.py:400-402`
  - `examples/pointcloud_process_example.py:390-392`
- 可执行证据：`checks/offline/check_atomic_calibration.py`（round-trip + 备份 copy）。

## A17 — synthetic hold 清 send 事件 / replay 尊重 send_mask

- `dexmani_real/teleop/episode_samples.py:181-185`：无 action 时 `action_id` 与
  `action_created/target/valid_until_monotonic_ns` 均记 0；`action_queued = action_candidate is not None`
  （合成 hold 即「无 action」，不伪造发送事件）。
- `dexmani_real/recording/recorder_client.py:122`：`flag_action_queued` ← `action_queued`。
- `examples/replay_episode.py:1179`：replay 尊重 `send_mask`（未置位不发送）。
- 可执行证据：`checks/offline/check_recording_send_semantics.py`。

## A18 — v16 已含 `flag_action_queued`，无需 bump schema

- `dexmani_real/utils/schema.py:280`：v16 sample dtype 含 `("flag_action_queued", "<u1")`。
- `dexmani_real/recording/episode_recorder.py:483,592` / `episode_reader.py:215,342` 读写该字段。

---

## 汇总

| 项 | 结论 | 证据类别 |
|---|---|---|
| A1 | audit-pass | file:line + check |
| A2 | 已改动 | commit + check |
| A3 | audit-pass | file:line |
| A4 | audit-pass | file:line |
| A5 | audit-pass | file:line |
| A6 | audit-pass | file:line |
| A7 | audit-pass | file:line（无 stop） |
| A8 | audit-pass | file:line + check |
| A9 | 已改动 | commit + check |
| A10 | audit-pass | file:line + check |
| A11 | audit-pass | file:line + check |
| A12 | 已改动 | commit + check |
| A13 | audit-pass | file:line + check |
| A14 | audit-pass | file:line + check |
| A15 | 已改动（meta 入档） | commit + check |
| A16 | audit-pass | file:line + check |
| A17 | audit-pass | file:line + check |
| A18 | audit-pass | file:line（schema v16 已含） |

离线回归：`python checks/offline/run_all.py` → **12/12 checks passed**（A0 harness + 各 check）。
硬件验证：**NOT RUN**（Phase A 离线边界内）。
