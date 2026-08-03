# CLAUDE.md -- DexMani Real

Dexterous manipulation teleop & data collection for **xArm7 (7-DOF) + XHand (12-DOF)** with VR control.
**Env:** conda `real_robot` (Python 3.10). All scripts: `PYTHONPATH=.` from repo root.
Activate: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate real_robot`

---

## Quick Reference

| Task | Key File(s) |
|------|------------|
| **Main entry point** | `examples/real/vr_teleop_hand_record.py` (canonical, 5-process arch) |
| **Policy process** | `policy/vr_teleop_policy.py` — `policy_loop(shared, config)`, owns recording |
| **SharedStorage (data plane)** | `shm/shared_storage.py` — all rings, queues, flags in one place |
| Arm servo loop | `robot/inner_loop.py` — `arm_loop(shared)` (canonical) |
| Hand control | `robot/hand_process.py` — `hand_loop(shared)` (canonical) |
| Policy (VR→IK, recording) | `policy/vr_teleop_policy.py` — reads rings, writes actions, owns recording |
| VR receiver | `sensor/vr_receiver_process.py` — `vr_loop(shared)` writes to `vr_ring` |
| Camera | `sensor/camera_process.py` — independent process, unchanged |
| IK / retargeting | `planning/ik.py`, `teleop/vr/arm_mapper.py`, `hand_retarget.py` |
| Safety state machine | `robot/safety.py` — SafetyState enum (DISARMED/ARMED/RUNNING/FAULT) + transition helpers |
| Recording format / lifecycle | `recording/episode_recorder.py` |
| SHM primitives | `shm/ring_buffer.py` (CameraRingBuffer + SeqlockRingBuffer base), `shm/robot_ring.py` (SeqlockRingBuffer) |
| Core types | `robot/types.py` — RobotState, RobotAction, ArmState, HandState, HandTactile |
| Episode tools (viz/health/export) | `tools/` |
| Type-check | `conda run -n real_robot mypy dexmani_real/` |

---

## Architecture (simplified — Phase 2, 2026-08-02)

### 5-Process Model + Thin Main

```
Main (140 lines) — spawns 5 processes, monitors is_running
  │
  ├─ Camera (camera_loop) ──camera_ring──┐
  ├─ VR (vr_loop) ──────vr_ring─────┤
  │                               ▼
  ├─ Policy (policy_loop) ──arm_action_q──→ Arm (arm_loop)
  │                ──hand_cmd_ring─→ Hand (hand_loop)
  │                ◄──arm_state_ring, hand_state_ring, hand_tactile_ring
  │                owns EpisodeRecorder (single-clock recording)
  │
  ├─ Arm (arm_loop, Mode 6, 30Hz) — reads arm_action_q, servos xArm7, writes arm_state_ring
  └─ Hand (hand_loop, 30Hz) — reads hand_cmd_ring, servos XHand, writes hand_state_ring + hand_tactile_ring
```

**Key principles (from ManiUniCon):**
- Main does NOT touch data plane — only orchestrates
- SharedStorage is the sole data plane — one class, all rings/queues/flags
- Processes exchange structured data only — no Python objects, no RPC
- Policy owns recording — single-process TimestampAlignedBuffer, natural alignment

### Data Flow

```
VR Tracker → ArmWristMapper → EMA → WorkspaceClamp → solve_teleop_ik → DeltaClamp
                                                                              │
                                                          ┌───────────────────┘
                                                          ▼
                                         shared.arm_action_q.put(ArmAction)
                                         shared.hand_cmd_ring.write(HandCmd)
```

**Rates:** Policy loop 16 Hz. Arm/Hand inner loops 30 Hz. Mode 6 firmware handles all trajectory smoothing (120 deg/s, acc configurable). No inner-loop interpolation.

### SharedStorage Data Plane (`shm/shared_storage.py`, ~340 lines)

| Transport | Type | Direction | Semantics |
|-----------|------|-----------|-----------|
| `arm_action_q` | mp.Queue(maxsize=2) | Policy → Arm | Ordered, bounded backpressure |
| `hand_cmd_ring` | SeqlockRingBuffer(8) | Policy → Hand | Latest-wins (position servo) |
| `arm_state_ring` | SeqlockRingBuffer(3) | Arm → Policy | Read-latest (~265B) |
| `hand_state_ring` | SeqlockRingBuffer(3) | Hand → Policy | Read-latest (~328B, no tactile_force) |
| `hand_tactile_ring` | SeqlockRingBuffer(8) | Hand → Policy | Sparse writes (~14.4KB, only on contact) |
| `vr_ring` | SeqlockRingBuffer(8) | VR → Policy | ~600B/frame |
| `camera_ring` | CameraRingBuffer(5) | Camera → Policy | ~1.5MB/frame |
| `is_running` | mp.Value | Main → all | Sole writer: Main |
| `is_recording` | mp.Value | Policy → Arm/Hand/Camera | Sole writer: Policy |
| `error_state` | mp.Value | Arm/Hand → all | Sticky latch (set-only) |
| `estop_request` | mp.Value | Policy → Arm/Hand | ESC key |
| `safety_state` | mp.Value('i') | Main + Policy → all | SafetyState enum (0-3). Main: DISARMED↔ARMED, →FAULT. Policy: ARMED↔RUNNING |
| `arm_heartbeat_s` | mp.Value('d') | Arm → Main | `time.monotonic()` per tick, timeout=1.0s |
| `hand_heartbeat_s` | mp.Value('d') | Hand → Main | `time.monotonic()` per tick, timeout=1.0s |
| `policy_heartbeat_s` | mp.Value('d') | Policy → Main | `time.monotonic()` per tick, timeout=1.0s |
| `vr_heartbeat_s` | mp.Value('d') | VR → Main | `time.monotonic()` per event, timeout=5.0s |
| `camera_heartbeat_s` | mp.Value('d') | Camera → Main | `time.monotonic()` per tick, timeout=2.0s |

### Process Entries

```python
# Each function is an mp.Process target, accepting SharedStorage + optional config:
arm_loop(shared, config)    # robot/inner_loop.py — Mode 6 servo, FK, tracking error
hand_loop(shared, config)   # robot/hand_process.py — XHand position servo, sets error_state
policy_loop(shared, config) # policy/vr_teleop_policy.py — VR→IK + recording, sets is_recording
vr_loop(shared)             # sensor/vr_receiver_process.py — HTS TCP
camera_loop(shared)         # Main — bridges frames from CameraSession → shared.camera_ring
```

### Core Types (`robot/types.py`)

- **`ArmState`** — `qpos(7) qvel(7) tau(7) eef_pos(3) eef_rot6d(6) error_code connected mode tracking_err timestamp` (~294B, from arm_state_ring; eef via `get_position_aa`, tracking_err = max|qpos - last_target|)
- **`HandState`** — `qpos(12) current(12) tactile_sum(5,3) tactile_contact(5) error_state connected timestamp` (328B, from hand_state_ring)
- **`HandTactile`** — `tactile_force(5,120,3)` (14.4KB, from hand_tactile_ring, sparse)
- **`RobotState`** — legacy 22-field monolithic state (Policy assembles from ArmState+HandState+HandTactile for recording)
- **`RobotAction`** — `arm_qpos_cmd(7) hand_qpos_cmd(12)` + optional `target_eef_pos/rot6d`

---

## Key Invariants

1. **All cross-process data through SharedStorage** — never direct SDK calls across processes
2. **Policy owns recording** — single-clock domain, natural (state, action, camera) alignment
3. **Mode 6 handles trajectory** — do NOT interpolate arm commands (double-interpolation → overshoot)
4. **Arm Queue (maxsize=2)** — bounded backpressure; Policy blocks if Arm falls behind
5. **Hand Ring (latest-wins)** — position servo; old targets overwritten
6. **Recording grid-aligned to 16 Hz** (`dt=1/control_hz`) — breaking alignment corrupts downstream
7. **State = bool flags, recording = bool** — not an enum. **Safety state IS an enum** (SafetyState, 0-3), stored in `shared.safety_state`
8. **Seqlock on all control rings** — torn-read protection for arm_state and hand_cmd

---

## Safety State Machine (ManiUniCon P0 — 2026-08-02)

Four-state machine per ManiUniCon §13.2:

```
DISARMED(0) --[Main: all ready]--> ARMED(1) --[Policy: B key]--> RUNNING(2)
     ^                                |  ^                           |  |
     |                                v  |                           v  |
     +---[Main: Q/shutdown]----------+  +---[Policy: C/S/D/H]------+  |
                                                                       |
     FAULT(3) <--[Main: error_state | proc death | heartbeat timeout]-- ANY
       |
       +--[Main: shutdown only]--> DISARMED(0)
```

- **Main** owns: DISARMED↔ARMED, →FAULT, →DISARMED
- **Policy** owns: ARMED↔RUNNING (teleop start/stop)
- **Arm/Hand** read-only: gate servo on `safety_state in (ARMED, RUNNING)`
- **5 process heartbeats** (`time.monotonic()` per tick) monitored by Main at 10Hz
- **Heartbeat timeouts** (from `config/defaults.py` `safety.heartbeat_timeouts`): arm/hand/policy=1.0s, vr=5.0s, camera=2.0s
- **Existing bool flags preserved** (`is_running`, `error_state`, `estop_request`) — state machine is additive
- **See**: `robot/safety.py` for SafetyState enum + transition validation

---

## Simplified Safety Architecture

### Design principle: firmware is the safety backstop

xArm7 Mode 6 firmware already enforces: C22 (position), C24 (velocity), C31 (joint limit),
collision detection, torque limit.  App-level safety prevents firmware trips (1-2s recovery
interrupts collection) and ensures data quality — it does NOT duplicate what the firmware
already catches.

### Layers (single-writer, no defense-in-depth redundancy)

1. **Arm-level:** NaN guard (protects `last_target`) + Mode 6 error handling (C22/C24/C31
   auto-recover → `clean_error+set_mode+set_state`; consecutive recoveries > `_RECOVERY_MAX`
   (30) → FAULT; non-recoverable → immediate FAULT) + `except Exception` path
   also escalates to FAULT after `_RECOVERY_MAX` consecutive failures
2. **Policy-level:** arm connected gate + NaN guard for arm/hand + workspace clamp +
   safety_state gate (ARMED required for B, FAULT blocks send) + hand_qpos_stale hold
3. **IK-level:** workspace clamping + elbow-flip detection + hold-on-failure + delta clamp
4. **E-stop:** Policy sets `estop_request=True` → Arm/Hand detect flag → `set_state(4)`
5. **Error state:** sticky latch (`error_state` mp.Value) — Arm/Hand set, Main detects → FAULT
6. **FK zero-pose guard:** throttled warning on FK failure (code≠0 or exception) — consumers
   see zero EEF with log trail
7. **Heartbeat supervisor:** Main monitors 5 process heartbeats at 10Hz → FAULT on timeout
8. **Safety state machine:** formal DISARMED/ARMED/RUNNING/FAULT states with validated
   transitions (Main owns DISARMED↔ARMED/→FAULT, Policy owns ARMED↔RUNNING)

### Removed (2026-08-03 audit)

- ~~Arm-level joint limit clip~~ (Policy clips, firmware C31 is backstop — arm_loop don't re-clip)
- ~~Policy arm error code gate~~ (arm_loop independently handles all error codes)
- ~~VR quat continuity gate~~ (firmware speed/accel limits + IK delta clamp sufficient)
- ~~startup_error mp.Array~~ (child processes `logger.error()` + `return`; Main detects via
  ready-event timeout + process exit)
- ~~Tactile force safety gate~~ (not a safety concern for this system)
- ~~Arm-level stale target timeout~~ (heartbeat supervisor is the real safety net; redundant
  `set_servo_angle` calls to Mode 6 are no-ops — same target = no motion)

### Added (2026-08-03)

- **Hand qpos_stale hold:** when hand_loop detects driver board lockout (qpos unchanged 15+
  frames), Policy holds prev_hand_qpos and sets retarget_ok=False — prevents gap jump on
  recovery, marks frames for offline filtering
- **Hand cmd NaN ring guard:** Policy no longer pushes NaN-rejected hand commands into
  hand_cmd_ring (hand_loop NaN guard retained as zero-cost backstop)
- **Recovery counter FAULT escalation (2026-08-03 ultracode review):** arm_loop's
  `except Exception` path and state-read C22/C24/C31 recovery path now share
  `_RECOVERY_MAX=30` escalation — persistent servo exceptions or state-read errors
  trigger FAULT instead of silent infinite retry. Separate counters
  (`_consecutive_recoveries` for servo, `_consecutive_state_errors` for state-read)
  prevent cross-contamination. Hand send-error watchdog intentionally excluded —
  hand comm errors are frequently intermittent and self-recovering; the clear_error()
  retry loop is correct.

### Earlier removals (pre-2026-08-03)

- ~~Dual-path estop fallback~~ (flag → loop exit → cleanup, single path)
- ~~Three-tier tracking error~~ (single warning threshold)
- ~~RPC macro subsystem~~ (HOME sentinel through action queue)
- ~~No heartbeat~~ (2026-08-02: per-process heartbeat + Supervisor, ManiUniCon P0)

---

## Known Footguns

- **C24 mid-motion:** IK spike → hold-on-failure → ramp reset → overspeed trip (`c24-ramp-reset-midmotion.md`)
- **Frozen camera:** L515 mid-run silent stall ~35-60s; forward-fill masks it (`l515-midrun-stream-stall.md`)
- **ENOSPC false positive:** Disk check races with async writer (`arm-only-record-session-2026-07-18.md`)
- **Velocity tuning ineffective:** Mode 6 bottleneck is acc/jerk, not velocity (`mode6-tracking-error-root-cause.md`)
- **Arm Queue backpressure:** `maxsize=2` means Policy blocks if Arm falls >125ms behind — monitor with status print

---

## Recording Format

HDF5 v8-10 (auto-selected). All streams grid-aligned to 16 Hz. Pipeline: `TimestampAlignedBuffer` → `EpisodeRecorder` (accumulate-then-dump, async writer). Field catalog: `episode_recorder.py` docstring. Tactile force stored separately in `hand_tactile_ring` (sparse writes).

---

## Conventions

| Aspect | Convention |
|--------|-----------|
| Python | 3.10+, **conda: `real_robot`** |
| Formatting | black (line-length 120), isort (black profile), mypy |
| Imports | `import numpy as np` (universal); `from __future__ import annotations` (preferred); `if TYPE_CHECKING:` for circular deps |
| Logger | `logger = get_logger(__name__)` after ALL imports, before any class/function |
| Types / Naming | `dataclass` for config/state, `numpy` for math; `snake_case`, `PascalCase`, `UPPER_SNAKE` |
| Error handling | fail-safe (NaN→neutral); always `logger.warning("msg", exc_info=True)` |
| Process isolation | mp.Process targets are plain functions (`*_loop(shared)`), not class methods |
| Lazy SDK imports | SDK imports inside process functions (not at module level) — avoids import errors in Main |

---

## Anti-Patterns

- Calling XArm7/XHand SDK from Policy or Main (SDK imports only in arm_loop/hand_loop)
- Creating SHM rings outside SharedStorage (use `shared.xxx_ring`)
- Blocking I/O in 16Hz loop (camera read, file write → silent frame drop)
- Assuming hand is connected without checking `hand_state.connected`
- Mutating RobotState/RobotAction arrays in-place (shape validation only at construction)
- Interpolating arm commands in app code (Mode 6 double-interpolation → overshoot)
- `logger.warning(f"foo: {e}")` without `exc_info=True` (loses stack)
- Circular imports without `TYPE_CHECKING` + lazy imports
- Mutable defaults in dataclass fields — use `field(default_factory=...)`
- Hardcoding rate assumptions (use `control_hz` from config)
- Silently swallowing exceptions without logging (`pass` in except — always `logger.warning(..., exc_info=True)`)
- Putting business logic in Main (Main = spawn + monitor + shutdown, nothing else)

---

## Typical Edit Patterns

| When you... | Also update... |
|-------------|---------------|
| Add a field to ArmState/HandState | `shared_storage.py` (dtype) + `types.py` (dataclass) + arm_loop/hand_loop (write) + policy (read) |
| Add a recording dataset | `episode_recorder.py` + `episode_reader.py` + `check_episode_health.py` |
| Change IK solver | `planning/ik.py` + `policy/vr_teleop_policy.py` |
| Add a new ring to SharedStorage | `shared_storage.py` + producer process + consumer process |
| New entry point (new architecture) | Follow Main pattern: `SharedStorage.create()` → spawn `*_loop(shared)` → monitor |
| Tune arm dynamics | `inner_loop.py` (ArmInnerLoopConfig) + Mode 6 acc/jerk; velocity alone has near-zero impact |

---

## Entry Points

**Primary (new architecture):** `examples/real/vr_teleop_hand_record.py` (~310 lines, canonical entry point — 5-process SharedStorage model, `--task`/`--operator`/`--acc`/`--speed`/`--no-hand` CLI).
**Also new arch:** `keyboard_teleop_real.py`, `calibrate_vr_heading.py`, **`replay_traj.py`**, **`calibrate_camera.py`**.
**Diag:** `calibrate_l515_depth.py`, `test_*.py`.
**Sim:** `examples/sim/vr_teleop_sim.py`, `test_motion_planning_sim.py`.

---

## Hardware Notes

**xArm7 Mode 6:** Firmware trajectory planning, targets at 16 Hz. No inner-loop interpolation. Tracking error bottleneck is acc/jerk, not velocity. Default: 120°/s, acc adjustable via `ArmInnerLoopConfig.joint_max_acc_rad_per_s2`.

**XHand:** 12-DOF EtherCAT position servo. Latest-wins semantics (hand_cmd_ring). Tactile: 5 fingers × 120 taxels × 3 axes. Board errors auto-logged.

**L515:** Direct motherboard USB 3.0 only (no hub; verify `lsusb -t`, 8086:0b64 under root hub). Depth intrinsics bad state: `hardware_reset()`. Mid-run stream stall ~35-60s. XU flaky: use `set_option` fallback.

**Quest VR:** HTS TCP on port 8000. `adb reverse tcp:8000 tcp:8000` for USB. `vr_loop` handles coordinate conversion (Unity left-hand → FLU).

**Deps:** `mplib`, `pinocchio`, `h5py`, RealSense SDK, XArm7/XHand SDKs, `sapien`, `numpy`, `pyav`.

---

## ToDo

- **P0 — 采集入口加 task_label 参数**: ✅ `--task`/`--operator` CLI 已在 `vr_teleop_hand_record.py`（主入口）接入。
- **P1 — held 帧过滤工具**: ✅ `tools/filter_training_frames.py` 已实现 (2026-08-03)。训练前按 `flag_held == True` 过滤（`data.h5` 已有此字段）。
- **P2 — 跟踪误差过滤阈值**: ✅ tracking error 已由 arm_loop 计算并发布到 arm_state_ring，Policy 记录到 HDF5。filter_training_frames.py 支持 `--max-tracking-error`。
- **P3 — camera_loop 独立模块**: ✅ camera_loop 已提取到 `sensor/camera_process.py`。
- **P4 — 旧入口点迁移**: ✅ 全部完成 (2026-08-03)。旧入口点已删除，所有功能已迁移至 SharedStorage 架构。
- **P5 — Config 常量整合**: ✅ 全部完成 (2026-08-03)。SharedStorageConfig 集中 ring maxlen、camera 默认分辨率、workspace bounds。
- **P6 — 异常处理加固 (2026-08-02)**: ✅ S02 validate gate 接入 policy_loop；✅ ERR-1~ERR-11 全部修复；✅ PI-4 hand_loop home settle；✅ PI-5 policy init 异常；✅ ThrottledWarner 支持 kwargs。
- **P6c — 安全简化 (2026-08-03)**: ✅ Arm 层关节限位裁剪移除；✅ Policy 层 arm 错误码门控移除；✅ startup_error mp.Array 移除；✅ VR/NAN/tactile/hand 门控简化；✅ arm_loop stale target timeout 移除。
- **P6b — Safety State Machine + Heartbeat (2026-08-02)**: ✅ SafetyState enum + transition validation；✅ 5 per-process heartbeats；✅ Main 10Hz heartbeat supervisor；✅ arm_loop/hand_loop/policy_loop safety gate；✅ FAULT transition on non-recoverable errors。Ref: ManiUniCon §13.2 P0 #2, #4。
- **Phase 3~7 — 入口点全量迁移 (2026-08-03)**: ✅ vr_teleop_hand_record.py (canonical)、keyboard_teleop_real.py、calibrate_vr_heading.py、replay_traj.py、calibrate_camera.py 全部迁移至 SharedStorage 架构。
- **Phase 9b — SharedStorageConfig 常量整合 (2026-08-03)**: ✅ SharedStorageConfig dataclass 集中 ring maxlen、camera 默认分辨率、workspace bounds。
- **Dead code cleanup (2026-08-03)**: ✅ 删除 robot/interface.py、robot/validate.py、robot/preflight.py、robot/arm_process.py；✅ hand_process.py 移除所有 legacy 类（HandControlProcess/HandSHMFaçade/HandSHMAdapter 等）；✅ types.py 移除 RobotInterfaceConfig；✅ (2026-08-03 后续) 删除 defaults.py 6 个 _Deprecated* 类 + TeleopDefaults；✅ 删除 planning/collision_config.py；✅ 删除 test_quest_hand_teleop.py；✅ shared_storage.py 移除 4 个 re-export；✅ 清理所有过期注释/docstring 中的 RobotInterface/RobotInterfaceConfig/ArmProcess 引用；✅ 更新 CLAUDE.md 架构图。
- **Code review bugfixes (2026-08-03)**: ✅ calibrate_camera.py CRITICAL NameError 修复（_get_ee_pose 中未定义的 Rotation）；✅ replay_traj.py queue.put 死锁修复（error_state 门控 + estopped 守卫）；✅ 多处 heartbeat wait loop 修复。
- **Code review round 2 (2026-08-03)**: ✅ CRITICAL: keyboard_teleop NameError 修复（XArm7Config 未导入）；✅ CRITICAL: `--acc`/`--speed` CLI 参数传递到 arm_loop（hand_record + keyboard_teleop）；✅ CRITICAL: arm_action_q blocking put() 改为 timeout-protected（policy_loop）；✅ HIGH: `--no-hand` 标志修复（hand_enabled 字段 +hand_loop 跳过）；✅ HIGH: arm_loop/hand_loop 资源泄漏修复（init 失败时 disconnect）；✅ HIGH: 重复辅助函数提取（read_arm_state/read_hand_state/write_hand_cmd → shared_storage.py）；✅ HIGH: 重复 rot6d 转换函数去重；✅ vm_loop 心跳竞态修复；✅ _simple_homing 心跳阻塞修复；✅ HandState.from_ring() 添加 qpos_stale；✅ arm.home_qpos 常量集中；✅ 死配置字段/无用导入/过时 docstring 清理。
- **Ultracode 全库审查修复 (2026-08-03)**: ✅ M1: arm_loop 恢复路径 FAULT 升级（`_RECOVERY_MAX=30` 常量 + `except Exception` 路径 + state-read C22/C24/C31 独立计数器）；✅ F#15: ArmWristMapper.reset() NaN 守卫；✅ F#3: camera_loop 元数据轮询；✅ F#13: camera_serial 校验接通；✅ F#14: 相机帧读取错误日志 DEBUG→WARNING；✅ F#4: smoothing_alpha 死参数修复（默认 None，YAML 仅在未指定时覆盖）；✅ F#18: HandRetarget.reset() 关节顺序重映射；✅ F#23: 仿真 np.clip NaN 守卫；✅ F#17: GlobalKeyState stop-before-start 修复；✅ F#12: 空 MP4 守卫；✅ F#8: align 丢弃相机→报错；✅ 死代码清理（vr_tracker event/last_read_key, sim_adapter last_delta_limited, xarm7_xhand 3 dead methods, serialization 不可达分支, log.py _loggers 冗余, visualize sys.path hack）；✅ 代码质量（audio bare except, pointcloud FPS warning, is_error 简化, types string concat）。Hand 发送错误看门狗**保留现状**（通信错误偶发可恢复，clear_error 重试是正确策略）。
