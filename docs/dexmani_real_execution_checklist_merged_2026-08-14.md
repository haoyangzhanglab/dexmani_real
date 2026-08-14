# DexMani Real — 合并执行清单（剔除已过时部分）

> 生成日期：2026-08-14
> 来源：合并 `dexmani_real_claude_code_execution_plan_reviewed_2026-08-14.md` 与
> `dexmani_real_xarm_xhand_modification_plan.md`，并对照当前 `main`
> （`fc0bd9f` 之后）逐项 fact-check。
> 适用对象：Claude Code / coding agent
>
> 状态标记：🔧 待执行 · ✅ 已完成/无需改 · 🧪 硬件门控 · 🚫 已剔除

---

## 0. Fact-check 结果总表

| Item | 来源 | 状态 | 结论 |
|---|---|---|---|
| A1 C24 recovery 缺 `motion_enable` | 文档二 | 🔧 | `_recover_c24_measured_hold` 无 `motion_enable(True)`（`arm_loop.py:186-209`）；`_enter_mode6_ready` 已存在可复用（`:229-233`） |
| A2 setter 失败读缓存 error | 文档二 | 🔧 | `arm_loop.py:841` 仍 `int(getattr(arm,"error_code",0))`；`_read_live_status`/`get_err_warn_code` 已存在但未用于该分支 |
| A3 homing 恢复读缓存 error | 文档二 | 🔧 | `arm_loop.py:1398` 仍用缓存 `error_code` 决定 restore Mode 6 |
| A4 `motion_enabled` 命名歧义 | 文档二 | 🔧 | `arm_loop.py:586` 局部变量存在 |
| A5 `clear_error()` 误导命名 | 文档二 | 🔧 | `xhand.py:871-875` 仅清本地 latch；callsite 在 `hand_process.py:324,421` |
| A6 `stop()` 死代码 | 文档二 | 🔧 | `xhand.py:882-902` 存在但**零真实调用**（`rg 'hand\.stop'` 无结果）→ 直接删除 |
| A7 TCP load 硬编码 | 文档二 | 🔧 | `arm_loop.py:516` 硬编码 `weight=1.1, cog=[16.3,7.9,109.5]`；`ArmParams`(defaults.py:216) 与 `ArmLoopConfig`(arm_loop.py:42) 均存在 |
| B1 两-controller discovery | 文档二 | 🧪 | `xhand.py:293/340` 分离 discovery/open control，含详尽注释 |
| B2 INIT+close+watchdog | 文档二 | 🧪 | `xhand.py:791-814` `_request_slave_init`→`close_device`→`sleep(2.0)` |
| W0 offline checks 骨架 | 文档一 | 🔧 | 仍无 `checks/offline/` |
| W1 执行器异常安全清理 | 文档一 | 🔧 | arm/hand 清理块存在但**无 `try/finally`**，循环内意外异常会跳过 disconnect |
| W2 ring 提交/发布契约 | 文档一 | 🔧 | `write_seq` 在 payload 之前发布（`ring_buffer.py:229`），`write_idx` 最后（`:243`）；`get_last_k` 遇 seq-mismatch 立即返回丢弃旧历史（`:288-292`）；camera 时间戳在拷贝前（`:572`） |
| W3A hand delta 预检 | 文档一 | 🚫→可选 | learned 路径已删；teleop `_sanitize_hand_command`（`hand_control.py:17`）已内联 delta 校验 |
| W3B `wait_applied` ACK 语义 | 文档一 | 🔧 | `safety.py:596-607` 只等 arm；`publish_hand_home_and_wait_applied`（`:623`）可复用 |
| W4 VR 旋转尖峰恢复 | 文档一 | ✅ | **已修复**（`arm_mapper.py:152-154,203` 基线存 raw 而非 gated），剔除 |
| WA1A 感知运行时契约 | 文档一 | 🔧 | `pointcloud_processor.py:109/176` 硬编码 `desk_plane.json`；无 `align_mode!=depth_to_color` 早失败 |
| WA1B 点云元数据 | 文档一 | ✅~复核 | `pointcloud_valid`/`pc_*` 元数据字段已大量存在（schema.py:156-161,285-304） |
| WA1C 标定原子写 | 文档一 | 🔧 | 标定写为 `open('w')` 或 rename-to-backup，非原子；`transaction.py:35-45` 已有 `atomic_publish` 可借鉴 |
| WA2 grid action vs send-event | 文档一 | 🔧 | `timestamp_buffer.py:245-251` gap-fill 全量拷贝 `_last_source_data`，synthetic hold 继承 `flag_action_queued/action_id` |
| WB1/WB2/WB3 learned policy | 文档一 | 🚫 | `policy/` 仅剩 `runtime.py/safety.py/loop_timing.py`，learned 已删 |
| 文档一 §1.9-1.11 learned contracts | 文档一 | 🚫 | 目标文件不存在 |

---

## 1. 已剔除的过时部分

以下内容因 `fc0bd9f`（删除 learned-policy/inference + 诊断层）或已修复而**不再适用**，后续会话不得执行：

1. **文档一 Lane B 全部**：WB1（观察历史语义）、WB2（fail-close/rewarm/`reset_policy`）、WB3（adapter/copy 优化）。
2. **文档一 §1.9 / §1.10 / §1.11**：learned observation history / learned fail-close / stateful policy lifecycle 三条 Contract。
3. **文档一 W3A 的 learned 路径**：依赖已删除的 `policy/learned_coordinator.py`。
4. **文档一 W4（VR 旋转尖峰恢复）**：已修复（见 §2 证据）。
5. **文档一 §13 `TimestampAlignedBuffer → GridBuffer` 重写**：本已 DEFER，且 `max_record_steps` 截断 + 固定网格 gap-fill 有研究价值。
6. **文档一 §15 WB3**：Policy adapter/copy-path 重构（无真实 adapter + profiler 数据）。
7. **文档一 §23 显式 DEFER 清单**中与 learned 相关项。

---

## 2. 仍有效的全局 Contract（精简）

优先级 `AGENTS.md → CLAUDE.md → 本清单 → README/comments`。冲突时停止并报告。

| # | 主题 | 核心规则 |
|---|---|---|
| C1 | 硬件所有权 | xArm/XHand/RealSense SDK 只在各自 worker 内；Main/policy/recorder 禁调 SDK |
| C2 | 硬件生命周期 | arm：所有退出 `state=4`→disconnect；hand：**只 disconnect，不 stop/unforce/mode0** |
| C3 | 发布时间所有权 | `source ≤ receive ≤ publish ≤ anchor`；ring 拥有权威 publish 时间 |
| C4 | Ring 序列 | 提交后的逻辑序列才代表"已提交 slot"；禁止先发布序列再写 payload |
| C5 | 安全边界 | SafetyGate = well-formed+limits+workspace；固件 = 速度/加速度/碰撞兜底；**禁止加回软件碰撞几何** |
| C6 | Hand command-delta | command-to-command bound；拒绝而非 clip |
| C7 | Action transport | arm 有序队列、hand latest-wins ring；**非事务**，禁止 PREPARE/COMMIT |
| C8 | Action ACK | `wait_applied=True`：arm `>=`、hand `==`；hand 被 supersede 立即失败 |
| C9 | Recording 语义 | 区分"grid 有效 action"与"是否真的产生 send event" |
| C10 | Runtime config | `CLI > YAML > defaults`；已 resolve 的事实不重读 |
| C11 | 架构权威 | `examples/` = thin CLI，domain = lifecycle/behavior |

---

## 3. 执行清单

### 3.1 Phase A — xArm/XHand SDK 契约（🔧 可离线验收，P0 优先）

> 不重构硬件架构：保持 Mode 6 + `set_servo_angle(wait=False)`、Mode 0 HOME、arm 队列 `maxsize=2`、hand latest-wins、persistent HandCommand + Mode 3、整条 delta 拒绝。

#### A1 — C24 recovery 补 `motion_enable(True)`（P0）
- **文件**：`dexmani_real/robot/arm_loop.py`
- **事实**：`_recover_c24_measured_hold()`（:186-209）现为 `clean_error → clean_warn → set_mode(6) → set_state(0) → get_joint_states → set_servo_angle`，缺 `motion_enable(True)`；`_enter_mode6_ready()`（:229-233）已含 set_mode(6)+set_state(0)+`_wait_live_status(state=2,mode=6)`。
- **改法**：clean_error/clean_warn 后 `motion_enable(True)`，再复用 `_enter_mode6_ready`（不手写第二套 postcondition）。可选 best-effort `get_c24_error_info()`（`hasattr` gate，失败不影响恢复）。
- **不要改**：2 秒内第二次 C24 → sticky fault 策略、`max_consecutive_recoveries`、measured-hold 数量（仍 exactly one）、speed/mvacc、`wrap_nearest_equivalent`。

#### A2 — setter 失败改用 live controller error（P0）
- **文件**：`dexmani_real/robot/arm_loop.py`
- **事实**：`set_servo_angle` 非零返回后 `:841` 立即读缓存 `arm.error_code`；`_read_live_status()`（:295）与 `get_err_warn_code()` 已存在；`collision_fault_errors` 字段存在（defaults.py:271）。
- **改法**：加窄 helper `_read_live_error_code()`（同步 `get_err_warn_code()`，live 失败→generic sticky fault，**不 fallback 缓存**）。非零返回分类改用该 helper；`err_code==0` 但 setter code 非零也按 unknown fault。
- **保留**：steady-state telemetry（`arm_loop.py:930` `arm.error_code`）继续用缓存，**不改**。

#### A3 — homing 恢复决策改用 live error（P1）
- **文件**：`dexmani_real/robot/arm_loop.py`
- **事实**：`_planned_homing` 尾部 `:1398` 用缓存 `error_code` 决定 restore Mode 6。
- **改法**：`_read_live_error_code()`；diagnosis 失败 → `-1` → fail-closed 不恢复 Mode 6，走现有 stop path。

#### A5 — `clear_error()` → `clear_local_error()`（P1）
- **文件**：`dexmani_real/robot/xhand.py`、`dexmani_real/robot/hand_process.py`
- **事实**：`clear_error()`（xhand.py:871-875）只清 `error_state/last_error_code/last_error_message`，无硬件调用；callsite 仅 `hand_process.py:324,421`。
- **改法**：重命名 `clear_local_error()`，同步 callsite 与日志文案（明确"local driver latch reset"，非硬件 clear）。
- **不要改**：retry 次数、board error watchdog、`shared.error_state` sticky 行为、xArm `clean_error()`（勿误改）。

#### A6 — 删除 `XHand.stop()`（P1）
- **文件**：`dexmani_real/robot/xhand.py`
- **事实**：`stop()`（:882-902）构造 mode=0/tor_max=0/kp=ki=kd=0 命令并置 error_state；`rg 'hand\.stop'` 与全仓 `.stop()` 均无 XHand 调用 → 死代码。
- **改法**：**直接删除**。不要为"未来可能用"保留语义不清的 API，不加 deprecated alias。

#### A4 — `motion_enabled` 语义澄清（P2，小改动）
- **文件**：`dexmani_real/robot/arm_loop.py`
- **事实**：局部变量 `motion_enabled`（:586）实为"控制器已接受 Mode-6 command 的状态"，易与 SDK `motion_enable()` 混淆。
- **改法**：仅局部重命名为 `accepts_motion_commands`（或 `controller_motion_ready`）。不改 `SafetyState`、不新增 enum、不调 `motion_enable(False)`。

#### A7 — TCP load 配置化（P2）
- **文件**：`dexmani_real/config/defaults.py`、`dexmani_real/robot/arm_loop.py`
- **事实**：`arm_loop.py:516` 硬编码 `weight=1.1, cog=[16.3,7.9,109.5]`；`ArmParams`(defaults.py:216, frozen) 与 `ArmLoopConfig`(arm_loop.py:42, 含 `from_runtime`) 均存在。
- **改法**：`ArmParams` 增 `tcp_load_mass_kg: float = 1.1`、`tcp_load_cog_mm: tuple = (16.3,7.9,109.5)`（`__post_init__` 最小 finite/shape 校验）；`ArmLoopConfig` 增对应字段并从 runtime 读取；`set_tcp_load` 改用 cfg 值。
- **不要改**：`config/runtime.py` 的泛化 resolver（会自动重建 dataclass 字段）；不新建 `TcpLoadParams` dataclass。

---

### 3.2 Common Core — IPC / 生命周期（🔧 可离线验收）

#### W1 — 执行器异常安全清理（P0）
- **文件**：`dexmani_real/robot/arm_loop.py`、`dexmani_real/robot/hand_process.py`
- **事实**：arm 清理块（arm_loop.py:1005-1021）与 hand 清理（hand_process.py:476-487）均在 `while` 循环**之后直线代码**，**无 `try/finally`**；循环内意外 Python 异常会跳过 disconnect。arm 早退路径用 `_disconnect_arm()`（:502/528/562/576/601），但终端清理未复用。
- **改法**：把 run-loop 包进 `try/finally`，`finally` 里 best-effort 清理——arm：`state=4`→verify→`disconnect()`；hand：**只 `disconnect()`**，绝不新增 motion。复用 `_disconnect_arm`，消除分散的早退清理碎片。
- **不要改**：`xhand.py` 命令语义、SafetyGate、shared-memory schema；hand 清理不加 stop/unforce/mode0。

#### W2 — Ring 提交/发布契约（P0，IPC 正确性）
- **文件**：`dexmani_real/shm/ring_buffer.py`
- **事实**（已核对 write/get_last_k）：
  1. `write()`（:217-245）：`_write_seq[0]` 在 payload 之前递增发布（:228-229），`write_idx` 最后（:243）→ 违反 C4"序列最后发布"。
  2. `get_last_k()`（:268-304）：seq-mismatch（slot 已被更新覆盖）时 `:288-292` **立即 return**，丢弃更旧可用历史；torn slot 则 `continue`。
  3. Camera ring：`now_ns` 在 RGB/depth/pointcloud 大拷贝**之前**采样（:572），publish 时间偏早。
- **改法**：ring 拥有权威 publish 时间并**覆盖** payload 的 `publish_monotonic_ns`（camera header 已覆盖，但要把 `now_ns` 采样移到拷贝完成后、even commit 前）；`write_seq` 移到 even commit 之后（最后）；`get_last_k` 遇 mismatch 标记 dropped 并继续向更旧序列，返回 oldest-first。
- **不要加**：mutex、读写锁、atomic 包、multi-producer；保持单生产者 seqlock。

#### W3B — `wait_applied` 显式 ACK 语义（P1）
- **文件**：`dexmani_real/policy/safety.py`
- **事实**：`publish_joint_targets(wait_applied=True)`（:596-607）只读 `arm_state_ring` 等 arm ACK，含 hand 的 action 不查 hand；可复用的精确 hand-ack 路径 `publish_hand_home_and_wait_applied` 已存在（:623）。
- **改法**：arm-only 保持 `last_cmd_seq >= action_id`；arm+hand 需 `arm >= action_id AND hand == action_id`；hand `> action_id` → 立即失败（superseded）。复用现有 hand-home ack 语义，不发明第三种"applied"定义。
- **不要**：让普通 16Hz action 阻塞等 ACK（仅显式 `wait_applied=True` 调用者）。

#### W3A — 降级为可选
- learned 路径已删除，W3A 的"阻止 coupled action 因 hand delta 违规而 arm 先入队"主要对象消失。teleop 已由 `_sanitize_hand_command()`（`teleop/hand_control.py:17,54-58`）内联 delta 校验。**仅当需要跨路径复用时**才抽取纯 helper `validate_hand_command_delta()`；否则不动。

---

### 3.3 Lane A — 3D 采集（🔧 采新数据集前）

#### WA1A — 感知运行时契约（P0，采数据前）
- **文件**：`dexmani_real/sensor/camera_process.py`、`dexmani_real/sensor/pointcloud_processor.py`、`dexmani_real/config/runtime.py`（仅当 cross-section 校验最干净时）
- **事实**：`pointcloud_processor.py:109` 硬编码 `desk_plane_path="dexmani_real/config/desk_plane.json"`，`:176-179` 自动加载；runtime 已 resolve `environment.table.plane_abcd` 但点云处理器未消费 → **两个独立 desk-plane 来源**。无 `align_mode != depth_to_color` 早失败（camera_process 不校验）。
- **改法**：生产点云消费 resolved `plane_abcd`；`table.enabled==false → desk_plane=None`，不 fallback 加载；生产 camera loop 在 `align_mode != depth_to_color` 时 fail early。不把每个调优 knob 塞进 YAML。

#### WA1B — 点云元数据（✅ 已基本完成，仅复核）
- **事实**：`pointcloud_valid`、`pc_num_points`、`pc_valid_depth_ratio`、`camera_health` 等字段已在 schema.py:156-161/285-304、episode_recorder.py:488-499、pointcloud_processor `to_meta_dict()` 中落地。
- **仅复核**：`has_pointcloud` 是否 = `pointcloud_valid_frames > 0`（而非 `camera_frame_count > 0`）；缺失的有效设置（voxel/DBSCAN/FPS 等）若影响输出再补，复用现有 metadata channel，不建新 metadata 服务。

#### WA1C — 标定原子写（P1）
- **文件**：`dexmani_real/sensor/pointcloud_processor.py`（`save_desk_plane`）、`examples/calibrate_camera.py`、`examples/calibrate_vr_heading.py`
- **事实**：标定写为 `open('w')` 或 rename-to-backup-then-write，非原子；`recording/transaction.py:35-45` 已有 `atomic_publish`（temp+fsync+rename+fsync parent）可借鉴。
- **改法**：`build JSON → write temp（同目录）→ flush → fsync temp → os.replace → fsync parent`；backup 用 copy 而非 rename。两处真实 caller 时可抽一个极小 `atomic_json_dump()`，不建事务框架。

#### WA2 — synthetic hold 不伪造 send event（P0，依赖 `--source sent`/延迟分析前）
- **文件**：`dexmani_real/recording/timestamp_buffer.py`、`episode_recorder.py`、`episode_reader.py`、`examples/replay_episode.py`
- **事实**：gap-fill（timestamp_buffer.py:245-251）全量拷贝 `_last_source_data`，`CAUSAL_HOLD_LAST` slot 继承 `flag_action_queued/action_id/action_created_monotonic_ns`（episode_recorder.py:478-483 构建，未清零）。`FillReason` 为 `SOURCE/CAUSAL_HOLD_LAST/LEADING_PLACEHOLDER`（非文档旧名）。
- **改法**：synthetic hold 继承 effective target，但**清零 send-event 字段**（用现有 schema/reader 的 sentinel）；replay 在 `send_mask==false` 的 slot 不 republish。修改前先重 trace 当前 sampled-hold 路径，勿带旧结论；不 bump schema 除非不可避免。

---

### 3.4 Phase B — XHand lifecycle（🧪 硬件门控实验，Claude 不得声称完成）

> 用 git commit/branch 做 A/B，一次只改一个变量，真人 soak test 后决定保留。**Claude Code 只准备 patch + 静态 review + fake lifecycle check + 人工 A/B checklist，不执行硬件。**

#### B1 — connect 改 single-controller-per-attempt（🧪）
- **文件**：`dexmani_real/robot/xhand.py`
- **事实**：`_retry_open_device`（:271-433）用 `temp_control` discovery → close → 新 `self.control` open；注释记录了 SDO/raw-socket 现象。
- **实验**：每次 attempt 用同一 `XHandControl` enumerate→open；成功后删临时 discovery controller 逻辑。隔离变量：保留 retries/delay/disconnect INIT-watchdog。
- **判定**（真人）：无 reproducible `write sdo failed` 回归、failure rate 不高于 baseline、不需 power-cycle、无 close/reconnect hang、无资源泄漏。仅当出现"single-controller→SDO fail，isolated discovery→成功"的可重复证据，才加**极窄** compatibility switch。

#### B2 — disconnect 改 close-only（🧪，仅 B1 选定后）
- **文件**：`dexmani_real/robot/xhand.py`
- **事实**：`disconnect()`（:791-814）`_request_slave_init`→`close_device`→`sleep(2.0)`。
- **实验**：`close_device` + `connected_flag=False` + `control=None`（前置：command loop 已退出、无后续 send/read）。测 normal exit / SIGTERM / exception finally / supervisor 回收 / SIGKILL（只记录恢复）。
- **通过后**：删 `_request_slave_init`、`_EC_STATE_*`、`_POST_DISCONNECT_WATCHDOG_WAIT_S` 及"必须等 watchdog"注释。仅当某 SDK/firmware 组合需旧路径才保留 narrow fallback（注释写清 SDK 版本/固件/复现/日期）。

---

### 3.5 结构减法（最后，行为保持）

- **WC1** 运行时拓扑/生命周期简化：`examples` 保持 thin CLI；用一个小 `WorkerSpec` frozen dataclass 收敛重复拓扑，不加 registry/factory；readiness timeout 从组启动时间测。先简化 collect teleop 与 replay 生命周期。
- **WC2** 死配置/状态/注释清理：每删一处先 `rg "<symbol>" dexmani_real examples`；候选如 `apply_timeout_s`、`workspace_bounds`、`action_validity_s`、producer 侧 `publish_monotonic_ns` prefill（W2 后）、stale alias。**不删 v16 reserved/zero-filled 持久字段**（schema 清理需 DG-3）。
- **WC3** 大文件可读性：只按单一 domain 拆（如 arm_loop homing → `robot/homing.py`），禁止"顺手 format"。

---

## 4. 执行顺序与 commit 切分

```
1. W0 建 checks/offline/（结构，仅 Python+assert）
2. A1 → A2/A3 → A5/A6 → A4 → A7        （xArm/XHand SDK 契约）
3. W1（异常安全清理）
4. W2（ring 契约）
5. W3B（wait_applied ACK）
6. 人工复核门：git log/diff、offline checks、compileall、无 SDK 所有权变化、无 SafetyGate 几何回归
7. 若采 3D：WA1A → WA1C → WA2（WA1B 仅复核）
8. 若做 XHand lifecycle：B1 → B2（真人 soak，逐项合入）
9. 最后：WC1 → WC2 → WC3
```

建议 commit（一个语义一个）：
```text
test: add minimal offline regression checks
fix(xarm): follow SDK contract when recovering C24
fix(xarm): use live controller errors for control decisions
refactor(robot): clarify arm gate and XHand local error semantics   # A4+A5+A6
refactor(xarm): move TCP load calibration into runtime config        # A7
fix: make actuator worker cleanup exception-safe                     # W1
fix: publish ring sequence only after commit                         # W2
fix: require hand ack for synchronous coupled action                 # W3B
fix: use resolved perception calibration contract                    # WA1A
fix: atomically publish calibration json                             # WA1C
fix: separate synthetic grid hold from send event                    # WA2
refactor(xhand): use one controller for EtherCAT discovery and open  # B1（硬件实验）
refactor(xhand): simplify validated EtherCAT disconnect lifecycle    # B2（硬件实验）
```

---

## 5. Definition of Done（剔除 learned 部分）

- actuator Python 异常不能绕过 cleanup 契约（arm state4+disconnect / hand disconnect-only）
- ring 全局序列/publish 时间代表已提交帧
- setter/homing 关键决策用 live error，不误信缓存
- 同步 coupled `wait_applied` 等所有含入 actuator 的 ACK
- 点云用 resolved table/workspace 契约；生产 alignment 显式；标定原子写；synthetic hold 不伪造 send event
- C24 recovery 遵循 SDK contract（clean→enable→Mode6 ready→一次 measured hold）
- XHand 无死代码/误导命名；shutdown 不造新 motion
- README/CLAUDE/AGENTS 对 active contract 无矛盾
- 无新增 manager/plugin/registry 框架

---

## 6. 决策门（Claude 不得自行改）

- **DG-1** 在线几何检查：保持 workspace/firmware 兜底 + 显式 homing/replay 几何检查，不加回 generic collision/transition 到 SafetyGate（改需 owner 批准 + 延迟预算）。
- **DG-2** 眼在手相机：不提前加动态 eye-in-hand；生产路径保持显式静态 eye-to-hand。
- **DG-3** HDF5 schema bump：不因清理 reserved 字段或改名 sent/provenance 语义而 bump v16；仅当数据消费者需要不兼容表示变更时。

---

## 7. 补充：参考依据

以下来源用于确定 SDK 语义与社区实践。执行修改时**不复制**这些项目的架构，只用于确认 API contract 与常见做法。证据权重：`vendor contract / official integration > 多个独立实现的共同实践 > 单个研究仓库的 workaround`。

### 本仓库基线
- Repository: https://github.com/haoyangzhanglab/dexmani_real
- `robot/arm_loop.py`: https://raw.githubusercontent.com/haoyangzhanglab/dexmani_real/main/dexmani_real/robot/arm_loop.py
- `robot/xhand.py`: https://raw.githubusercontent.com/haoyangzhanglab/dexmani_real/main/dexmani_real/robot/xhand.py
- `robot/hand_process.py`: https://raw.githubusercontent.com/haoyangzhanglab/dexmani_real/main/dexmani_real/robot/hand_process.py
- `config/defaults.py`: https://raw.githubusercontent.com/haoyangzhanglab/dexmani_real/main/dexmani_real/config/defaults.py
- `CLAUDE.md` / `AGENTS.md`

### xArm 官方
- UFACTORY xArm Python SDK: https://github.com/xArm-Developer/xArm-Python-SDK
- Python API（`clean_error` 契约）: https://github.com/xArm-Developer/xArm-Python-SDK/blob/master/doc/api/xarm_api.md
- C++ API: https://github.com/xArm-Developer/xArm-CPLUS-SDK/blob/master/doc/xarm_cplus_api.md
- UFACTORY ROS / mode 示例: https://github.com/xArm-Developer/xarm_ros
- UFACTORY LeRobot adapter: https://github.com/xArm-Developer/lerobot_robot_ufactory

### XHand / 研究对照
- PF-DAG: https://github.com/XiaohanLei/PF-DAG
- pi-r2-flow: https://github.com/pi-r2-flow/pi-r2-flow
- DexUMI: https://github.com/real-stanford/DexUMI
- LeFranX: https://github.com/wengmister/LeFranX
- Robotology XHand YARP 集成: https://github.com/robotology/yarp-device-xhand

---

## 8. 补充：Phase B 人工 A/B checklist 明细

> Claude Code **不执行**以下操作，只生成 checklist / 记录模板。真人执行。

### 8.1 环境记录（每次实验）
```text
Git commit
XHand sdk_version
XHand serial_number
hand type
Python version
xhand_controller package/native library location
EtherCAT interface/device_name
process start method
OS/kernel
```

### 8.2 正常 reconnect soak
```text
至少 100 次起步；若成本可接受，500 次更有判别力。
每次循环：
  create worker/driver → connect → 一次 fresh read
  → (optional non-motion/read-only health check)
  → disconnect → destroy → short interval → reconnect
```
- 没有明确运动授权时**不要发送 finger motion command**。
- 记录：`connect success/fail`、`open_ethercat error code`、vendor stdout/stderr diagnostic、SDO failure string、connect latency、disconnect latency、whether power cycle required、whether next session can reconnect。

### 8.3 B1 判定（single-controller 是否值得取代旧方案）
至少满足：
- 正常循环无 reproducible `write sdo failed` 回归
- failure rate 不高于 baseline
- 不增加 power-cycle requirement
- 不出现 close/reconnect hang
- repeated run 后资源不明显泄漏
- 失败时 diagnostics 仍足以定位问题

仅当出现**可重复**现象 `single-controller → SDO fail；isolated discovery → 同 setup 成功`，才有证据保留 isolated discovery workaround；此时才加一个**极窄** compatibility switch，不提前加。

### 8.4 B2 实机验收（在正常 reconnect soak 之外）
至少测试：
- normal exit
- SIGTERM / orderly process termination
- Python exception 触发 finally cleanup
- worker crash 后由 supervisor 回收
- **SIGKILL 只记录恢复行为，不要求 Python cleanup 能执行**

判定重点：`close-only 后下一次 reconnect 是否稳定`、`是否仍出现 stale OP / SDO failure`、`是否仍需 2–3s watchdog wait`。

---

## 9. 补充：C24 调用顺序与 live-error 离线 fake-check

仓库无 conventional test suite。从 repo root 运行一次性 deterministic fake（若真实 helper contract 与这里稍有不同可调整 fake，但**不能为了让 fake 通过而弱化 production postcondition**）。

### 9.1 C24 recovery 调用顺序
```bash
conda run -n real_robot python - <<'PY'
import numpy as np
from dexmani_real.robot.arm_loop import ArmLoopConfig, _recover_c24_measured_hold

class FakeArm:
    def __init__(self):
        self.calls = []
        self.mode = 6

    def get_c24_error_info(self):
        self.calls.append("get_c24_error_info")
        return 0, [3, 1.23]

    def clean_error(self):
        self.calls.append("clean_error")
        return 0

    def clean_warn(self):
        self.calls.append("clean_warn")
        return 0

    def motion_enable(self, value):
        self.calls.append(f"motion_enable({value})")
        return 0

    def set_mode(self, value):
        self.calls.append(f"set_mode({value})")
        self.mode = value
        return 0

    def set_state(self, value):
        self.calls.append(f"set_state({value})")
        return 0

    def get_state(self):
        self.calls.append("get_state")
        return 0, 2

    def get_err_warn_code(self):
        self.calls.append("get_err_warn_code")
        return 0, [0, 0]

    def get_joint_states(self, **kwargs):
        self.calls.append("get_joint_states")
        return 0, [np.zeros(7)]

    def set_servo_angle(self, **kwargs):
        self.calls.append("set_servo_angle")
        return 0

arm = FakeArm()
q = _recover_c24_measured_hold(arm, ArmLoopConfig())
assert q.shape == (7,)

order = arm.calls
assert order.index("clean_error") < order.index("motion_enable(True)")
assert order.index("motion_enable(True)") < order.index("set_mode(6)")
assert order.index("set_state(0)") < order.index("get_joint_states")
assert order.index("get_joint_states") < order.index("set_servo_angle")
print("OK", order)
PY
```

### 9.2 live error diagnosis（A2/A3）
至少验证：
```text
cached arm.error_code = 0, live get_err_warn_code = C24
  → command failure classification 必须选择 C24（而非 generic zero-error path）

cached arm.error_code = 24, live get_err_warn_code = C31
  → 必须选择 C31 collision fault

live getter 本身失败
  → 必须 fault，不允许 fallback 后继续 recovery
```
若为测试需要提取一个很小的 pure classification helper 可以做；但优先保持现有 inline control flow，**不要为两个 case 创建新的 error-policy abstraction**。
