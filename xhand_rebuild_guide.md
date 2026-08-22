# DexMani Real — XHand 重构修改指南

> 目标：面向个人研究与 Robot Learning 实验，将当前 XHand 控制链从“多层防御 + 多状态恢复”重构为“单一 owner + latest target + 简单 motion safety + 简单 SDK error semantics”的可维护实现。
>
> 核心参考：
>
> - [LeFranX / XHand](https://github.com/wengmister/LeFranX/tree/main/src/lerobot/robots)
> - [pi-r2-flow / XHandRobot](https://github.com/pi-r2-flow/pi-r2-flow/tree/main/deployment/mindex/robots)
> - [DexUMI / hand_sdk](https://github.com/real-stanford/DexUMI/tree/main/dexumi/hand_sdk)
> - [PF-DAG / xhand_robot.py](https://github.com/XiaohanLei/PF-DAG/tree/main/pf_dag/robots)
>
> 本方案的主体设计明确来自 **LeFranX + pi-r2-flow**：
>
> - **LeFranX**：SDK non-zero error 不等价于 joint observation 无效；已知 sensor/CRC warning 下可以继续使用有效 joint state。
> - **pi-r2-flow**：runtime read/send 不做 transaction retry，不建立 recovery state machine；失败只影响本次 operation，下一 control tick 自然继续。
>
> PF-DAG 只借鉴 measured-state max-delta 的 servo 思路；DexUMI 只借鉴 thin driver / error 不上升为系统 safety state 的理念。

---

## 1. 重构目标

当前 `dexmani_real` 的 XHand 路径同时承担了：

- XHand SDK 封装；
- RS485 CRC retry/backoff；
- sensor error 分类；
- grasp overcurrent 特判；
- send uncertainty / command-path resynchronization；
- read/send/board 三套 watchdog；
- persistent error → global `error_state`；
- 多层 joint limit；
- stale/synthetic feedback；
- command acknowledgement / provenance。

对于个人研究场景，这些机制显著增加：

- runtime latency jitter；
- 分支数量；
- 状态组合；
- 测试负担；
- debug 难度；
- 对 SDK error 的过度敏感。

重构后的目标是：

```text
Teleop / Policy / Replay
        │
        │ q_target[12]
        ▼
   hand_cmd_ring
    latest-wins
        │
        ▼
    hand_loop
        │
        ├── read_state(True)
        │       │
        │       ├── usable → publish fresh state
        │       └── unusable → publish stale previous state
        │
        ├── latest target
        │
        ├── measured-state max-delta
        │
        ├── hard mechanical clip
        │
        └── send_command once
                │
                ├── success / accepted soft status
                └── warning + False

下一 tick 继续
```

核心要求：

1. **XHand SDK 仍由独立 hand process 单一持有。**
2. **runtime SDK I/O 不 retry。**
3. **read error 与数据有效性分离。**
4. **send error 不建立 recovery state。**
5. **低层 safety 只保留 max-delta + hard mechanical limit。**
6. **持续异常通过 freshness / process heartbeat 暴露，而不是 SDK watchdog。**
7. **raw data collection 继续保持 fixed-grid publication；hard read failure 发布 stale frame，而不是直接跳 frame。**

---

## 2. 参考项目设计取舍

### 2.1 LeFranX：主要借鉴 Read Error Semantics

LeFranX 的关键设计：

```text
read_state() 返回 non-zero error
        │
        ├── 已知 sensor / CRC warning
        │       └── 继续使用 finger_state.position / torque
        │
        └── 其他 error
                └── warning + return None
```

其本质是：

> **SDK status != observation validity**

对于 XHand，force / temperature / CRC warning 不应自动废掉本轮 12-DoF joint state。

#### 本项目借鉴

- sensor-related SDK error 不影响 joint qpos/current；
- CRC 若 SDK 同时返回完整、finite 的 joint payload，可继续使用 joint state；
- runtime error 只影响当前 operation；
- hard joint limit 简单明确。

#### 不直接照搬

LeFranX 使用 `error_message` 字符串匹配 error 类型，容易受到 SDK 文案变化影响。

本项目使用 **error code set**。

---

### 2.2 pi-r2-flow：主要借鉴 Runtime Error Handling

pi-r2-flow 的 XHand runtime 基本是：

```text
READ:
    error != 0
    → warning
    → return None

SEND:
    error != 0
    → warning
    → return

NO:
    retry
    backoff
    watchdog
    resync
```

只有 sensor reset / calibration 等 startup 操作允许 bounded retry。

#### 本项目借鉴

> **startup 可以 retry；runtime 不 retry。**

尤其是：

```text
send CRC
→ 不 sleep
→ 不 resend
→ 本次 send False
→ 下一 tick直接发送 newest absolute target
```

这比在 30 Hz control loop 内 sleep 80–240 ms 更符合 robot learning servo 需求。

---

### 2.3 PF-DAG：只借鉴 measured-state max-delta

PF-DAG 的核心 servo：

```text
current state
    ↓
target - current
    ↓
max_delta
    ↓
command
```

这一点适合保留。

#### 不借鉴

PF-DAG 对所有 SDK error 统一：

```text
retry 3 次
sleep 100 ms
```

同时 joint/tactile 分开做 fresh read，RS485 traffic 较重，不适合作为 error handling 基线。

---

### 2.4 DexUMI：只借鉴 thin-driver 理念

DexUMI 的 error：

```text
read error → None
send error → print
```

其 driver 很薄。

但其 teleop hand path基本不依赖 XHand feedback，因此“read error 直接不更新 queue”并不适合需要连续 hand observation 的 DexMani data collection。

因此本项目：

- 借 thin driver；
- **不采用直接 drop state publication 的策略。**

---

## 3. 新的 SDK Error Contract

这是本次重构最重要的设计约束。

### 3.1 三种错误语义

XHand 层最终只保留三类错误：

| 类型 | 示例 | 行为 |
|---|---|---|
| Startup / Connection Error | open device失败、hand ID不存在、无初始state | `raise` |
| Programming / Contract Error | action shape错误、NaN、invalid config、未connect调用send | `raise` |
| Runtime SDK Error | CRC、sensor warning、unknown SDK code、偶发read/send失败 | `warn + None/False` |

原则：

> **SDK runtime error 不 raise。系统没初始化好或程序调用错了才 raise。**

---

## 4. Read Error Policy

### 4.1 最小 whitelist

基于当前 `dexmani_real` 已知 XHand code：

```python
READ_USABLE_CODES = {
    0,
    1_501_018,  # combined force unavailable
    1_501_019,  # distributed force unavailable
    1_501_020,  # temperature unavailable
    1_501_070,  # communication CRC
}
```

第一版不再做复杂 partial tactile salvage。

### 4.2 Read 判定逻辑

#### Case A：`code == 0`

要求：

- `state is not None`
- 12 joints 完整；
- joint id / order合法；
- qpos finite；
- current finite。

结果：

```text
joint_valid = True
tactile_valid = tactile parse success
```

#### Case B：`code in READ_USABLE_CODES - {0}`

例如：

- combined force unavailable；
- distributed force unavailable；
- temperature unavailable；
- CRC warning。

如果返回的 joint payload：

- 12 joints完整；
- qpos/current finite；

则：

```text
joint_valid = True
tactile_valid = False
```

并正常发布 fresh joint state。

即：

```text
qpos_stale = 0
state_valid = 1
```

这是整个方案最重要的 LeFranX-style 行为。

#### Case C：unknown non-zero code

```text
warning
return None
```

不 retry，不 raise。

#### Case D：`state is None`

```text
warning
return None
```

#### Case E：joint payload malformed / NaN

```text
warning
return None
```

不要让 parse error 冒到 `hand_loop`。

从 runtime 角度，这只是“本次 measurement 不可用”。

---

## 5. Send Error Policy

Send 采用 pi-r2-flow 风格：

```text
one SDK call only
```

建议：

```python
SEND_ACCEPTED_CODES = {
    0,
    1_501_018,  # sensor warning
    1_501_019,
    1_501_020,
    1_501_035,  # configured-current overrun / grasp contact
}
```

### 5.1 Send success

```text
code in SEND_ACCEPTED_CODES
→ True
→ update last_qpos_cmd
```

其中 `1501035` 只保留为：

```text
accepted soft status
```

不要再：

- 累计特殊 watchdog；
- 进入 grasp error state；
- 改变 command synchronization。

### 5.2 Send CRC

`1501070` 不放入 accepted code。

处理：

```text
warning
return False
```

必须满足：

```text
exactly one send_command()
NO retry
NO sleep
NO backoff
NO resend
NO resync
```

下一 tick发送 newest latest-wins target。

### 5.3 Unknown Send Error

```text
warning
return False
```

同样不 raise。

### 5.4 SDK Python Binding 抛异常

例如 pybind/native wrapper 自身抛异常：

```python
try:
    error = device.send_command(...)
except Exception:
    logger.warning(...)
    return False
```

runtime 不让偶发 SDK exception 进入 hand worker recovery state。

---

## 6. 哪些情况继续 `raise`

### 6.1 Startup

以下仍然 fail-fast：

- `open_serial/open_ethercat` 最终失败；
- `list_hands_id()` 不包含目标 hand；
- SDK object 初始化失败；
- startup 无法获取一次 usable 12-DoF joint state；
- command structure 初始化失败。

启动失败时：

```text
raise XHandError / RuntimeError / ConnectionError
```

第一阶段可以暂时保留 `XHandError`。

### 6.2 Programming Error

以下必须 `raise ValueError`：

```text
action.shape != (12,)
action contains NaN/Inf
joint limits invalid
config vector length != 12
lower > upper
```

这些不是设备通信错误，而是程序 bug。

### 6.3 Lifecycle Error

例如：

```text
send_action() before connect
_command is None
_control is None
```

应：

```python
raise RuntimeError(...)
```

不要 `return False`。

否则“程序调用错误”和“偶发 SDK error”会失去区分。

---

## 7. `xhand.py` 修改指南

目标：

> `xhand.py` 只负责 connect / parse / read / send / tactile calibration。

### 7.1 删除

删除 runtime：

```text
_RS485_CRC_ERROR
_send_with_crc_retry()
_read CRC retry loop
_crc_backoff()

self._sensor_status
self._last_tactile_valid
_update_sensor_status()

runtime grasp-overcurrent state handling
runtime retry counters
```

同时后续删除 config：

```text
rs485_crc_retry_count
rs485_read_crc_retry_count
rs485_crc_retry_backoff_s
```

### 7.2 暂时保留

第一阶段不要同时动：

```text
connect retry
post-open settle
identity read
initial state seed
tactile calibration
EtherCAT startup cleanup
```

原因：

这些不在 realtime hot path。

pi-r2-flow 也允许 startup / sensor reset做 bounded retry。

原则：

```text
startup retry: allowed
runtime retry: forbidden
```

---

## 8. 新 `get_state()` 参考结构

```python
READ_USABLE_CODES = {
    0,
    1_501_018,
    1_501_019,
    1_501_020,
    1_501_070,
}


def get_state(self) -> XHandState | None:
    if self._control is None or not self.connected_flag:
        raise RuntimeError("XHand is not connected")

    try:
        error, raw = self._control.read_state(
            self.cfg.device_id,
            True,
        )
    except Exception:
        logger.warning("XHand read_state raised", exc_info=True)
        return None

    code = _error_code(error)

    if raw is None:
        logger.warning(
            "XHand read returned no state: code=%s msg=%s",
            code,
            getattr(error, "error_message", ""),
        )
        return None

    if code not in READ_USABLE_CODES:
        logger.warning(
            "XHand read failed: code=%s msg=%s",
            code,
            getattr(error, "error_message", ""),
        )
        return None

    try:
        qpos, current, board_errors = self._parse_joints(raw)
    except (ValueError, TypeError, AttributeError):
        logger.warning("XHand joint payload invalid", exc_info=True)
        return None

    tactile_valid = (code == 0)

    if tactile_valid:
        try:
            tactile_force, tactile_sum = self._parse_tactile(raw)
        except (ValueError, TypeError, AttributeError):
            tactile_valid = False
            tactile_force = np.zeros(HAND_TACTILE_FORCE_SHAPE)
            tactile_sum = np.zeros(HAND_TACTILE_SUM_SHAPE)
    else:
        tactile_force = np.zeros(HAND_TACTILE_FORCE_SHAPE)
        tactile_sum = np.zeros(HAND_TACTILE_SUM_SHAPE)

    tactile_contact = (
        np.linalg.norm(tactile_sum, axis=1) > threshold
        if tactile_valid
        else np.zeros(HAND_CONTACT_SHAPE, dtype=bool)
    )

    return XHandState(
        qpos=qpos,
        current_ma=current,
        tactile_force=tactile_force,
        tactile_sum=tactile_sum,
        tactile_contact=tactile_contact,
        tactile_valid=tactile_valid,
        ...
    )
```

---

## 9. 新 `send_action()` 参考结构

```python
SEND_ACCEPTED_CODES = {
    0,
    1_501_018,
    1_501_019,
    1_501_020,
    1_501_035,
}


def send_action(self, action: np.ndarray) -> bool:
    if self._control is None or self._command is None or not self.connected_flag:
        raise RuntimeError("XHand command path is not initialized")

    target = np.asarray(action, dtype=np.float64)

    if target.shape != (12,):
        raise ValueError(
            f"XHand action must have shape (12,), got {target.shape}"
        )

    if not np.all(np.isfinite(target)):
        raise ValueError("XHand action contains NaN/Inf")

    target = np.clip(
        target,
        self.cfg.mechanical_qpos_min_rad,
        self.cfg.mechanical_qpos_max_rad,
    )

    for i, value in enumerate(target):
        self._command.finger_command[i].position = float(value)

    try:
        error = self._control.send_command(
            self.cfg.device_id,
            self._command,
        )
    except Exception:
        logger.warning("XHand send_command raised", exc_info=True)
        return False

    code = _error_code(error)

    if code not in SEND_ACCEPTED_CODES:
        logger.warning(
            "XHand send failed: code=%s msg=%s",
            code,
            getattr(error, "error_message", ""),
        )
        return False

    self.last_qpos_cmd = target.copy()
    return True
```

---

## 10. XHand State 建议

第一阶段保持现有 schema compatibility，但 driver object 可以先简化。

推荐最终：

```python
@dataclass
class XHandState:
    qpos: np.ndarray
    current_ma: np.ndarray

    tactile_force: np.ndarray
    tactile_sum: np.ndarray
    tactile_contact: np.ndarray
    tactile_valid: bool

    commboard_err: np.ndarray
    jointboard_err: np.ndarray
    tipboard_err: np.ndarray
```

建议逐步删除 driver-level：

```text
tactile_sum_valid
has_hardware_fault → control safety semantics
sensor status transition state
```

如果 recorder/policy 尚依赖 `tactile_sum_valid`，第一阶段兼容保留，后续统一成单一 `tactile_valid`。

---

## 11. `hand_process.py` 修改指南

这是第二阶段主要减法。

### 11.1 删除

删除：

```text
RetryCounter import

_send_error_counter
_read_error_counter
_error_state_counter

command_path_synchronized

_overcurrent_error_count_total
_read_error_count_total

send failure → global error_state
read failure → global error_state
board error → global error_state

send uncertainty recovery barrier
```

### 11.2 保留

保留：

```text
single SDK owner process
heartbeat
ready flag
estop
SafetyState software arm/disarm
latest-wins hand_cmd_ring
worker_validate_hand()
action_id
run_generation
fresh/stale state publication
last_cmd_qpos
RateManager
```

---

## 12. 新 hand loop 推荐顺序

建议从当前：

```text
SEND → READ
```

改成：

```text
READ → PUBLISH → BOUND → SEND
```

原因：

- motion delta基于 fresh measured state；
- hard read failure天然禁止发送新动作；
- 不需要 send failure后再做“resync”；
- 对数据采集 observation/action timestamp更容易解释。

---

## 13. 新 `hand_loop` 参考伪代码

```python
last_state = initial_state
last_applied_action_id = 0

while shared.is_running.value:
    shared.set_heartbeat("hand", time.monotonic())

    if shared.estop_request.value:
        break

    # ---------------------------------
    # 1. READ
    # ---------------------------------
    state = hand.get_state()

    if state is None:
        # Fixed-grid publication for recorder.
        publish_stale(
            shared,
            state=last_state,
            last_cmd_seq=last_applied_action_id,
            last_cmd_qpos=hand.last_qpos_cmd,
        )

        rate_mgr.wait()
        continue

    last_state = state

    # ---------------------------------
    # 2. PUBLISH FRESH STATE
    # ---------------------------------
    publish_fresh(
        shared,
        state=state,
        last_cmd_seq=last_applied_action_id,
        last_cmd_qpos=hand.last_qpos_cmd,
    )

    # ---------------------------------
    # 3. SOFTWARE DISARM
    # ---------------------------------
    if shared.safety_state.value not in (
        SafetyState.ARMED,
        SafetyState.RUNNING,
    ):
        rate_mgr.wait()
        continue

    # ---------------------------------
    # 4. LATEST TARGET
    # ---------------------------------
    command = read_latest_command()

    if command is None:
        rate_mgr.wait()
        continue

    target = np.asarray(
        command["qpos_cmd"][0],
        dtype=np.float64,
    )

    # ---------------------------------
    # 5. MOTION BOUND
    # ---------------------------------
    target = limit_hand_delta(
        target=target,
        measured=state.qpos,
        max_delta=config.max_delta_rad_per_tick,
    )

    # ---------------------------------
    # 6. ONE SDK SEND
    # ---------------------------------
    if hand.send_action(target):
        last_applied_action_id = int(
            command["action_id"][0]
        )

    rate_mgr.wait()
```

---

## 14. Hard Read Failure 的数据采集处理

这里明确不采用 DexUMI-style“直接不更新”。

对于 raw data fixed-grid：

```text
hard read failure
      │
      ▼
publish previous qpos/current
      │
      ├── qpos_stale = 1
      ├── state_valid = 0
      └── source_monotonic_ns 保持上一真实 measurement timestamp
```

这样：

- recorder仍有固定 grid sample；
- stale状态明确；
- 不伪造新的 source timestamp；
- downstream可选择丢弃 / hold / mask。

注意：

> stale publication 是 data-plane compatibility，不是 SDK recovery。

---

## 15. Soft Read Error 的 publication

例如：

```text
1501019
+
joint payload valid
```

应：

```text
qpos/current = fresh
qpos_stale = 0
state_valid = 1
source timestamp = current read time

tactile_valid = 0
```

不能因为 tactile error 把 hand joint observation标 stale。

---

## 16. Board Error 改为 Telemetry

当前：

```text
commboard_err
jointboard_err
tipboard_err
```

建议继续：

- parse；
- log transition；
- record。

但不再：

```text
board error
→ hand feedback unhealthy
→ stop publication
→ watchdog
→ global error_state
```

除非 vendor specification 后续明确某个具体 code 对 continued motion 有危险。

原则：

```text
board register != safety policy
```

---

## 17. Motion Safety 重构

SDK error handling简化后，motion safety继续保留，但收敛到两层。

### 17.1 Layer A：Measured-State Max Delta

统一所有 control source：

```python
delta = np.clip(
    target - measured,
    -max_delta,
    +max_delta,
)

bounded = measured + delta
```

建议 per-joint，不使用 12D global L2 norm。

例如：

```python
hand_max_delta_rad_per_tick = 0.2
```

该 bound 应覆盖：

- teleop；
- learned policy；
- replay。

### 17.2 Layer B：Hard Mechanical Joint Limit

最终 device boundary：

```python
bounded = np.clip(
    bounded,
    mechanical_lower,
    mechanical_upper,
)
```

不要重复：

```text
operational
rated
mechanical
SafetyGate
worker
driver
```

多个相同 static invariant。

---

## 18. `worker_validate_hand()` 保留

它负责 IPC correctness：

```text
dtype
shape
finite
run_generation
expiry
```

这是合理边界。

不要把它和 hardware SDK safety 混在一起。

---

## 19. `hand_health.py` 修改指南

当前 hand feedback validation 包含：

```text
connected
error_state
state_valid
send_healthy
read_healthy
freshness
qpos
```

最终建议缩成：

```text
connected
state_valid
source freshness
finite qpos
```

删除 command gate：

```text
send_healthy
read_healthy
error_state
```

否则 driver 虽然简化，上层仍会继续把偶发 SDK error放大成控制停机。

---

## 20. `command_publication.py` 修改指南

Hand feedback snapshot 应只回答：

> 当前是否存在一个 fresh、finite、可用于 safety / action computation 的 measured qpos？

检查：

```text
connected
state_valid
freshness
qpos shape
qpos finite
```

不检查历史 SDK send/read health。

---

## 21. `HAND_STATE_DTYPE` 分阶段处理

### Phase 1 / 2

保持兼容，不立刻 breaking change。

已不再使用的字段可以先写兼容值：

```text
send_healthy = 1
read_healthy = state_valid
read_error_count = 0
overcurrent_error_count = 0
```

### Phase 4

稳定后删除：

```text
send_healthy
read_healthy
read_error_count
overcurrent_error_count
error_state
```

最终建议保留：

```text
qpos[12]
current[12]

tactile_sum[5,3]
tactile_force[5,120,3]
tactile_valid
tactile_contact[5]

qpos_stale
state_valid

last_cmd_seq
last_cmd_qpos[12]

commboard_err[12]
jointboard_err[12]
tipboard_err[12]

source_monotonic_ns
publish_monotonic_ns
```

---

## 22. Hand Config 清理

稳定后删除：

```text
rs485_crc_retry_count
rs485_read_crc_retry_count
rs485_crc_retry_backoff_s

send_err_watchdog_count
error_state_watchdog_frames
```

保留真正硬件参数：

```text
comm_type
device_name
baudrate
device_id

rs485_post_open_settle_s

kp
ki
kd
tor_max_ma

mechanical_qpos_min_rad
mechanical_qpos_max_rad

loop_hz
hand_max_delta_rad_per_tick
```

---

## 23. Home 处理

第一阶段不必改。

长期建议 home 也走 normal servo path：

```text
current measured pose
→ interpolation / gradual target
→ normal hand_cmd_ring
→ max-delta
→ send
```

减少专用 ACK protocol。

但属于后续简化，不和 SDK error 重构绑在同一次修改中。

---

## 24. 分阶段实施计划

### Phase 1 — Driver Simplification

只改：

```text
dexmani_real/robot/xhand.py
```

目标：

- runtime single read；
- runtime single send；
- LeFranX-style read whitelist；
- pi-r2-flow-style send drop；
- 无 CRC retry；
- 无 backoff；
- 无 sensor transition state。

保持现有上层 contract。

#### 验证

- palm / fist循环；
- 5–10 min手部空载；
- 手指接触；
- 主动制造 USB/RS485轻微通信扰动；
- 统计 actual loop Hz / error log。

### Phase 2 — Worker Simplification

改：

```text
robot/hand_process.py
utils/hand_health.py
```

删除：

```text
RetryCounter
command_path_synchronized
send/read/board watchdog
SDK runtime → global error
```

保留：

```text
fresh/stale publication
latest-wins command
action_id
heartbeat
estop
```

### Phase 3 — Unified Motion Safety

统一：

```text
fresh measured qpos
→ per-joint max-delta
→ mechanical clip
```

覆盖：

```text
teleop
policy
replay
```

然后删除重复 hand static limit checks。

### Phase 4 — Schema / Config Cleanup

最后删除：

```text
send_healthy
read_healthy
error counters
watchdog config
CRC retry config
```

避免第一轮同时破坏：

- recorder；
- teleop；
- deployment；
- replay；
- processed dataset。

---

## 25. 单元测试矩阵

必须固定以下行为。

| Test | Expected |
|---|---|
| read code=0 + valid state | fresh qpos/current + tactile valid |
| read 1501018 + valid joints | fresh qpos/current + tactile invalid |
| read 1501019 + valid joints | fresh qpos/current + tactile invalid |
| read 1501020 + valid joints | fresh qpos/current + tactile invalid |
| read 1501070 + valid joints | fresh qpos/current + tactile invalid |
| read unknown non-zero | `None` |
| read `state=None` | `None` |
| read malformed joints | `None` |
| send code=0 | `True` |
| send 1501018/19/20 | `True` |
| send 1501035 | `True` |
| send 1501070 | `False`, exactly one SDK call |
| send unknown error | `False`, exactly one SDK call |
| send SDK raises | `False` |
| invalid action shape | `ValueError` |
| action NaN/Inf | `ValueError` |
| send before connect | `RuntimeError` |
| hard read failure | stale publication |
| soft read error | fresh qpos + invalid tactile |
| repeated runtime SDK error | never trigger SDK-specific global fault |

---

## 26. Hardware Acceptance Criteria

重构是否成功，不看“是否完全没有 SDK error”，而看：

1. hand loop是否稳定接近 nominal frequency；
2. CRC发生时不再出现人为 80/160/240 ms backoff；
3. sensor error时 joint qpos仍连续 fresh publication；
4. raw episode仍保持固定 control grid；
5. hard read failure明确标 stale；
6. grasp/contact不再触发复杂 error state；
7. send failure后下一 tick直接执行 latest command；
8. runtime没有 command resync / watchdog 分支；
9. error log足以定位问题，但不改变长期控制状态；
10. driver/worker主循环可以快速人工审阅。

---

## 27. 代码复杂度目标

### `xhand.py`

核心 runtime：

```text
connect
get_state
send_action
parse
disconnect
```

目标约：

```text
150–250 LOC
```

不含 vendor diagnostic / EtherCAT特殊初始化时可更短。

### `hand_process.py`

主 `while` loop：

```text
30–60 行
```

一次 control tick应该能一屏看清：

```text
read
publish
gate
latest command
bound
send
wait
```

如果后续再次出现：

```text
多个 RetryCounter
多个 health bool
多个 command recovery state
```

说明设计重新开始过度工程化。

---

## 28. 最终职责边界

```text
┌──────────────────────────────────────┐
│ Teleop / Policy / Replay             │
│                                      │
│ retarget / inference                 │
│ workspace / collision                │
│ high-level trajectory semantics      │
└───────────────────┬──────────────────┘
                    │ q_target
                    ▼
┌──────────────────────────────────────┐
│ hand_loop                            │
│                                      │
│ latest-wins IPC                      │
│ fresh / stale observation            │
│ measured-state max-delta             │
│ generation / expiry                  │
└───────────────────┬──────────────────┘
                    │ bounded q
                    ▼
┌──────────────────────────────────────┐
│ XHand driver                         │
│                                      │
│ connect                              │
│ one read_state                       │
│ one send_command                     │
│ parse payload                        │
│ hard mechanical clip                 │
│ warn/drop runtime SDK errors         │
└───────────────────┬──────────────────┘
                    ▼
                  XHand
```

---

## 29. 明确禁止重新引入的机制

除非有新的硬件实验证据，否则不要重新加入：

```text
runtime CRC retry
runtime fixed backoff
send response uncertainty state
command_path_synchronized
SDK error consecutive watchdog
board-register global watchdog
read/send health state machine
SDK error → global error_state
多级重复 hand joint limit
复杂 tactile partial-validity state machine
```

任何新机制都应先回答：

```text
1. 它解决了哪个已经复现的问题？
2. 不加它会导致什么明确错误？
3. 是否可以由 next control tick / freshness自然解决？
4. 是否已有 reference project采用？
5. 是否有单变量 hardware ablation证明收益？
```

如果回答不了，不应加入。

---

## 30. 推荐的代码注释 / Design Contract

建议直接写在新的 `xhand.py` module docstring 中：

```text
Runtime XHand I/O is intentionally single-shot.

Read follows LeFranX-style semantics: known sensor/CRC warnings may
still yield usable joint feedback when the returned 12-DoF joint
payload is complete and finite. Tactile data is invalidated on such
soft statuses.

Send follows pi-r2-flow-style semantics: one SDK send_command call per
servo tick. Runtime SDK failures are logged and dropped; no retry,
backoff, resend, watchdog, or command-path recovery state is used.
The next tick naturally sends the newest absolute joint target.

Persistent runtime safety is handled by observation freshness,
process heartbeat, software arm/disarm, and e-stop—not by SDK error
state machines.
```

---

## 31. 最终总结

本次重构应坚持：

```text
LeFranX 决定：
“SDK error 是否真的使 joint state 不可用？”

pi-r2-flow 决定：
“runtime error 后不要过度恢复。”

PF-DAG 提供：
“measured state → max-delta → send。”

DexMani 保留：
“single-owner hand process + latest-wins shared memory。”
```

最终得到：

> **一个 XHand runtime error 只影响一次 transaction，不应该自动成为长期 robot fault。一个 joint observation 是否可用，应由 joint payload本身决定，而不是仅由 SDK status code决定。**

这应作为后续 XHand 所有修改的基本设计准则。
