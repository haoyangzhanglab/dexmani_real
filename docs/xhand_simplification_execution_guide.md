# DexMani Real — XHand 简化重构执行指南

> 适用仓库：`haoyangzhanglab/dexmani_real`  
> 核心文件：`dexmani_real/robot/xhand.py`、`dexmani_real/robot/hand_process.py`  
> 目标：**显著降低 XHand 控制链路的认知复杂度和代码体量，同时不削弱已经验证有价值的实机可靠性机制。**
> 状态（2026-08-20）：Phase A–F 已完成并通过离线验证；Phase G（runtime schema）与 Phase H（实机语义验证）保持独立、尚未执行。本文保留实施前审计内容，历史描述以本状态和 §36 的实施记录为准。

---

## 0. 重构目标

重构前的 XHand 路径存在较明显的 accidental complexity：

```text
HandParams
    ↓
ResolvedRuntimeConfig.hand
    ↓
HandProcessConfig
    ↓
XHandConfig
    ↓
XHand
```

同一组配置被多次：

```text
声明
→ 转换
→ 校验
→ 再转换
→ 再校验
```

同时错误与健康状态也存在多套表达：

```text
bool return
exception
last_error_code
last_error_message
XHand.error_state

state_valid
qpos_stale
read_healthy
send_healthy
shared.error_state
```

本次重构最终要得到：

```text
ResolvedRuntimeConfig.hand
        │
        ▼
     HandParams
        │
   ┌────┴───────────────┐
   ▼                    ▼
hand_process          XHand
   │                    │
IPC / lifecycle      SDK / hardware
watchdog             fresh read
publication          CRC retry
                     parsing
                     hard limits
```

最终日常使用 API 应接近：

```python
hand = XHand(runtime.hand)

hand.connect()

state = hand.get_state()

hand.send_action(qpos)

hand.disconnect()
```

当前实现还提供：

```python
hand.calibrate_tactile()
state.has_hardware_fault
hand.is_connected
```

---

# 1. 本次重构的边界原则

## 1.1 一个事实只有一个配置来源

以下配置只允许由：

```python
runtime.hand
```

提供：

```text
comm_type
device_name
baudrate
device_id

RS485 retry parameters

joint limits
mechanical limits
home pose

kp / ki / kd
tor_max_ma

loop_hz
```

禁止重新创造：

```text
XHandConfig
HandProcessConfig
DriverConfig
ResolvedXHandConfig
HardwareHandConfig
```

## 1.2 三层职责固定

### Producer / `policy/safety.py`

负责：

```text
policy semantic safety
operational joint limits
workspace
arm-hand coupled preflight
home command preflight
```

### `hand_process.py`

只负责：

```text
process lifecycle
heartbeat / ready
SafetyState gate

command ring
dtype / finite
run_generation
TTL

send/read/board watchdog
SHM state publication
```

### `xhand.py`

只负责：

```text
vendor SDK
RS485 / EtherCAT
connect / disconnect

fresh state read
command send
CRC retry

mechanical hard limits

joint parsing
tactile parsing
tactile calibration
hardware errors
```

---

# 2. 明确禁止删除的机制

以下内容虽然增加了一定代码量，但属于**真实可靠性逻辑**，此次不得以“简化”为由删除。

- [ ] XHand 独立 worker process
- [ ] native SDK 只在 hand worker 中加载
- [ ] `read_state(..., True)` fresh hardware transaction
- [ ] RS485 CRC retry
- [ ] EtherCAT cleanup / INIT recovery
- [ ] mechanical hard limits
- [ ] joint ID validation
- [ ] duplicate/missing joint detection
- [ ] tactile partial-degradation handling
- [ ] tactile validity
- [ ] tactile calibration/load precheck
- [ ] invalid tactile frame publication
- [ ] command `run_generation`
- [ ] command TTL / `valid_until_monotonic_ns`
- [ ] latest-wins hand ring
- [ ] `last_cmd_seq` SDK acceptance acknowledgement
- [ ] send watchdog
- [ ] read watchdog
- [ ] persistent board-error watchdog
- [ ] heartbeat
- [ ] initial state publication before `ready`
- [ ] explicit homing workflow
- [ ] reject-whole joint limits，禁止 silent clipping
- [ ] connect 初始状态读取重试（`_seed_command_history` 的 3 次尝试，不得因重构 connect 而丢失）

> 范围说明：
> - 本清单针对 XHand driver/worker。三层 fault 阈值互补：board watchdog 5 帧、send watchdog 30 帧（`send_err_watchdog_count`）、loop 层 `hand_disconnect_timeout_s = 1.0s` 去抖。最后一项属于 teleop loop 层，不在本指南范围，重构期间不得顺手改动。
> - 「禁止 silent clipping」针对关节限位校验链（driver/worker 必须 reject-whole）。`teleop/loop.py:1443` 的 anti-clogging command box（仅投影命令、绝不裁剪 measured feedback，投影后仍走 reject-whole 校验）是 Producer 层的**有意设计**，不属于本禁令范围，不得删除。

---

# 3. 明确冻结的硬件语义

本轮代码重构中，以下行为**不得顺手修改**。

## 3.1 Tactile scale

保持：

```python
_TACTILE_SCALE = 0.1
```

当前数据集 metadata 已经明确记录：

```text
tactile_sdk_scale_factor = 0.1
tactile_unit = sdk_scaled_unknown_si
tactile_si_unit_verified = False
```

因此：

```text
代码重构
≠
触觉物理标定
```

## 3.2 Tactile sensor IDs

保持：

```python
TACTILE_SENSOR_IDS = tuple(range(0x11, 0x16))
```

即：

```text
17, 18, 19, 20, 21
```

不要改成：

```text
2, 5, 7, 9, 11
```

这些是 fingertip joint IDs，不是 `reset_sensor()` 使用的 sensor component IDs。

## 3.3 Overcurrent

`1501035` 对应：

```text
JOINT_ERROR_CURRENT_PROCTED
```

当前实现已将 1501035 作为**可恢复警告**处理（`hand_process.py` read 路径过流分支）：不计入 read 失败、不置 `state_valid`/`read_healthy` 失效、保留 last-known 电流、仅累计 `overcurrent_error_count` 供 episode 质量统计；电流兜底由固件 `tor_max` 负责。该策略与 CLAUDE.md「不可破坏的协议」一致。

仍缺乏证据的是：

```text
read_state 返回 1501035 时
raw state 是否依然完整有效
```

所以：

> **此次简化不改变 overcurrent 行为。** 不得把 1501035 升级为 fault，也不得改动其计数/降级路径。

后续单独做硬件 contract test。

## 3.4 Shutdown

保持当前：

```text
不主动回 home
不自动切 passive
保持最后 commanded state
关闭 device
```

本轮不讨论：

```text
hold vs passive
```

---

# 4. 推荐执行顺序

```text
Phase A — Config Collapse
        ↓
Phase B — Driver Surface
        ↓
Phase C — Error Model
        ↓
Phase D — State Model
        ↓
Phase E — Safety Deduplication
        ↓
Phase F — Worker Flattening
        ↓
Phase G — Runtime Schema Cleanup
        ↓
Phase H — Hardware Semantic Verification
```

其中：

> **Phase A–F 是当前 XHand 简化的主体。**

Phase G 和 H 不应该与前面的代码瘦身混在同一个大 PR 中。

顺序硬约束（2026-08-19 审计确认）：

```text
A 先于 E        §E3 示例使用 HandParams 字段名（_rad 后缀），XHandConfig 现名不一致
C 先于 D        error_state 的非 board 赋值点必须由 XHandError 异常路径替代（见 §13）
A–F 先于 G      schema 变更必须独立 PR
```

实施前已定决定（审计修订）：

```text
tactile_contact              保留，进入 XHandState 目标字段（见 §11）
error_state_watchdog_frames  提升到 HandParams（见 §6 A6）
```

实施状态（2026-08-20）：

| Phase | 状态 | 说明 |
|---|---|---|
| A | 已完成 | `HandParams` 是 worker 与 driver 的唯一配置来源。 |
| B | 已完成 | gain fallback、可配置 mode 和 driver home 配置已删除。 |
| C | 已完成 | SDK 失败统一为 `XHandError`。 |
| D | 已完成 | `XHandState` 在 SDK payload 边界验证，并派生 hardware fault。 |
| E | 已完成 | Producer 保留限位 preflight；Worker 保留 IPC 时效检查；Driver 保留机械硬限位。 |
| F | 已完成 | startup/loop 统一经 `_publish_feedback()` 序列化。 |
| G | 未开始 | 不修改 runtime/shared-memory/HDF5 数据合同。 |
| H | 未开始 | 需要受控实机执行 H0–H7。 |

---

# 5. Phase A — Config Collapse

## 目标

删除：

```text
HandProcessConfig
XHandConfig
```

建立：

```text
HandParams = XHand 唯一配置来源
```

## A1. 删除 `HandProcessConfig`

文件：

```text
dexmani_real/robot/hand_process.py
```

删除：

```python
@dataclass
class HandProcessConfig:
    ...
```

删除：

```python
HandProcessConfig.from_runtime(...)
```

注意（审计）：`HandProcessConfig.__post_init__`（limit nesting、shape、finite）与 `XHandConfig.__post_init__`（limit nesting、shape、comm_type）各有一条校验链。删除两个类时，必须确认 HandParams 层或 `hand_loop` 入口已有**等价校验**，安全网不得丢失。

## A2. 修改 `hand_loop`

从：

```python
def hand_loop(
    shared,
    config: HandProcessConfig | None = None,
) -> None:
    cfg = config or HandProcessConfig()
```

改为：

```python
def hand_loop(
    shared,
    config: HandParams,
) -> None:
    cfg = config
```

最好甚至统一命名为：

```python
def hand_loop(shared, config: HandParams) -> None:
```

不要继续使用 `cfg1 / cfg2` 等转换对象。

## A3. 修改 worker 创建代码

当前类似：

```python
WorkerSpec(
    "hand",
    hand_loop,
    (
        shared,
        HandProcessConfig.from_runtime(runtime),
    ),
    ready_name="hand",
)
```

改成：

```python
WorkerSpec(
    "hand",
    hand_loop,
    (shared, runtime.hand),
    ready_name="hand",
)
```

需要搜索并修改（2026-08-19 审计确认的完整调用点）：

```text
dexmani_real/deployment/lifecycle.py
dexmani_real/teleop/session.py
dexmani_real/teleop/keyboard_session.py
dexmani_real/robot/episode_replay.py
```

## A4. 删除 `startup_failure_is_fatal`

如果该参数只用于 fault-injection/offline test：

```python
startup_failure_is_fatal=False
```

建议彻底从 production API 移除。

production 行为统一：

```text
hand startup failure
→ shared.error_state=True
→ worker 不 ready
→ startup fail-closed
```

测试场景通过：

```text
FakeXHand
Mock XHand
dependency patch
```

完成。

不要为了测试在 production control path 中长期保留双行为分支。

审计确认：`startup_failure_is_fatal` 在 `hand_loop` 内有四处使用；唯一显式传参原位于 keyboard teleop 会话。删除后四处分支统一为 fail-closed 路径。

---

# 6. Phase A — 删除 `XHandConfig`

文件：

```text
dexmani_real/robot/xhand.py
```

删除：

```python
@dataclass
class XHandConfig:
    ...
```

修改：

```python
class XHand:
    def __init__(self, config: HandParams):
        self.cfg = config
```

## A5. 迁移 `XHandConfig` 独有参数

不能直接丢掉：

```python
tactile_contact_threshold = 1.0
raw_force_contact_threshold = 1.0
mode = 3
```

改为 driver constants：

```python
_POSITION_MODE = 3

_TACTILE_CONTACT_THRESHOLD = 1.0
_RAW_FORCE_CONTACT_THRESHOLD = 1.0
```

不要为了这三个固定参数再创建新 config。

审计注：`XHandConfig` 的标量 `kp=100` / `tor_max=300` 是死代码——`hand_process.py:273,276` 始终传入 `kp_per_joint` / `tor_max_per_joint` 数组，标量 fallback 从未被 `_make_command` 消费。删除无实机行为变化。

## A6. watchdog 阈值归位

`HandProcessConfig` 中硬编码的：

```python
error_state_watchdog_frames: int = 5
```

驱动 board/read 两个 RetryCounter，但 HandParams 无对应字段、`from_runtime` 也不传递。决定：

```text
error_state_watchdog_frames 提升到 HandParams
（与 send_err_watchdog_count 并列，统一来自 runtime.hand）
```

同时消除 `hand_process.py` 内部把 `send_err_watchdog_count` 改名为 `send_err_watchdog_frames` 的隐式转换：直接使用 HandParams 原名，或做一处显式映射。

## Phase A 验收

仓库搜索必须满足：

```bash
rg "HandProcessConfig"
```

结果：

```text
0 production references
```

以及：

```bash
rg "XHandConfig"
```

结果：

```text
0 production references
```

调用关系变成：

```text
runtime.hand
    ↓
hand_loop
    ↓
XHand
```

---

# 7. Phase B — 简化 Driver Surface

## B1. 删除 gain 双表示

删除：

```python
kp: int
kp_per_joint

ki_per_joint
kd_per_joint

tor_max
tor_max_per_joint
```

直接使用 `HandParams` 已经确定的语义：

```text
kp         = 12-vector
ki         = scalar
kd         = scalar
tor_max_ma = 12-vector
```

## B2. 简化 `_make_command`

目标：

```python
def _make_command(self, qpos: np.ndarray):
    command = xhc.HandCommand_t()

    for i in range(HAND_DOF):
        joint = command.finger_command[i]

        joint.id = i
        joint.position = float(qpos[i])

        joint.kp = int(self.cfg.kp[i])
        joint.ki = int(self.cfg.ki)
        joint.kd = int(self.cfg.kd)
        joint.tor_max = int(self.cfg.tor_max_ma[i])

        joint.mode = _POSITION_MODE

        joint.res0 = 0
        joint.res1 = 0
        joint.res2 = 0
        joint.res3 = 0

    return command
```

这里不要为了减少几行代码写：

```python
for name, values, fallback in ...
```

硬件 command builder 应优先显式。

审计注（B1）：标量 `kp`/`tor_max` 为死 fallback（见 §6 A5 审计注），B1 的实质收益是删除死代码与 Optional 间接层，不改变任何实机 gain；HandParams.kp 的 per-joint 语义（拇指 120 / 其余 100）当前已经生效。

---

# 8. Phase B — 从 Driver 移除 Home 概念

当前 driver 连接时知道 `home_qpos`。这属于错误职责边界。

目标：

```python
def connect(self):
    self._open_device()
    ...
    self._connected = True

    # 保留现有 3 次重试（_seed_command_history，0.02s 间隔）
    state = self._read_initial_state_with_retry()

    self._command = self._make_command(state.qpos)
    self._last_command = state.qpos.copy()
```

审计注（行为变化，必须在 commit 说明中显式声明）：

1. 现有 `_seed_command_history` 带 3 次重试。目标代码若写成单次 `get_state()` 会丢失该重试——上面已保留，落地时不得简化掉。
2. 当前 `_command` 用 `home_qpos` 初始化、`last_qpos_cmd` 用 live state clip 初始化，是「应该去哪 / 实际在哪」的有意区分。统一为 live state 后，连接时手不在 home 的首帧命令行为会变化，需实机确认（H3 覆盖）。
3. `_last_command` 对应现有 `last_qpos_cmd`，落地时明确是重命名还是保留原名。
4. `connect() -> None` 是 Phase C 之后的形态；Phase B 落地时 connect 仍返回 bool，只改初始化来源。

driver 以后只知道：

```text
current measured position
absolute target position
```

而不知道：

```text
home
teleop home
policy initial pose
```

## B3. 注意行为保持

当前初始 `last_qpos_cmd` 会经过 operational clipping。

因此建议：

### 第一轮

仅从 `XHandConfig` 中删除 `home_qpos`，但暂时保持原有：

```text
last_qpos_cmd
```

语义。

### 后续独立 commit

再讨论：

```text
last accepted command 尚不存在时
last_qpos_cmd 应等于 measured qpos
还是 clipped measured qpos
```

不要和 config collapse 混在一起。

---

# 9. Phase C — 统一 Error Model

## 目标

删除多套 error channels：

```text
bool
exception
last_error_code
last_error_message
error_state
```

统一成：

```text
successful return
or
XHandError
```

## C1. 定义单一异常

```python
class XHandError(RuntimeError):
    def __init__(
        self,
        operation: str,
        code: int,
        message: str,
    ):
        self.operation = operation
        self.code = int(code)
        self.message = str(message)

        super().__init__(
            f"XHand {operation} failed: "
            f"code={self.code} msg={self.message}"
        )
```

## C2. `connect`

从：

```python
if not hand.connect():
    ...
```

改成：

```python
try:
    hand.connect()
except XHandError:
    ...
```

正常：

```python
connect() -> None
```

审计注：connect 现有 4 个 return 点（:281 已连接短路 / :285 open 失败 / :304 成功 / :309 异常 cleanup），异常化时逐一映射，不得遗漏短路路径语义。

## C3. `send_action`

从：

```python
sent = hand.send_action(qpos)

if sent:
    ...
else:
    ...
```

改成：

```python
try:
    hand.send_action(qpos)
except XHandError:
    send_failures.inc()
else:
    send_failures.reset()
    last_applied_action_id = action_id
```

审计注：send_action 现有 3 条 `return False` 路径（限位/参数 validation 失败、command path 未初始化、SDK 发送失败）。异常化时区分：参数非法 → `ValueError`（见 §E3 `_validate_action`）；SDK/通信失败 → `XHandError`。

## C4. `get_state`

统一：

```python
try:
    state = hand.get_state()
except XHandError as exc:
    ...
```

删除独立：

```text
XHandReadError
```

除非测试证明 read error 作为独立 exception type 有真实消费者。

当前没有必要为了 operation 分类创建异常继承树。

审计注：**真实消费者已确认存在**——`hand_process.py:510` 用 `isinstance(exc, XHandReadError)` 提取 `.code` 做过流判断。合并为 `XHandError` 时必须保持 `code` 可达，过流分支语义不变。现有 `XHandReadError` 含 code/message/connected 三字段，与目标结构差异小（缺 operation 字段；connected 字段不在目标规格内，合并前确认无消费者依赖它）。

## C5. 删除

```text
last_error_code
last_error_message
_set_error()
```

operation/code/message 直接跟随 exception。

---

# 10. Phase C — 保留 Connect Cleanup

异常化时必须保持：

```python
def connect(self):
    try:
        ...
    except Exception:
        self.disconnect()
        raise
```

禁止产生：

```text
open_device success
→ identity/read failure
→ exception
→ device handle 泄漏
```

---

# 11. Phase D — `XHandSample` → `XHandState`

重命名：

```text
XHandSample
→
XHandState
```

目标：

```python
@dataclass
class XHandState:
    qpos: np.ndarray
    current_ma: np.ndarray

    tactile_force: np.ndarray
    tactile_sum: np.ndarray
    tactile_contact: np.ndarray

    tactile_valid: bool
    tactile_sum_valid: bool

    commboard_err: np.ndarray
    jointboard_err: np.ndarray
    tipboard_err: np.ndarray
```

第一轮不要增加没有真实消费者的新 field。

审计决定：**`tactile_contact` 保留**。它有 episode 级持久化与约 28 处引用（`HAND_CONTACT_SHAPE`、`episode_schema.py`、teleop samples、recorder 链等 10 个文件），且是 `HAND_STATE_DTYPE` 字段；移除等同数据合同变更，超出本次重构范围。若未来要删，单独立项并同步 `docs/dataset/` 合同。

重命名注意：`current` → `current_ma` 仅限 driver 内 dataclass；SHM `HAND_STATE_DTYPE` 的 `current` 字段名不动（Phase G 范围之外）。同步点：`hand_process.py`、`_parse_sample`/`get_state` 返回注解。

---

# 12. Phase D — 迁移 Validation，再删除 `__post_init__`

这一项必须严格按顺序操作。

当前 `_parse_joints()` 检查：

```text
joint ID range
duplicate IDs
12 joints reported
```

但是没有独立完成：

```text
qpos finite
current finite
```

因此：

## D1. 先把 invariant 放进 parser

```python
def _parse_joints(...):
    ...
    if len(seen) != HAND_DOF:
        raise XHandError(
            "read",
            -1,
            f"{len(seen)}/{HAND_DOF} joints reported",
        )

    if not np.all(np.isfinite(qpos)):
        raise XHandError(
            "read",
            -1,
            "non-finite joint position feedback",
        )

    if not np.all(np.isfinite(current)):
        raise XHandError(
            "read",
            -1,
            "non-finite joint current feedback",
        )

    return qpos, current, errors
```

依赖：C1 先行——parser 抛出的 `XHandError` 由 Phase C 定义（当前 `_parse_joints` 抛 `RuntimeError`）。

## D2. tactile parser 保持验证

继续验证：

```text
sensor count == 5
raw points == 120
force fields finite
```

## D3. 最后删除

```text
_ARRAYS
__post_init__()
object.__setattr__()
setflags(write=False)
```

原则：

> Validation 从来没有删除，只是统一迁移到 `SDK payload → canonical XHandState` 的唯一边界。

---

# 13. Phase D — 删除 mutable `XHand.error_state`

当前该状态本质来自：

```text
commboard_err
jointboard_err
tipboard_err
```

审计修正：上述表述只对一处成立。`self.error_state` 共 8 处赋值，仅 `get_state` 内 1 处（:563–566）由三块 board error 数组派生；其余 5 处——connect 异常（:306）、无设备（:329）、重试耗尽（:371）、send 未初始化（:595）、`_set_error`（:846）——与 board error 无关，必须由 Phase C 的 `XHandError` 异常路径替代。**这是 C 必须先于 D 的原因。**

目标：

```python
@dataclass
class XHandState:
    ...

    @property
    def has_hardware_fault(self) -> bool:
        return bool(
            np.any(self.commboard_err)
            or np.any(self.jointboard_err)
            or np.any(self.tipboard_err)
        )
```

worker：

```python
if state.has_hardware_fault:
    board_faults.inc()
else:
    board_faults.reset()
```

删除：

```python
self.error_state
```

同步迁移点（审计）：`hand_process.py` 直接读取 `hand.error_state` 的位置为 :335（启动校验）、:355（frame0 发布）、:490（loop 内复制到局部变量）。:558 的 `error_state = False` 是「瞬态读失败不伪造 board fault」的有意设计，由独立 read watchdog 负责升级——迁移后该语义必须保留（read_failures 计数，不置 board fault）。

---

# 14. Phase D — `tactile_calibrated` 改 Derived Property

目标：

```python
@property
def tactile_calibrated(self) -> bool:
    return (
        self._tactile_bias_sum is not None
        and self._tactile_bias_raw is not None
    )
```

删除：

```python
self.tactile_calibrated = False
self.tactile_calibrated = True
```

这样 calibration truth 只有一个来源：

```text
bias 是否存在
```

---

# 15. Phase E — Safety Deduplication

最终责任：

```text
Producer
    operational + mechanical preflight

Worker
    IPC + generation + TTL

Driver
    mechanical hard boundary
```

## E1. 简化 `worker_validate_hand`

最终：

```python
def worker_validate_hand(
    command: np.ndarray,
    *,
    expected_run_generation: int | None = None,
    now_monotonic_ns: int | None = None,
) -> bool:
    well_formed = (
        isinstance(command, np.ndarray)
        and command.shape == (1,)
        and command.dtype == HAND_COMMAND_DTYPE
        and np.all(
            np.isfinite(command["qpos_cmd"][0])
        )
    )

    return bool(
        well_formed
        and _worker_command_is_current(
            command,
            expected_run_generation=expected_run_generation,
            now_monotonic_ns=now_monotonic_ns,
        )
    )
```

## E2. 修改 `hand_process.py`

从：

```python
worker_validate_hand(
    data,
    qpos_lower_rad=...,
    qpos_upper_rad=...,
    mechanical_lower_rad=...,
    mechanical_upper_rad=...,
    expected_run_generation=...,
    now_monotonic_ns=...,
)
```

改成：

```python
worker_validate_hand(
    data,
    expected_run_generation=int(
        shared.run_generation.value
    ),
    now_monotonic_ns=time.monotonic_ns(),
)
```

## E3. Driver 仅保留机械硬限位

目标：

```python
def _validate_action(self, qpos: np.ndarray) -> None:
    if qpos.shape != (HAND_DOF,):
        raise ValueError(...)

    if not np.all(np.isfinite(qpos)):
        raise ValueError(...)

    lower = np.asarray(
        self.cfg.mechanical_qpos_min_rad
    )
    upper = np.asarray(
        self.cfg.mechanical_qpos_max_rad
    )

    if np.any(qpos < lower) or np.any(qpos > upper):
        raise ValueError(
            "XHand mechanical joint limit violation"
        )
```

删除 driver 中：

```text
operational qpos_min/qpos_max validation
```

但保留：

```text
mechanical_qpos_min/max
```

字段名注意：上面示例使用 HandParams 命名（`mechanical_qpos_min_rad`）；当前 `XHandConfig` 字段为 `mechanical_qpos_min`（无 `_rad` 后缀，xhand.py:130–138）。**E 必须在 A 之后执行**，否则示例代码无法编译。

安全依据（2026-08-19 审计）：`send_action` 全仓库唯一调用方是 `hand_process.py:460`；`hand_cmd_ring` 仅两个写点（`policy/safety.py:537` 与 :1132），均在 `validate_hand_command_bounds`（operational + mechanical reject-whole + rated envelope 一致性断言）之后。命令必经 Producer preflight，删除 Worker/Driver 的冗余限位检查**不产生保护缺口**。

另注：teleop/loop.py:1443 的 anti-clogging command box 是 Producer 层有意 clip（见 §2 范围说明），E 不涉及。`HandParams.qpos_max_rad` 当前默认等于 mechanical 上限（rated envelope），operational 上界实际约束力弱，属既有事实，本轮不改。

---

# 16. Phase F — Flatten `hand_process.py`

目标不是“多写 helper”，而是让主流程肉眼可见。

实施状态（2026-08-20）：`hand_process.py` 主流程可从上往下直读（connect → calibrate tactile → initial read → publish → ready → loop → finally disconnect），三个 `RetryCounter`（send/board/read）齐备。startup 与 loop 均通过 `_publish_feedback()` 序列化，且没有合并 startup acceptance policy；`HandProcessConfig` 的字段映射已随 Phase A 删除。

最终应该能直接读成：

```text
connect
↓
calibrate tactile
↓
initial read
↓
publish
↓
ready
↓
loop:
    heartbeat
    command
    send
    read
    publish
    watchdog
↓
disconnect
```

## F1. 推荐主循环结构

```python
def hand_loop(shared, config: HandParams) -> None:
    from dexmani_real.robot.xhand import (
        XHand,
        XHandError,
    )

    hand = XHand(config)

    send_failures = RetryCounter(...)
    read_failures = RetryCounter(...)
    board_faults = RetryCounter(...)

    try:
        hand.connect()

        try:
            hand.calibrate_tactile()
        except XHandError:
            logger.warning(
                "hand tactile calibration failed",
                exc_info=True,
            )

        state = hand.get_state()

        _publish_feedback(
            shared,
            hand,
            state,
            ...
        )

        shared.set_heartbeat(
            "hand",
            time.monotonic(),
        )
        shared.set_ready("hand")

        rate = RateManager(
            config.loop_hz,
            label="hand",
        )

        while shared.is_running.value:
            shared.set_heartbeat(
                "hand",
                time.monotonic(),
            )

            if shared.estop_request.value:
                break

            if _can_send(shared):
                command = _read_latest_command(
                    shared
                )

                if command is not None:
                    try:
                        hand.send_action(
                            command["qpos_cmd"][0]
                        )
                    except XHandError:
                        send_failures.inc()
                    else:
                        send_failures.reset()
                        last_action_id = int(
                            command["action_id"][0]
                        )

            try:
                state = hand.get_state()
            except XHandError:
                read_failures.inc()
                _publish_invalid_feedback(...)
            else:
                read_failures.reset()

                if state.has_hardware_fault:
                    board_faults.inc()
                else:
                    board_faults.reset()

                _publish_feedback(...)

            if (
                send_failures.triggered
                or read_failures.triggered
                or board_faults.triggered
            ):
                shared.error_state.value = True

            rate.wait()

    finally:
        hand.disconnect()
```

这是结构示意，不要求逐字采用。

---

# 17. Phase F — 合并重复 Publication

重构前 startup publication 与 loop publication 都在逐字段填：

```text
qpos
current
tactile
board errors
timestamps
health
last command
...
```

现已统一。

建议：

```python
_publish_feedback(...)
```

startup：

```python
state = hand.get_state()

_validate_startup_state(state)

_publish_feedback(...)

shared.set_ready("hand")
```

loop：

```python
state = hand.get_state()

_publish_feedback(...)
```

注意：

> **只合并 serialization，不合并 startup acceptance policy。**

当前 `_publish_feedback()` 仅统一 serialization；startup 的有效反馈要求和 loop 的动态健康计算仍在各自控制路径中明确决定。

---

# 18. Phase F — Helper 上限

最终 `hand_process.py` 最多建议：

```python
_read_latest_command(...)
_publish_feedback(...)
_publish_invalid_feedback(...)
_log_board_fault_changes(...)
_can_send(...)
hand_loop(...)
```

如果出现：

```text
_handle_read
_handle_send
_process_read_failure
_update_send_health
_update_read_health
_build_health
_validate_health
```

说明正在重新制造抽象层。

审计注：`_read_latest_command` 目前是 `hand_loop` 内闭包（依赖 `nonlocal last_consumed_ring_sequence`）；提取为模块级 helper 时须显式传 shared/config/ring 游标，不得依赖闭包捕获。

---

# 19. Phase G — Runtime Schema Cleanup

这个阶段必须单独进行。

第一轮重构：

```text
不要改 HAND_STATE_DTYPE
```

等 A–F 稳定后再处理。

## G1. `read_healthy`

优先删除候选。

因为一次 read failure 时：

```text
state_valid = False
read_healthy = False
```

二者高度重合。

删除时同步修改：

```text
utils/hand_health.py
runtime/supervisor.py
teleop
keyboard teleop
replay
policy safety
```

审计注：`read_healthy` 与 `state_valid` **非等价**——`read_healthy = not read_failed and not watchdog_triggered`，`state_valid = connected and not read_failed`；watchdog 触发但当前帧成功时二者分叉。删除前须把 watchdog 语义折叠进 `state_valid` 或 `validate_hand_feedback`。实际消费者 13 处（含 `deployment/worker.py` 以 `required_true_fields` 过滤历史帧，需替换为等价条件）。

## G2. `send_healthy`

也是强删除候选。

send watchdog 到 threshold 时：

```text
send_healthy=False
```

同时 worker 已经：

```python
shared.error_state.value = True
```

因此 health bit 很可能只是重复 sticky fault。

审计注：`send_healthy` 消费者 10 处；`send_healthy=False ⇒ shared.error_state=True`（sticky），反向不成立。删除时逐一核对消费者是否已依赖 `error_state` 覆盖。

---

# 20. `qpos_stale` 的正确迁移方法

**不要直接删 HDF5 `hand_qpos_stale`。**

Runtime 中：

```text
qpos_stale
```

可以考虑由：

```text
not state_valid
```

替代。

但是当前受支持的 schema v17/v18 保存的是：

```text
hand_qpos_stale
```

正确迁移：

```text
HAND_STATE_DTYPE:
    删除 qpos_stale

Recorder:
    hand_qpos_stale = not state_valid

HDF5 schema:
    暂时继续保存 hand_qpos_stale
```

这样 runtime contract 可以变干净，而 dataset 兼容性不受影响。

审计补充：消费侧同步点为 `teleop/episode_samples.py:474`（当前从 dtype 读 `qpos_stale`，改为 `not state_valid`）与 `data_processing/cleaning.py:278`（读 HDF5 `hand_qpos_stale`，保持不变）。

---

# 21. 明确保留 Error Counters

保留：

```text
read_error_count
overcurrent_error_count
```

不要移到新 metrics subsystem。

原因：recorder 当前使用它们计算 episode 期间：

```text
hand_read_error_count
hand_overcurrent_count
```

为了“状态纯洁”新建：

```text
metrics ring
diagnostics channel
health process
```

只会重新制造过度工程化。

审计补充：两个 counter 不仅用于日志——`episode_recorder.py:1022–1023` 将其作为 quality metrics 写入 HDF5 meta attrs；`io_process.py:439–451` 从 `hand_state_ring` 差值计算 episode 级 `hand_read_error_count` / `hand_overcurrent_count`。字段必须留在 `HAND_STATE_DTYPE`。

---

# 22. `timestamp` 的处理

当前如果同时存在：

```text
source_monotonic_ns
publish_monotonic_ns
timestamp
```

那么：

```text
timestamp
```

可能只是：

```python
source_monotonic_ns / 1e9
```

最终可以考虑删除。

审计结论（2026-08-19，验证已完成）：`shared_storage.read_hand_state_dict()` 不含该字段；全仓库对 `"timestamp"` 的引用均属 `ARM_STATE_DTYPE` 或 episode grid，`hand_state['timestamp']` **零 raw consumer**。可在 Commit 7 一并删除；仍单独 commit，不与 A–F 混合。

---

# 23. 最终 `xhand.py` 应该长什么样

推荐结构：

```python
"""XHand hardware driver."""

# imports

# constants
_POSITION_MODE = 3
_TACTILE_SCALE = 0.1
_TACTILE_SENSOR_IDS = tuple(range(0x11, 0x16))
_TACTILE_CONTACT_THRESHOLD = 1.0
_RAW_FORCE_CONTACT_THRESHOLD = 1.0


class XHandError(RuntimeError):
    ...


@dataclass
class XHandState:
    qpos: np.ndarray
    current_ma: np.ndarray

    tactile_force: np.ndarray
    tactile_sum: np.ndarray

    tactile_valid: bool
    tactile_sum_valid: bool

    commboard_err: np.ndarray
    jointboard_err: np.ndarray
    tipboard_err: np.ndarray

    @property
    def has_hardware_fault(self) -> bool:
        ...


class XHand:
    def __init__(self, config: HandParams):
        ...

    @property
    def is_connected(self) -> bool:
        ...

    @property
    def tactile_calibrated(self) -> bool:
        ...

    def connect(self) -> None:
        ...

    def disconnect(self) -> None:
        ...

    def get_state(self) -> XHandState:
        ...

    def send_action(
        self,
        qpos: np.ndarray,
    ) -> None:
        ...

    def calibrate_tactile(self) -> bool:
        ...

    def _open_device(self):
        ...

    def _make_command(self, qpos):
        ...

    def _read_raw_state(self):
        ...

    def _parse_state(self, raw):
        ...

    def _parse_joints(self, raw):
        ...

    def _parse_tactile(self, raw):
        ...
```

---

# 24. `xhand.py` 最终禁止出现

```text
XHandConfig

home_qpos workflow

generic mode configuration

scalar/per-joint fallback abstraction

last_error_code
last_error_message
_set_error

mutable error_state mirror

多层 exception hierarchy

SHM logic

policy SafetyState logic
```

---

# 25. `hand_process.py` 最终职责

只留下：

```text
process startup
XHand lifecycle

heartbeat
ready

read latest command
worker IPC validation
SafetyState command gate

send watchdog
read watchdog
board watchdog

state publication
tactile publication

disconnect
```

不再包含：

```text
hardware config dataclass
transport validation
joint-limit config validation
PID construction
SDK parsing
tactile parsing
```

---

# 26. 每个 Phase 的测试要求

## Phase A：Config

必须验证：

- [ ] runtime YAML override 仍然生效
- [ ] serial device name 生效
- [ ] baudrate 生效
- [ ] device ID 生效
- [ ] kp 生效
- [ ] tor_max 生效
- [ ] mechanical limits 未变化
- [ ] loop_hz 未变化
- [ ] tactile thresholds 保持 `1.0`
- [ ] mode 保持 `3`

## Phase B/C：Driver

### Connection

- [ ] serial connect
- [ ] EtherCAT connect
- [ ] no device
- [ ] wrong device ID
- [ ] repeated connect
- [ ] repeated disconnect
- [ ] partial-connect failure 会 cleanup

### Command

- [ ] shape `(12,)`
- [ ] wrong shape rejected
- [ ] NaN rejected
- [ ] Inf rejected
- [ ] lower mechanical violation rejected
- [ ] upper mechanical violation rejected
- [ ] success 后才更新 last command
- [ ] failure 不更新 last command

### Command construction

验证每 joint：

- [ ] `id`
- [ ] `position`
- [ ] `kp`
- [ ] `ki`
- [ ] `kd`
- [ ] `tor_max`
- [ ] `mode == 3`

---

# 27. CRC Tests

### Send

```text
CRC error
→ retry
→ success
```

以及：

```text
CRC error
→ retry exhausted
→ XHandError
```

### Read

同样覆盖：

```text
CRC retry success
CRC retry exhaustion
```

---

# 28. Parser Tests

## Joint

- [ ] 12 valid joints
- [ ] duplicate joint ID
- [ ] missing joint
- [ ] invalid joint ID
- [ ] NaN qpos
- [ ] Inf qpos
- [ ] NaN current
- [ ] Inf current

## Tactile

- [ ] exactly 5 sensors
- [ ] wrong sensor count
- [ ] exactly 120 points
- [ ] wrong raw point count
- [ ] malformed force
- [ ] combined force valid
- [ ] raw invalid / combined valid
- [ ] raw + combined valid
- [ ] bias subtraction
- [ ] startup load rejection
- [ ] load during bias capture rejection
- [ ] reset IDs exactly `17..21`

---

# 29. Worker Tests

必须覆盖：

- [ ] initial state 在 ready 前发布
- [ ] heartbeat 在 ready 前至少写一次
- [ ] DISARMED 不发送 command
- [ ] ARMED/RUNNING 接受 command
- [ ] FAULT 不发送
- [ ] ESTOP 退出
- [ ] latest ring sequence 不 replay
- [ ] wrong generation 丢弃
- [ ] expired command 丢弃
- [ ] NaN command 丢弃
- [ ] send success 更新 `last_cmd_seq`
- [ ] send failure 不更新
- [ ] N send failures latch fault
- [ ] N read failures latch fault
- [ ] transient board error 不立即 fault
- [ ] persistent board error latch fault
- [ ] failed read `state_valid=False`
- [ ] failed read 不更新 source timestamp
- [ ] failed tactile publication `fresh=False`

---

# 30. 实机回归顺序

不要一上来跑完整 teleop。

## H0 — Connect only

验证：

```text
connect 不产生 motion
```

## H1 — Read only

读取：

```text
qpos
current
board error
tactile
```

## H2 — Explicit home

通过现有 home workflow：

```text
home command
```

验证动作与重构前一致。

注：hand home workflow 位于 `teleop/safety.py` 的 `publish_hand_home_and_wait_applied`（hand-home-first，再 arm home）；`robot/homing.py` 仅负责 arm。

## H3 — Basic position command

依次：

```text
open
partial close
open
```

确认：

```text
joint order
sign
magnitude
```

均无变化。

## H4 — Mechanical rejection

发送越界 target。

要求：

```text
driver reject
hardware 不动作
```

## H5 — Tactile calibration

空载：

```text
calibration success
```

带接触：

```text
calibration refuse
```

## H6 — Communication failure

运行时断 USB / RS485。

要求：

```text
state_valid=False
read watchdog accumulates
最终 global fault
旧 command 不被反复 replay
```

## H7 — Shutdown

确认：

```text
没有自动 home
没有意外 passive
close_device 执行
```

---

# 31. 每个 Commit 的建议范围

Phase A–F 已按以下边界完成。这里保留建议拆分，便于审查或回溯；Phase G 仍必须保持独立。

## Commit 1 — `refactor: collapse xhand configuration`

修改：

```text
hand_process.py
xhand.py
deployment/lifecycle.py
examples/*
```

只做：

```text
delete HandProcessConfig
delete XHandConfig
runtime.hand direct pass
constants migration
```

禁止：

```text
SHM schema change
error model change
tactile semantic change
```

## Commit 2 — `refactor: simplify xhand command construction`

只做：

```text
delete gain fallback abstraction
position mode constant
remove home from driver config
```

## Commit 3 — `refactor: unify xhand errors`

只做：

```text
XHandError
connect/send/read exception semantics
delete last_error_*
```

## Commit 4 — `refactor: simplify xhand state model`

只做：

```text
XHandSample → XHandState
move finite validation into parser
delete __post_init__
delete mutable error_state
derived tactile_calibrated
```

## Commit 5 — `refactor: deduplicate hand command validation`

只做：

```text
simplify worker_validate_hand
keep driver mechanical bounds
```

## Commit 6 — `refactor: flatten xhand worker loop`

只做：

```text
shared serializer
simplify main loop
reduce helpers
```

## Commit 7 — `refactor: simplify hand health schema`

独立处理：

```text
read_healthy
send_healthy
qpos_stale runtime representation
duplicate timestamp
```

不要与前六个 commit 混合。

---

# 32. 代码 Review Checklist

每个 PR 都逐项审核。

## Config

- [ ] 是否出现新的 config wrapper？
- [ ] 是否重新复制 `HandParams`？
- [ ] 是否出现第二套 limits？
- [ ] 是否出现第二套 gains？

只要答案是“是”，基本说明重构方向开始倒退。

## Driver

- [ ] public API 是否仍然只有少数几个方法？
- [ ] hardware errors 是否只有一个主 error channel？
- [ ] parser 是否完整检查 SDK payload？
- [ ] driver 是否仍做 mechanical hard-limit validation？
- [ ] driver 是否误加入 policy/home/SHM 逻辑？

## Worker

- [ ] 主 loop 能否从上往下阅读？
- [ ] heartbeat 是否明显？
- [ ] command path 是否明显？
- [ ] feedback path 是否明显？
- [ ] watchdog 是否明显？
- [ ] 是否出现大量仅调用一次的 helper？

## Safety

- [ ] operational limits 是否仍在 producer？
- [ ] coupled arm-hand command 是否仍先 preflight hand？
- [ ] mechanical limits 是否仍在 driver？
- [ ] stale generation 是否仍拒绝？
- [ ] expired command 是否仍拒绝？
- [ ] 是否存在任何 silent clipping？

---

# 33. 最终验收标准

重构完成后：

## 配置层

只剩：

```text
HandParams
```

作为 XHand configuration source。

## Driver 层

public surface：

```python
XHand(...)
.is_connected

.connect()
.disconnect()

.get_state()
.send_action()

.calibrate_tactile()
```

## State 层

不再有：

```text
XHand.error_state
last_error_code
last_error_message
```

hardware fault 从：

```python
state.has_hardware_fault
```

直接获取。

## Worker 层

第一次打开 `hand_process.py`，应该很快看到：

```text
command
→ send
→ read
→ publish
→ watchdog
```

而不是：

```text
config conversion
→ validation
→ abstraction
→ status conversion
→ health conversion
```

---

# 34. 最终 KEEP / DELETE / LATER

| 项目 | 决策 |
|---|---|
| `HandParams` | **KEEP** |
| `HandProcessConfig` | **DELETE** |
| `XHandConfig` | **DELETE** |
| process isolation | **KEEP** |
| fresh read | **KEEP** |
| CRC retry | **KEEP** |
| EtherCAT cleanup | **KEEP** |
| `XHandReadError` | **MERGE** |
| `XHandError` | **KEEP one** |
| `last_error_code` | **DELETE** |
| `last_error_message` | **DELETE** |
| `XHand.error_state` | **DELETE** |
| `XHandSample` | **RENAME** |
| `XHandSample.__post_init__` | **DELETE after invariant migration** |
| gain fallback abstraction | **DELETE** |
| generic position mode config | **DELETE** |
| home inside driver | **DELETE** |
| strict joint parser | **KEEP** |
| mechanical driver limit | **KEEP** |
| operational driver limit | **DELETE** |
| worker operational limits | **DELETE after producer audit** |
| command TTL | **KEEP** |
| generation | **KEEP** |
| send watchdog | **KEEP** |
| read watchdog | **KEEP** |
| board watchdog | **KEEP** |
| tactile validity | **KEEP** |
| tactile calibration | **KEEP** |
| tactile sensor IDs 17..21 | **KEEP** |
| tactile scale 0.1 | **FREEZE** |
| overcurrent semantics | **FREEZE** |
| raw tactile ring | **KEEP** |
| `read_error_count` | **KEEP** |
| `overcurrent_error_count` | **KEEP** |
| `read_healthy` | **LATER / likely delete** |
| `send_healthy` | **LATER / likely delete** |
| SHM `qpos_stale` | **LATER / likely derive** |
| HDF5 `hand_qpos_stale` | **KEEP for compatibility** |
| duplicate float timestamp | **LATER**（零消费者已验证，可随 Commit 7 删） |
| `tactile_contact` | **KEEP**（episode 级消费者，见 §11 审计决定） |
| connect 初始读重试 | **KEEP** |
| `error_state_watchdog_frames` | **MOVE → HandParams**（见 §6 A6） |
| anti-clogging command box（teleop 层） | **KEEP**（范围外，见 §2 范围说明） |

---

# 35. 最核心的执行准则

整个重构过程中始终使用这一判断标准：

```text
这个复杂度是在表达真实硬件/并发语义吗？
```

如果是：

```text
保留。
```

例如：

```text
fresh read
CRC retry
TTL
run generation
process isolation
mechanical hard limit
watchdog
tactile validity
```

如果只是：

```text
同一事实换了一层对象重新表达
```

例如：

```text
HandParams
→ HandProcessConfig
→ XHandConfig
```

或者：

```text
SDK error
→ bool
→ last_error_code
→ last_error_message
→ error_state
```

则应优先删除。

**XHand 简化的目标不是让系统变得“简单”，而是让代码复杂度与真实机器人问题的复杂度一一对应。**

---

# 36. 变更记录

## 2026-08-20 — Phase A–F 实施完成

- `runtime.hand` 直接传入 hand worker 和 `XHand`；删除 `HandProcessConfig`、`XHandConfig` 与 `startup_failure_is_fatal`。
- `error_state_watchdog_frames` 归入 `HandParams`；gain、current limit、transport 与机械限位不再经 worker 映射复制。
- `XHandError` 统一 connect/read/send 的 SDK 失败；`XHandState` 取代 `XHandSample`，在 parser 验证 joint payload，并以 `has_hardware_fault` 派生 board fault。
- 保留 fresh read、RS485 CRC retry、EtherCAT cleanup、joint-ID 检查、tactile 降级/校准、run generation、TTL、latest-wins、三个 watchdog 和 overcurrent 降级语义。
- `worker_validate_hand()` 只检查 IPC 结构、finite、generation 和 TTL；Producer 仍做 operational/mechanical preflight，Driver 仍拒绝机械越界。
- startup 与 loop 反馈统一经 `_publish_feedback()` 写入既有 `HAND_STATE_DTYPE`；本次未修改 schema 或持久化字段。
- 离线验证：compileall、diff check、runtime override、worker command、driver parser/command/error/hard-limit 与 feedback serialization。未运行硬件。

## 2026-08-19 — 审计修订（实施前定稿）

对代码库 `dexmani_real@main(c445373)` 做了逐阶段审计（8 区域 × 审计 + 对抗复核），确认 Phase A–E、G 未开始，F 主循环可读性已提前达标。本次修订：

- **§2**：补「connect 初始读重试」为禁删项；补范围说明（loop 层 `hand_disconnect_timeout_s` 去抖、anti-clogging box 的边界）。
- **§3.3**：过流前提更新——1501035 降级策略已在 `hand_process.py` 落地并实机验证；冻结决定不变。
- **§4**：补顺序硬约束（A 先于 E、C 先于 D、A–F 先于 G）与两项已定决定。
- **§5/§6**：补 `__post_init__` 校验链迁移要求、调用点审计行号、`startup_failure_is_fatal` 使用点、A6 watchdog 阈值归位、标量 gain 死代码确认。
- **§7/§8**：B1 死代码注；§8 connect 目标保留重试，显式声明行为变化与签名依赖。
- **§9**：C2/C3/C4 补 return 点清点、`XHandReadError` 真实消费者与合并条件。
- **§11**：`tactile_contact` 决定保留并进入目标字段；`current → current_ma` 边界说明。
- **§12/§13**：D1 依赖 C1；§13 修正 `error_state` 来源表述（8 处赋值仅 1 处 board 派生），补同步迁移点。
- **§15**：字段名修正、E 在 A 之后、安全依据（`send_action`/`hand_cmd_ring` 唯一路径审计）。
- **§16–§18**：F 现状与剩余差距、publication 两处定位、闭包提取注意。
- **§19–§22**：`read_healthy` 非等价性与消费者数、`qpos_stale` 消费侧同步点、两个 counter 的 HDF5 meta 持久化、`timestamp` 零消费者验证完成。
- **§30**：H2 home workflow 定位修正。
- **§34**：新增 4 行决策。
