# XHand SDK 使用模式：LeFranX / DexUMI / Dexora / pi-r2-flow / DexScrew 参考文档

**编制日期:** 2026-08-02（同日简化：-118 行，见 [§13 已执行的简化](#13-已执行的简化-2026-08-02)）
**目的:** 对照六个项目的 XHand SDK 使用模式，客观评估 DexMani 的架构取舍与改进方向

---

## 1. 项目概览

| 维度 | LeFranX | DexUMI | Dexora | pi-r2-flow | DexScrew | DexMani |
|------|---------|--------|--------|------------|----------|---------|
| **SDK 名称** | `xhand_controller` | `xhand_controller` | `xhand_tele_ops` | `xhand_controller` | `xhand_controller` | `xhand_controller` |
| **手数量** | 1 (右手) | 1 (右手) | 2 (双手) | 1 (右手，支持双手扩展) | 1 (左手) | 1 (右手) |
| **通信协议** | RS485 | RS485 | RS485 (双路) | RS485 (EtherCAT 可选) | **EtherCAT** (RS485 可选) | **EtherCAT** |
| **物理端口** | `/dev/ttyUSB0` | `/dev/ttyUSB0` | `/dev/ttyUSB0`, `/dev/ttyUSB1` | `/dev/serial/by-id/usb-FTDI_...` | EtherCAT (默认) 或 `/dev/ttyUSB0` (RS485) | 以太网接口 (如 enp1s0) |
| **波特率** | 3 Mbps | 3 Mbps | 3 Mbps | 3 Mbps | 115200 (RS485 fallback) | N/A (100 Mbps Ethernet) |
| **控制频率** | 30 Hz | 30 Hz | 20 Hz | 25 Hz | 自由运行 (无速率控制) | 16 Hz |
| **控制模式** | Mode 3 (位置控制) | Mode 3 (位置控制) | SDK 内部 | Mode 3 (位置控制), Mode 5 (力控，未使用) | Mode 3 (位置控制) | Mode 3 (位置控制) |
| **PID** | Kp=80, Ki=0, Kd=0 | Kp=150, Ki=0, Kd=0 (J11: Kp=100) | SDK 内部 | Kp=100, Ki=0, Kd=0 | **Kp=100, Ki=0, Kd=1** (唯一非零 Kd) | - |
| **扭矩限制** | tor_max=400 mA | tor_max=400 mA | SDK 内部 | tor_max=300 mA (固件级) | **tor_max=100** (最低) | 固件级 tor_max (无应用层力门控) |
| **进程隔离** | 无 | ZMQ (仅 eval 路径) | ZMQ (独立 conda env) | 无 (threading.Lock 串行 RS485) | 无 (单线程) | fork + SHM seqlock ring |
| **代码规模** | ~386 行 | ~441 行 | ~414 行 | ~339 行 | **~288 行** (最精简) | ~3000 行 (曾 ~3100，2026-08-02 简化 -118 行) |
| **接口风格** | lerobot Robot 基类 | 自定义 | ZMQ 协议 | LeRobot 标准接口 | 自定义 (单脚本) | RobotInterface 自定义 |

> **关键发现:** 四个参考项目默认 RS485，pi-r2-flow 是 RS485 项目中唯一同时提供完整 EtherCAT 可选路径的。DexScrew 和 DexMani 是两个默认 EtherCAT 的项目。DexScrew 是唯一使用非零 Kd (阻尼项) 的项目。
>
> **代码规模说明:** DexMani 的 XHand 相关代码 (~3000 行，曾 ~3100，2026-08-02 简化 -118 行) 是五个参考项目 (平均 ~370 行) 的 **8 倍以上**。这一差异主要来自进程隔离 (fork+SHM seqlock ~1700 行)、EtherCAT 从站状态管理 (~600 行) 和安全门体系 (~150 行)。五个参考项目均用单线程同步模式完成数据采集，代码量的差异不代表采集质量的差异。详见[附录 A](#附录-aethercat-vs-rs485-详解)和 [§13 已执行的简化](#13-已执行的简化-2026-08-02)。

---

## 2. SDK 初始化

### 2.1 LeFranX (`src/lerobot/robots/xhand/xhand.py`)

```python
# 懒加载, import 失败回退到 stub 模式
try:
    from xhand_controller import xhand_control
except ImportError:
    self._connect_stub()   # device=None, _is_connected=True, 仅测试用
    return

self._device = xhand_control.XHandControl()
self._device.open_serial(port, 3000000)
self._hand_id = self._device.list_hands_id()[0]

# 预填充 HandCommand_t 结构（connect 时一次，后续复用）
self._hand_command = xhand_control.HandCommand_t()
for i in range(12):
    self._hand_command.finger_command[i].id = i
    self._hand_command.finger_command[i].kp = 80
    self._hand_command.finger_command[i].ki = 0
    self._hand_command.finger_command[i].kd = 0
    self._hand_command.finger_command[i].position = 0.0
    self._hand_command.finger_command[i].tor_max = 400
    self._hand_command.finger_command[i].mode = 3
```

**特点:**
- 无连接重试、无心跳检测
- `configure()`/`calibrate()` 是空桩
- `is_calibrated` 硬编码 `True`
- 连接后不验证手是否真正可达

### 2.2 DexUMI (`dexumi/hand_sdk/xhand/hand_api_cls.py`)

```python
self._device = xhand_control.XHandControl()

def connect(self):
    device_identifier = {
        "protocol": "RS485",
        "serial_port": "/dev/ttyUSB0",
        "baud_rate": 3000000
    }
    self.open_device(device_identifier)
    self.list_hands_id()
    return True  # 无条件返回 True, 不检查 open_device 结果

def open_device(self, device_identifier):
    rsp = self._device.open_serial(port, baud_rate)
    print(f"open RS485 result: {rsp.error_code == 0}")  # 只打印, 不抛异常
```

**ExoXhandSDK 扩展:**
- 加载 6 个 pickle 校准模型（拇 swing ×1, 拇 bend ×1, 四指 bend ×4）
- 模型加载失败 → `exit()` 直接终止进程
- 硬编码 `calibrate_angle` 数组（12 个编码器参考位置）

**特点:**
- 无连接重试
- `open_device` 失败不抛异常，依赖后续 `read_state` 失败来发现
- 校准模型路径是必填参数

### 2.3 Dexora (`deploy/xhand_forwarder.py`)

```python
def _initialize_xhand(self):
    os.chdir(self.xhand_code_path)       # SDK 依赖 cwd 找配置文件
    from xhand_tele_ops import XHandTeleOps
    self.xhand_controller = XHandTeleOps(self.xhand_config)
    # 无 try/except — 失败则进程崩溃

# 启动后归位
def _reset_to_init_pose(self):
    left_rad = np.deg2rad(INIT_JOINTS_DEG["left_hand"])
    right_rad = np.deg2rad(INIT_JOINTS_DEG["right_hand"])
    self.execute_action({"left_hand": left_rad, "right_hand": right_rad})
    # 返回值未验证
```

**特点:**
- `os.chdir()` 脆弱依赖 SDK 的相对路径配置
- 初始化失败直接崩溃（无回退）
- `_reset_to_init_pose()` 返回值未检查

### 2.4 pi-r2-flow (`deployment/mindex/robots/xhand_robot.py`)

```python
# __init__ (line 38-58)
self._protocol = "RS485"         # 也支持 EtherCAT
self._serial_port = "/dev/serial/by-id/usb-FTDI_USB-RS485-WE_FTAAUYBU-if00-port0"
self._baud_rate = 3_000_000
self._lock = threading.Lock()    # 串行化 RS485 总线访问

# connect() — RS485 端口级回退 (lines 101-128)
if self._protocol == "RS485":
    enum_ports = self._device.enumerate_devices("RS485") or []
    candidates = [self._serial_port] + [
        p for p in enum_ports
        if p.startswith("/dev/ttyUSB") and p != self._serial_port
    ]
    for port in candidates:
        rsp = self._device.open_serial(port, self._baud_rate)
        if rsp.error_code != 0:    # 失败 → 下一候选
            continue
        hand_ids = self._device.list_hands_id()
        if not hand_ids:           # 开成功了但手不应答
            continue               # ⚠️ 端口句柄泄漏
        self._serial_port = port   # 成功
        break

# connect() — EtherCAT 路径 (lines 86-98)
elif self._protocol == "EtherCAT":
    ports = self._device.enumerate_devices("EtherCAT")
    rsp = self._device.open_ethercat(ports[0])
    # 需要 cap_net_raw+ep 权限

# Post-connect: 预分配 HandCommand_t 复用 (lines 133-143)
self._hand_command = xhand_control.HandCommand_t()
for i in range(12):
    self._hand_command.finger_command[i].kp = 100
    self._hand_command.finger_command[i].tor_max = 300
    self._hand_command.finger_command[i].mode = 3
```

**特点:**
- 唯一提供 RS485 **端口级回退**的项目（遍历 `/dev/ttyUSB*` 候选列表）
- 端口泄漏 bug: `open_serial` 成功但 `list_hands_id` 失败时不 close
- EtherCAT 路径**完整可用**（有别于 LeFranX 的 NotImplementedError）
- LeRobot 标准接口: `connect/disconnect/get_observation/send_action/is_connected`

### 2.5 DexScrew (`xhand-deploy/xhand_deploy.py`)

```python
from xhand_controller import xhand_control as xh

class XHandControl:
    def __init__(self, hand_id=0, position=0.1, mode=3):
        self._hand_id = hand_id
        self._device = xh.XHandControl()
        self._hand_command = xh.HandCommand_t()

        for i in range(12):
            finger_cmd = self._hand_command.finger_command[i]
            finger_cmd.id = i
            finger_cmd.kp = 100
            finger_cmd.ki = 0
            finger_cmd.kd = 1            # ⚠️ 唯一使用非零 Kd 的项目
            finger_cmd.position = position
            finger_cmd.tor_max = 100     # ⚠️ 3-4x 低于所有其他项目
            finger_cmd.mode = mode

# main() — 默认 EtherCAT，也支持 RS485
device_identifier = {"protocol": "EtherCAT"}
controller.open_device(device_identifier)

# open_device() 内部协议分发:
def open_device(self, device_identifier: dict):
    protocol = device_identifier.get("protocol")
    if protocol == "RS485":
        rsp = self._device.open_serial(port, baud_rate)   # baud_rate=115200
    elif protocol == "EtherCAT":
        ether_cat = self._enumerate_devices("EtherCAT")
        rsp = self._device.open_ethercat(ether_cat[0])    # 取第一个设备
```

**特点:**
- 默认 EtherCAT，RS485 为备选 — 与 DexMani 相同的协议选择
- **唯一使用 Kd=1**（阻尼项）的部署项目，所有其他项目 Kd=0
- **tor_max=100 是六个项目中最低的**（DexMani 320, LeFranX/DexUMI 400, pi-r2-flow 300）
- 无 stub 模式、无连接重试、无硬件身份验证
- 初始位姿发送后 `sleep(1)` 作为唯一的"复位"机制 — 无 `reset_sensors()`
- **归一化统计量嵌入 JIT 模型** — 唯一将 running_mean/running_var 冻结在 TorchScript 导出的项目

### 2.6 对比与借鉴

| 模式 | 来源 | 优点 | 缺点 |
|------|------|------|------|
| 懒加载 + stub 回退 | LeFranX | 无 SDK 也能跑测试 | 可能掩盖连接问题 |
| 预填充 HandCommand | LeFranX/pi-r2-flow/DexScrew | 减少每帧开销 | PID/Torque 固定不可调 |
| Exo SDK 校准模型 | DexUMI | 支持外骨骼→真手映射 | 模型加载失败处理粗暴(exit) |
| chdir + YAML config | Dexora | 配置外置，多进程共享 | chdir 脆弱，路径硬编码 |
| 端口级回退 | pi-r2-flow | 自动发现 USB 端口，容错 | 端口泄漏 bug |
| LeRobot 标准接口 | pi-r2-flow | 生态互操作 | 接口不涵盖 safety gates |
| 完整 EtherCAT 可选路径 | pi-r2-flow/DexScrew | 双协议一线切换 | 需要 cap_net_raw |
| 归一化嵌入 JIT 导出 | **DexScrew 独有** | 消除部署时统计量不匹配 | 需要训练时注册 buffer |
| 非零 Kd 阻尼 | **DexScrew 独有** | 减少振荡 | 无对比验证 |

**DexMani 建议:**
1. 保留预填充 HandCommand 模式（LeFranX/pi-r2-flow/DexScrew 均用此模式）
2. 对 SDK import 失败加 graceful fallback（LeFranX 的 stub 思路，但用于诊断模式而非静默跳过）
3. 将 SDK 配置路径作为绝对路径传入，避免 chdir（Dexora 的反面教材）
4. 避免 DexScrew 的 12x 逐关节读取反模式 — 保持当前单次 `read_state` + 批量解析
5. 考察是否引入非零 Kd（DexScrew 的 kd=1），用于抑制手指振荡 — 但需真机验证
6. 将归一化统计量嵌入模型导出（DexScrew），如果未来需要 sim-to-real 策略部署

---

## 3. 通信模式

### 3.1 主线程同步调用 (LeFranX — 全路径)

```
App → XHand SDK (xhand_control) → RS485 → 硬件
```

- `read_state()` / `send_command()` 阻塞调用
- 一个 `Exception` 直接传播到控制循环
- `_get_joint_states()` catch `Exception` → return `None`
- `_send_position_command()` catch `Exception` → return `False`

### 3.2 后台读线程 + 直连写 (DexUMI — teleop/replay 路径)

```
App → ExoXhandSDK.send_command() → RS485 → 硬件
       ↑ (读写分离)
       XhandSDK._read_loop (30Hz daemon thread) → deque(maxlen=10)
```

- 读：后台 daemon 线程 30Hz 填充 `deque`
- 写：主线程直接 SDK 调用
- teleop/replay **不启动读线程**（只写不读）
- `threading.Lock` 保护 `_state_queue`

### 3.3 ZMQ 进程隔离 (Dexora — inference 路径)

```
Policy Host (dexora_inference_zmq, env: dexora)
    │  ZMQ REQ tcp://*:5557
    ▼
XHand Forwarder (xhand_forwarder, env: xhand_tele_env)
    │  XHandTeleOps SDK
    ▼
RS485 → 2x XHand
```

**ZMQ 协议 (仅 3 条指令):**
```python
{"command": "get_observations"}  → {"left_hand": [12], "right_hand": [12]}
{"command": "execute_action", "action_data": {...}}  → {"status": "success/error/skipped"}
{"command": "ping"}  → {"status": "pong"}
```

**注意:** 遥操录制路径 (`receive_from_vision_pro.py`) **不使用** ZMQ forwarder，直接调用 SDK → 两种路径行为不一致（录制缺少 safety gates）

### 3.4 DexUMI ZMQ 路径 (eval 路径)

```
eval_xhand.py → DexClient ──ZMQ IPC──→ DexServer → ExoXhandSDK → 硬件
                 (ipc:///tmp/dex_req)   (ipc:///tmp/dex_stream)
```

- PUB/SUB 用于状态流，REQ/ROUTER 用于指令
- `MotorTrajectoryInterpolator` 做轨迹插值
- Bug: `open_server.py:93` 传 `inspire=hand` 但 DexServer 参数名是 `hand`

### 3.5 pi-r2-flow — 进程内 Lock 串行 (policy 路径)

```
Policy Loop (run_policy.py, ~25 Hz)
    │  threading.Lock 串行 RS485
    ▼
XHandRobot (xhand_robot.py) → xhand_controller SDK → RS485 → 硬件
    │
    └── send_action 响应的缓存状态 → get_observation(fresh=False) 零额外往返
```

- 读(fresh=True): 独立 RS485 往返 (~10-15ms)
- 读(fresh=False): 返回上一次 `send_command` 响应的缓存状态 (零成本)
- `threading.Lock` 保护所有 RS485 访问
- **无进程隔离** — SDK segfault → 整个臂+手+策略进程一起崩溃

### 3.6 DexScrew — 单线程自由运行 + 策略推理 decimation (policy 路径)

```
Policy Loop (xhand_deploy.py, while True 无速率控制)
    │  每步: 逐关节 read_joint_pos(i) ×12 (12x 总线往返)
    │  每10步: JIT model inference (decimation 1/10)
    │  send_command() — 返回值丢弃
    ▼
xhand_controller SDK → EtherCAT / RS485 → 硬件
```

- 读: `for i in range(12): pos = controller.read_joint_pos(i)` — 每个关节触发一次完整的 `read_state(force_update=True)`，12 次独立总线往返
- 写: 只更新 `position` 字段，Kp/Kd/Ki/tor_max/mode 初始化后不变
- 策略推理: `step % 10 == 0` 时才跑 JIT 模型，其余步发送上次 target（zero-order hold）
- **无速率控制** — `while True` 自由运行，只有 `time.sleep(0.005)` 的微弱限制
- **30 步 proprioceptive history buffer** — 滑动窗口 `(30, 24)` 保存配对 `(当前关节角, 目标关节角)`，给策略提供指令-执行动态的短期记忆
- **关节索引重映射** — 策略输出顺序 ≠ SDK 硬件顺序，12 元素排列桥接（因 URDF 关节序与 SDK 枚举序不同）
- **`send_command` 返回值完全丢弃** — `_ = self._device.send_command(...)`
- **无进程隔离** — 单线程，SDK segfault → 整个策略进程崩溃

### 3.7 对比

| 模式 | 项目 | 物理层 | 隔离性 | 延迟 | 复杂性 |
|------|------|--------|--------|------|--------|
| 主线程同步 | LeFranX | RS485 | 无 | 最低 | 最低 |
| 读写线程分离 | DexUMI | RS485 | 无 | 低 | 低 |
| ZMQ REP/REQ | Dexora | RS485 | 进程级 (不同 conda env) | 中 (序列化) | 中 |
| Lock 串行 + 缓存读 | pi-r2-flow | RS485 | 无 | 低 (fresh=False 零往返) | 最低 |
| 单线程自由运行 + decimation | DexScrew | **EtherCAT** | 无 | 低 (策略 1/10 步) | **最低** (288 行) |
| fork+SHM seqlock | DexMani | **EtherCAT** | 进程级 | 低 (零拷贝) | **高** (含 EtherCAT 从站状态管理) |

**架构对比总结:**

| 模式 | 项目 | 物理层 | 隔离性 | 延迟 | 代码复杂度 | 主要代价 |
|------|------|--------|--------|------|-----------|----------|
| 主线程同步 | LeFranX | RS485 | 无 | 最低 | 最低 | 无容错 |
| 读写线程分离 | DexUMI | RS485 | 无 | 低 | 低 | daemon 线程生命周期 |
| ZMQ REP/REQ | Dexora | RS485 | 进程级 (不同 conda env) | 中 (序列化) | 中 | 序列化开销 + 配置管理 |
| Lock 串行 + 缓存读 | pi-r2-flow | RS485 | 无 | 低 (fresh=False 零往返) | 最低 | 无隔离 |
| 单线程自由运行 + decimation | DexScrew | **EtherCAT** | 无 | 低 (策略 1/10 步) | **最低** (288 行) | 无 safety/cleanup/速率控制 |
| fork+SHM seqlock | DexMani | **EtherCAT** | 进程级 | 低 (零拷贝) | **最高** (~1700 行隔离层) | 调试困难、seqlock 协议脆弱、布局变更需多文件同步 |

DexMani 和五个参考项目代表了两种不同的设计哲学：参考项目以**最小代码量完成采集任务**（平均 ~370 行），DexMani 以**进程隔离 + 纵深防御**换取故障时不会同时损坏臂和手。两种取舍在不同场景下有不同合理性 — 学术数据采集对可靠性要求低于 7×24 生产部署，参考项目的简单架构在各自场景下完全够用。

---

## 4. 错误检测

### 4.1 各项目实际检测能力

| 检测能力 | LeFranX | DexUMI | Dexora | pi-r2-flow | DexScrew | DexMani | 借鉴价值 |
|----------|---------|--------|--------|------------|----------|---------|----------|
| read_state error_code != 0 | ✅ (whitelist 5 项) | ✅ (print then drop) | ✅ (code == 200 only) | ✅ (所有调用点) | ⚠️ (print + return None，调用方不检查) | ✅ | pi-r2-flow 最一致 |
| CRC 错误分类 | ❌ (whitelist 静默) | ❌ | ✅ (字符串匹配) | ❌ | ❌ | ✅ (EtherCAT 层) | **Dexora 的 RS485 CRC 分类可借鉴** |
| SDO 写入错误 | N/A (RS485) | N/A (RS485) | N/A (RS485) | N/A (RS485) | N/A (EtherCAT) | ✅ (EtherCAT 特有) | DexMani 独有 |
| 每关节错误寄存器 | ❌ (不读) | ✅ (读但不检查) | ❌ (不读) | ❌ (温度高字节故障标志被 `& 0xFF` 丢弃) | ❌ (不读) | ✅ | **DexUMI 有字段但死数据；pi-r2-flow 有数据但主动丢弃** |
| 温度错误 | ❌ (whitelist) | ❌ (读但不检查) | ❌ (丢弃) | ❌ (读但从不阈值检查) | ❌ (不读) | ✅ (70°C gate) | 五个参考项目都不可靠 |
| 通信超时 | ❌ | ❌ (queue 默默排空) | ❌ (ZMQ 5s 超时仅策略端) | ❌ (send_action 失败打印 warning 继续) | ❌ | ✅ (SM-watchdog) | 参考项目均无超时检测 |
| 数据新鲜度 | ❌ | ❌ | ❌ (force_update=False) | ❌ (fresh=False 可返回任意旧缓存) | ❌ (force_update=True 但无超时检查) | ✅ (qpos freshness) | 六个项目中仅 DexMani 有 |

### 4.2 关键代码模式

**Dexora CRC 分类 (借鉴 — RS485 场景):**
```python
# xhand_forwarder.py:257-264
if isinstance(resp, dict):
    msg = str(resp.get("msg") or resp.get("message") or "")
    if "crc" in msg.lower() or "CRC" in msg:
        self._crc_count += 1
        # 退避重试
        if attempt < retries:
            time.sleep(backoff)
            continue
# 也捕获异常字符串中的 CRC:
except Exception as e:
    if "crc" in str(e).lower():
        self._crc_count += 1
```

> **注意:** EtherCAT 的 CRC 错误语义与 RS485 不同 — EtherCAT 帧由 FPGA/MAC 层做硬件 CRC 校验，应用层通常不会看到 CRC 错误。Dexora 的 CRC 重试模式更适用于 RS485 总线。如果未来 DexMani 切换回 RS485，此模式可直接借鉴。

**LeFranX 错误白名单 (可借鉴 — 加计数):**
```python
# xhand.py:231-237
ignored_errors = [
    "Sensor fails to read the combined force",
    "Sensor fails to read the distributed force",
    "Sensor fails to read temperature",
    "Communication data CRC error",              # ⚠️ CRC 被静默!
    "This hardware version does not support force control mode"
]
if not any(ignored_error in error_struct.error_message for ignored_error in ignored_errors):
    logger.warning(...)  # 非白名单错误才告警
```

**DexUMI 错误寄存器字段 (可借鉴 — 使之可用):**
```python
# hand_api_cls.py:21-34 — 这些字段已被 SDK 填充但从未被任何应用代码检查
@dataclass
class JointState:
    commboard_err: int    # 通信板错误
    jonitboard_err: int   # 关节板错误 (SDK 拼写 bug)
    tipboard_err: int     # 指尖板错误
```

---

## 5. 错误恢复

### 5.1 恢复能力一览

| 恢复能力 | LeFranX | DexUMI | Dexora | pi-r2-flow | DexScrew | DexMani |
|----------|---------|--------|--------|------------|----------|---------|
| `clear_error()` | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ T1 |
| `reset_connection()` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ (2026-08-03 移除) |
| CRC 重试 | ❌ (whitelist 静默) | ❌ | ✅ (退避重试，RS485) | ❌ | ❌ | N/A (EtherCAT 硬件 CRC) |
| SDO 重试 | N/A | N/A | N/A | N/A | N/A | ✅ (open_ethercat_retries=2) |
| 急停 | ❌ (空桩) | ❌ | ❌ (kill 进程) | ⚠️ (disconnect 设 passive mode 放手) | ❌ (无任何 cleanup) | ✅ |
| `recover_from_errors()` | ❌ (空桩) | ❌ | ❌ | ❌ | ❌ | ✅ 分级自愈 |
| 断连检测 + 重连 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| EtherCAT 从站状态恢复 | N/A | N/A | N/A | N/A | ❌ (无状态管理) | ✅ |
| `reset_sensors()` 重试+验证 | ❌ | ❌ | ❌ | ✅ (迭代验证+偏置消除，**最优**) | ❌ (完全无) | ⚠️ (已有偏置消除，缺迭代验证) |

**DexMani 的 EtherCAT 专项恢复**（`xhand.py:530-615`，三个参考项目均不适用）：

| 机制 | 说明 |
|------|------|
| **Slave state 管理** | AL state 常量 (INIT=1/PRE_OP=2/SAFE_OP=4/OP=8)，`set_firmware_state()` 请求从站回 INIT |
| **SM-watchdog 等待** | `close_device()` 后等待 2.0s 让从站 watchdog 超时自动回 INIT |
| **Stale-OP 恢复** | 上次进程被 kill -9 后从站卡在 OP → 首次重试前额外等待 3.0s |
| **两阶段设备发现** | enumerate 和 open 使用**不同的 XHandControl 实例**，避免 raw socket 冲突导致 "write sdo failed" |

### 5.2 Dexora CRC 重试 (可借鉴 — RS485 场景)

```python
# xhand_forwarder.py:244-273
retries = int(self.config.get("xhand_send_retries", 1))
backoff = float(self.config.get("xhand_crc_backoff_s", 0.08))

for attempt in range(retries + 1):
    try:
        resp = self.xhand_controller.send_data_xhand(transform_data)
        # 成功: 更新状态, 重置错误
        self._last_send_t = now
        self._send_count += 1
        self._last_err = None

        if isinstance(resp, dict):
            msg = str(resp.get("msg") or resp.get("message") or "")
            if "crc" in msg.lower() or "CRC" in msg:
                self._crc_count += 1
                if attempt < retries:
                    time.sleep(backoff)
                    continue  # 重试
        return {"status": "success", ...}
    except Exception as e:
        if "crc" in str(e).lower():
            self._crc_count += 1
        if attempt < retries:
            time.sleep(backoff)
            continue
        break

# 重试耗尽
return {"status": "error", "error": str(last_exc), "crc_count": self._crc_count}
```

### 5.3 pi-r2-flow `reset_sensors()` 迭代验证 (可借鉴 — 五项目最优)

```python
# xhand_robot.py:252-326

MAX_OUTER_ITERS = 5
verify_thresh_n = 2.0  # N — 力阈值

for outer_iter in range(MAX_OUTER_ITERS):
    bad_fingers = []
    for finger_idx in range(5):
        for attempt in range(3):  # 每个传感器最多重试 3 次
            err = self._device.reset_sensor(self._hand_id, sensor_id)
            if err.error_code == 0:
                break

    # 取 5 帧新鲜读数计算软件偏置 (消除厂商残留 ~5-30N)
    force_samples = []
    for _ in range(5):
        _, state = self._device.read_state(self._hand_id, True)
        force = np.array([state.sensor_data[k].calc_force.{fx,fy,fz} for k in range(5)])
        force_samples.append(force)
    self._bias_ft = np.mean(force_samples, axis=0)  # 软件偏置

    # 验证: 每指 |F| > 2.0N 则标记为 bad
    ft_mag = np.linalg.norm(fingertip_force, axis=1)
    bad_fingers = [i for i, m in enumerate(ft_mag) if m > verify_thresh_n]

    if not bad_fingers:
        break  # 全部通过
    # 下一轮只重试仍 bad 的手指

# 结束后打印诊断: "Passed: index middle ring little | Still bad: thumb=3.1N"
```

**为什么 DexMani 应该借鉴:**
- 当前 DexMani `_reset_tactile_sensors()` (`xhand.py:449-488`) 只做一次 SDK `reset_sensor()` 调用，无偏置消除，无验证
- 厂商 `reset_sensor()` 后残留 ~5-30N 偏置 → 录制的触觉数据有 DC 偏移
- pi-r2-flow 的偏置消除和迭代验证是低风险纯软件改进

**实施位置:**
- 偏置消除: `xhand.py:447` (`_reset_tactile_sensors()` 调用后)，加 5 帧平均偏置计算
- 迭代验证: 同上，用 `verify_thresh_n=2.0N` 的选择性重试循环

---

## 6. 安全机制

### 6.1 发送前验证

| 安全门 | LeFranX | DexUMI | Dexora | pi-r2-flow | DexScrew | DexMani |
|--------|---------|--------|--------|------------|----------|---------|
| 连接检查 | `DeviceNotConnectedError` (标志永不重验证) | ❌ | ❌ | ✅ `_require_connected()` | ❌ | ✅ |
| NaN/Inf 消毒 | ❌ | ❌ | ✅ `nan_to_num(→0)` | ❌ | ❌ | ✅ (→neutral) |
| 关节限位 | ✅ per-joint 可配置 | ⚠️ clip [0, π]，joint 3 豁免 | ✅ 硬编码 rad 限位 | ❌ (driver 层无，`clip_to_state()` 存在但主循环中**未调用**) | ⚠️ (定义了 limit 数组但 deploy 中不钳制) | ✅ (IK-level + per-joint) |
| 扭矩/电流限制 | ⚠️ 固件级 tor_max=400mA | ⚠️ 固件级 tor_max=400mA | ❌ | ⚠️ 固件级 tor_max=300mA | ⚠️ 固件级 tor_max=**100** (最低) | ⚠️ 固件级 tor_max |
| 温度检查 | ❌ (whitelist 静默) | ❌ (读但不检查) | ❌ (丢弃) | ❌ (读但从不与阈值比较) | ❌ (不读温度) | ✅ (70°C) |
| 速度/加速度限制 | ❌ | ❌ | ❌ | ❌ | ❌ (自由运行无速率控制) | ❌ (已移除，见 §13.5) |
| 数据新鲜度 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (qpos freshness) |

### 6.2 安全门的代价

DexMani 的 5 道安全门 (error→connection→NaN→torque→temp) 提供了最完整的发送前验证，但也引入了相应代价：

| 代价 | 说明 |
|------|------|
| **假阳性停顿** | 温度门 70°C 和扭矩门 (J1-2=50, J3-5=30, J6-7=20 Nm) 在极端但安全的工况下可能误触发，中断采集 |
| **门配置耦合** | （已移除 — DexMani 不做应用层触觉力安全门控） |
| **维护负担** | 每个新状态字段需要同步更新 validate 逻辑，增加一致性问题 |
| **参考项目都不需要** | 五个项目用固件级 tor_max 完成相同任务，说明应用层安全门对数据采集而言并非刚性需求 |

**结论:** 安全门体系在 7×24 生产部署中是合理投资，在数据采集场景中属于防御性过度设计 — 有用但边际收益递减。五个参考项目依赖固件级保护 (tor_max + 温度 fuse) 采集了数千个 episode 未出现安全事故，证明简化方案在采集场景下是可接受的。

### 6.3 Dexora send 节流 + epsilon 门 (参考 — RS485 带宽受限场景)

```python
# xhand_forwarder.py:222-231
min_interval = float(self.config.get("xhand_min_send_interval_s", 0.12))  # ~8.3Hz
eps = float(self.config.get("xhand_forwarder_eps", 0.0))      # mean abs delta (rad)

now = time.time()
mean_delta = float(
    np.mean(np.abs(left_arr - self._last_left))
    + np.mean(np.abs(right_arr - self._last_right))
) / 2.0

if (now - self._last_send_t) < min_interval or (eps > 0.0 and mean_delta < eps):
    self._skip_count += 1
    return {"status": "skipped", ...}
```

> **注意:** Dexora 的 RS485 节流每秒 ~8.3 次。EtherCAT 带宽高两个数量级（100M vs 3M），且总线周期由 master 精确调度，不需要人为限速。**epsilon 门的思路**（静止时跳过相同目标的重复发送）对任何总线都有降低负载的价值，但 DexMani 在 EtherCAT 上无需 min_interval 节流。可以只采用 epsilon 门部分。

### 6.4 Dexora NaN 消毒模式 (可借鉴但改进默认值)

```python
# xhand_forwarder.py:110-112
if not np.all(np.isfinite(arr)):
    logging.warning("[XHAND] action contains NaN/Inf, sanitizing to finite values")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
```

⚠️ NaN→0 对某些关节不安全（例如拇指关节 0° 是极限位置）。**建议改为** NaN→上次已知正常值或 home position。

---

## 7. 状态监控

### 7.1 读取的状态维度

| 状态维度 | LeFranX | DexUMI | Dexora | pi-r2-flow | DexScrew | DexMani |
|----------|---------|--------|--------|------------|----------|---------|
| 关节位置 (12-DOF) | ✅ | ✅ | ✅ | ✅ | ✅ (逐关节读 ×12) | ✅ |
| 关节力矩/电流 | ✅ (标为 torque，实为 mA) | ✅ (读但从不检查) | ❌ (SDK 返回但丢弃) | ✅ (读但从不阈值检查) | ❌ (不读) | ✅ |
| 温度 | ❌ (whitelist 静默) | ✅ (读但从不检查) | ❌ (丢弃) | ✅ (读但从不阈值检查) | ❌ (不读) | ✅ (70°C gate) |
| 触觉/压力 | ❌ | ✅ (独立 UART, 3 指 FSR) | ❌ (丢弃) | ✅ (fingertip_force 5x3 + xhand_tactile 5x120x3) | ❌ (完全无触觉) | ✅ (tactile_sum + tactile_force 5x120x3) |
| 指尖位置 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (chained FK) |
| 错误寄存器 | ❌ | ✅ (读但从不检查) | ❌ | ❌ (温度高字节故障标志被 `& 0xFF` 丢弃) | ❌ | ✅ |
| 连接状态 | ⚠️ (一次性标志) | ❌ | ❌ | ✅ `_connected` flag | ❌ | ✅ (持续更新) |
| 数据时间戳 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 速度 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (computed) |

### 7.2 观测失败处理

| 项目 | 行为 | 危险程度 |
|------|------|----------|
| LeFranX | return None → 观测 dict 中静默缺字段 | 中 |
| DexUMI | queue 慢慢排空 → ValueError("Queue is empty") 崩溃 | 高 |
| Dexora | return `[0.0] * 12` → 策略看到不可能的机械零点位置 | **最高** |
| pi-r2-flow | return None → `_build_flat_state` 返回 None → 策略用上次已知状态继续 | 低 |
| DexScrew | `read_joint_pos` 返回 None → `q_xhand.append(None)` → `torch.tensor([None,...])` 崩溃或 NaN | **高** |
| DexMani | hand_connected flag → zero-command fallback | 安全 |

### 7.3 Dexora 观测模式 (force_update 选择)

```python
# receive_from_vision_pro.py:102
# 如果同时读写，需要设置 force_update=False
# 如果只读不写，需要设置 force_update=True
resp = node.get_hand_full_info("hand_a", force_update=False, is_print=False)
```

**模式差异:**
- `force_update=True`: 强制刷新，保证新鲜但增加总线负载
- `force_update=False`: 可能返回缓存数据，适合频繁读写的控制循环

> **DexMani 当前做法:** `force_update_state: bool = True`（`xhand.py:75`），与 Dexora 的 teleop 录制路径相反。EtherCAT 带宽充足，force_update 的开销可以接受。

### 7.4 状态监控的维护成本

DexMani 读取 **9 个状态维度**（vs 参考项目平均 2-3 个），每增加一个维度需要的同步更新：

| 新增维度时需修改的文件 | 受影响模块 |
|----------------------|-----------|
| `RobotState` (types.py) | dataclass 字段定义 |
| `robot_layouts.py` | SHM dtype 布局 |
| `interface.py` (`get_state()`) | 读取逻辑 |
| `episode_recorder.py` | 录制数据集 |
| `validate.py` | 如需门控，加检查逻辑 |

五个参考项目中，状态字段与代码高度内聚（通常都在一个 300 行的文件中），新增字段只需改一处。DexMani 的跨文件同步要求是隔离架构的固有代价 — 五个参考项目之所以能在 300 行内完成相同采集任务，很大程度上是因为避免了这些间接层。

---

## 8. 抓握/负载处理

六个项目中抓握力/负载运行时处理的覆盖情况：

| 能力 | LeFranX | DexUMI | Dexora | pi-r2-flow | DexScrew | DexMani |
|------|---------|--------|--------|------------|----------|---------|
| 运行时电流监控 | ❌ | ❌ | ❌ | ❌ (记录但不门控) | ❌ | ✅ (per-joint Nm) |
| 过流失速检测 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 抓握力控制 | ❌ | ❌ | ❌ | ❌ (Mode 5 定义但未使用) | ❌ | ❌ (纯位置控制) |
| FSR 触觉用于控制 | ❌ | ❌ (只记录) | ❌ | ❌ (只记录+可视化) | ❌ (无触觉) | ❌ (只记录) |
| 温度降额 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (70°C) |
| 驱动板过流保护 | ⚠️ 固件 tor_max=400 | ⚠️ 固件 tor_max=400 | ❌ | ⚠️ 固件 tor_max=300 | ⚠️ 固件 tor_max=**100** | ✅ (应用层 320mA per-joint) |
| 触觉传感器校准 | ❌ | ❌ | ❌ | ✅ (迭代验证+偏置消除) | ❌ (无触觉) | ✅ (偏置消除，缺迭代验证) |

**结论:** DexMani 的安全防护为固件级 tor_max 保护。pi-r2-flow 是唯一有高质量触觉传感器校准的项目。

---

## 9. 跨项目差异分析

### 参考项目共性的简化取舍

五个参考项目存在以下共同缺失的能力，但这些"缺失"需要在其设计背景下理解 — 它们服务于**数据采集**场景，而非 7×24 生产部署：

1. **运行时错误恢复** — LeFranX `recover_from_errors()` 是空桩，其他项目根本没有这个方法。实际运行中，RS485 偶发 CRC 噪声被 Dexora 的退避重试处理，其他项目依赖人工重启。
2. **硬件急停** — LeFranX `stop()` 是空桩，Dexora kill 进程不断电，DexScrew Ctrl-C 后手停在 Mode 3。**但对于采集场景，操作员随时可以物理干预，急停的必要性低于无人值守部署。**
3. **每关节电流/温度门控** — 数据要么不读，要么读了不检查。所有五个项目依赖固件级 tor_max 单一保护。**采集了数千个 episode 均未出现烧毁事故，说明固件保护在此场景下是足够的。**
4. **数据新鲜度/连接心跳** — 无运行时检测。**RS485 USB 连接在采集场景下极少中断（无 EtherCAT SM-watchdog 机制），这项缺失的实际影响很小。**
5. **触觉用于安全决策** — 触觉只记录/可视化，不门控。**对位置控制模式的手部，触觉安全门的边际收益有限 — xHand 的 tor_max 固件限流已在驱动层提供过流保护。**

> **关键认知:** 五个参考项目缺失的不是"能力"，而是在各自场景下不必要的防御层。把这些列为"缺陷"是站在 DexMani 过度工程化的视角去评判。反过来，DexMani 多出来的每一层防御都有真实的代码和维护成本。

### DexScrew 特有的缺陷

1. **12x 读放大** — `for i in range(12): read_joint_pos(i)` 每个关节触发完整 `read_state(force_update=True)`，同时有关节间时序偏差（joint 11 比 joint 0 晚 ~12 次总线往返）
2. **`read_joint_pos` 返回 None 未检查** — 错误时 None 被 append 到列表 → `np.asarray([None,...])` → object dtype → `torch.tensor()` 崩溃
3. **无 cleanup/disconnect** — `while True` 无 try/finally，Ctrl-C 后手停留在 Mode 3 持续出力（电机过热风险）
4. **自由运行无速率控制** — `time.sleep(0.005)` 是唯一限制，EtherCAT 上可达 kHz 级 → 总线饱和
5. **关节限位定义了但不钳制** — `xhand_dof_lower_limits/upper_limits` 数组存在，但 deploy loop 只对策略输出 clip 而不对最终 `send_command` 的值钳制
6. **`send_command` 返回值丢弃** — `_ = self._device.send_command(...)` — 完全不知命令是否被接收
7. **Sim torque control / 真机 position control 范式不匹配** — Sim 用 PD 力矩控制 (`p_gain=3, d_gain=0.01, torque_limit=300`)，真机用 Mode 3 位置控制。桥接全靠 student distillation + domain randomization，无机增益匹配
8. **无触觉传感器初始化** — 即使策略不使用触觉，未初始化的传感器硬件可能干扰正常操作
9. **启动时 30 步历史缓冲区全为合成数据** — 初始 `proprio_hist_buffer` 30 步全部相同，需要 30 个真实步（可能 1.5-3 秒）才能完全冲掉

### pi-r2-flow 特有的缺陷

1. **`clip_to_state()` 存在但主循环不调用** — 代码注释说 EMA 比 clip 更平滑 → 策略输出**无硬上界直接送到硬件**
2. **温度高字节故障标志被 `& 0xFF` 主动丢弃** — board error 信息已读入但被位运算消除
3. **零自愈** — `send_command` 持久失败只打印 stderr，永不重连
4. **无进程隔离** — SDK segfault → 臂+手+策略全死

### pi-r2-flow 值得 DexMani 借鉴

**`reset_sensors()` 校准质量** — 五项目中最好的传感器复位实现（详见 [5.3 节](#53-pi-r2-flow-reset_sensors-迭代验证可借鉴)）

### DexMani 过度工程化的具体表现

DexMani 多出的 ~2700 行代码（vs 参考项目平均 ~370 行）分布在以下层级，每层都有真实的维护代价：

| 层级 | 代码量 | 核心能力 | 代价 / 风险 |
|------|--------|---------|------------|
| **进程隔离** (arm_process + hand_process + shm/*) | ~1700 行 | fork + SHM seqlock，臂手崩溃不互伤 | Seqlock odd/even 协议脆弱，读写竞态难以复现调试；布局变更需 arm/hand/layouts 三方同步 |
| **EtherCAT 从站管理** (xhand.py ~600 行) | ~600 行 | AL state machine + SM-watchdog + stale-OP 恢复 | 需要 PREEMPT_RT 内核；SM-watchdog 超时是神秘故障源；两实例设备发现是绕过 raw socket 冲突的 workaround |
| **安全门 + validate** (validate.py) | ~158 行 | 5 道发送前门控 | 假阳性中断采集；臂手门耦合；每个新状态字段需要同步门控逻辑 |
| **RobotInterface 抽象** (interface.py) | ~500 行 | 唯一硬件访问路径 | 间接层增加调用链长度；新功能需先加到 interface 再透传，两步改法 |
| **全量状态监控** (types.py + layouts.py + recorder.py) | ~250 行 | 9 维状态字段 | 加字段 = 改 5 个文件；大量字段被录制但从不在控制循环中使用 |
| **合计** | **~3000 行**（曾 ~3100，已简化 -118） | | **是参考项目平均的 ~8 倍** |

**综合评估:**
- 进程隔离是 DexMani 架构中最有价值的部分（臂手不会同时崩溃），但也是最大的复杂度来源
- EtherCAT 从站管理在数据采集场景下边际收益有限 — DexScrew 同用 EtherCAT 但 288 行完成，依赖 SDK 默认行为足矣
- 安全门体系中，只有 NaN 消毒和关节限位在与 IK 管线配合时有明确的实用价值；温度门和扭矩门的实际触发频率需要统计验证
- 五个参考项目用 ~370 行平均代码量完成相同采集任务的事实，表明 DexMani 的多层抽象主要服务于工程美学而非功能需求

---

## 10. 可借鉴清单 (按优先级)

### P0 — 直接针对驱动板锁死 / 数据质量

| # | 模式 | 来源 | 适用性 | 说明 |
|---|------|------|--------|------|
| R0.1 | **每关节错误寄存器解码** | DexUMI `hand_api_cls.py:29-31` | ✅ 通用 | commboard/jointboard/tipboard err bitfield → 针对性恢复 |
| R0.2 | **reset_sensors() 软件偏置消除** | pi-r2-flow `xhand_robot.py:285-299` | ✅ 通用 | 取 5 帧平均补偿厂商残留 ~5-30N 偏置 |
| R0.3 | **reset_sensors() 迭代验证+选择性重试** | pi-r2-flow `xhand_robot.py:301-326` | ✅ 通用 | 只重试验证失败的手指，外层最多 5 次 |
| R0.4 | **错误白名单 + 计数 + 速率升级** | LeFranX `xhand.py:231-237` | ✅ 通用 | 已知传感器毛刺不中断采集，但累计超阈值告警 |
| R0.5 | **观测失败回退到已知正常值** | Dexora `xhand_forwarder.py:192` (改默认值) | ✅ 通用 | 用上次正常值或 home position，而非 zeros |
| R0.6 | **epsilon 门跳过冗余发送** | Dexora `xhand_forwarder.py:222-231` (仅 epsilon 部分) | ⚠️ 部分适用 | EtherCAT 不需要 min_interval 节流，但 epsilon 门仍可减少总线负载 |

### P1 — 安全加固

| # | 模式 | 来源 | 适用性 | 说明 |
|---|------|------|--------|------|
| R1.1 | **CRC 分类 + 退避重试** | Dexora `xhand_forwarder.py:244-273` | ⚠️ RS485 场景 | EtherCAT 硬件 CRC 不会透传到应用层。如未来降级到 RS485 则直接复用 |
| R1.2 | **预填充 HandCommand 复用** | LeFranX/pi-r2-flow | ✅ 通用 | 每帧只更新 position，减少 malloc |
| R1.3 | **可配置安全开关** | LeFranX `franka_fer_xhand_config.py:27-28` | ✅ 通用 | enable_torque_gate 等独立开关，不硬编码 |
| R1.4 | **`_require_connected()` 独立方法** | pi-r2-flow `xhand_robot.py:337-339` | ✅ 通用 | 消除 6+ 处内联 `if not self.connected` 检查 |

### P2 — 架构参考

| # | 模式 | 来源 | 说明 |
|---|------|------|------|
| R2.1 | **ZMQ 协议用于手部进程隔离** | Dexora `xhand_forwarder.py:282-297` | 仅 3 条指令的极简协议，适合低耦合 |
| R2.2 | **独立触觉 UART 路径** | DexUMI `xhand_tactile.py` | 触觉与电机控制解耦，互不干扰 |
| R2.3 | **多手双路 RS485 配置** | Dexora `config.yaml:20-51` | 双路手共享配置结构，可复用 |
| R2.4 | **LeRobot 标准接口** | pi-r2-flow `xhand_robot.py` | `connect/disconnect/get_observation/send_action/is_connected` |
| R2.5 | **`send_command` 响应夹带缓存状态** | pi-r2-flow `xhand_robot.py:206-209` | 零额外 RS485 往返获取读状态 |
| R2.6 | **RS485 端口级回退** | pi-r2-flow `xhand_robot.py:104-128` | 枚举 `/dev/ttyUSB*` 候选列表 |
| R2.7 | **Proprioceptive history buffer (当前+目标配对)** | DexScrew `xhand_deploy.py:64,96-105` | 策略可学习指令-执行动态，推断手-物接触 |

### P3 — 未来参考

| # | 模式 | 来源 | 说明 |
|---|------|------|------|
| R3.1 | **策略推理 decimation (step % N)** | DexScrew `xhand_deploy.py:238` | 计算量大时降低推理频率，其余步保持全速指令发送 |
| R3.2 | **归一化统计量嵌入 JIT 导出** | DexScrew `xhand_deploy.py:87-93` | 消除部署时 normalization stats 不匹配风险 |
| R3.3 | **非零 Kd 阻尼项** | DexScrew `xhand_deploy.py:121` | 对手指振荡可能有抑制作用，但缺乏对比验证 |

### 反向借鉴 — 从参考项目的简洁性中学习

DexMani 也可以从参考项目中学习"少即是多"：

| # | 参考模式 | 来源 | 说明 |
|---|---------|------|------|
| S1 | **单文件手部控制** | 全部 5 个项目 | 手部逻辑集中在一个 ~300 行文件中 → 新增字段/修改 PID/加功能只需改一处 |
| S2 | **无 SHM 直接调用** | LeFranX/DexUMI/DexScrew | 数据采集场景下 SDK 直接调用完全够用，进程隔离对采集不是刚需 |
| S3 | **固件级 tor_max 作为唯一硬件保护** | 全部 5 个项目 | 省去 per-joint Nm 门控 + 温度门的代码和维护 |
| S4 | **忽略非必要状态字段** | DexScrew (只读位置，288 行) | 采集 12-DOF 位置即可训练策略，力矩/温度/触觉/指尖位置对采集不是必选项 |
| S5 | **无 EtherCAT 状态管理** | DexScrew (288 行 EtherCAT) | SDK 默认行为处理了大部分 EtherCAT 细节，不需要应用层手动管理从站状态 |

这些不是建议 DexMani 删掉已有功能，而是在添加新功能时的校验标准：**"五个参考项目没有这个也能完成采集 — 我们真的需要它吗？"**

---

## 11. 文件索引

### LeFranX

| 文件 | 内容 |
|------|------|
| `src/lerobot/robots/xhand/xhand.py` | SDK 包装器 (386 行): connect/read_state/send_command/错误白名单/急停空桩 |
| `src/lerobot/robots/xhand/xhand_config.py` | 配置 (65 行): PID/tor_max/关节限位/控制频率 |
| `src/lerobot/robots/franka_fer_xhand/franka_fer_xhand.py` | 复合机器人 (332 行): 臂+手同步 send_action |
| `src/lerobot/robots/franka_fer_xhand/franka_fer_xhand_config.py` | 安全 flag 声明 (未实现): check_arm_hand_collision/emergency_stop_both |

### DexUMI

| 文件 | 内容 |
|------|------|
| `dexumi/hand_sdk/xhand/hand_api_cls.py` | SDK 包装器 (441 行): XhandSDK + ExoXhandSDK/JointState(含错误寄存器)/后台读线程 |
| `dexumi/encoder/xhand_tactile.py` | 独立触觉 UART 读取器 (145 行) |
| `dexumi/real_env/common/dexhand.py` | ZMQ DexServer/DexClient (286 行): 轨迹插值+状态流 |
| `real_script/open_server.py` | 手部进程入口 (126 行): 含 `inspire=hand` bug |

### Dexora

| 文件 | 内容 |
|------|------|
| `deploy/xhand_forwarder.py` | ZMQ Forwarder (414 行): 关节钳制/NaN消毒/CRC重试/发送节流epsilon门 |
| `deploy/mmk_xhand_config.yaml` | 共享配置 (77 行): 所有可调参数 |
| `teleop/teleop_pkg/receive_from_vision_pro.py` | 遥操录制 (153 行): *绕过 forwarder 所有安全门* |
| `teleop/teleop_pkg/config.yaml` | 遥操配置 (51 行): 双路 RS485/认证(含未启用的 EtherCAT 占位) |
| `deploy/dexora_inference_zmq.py` | 策略推理 (488 行): ZMQ REQ 客户端，5s 超时 |

### pi-r2-flow

| 文件 | 内容 |
|------|------|
| `deployment/mindex/robots/xhand_robot.py` | SDK 包装器 (339 行): **LeRobot 标准接口**/RS485+EtherCAT 双协议/端口回退/预填充 HandCommand/cached obs/`reset_sensors()` 迭代验证+偏置消除(**五项目最优**) |
| `deployment/apps/run_policy.py` | 主控制循环 (1101 行): 策略推理+手部控制编排/`_FakeHand` dry-run stub/`_build_flat_state()` |
| `deployment/mindex/policy/control_utils.py` | 控制工具 (192 行): `clip_to_state()`(存在但主循环不调用)/EMA/action ensemble/interpolation |
| `deployment/mindex/policy/groot_client.py` | GR00T 推理客户端 (521 行): ZMQ REQ/state keys 映射 |
| `deployment/mindex/recording/dataset.py` | HDF5 录制 (153 行): LeRobot 格式/含 fingertip_force + xhand_tactile |
| `deployment/apps/_policy_args.py` | CLI 参数 (148 行): `--protocol RS485\|EtherCAT --hand-mode absolute\|delta --rate-hz 25` |
| `deployment/scripts/render_episode_modalities.py` | 可视化 (156 行): 手部力矩+指尖力 MP4 渲染 |
| `deployment/scripts/render_finger_focus.py` | 聚焦可视化 (137 行): 单指力+动作 DOF 面板 |

### DexScrew

| 文件 | 内容 |
|------|------|
| `xhand-deploy/xhand_deploy.py` | 真机部署脚本 (**288 行**，六项目最精简): XHandControl 包装器/EtherCAT+RS485 双协议/JIT 模型加载+归一化/30 步 proprioceptive history buffer/关节索引重映射/策略推理 decimation/无 safety gates/无 cleanup |
| `dexscrew/tasks/xhand_hora.py` | IsaacGym 任务定义 (1317 行): HoRA teacher-student 训练/力矩级 PD 控制 (`p_gain=3, d_gain=0.01`)/domain randomization/thumb joint regularization mask |
| `student_eval.py` | 学生策略评估入口 (173 行): Hydra 配置/ProprioAdapt 加载/JIT 导出 |
| `train.py` | 训练入口 (124 行): Hydra/PPO/ProprioAdapt/wandb |
| `configs/task/XHandHora*.yaml` | 任务配置: action_scale/controller gains/obs noise/domain rand |
| `configs/train/XHandHora*.yaml` | 训练配置: PPO params/ProprioAdapt params |
| `assets/xhand_left/urdf/xhand_left.urdf` | XHand 左手 URDF: 关节名/限位/惯性参数（关节序与 SDK 不同 → 需要索引重映射） |

### DexMani

| 文件 | 内容 |
|------|------|
| `dexmani_real/robot/xhand/xhand.py` | SDK 包装器 (**1204 行**，六项目最大): EtherCAT 从站状态管理/AL state machine/SM-watchdog/stale-OP 恢复/两实例设备发现/触觉传感器 reset+偏置消除/分级自愈 T1-T3 |
| `dexmani_real/robot/hand_process.py` | 手部进程隔离 (**1251 行**): HandControlProcess/HandSHMFacade/fork 子进程/seqlock SHM 读写/臂手协调 |
| `dexmani_real/robot/validate.py` | 安全门 (**158 行**): error→connection→NaN→torque→temp 五道发送前验证 |
| `dexmani_real/robot/interface.py` | RobotInterface 抽象层: 统一硬件访问/状态路由/action 分发 |
| `dexmani_real/robot/types.py` | RobotState/RobotAction 类型定义 |
| `dexmani_real/shm/robot_ring.py` | SeqlockRingBuffer (**281 行**): odd/even torn-read 协议 |
| `dexmani_real/shm/robot_layouts.py` | SHM dtype 布局 (**101 行**): numpy 结构化数组定义 |

> **代码量对比:** DexMani 的 XHand 代码 (~2550 行核心，曾 ~3100，2026-08-02 简化 -118 行) 仍超过五个参考项目之和 (~1868 行)。差异主要来自进程隔离和 EtherCAT 从站管理，而非手部控制本身。

---

## 12. DexMani 独有特性：五个参考项目完全不存在的设计

> 本节基于对 LeFranX、DexUMI、Dexora、pi-r2-flow、DexScrew 的全部手部控制代码的逐行审查，筛选出 DexMani 中**五个项目完全没有**的 17 项特性。每项特性的评估均基于数据采集场景，而非 7x24 生产部署。

### 12.1 总览

| # | 特性 | 类别 | 必要性评级 | 状态 | 一句话：为什么只有 DexMani 有 |
|---|------|------|-----------|------|----------------------------|
| 1 | Fork 进程隔离 (手部独立子进程) | 进程隔离 | 有益 | 保留 | 五个项目手部全在主线程，不需要进程隔离 |
| 2 | Seqlock odd/even 防撕裂协议 | 状态监控 | 有益 | 保留 | 五个项目无 SHM，都在进程内直接调用 SDK |
| 3 | 两阶段 EtherCAT 打开 (独立实例) | EtherCAT | 必要 | 保留 | 五个项目要么不用 EtherCAT，要么单实例打开 |
| 4 | EtherCAT stale-OP 恢复 | EtherCAT | 必要 | 保留 | 五个项目要么 RS485 要么不处理从站状态 |
| 5 | 两阶段断开 (INIT 请求+close+watchdog 等待) | EtherCAT | 有益 | 保留 | 五个项目要么无断开逻辑，要么只 close 不等待 |
| 6 | ~~Deadband 卡死防护~~ | 安全 | 必要 | **已删** | 随 deadband 一起删除 — 去掉 deadband 后每帧必发，无卡死场景 |
| 7 | ~~逐步 delta 钳制 (0.3 rad/step)~~ | 安全 | 必要 | **已删** | 五个项目均无；钳制导致 homing 无法收敛；2026-08-02 移除 |
| 8 | 板级错误瞬态自动清除 | 错误恢复 | 有益 | 保留 | 五个项目不区分瞬态板错误与持久通信错误 |
| 9 | 分级双重看门狗 (send 错误+qpos 僵死) | 错误恢复 | 必要 | 保留 | 五个项目无分级恢复，最多 CRC 重试或放弃 |
| 10 | ~~板级错误逐帧升级跟踪~~ | 错误恢复 | 有益 | **已删** | per-frame throttled warning 保留，去掉了 10 帧累积升级 |
| 11 | 宏 RPC 系统 (含 settle-to-target 诊断) | 架构 | 有益 | 保留 | 五个项目直接调用 SDK 函数，不需要 RPC |
| 12 | 手部子进程 Estop 抢占 | 安全 | 必要 | 保留 | 五个项目无子进程，estop 在主线程处理 |
| 13 | ~~C stdout 抑制~~ | 其他 | 鸡肋 | **已删** | 纯 cosmetic，DexUMI/pi-r2-flow 都不抑制 |
| 14 | 手部运动学 FK (世界系指尖位置) | 架构 | 有益 | 保留 | 五个项目只返回原始关节角 |
| 15 | 手部错误与臂部验证门控解耦 | 安全 | 有益 | 保留 | 五个项目要么全耦合要么无验证 |
| 16 | ~~SHM 命令 Producer-ID 门控~~ | 架构 | 有益 | **已删** | 永远 PRODUCER_TELEOP=1，无多 producer 场景 |
| 17 | 类型化 numpy dtype SHM schema | 架构 | 有益 | 保留 | 五个项目无 SHM，用 dataclass/dict 传状态 |

> **2026-08-02 简化:** 17 项独有特性中已删除 5 项（#6 deadband 卡死防护、#10 板级错误升级、#13 C stdout 抑制、#16 Producer-ID 门控、以及未列入此表的 deadband 节流本身），净减少 118 行。详见 [§13 已执行的简化](#13-已执行的简化-2026-08-02)。

**评级说明:**
- **必要 (Essential)**: 缺失会导致数据采集失败或产生不可检测的损坏数据。6 项。
- **有益 (Beneficial)**: 提升安全性/可靠性/数据质量，但存在替代方案或不上也不会直接失败。10 项。
- **鸡肋 (Marginal)**: 不影响采集，纯 cosmetic 或极少触发的边界情况。1 项。

---

### 12.2 进程隔离类

#### 12.2.1 Fork 进程隔离 (手部独立子进程)

**是什么:** 手部在 crash-isolated fork 子进程中运行，通过 SHM ring buffer 与主进程通信。非 daemon 子进程在父进程死亡后通过孤儿退出 (`os.getppid()==1`) 自毁。SIGINT 保持位置不断电。

**代码位置:** `hand_process.py:200-257,1037-1053`

**为什么五个参考项目没有:**
五个项目手部控制全在主线程同步调用 SDK（LeFranX/DexScrew 主线程直调，DexUMI 后台读线程但不隔离，Dexora ZMQ 隔离但手部 forwarder 是独立脚本非 fork）。进程隔离的前提是有独立的手部控制循环和 SHM 通信层，这五个项目都没有。

**必要性评级: 有益**

**诚实评估:** 进程隔离是 DexMani 架构中最"看起来厉害"但数据采集场景下边际收益最低的投入。五个项目主线程调 SDK 采集了数千 episode，手部崩溃的概率极低（XHand SDK 的 segfault 在实际使用中几乎从未发生）。DexMani 投入 ~1700 行代码（arm_process + hand_process + shm/*）主要换取了一个从未触发的故障场景的防护。此外，进程隔离引入了自己的故障模式：seqlock 竞态难以调试、SHM 布局变更需三方同步、子进程僵尸化需孤儿退出机制。

**如果重来:** 对纯数据采集场景，**不隔离**。手部和臂部在同一进程，SDK 直调。代码量从 ~3000 行降至 ~800 行（参考 DexScrew 288 行 + DexMani 必要的安全门）。进程隔离的价值仅在 7x24 无人值守部署中体现，那时手部崩溃后自动恢复而不中断整个系统的需求才成立。

#### 12.2.2 手部子进程 Estop 抢占

**是什么:** Estop 在子进程 tick loop 中**第一个**检查。一次性 detorque + latch。新命令清除 estopped 标志。宏执行在各步骤间检查 estop_event 以尽快释放 macro_lock。

**代码位置:** `hand_process.py:887-895,760-761,819-822`

**为什么五个参考项目没有:**
Estop 抢占是进程隔离的衍生品 — 没有手部子进程，自然没有子进程级 estop 抢占。五个项目的 estop（如果有的话）在主线程中处理。

**必要性评级: 必要**

**诚实评估:** 这是进程隔离**内部**的必要特性 — 如果你决定 fork 子进程，子进程内必须有 estop 抢占，否则 SIGINT 后子进程继续发指令。但如果不用进程隔离，这个特性就不需要了。它是为解决自己创造的问题而存在的。

**如果重来:** 如果保留进程隔离，必须保留此项。如果去掉进程隔离，整项随之消失。

#### 12.2.3 手部错误与臂部验证门控解耦

**是什么:** `validate_action` **仅**门控臂部错误。手部错误故意排除在外：独立子系统有独立恢复机制。阻塞臂部等待瞬态手部板卡 glitch（33ms 后自动清除）不必要地冻结臂部。

**代码位置:** `validate.py:41-59`

**为什么五个参考项目没有:**
LeFranX `FrankaFERXHand` 显式耦合错误通过 `emergency_stop_both` 标志。其他四个项目要么单设备无耦合问题，要么根本没有验证门控。DexMani 的解耦设计来自真机运维经验：手部板错误瞬态高频出现（每秒数次），如果与臂部耦合会导致臂部频繁停顿。

**必要性评级: 有益**

**诚实评估:** 这是正确的工程决策但不是数据采集的刚需。五个项目靠固件级 tor_max 和偶尔的人工重启完成采集。DexMani 从真机运维中发现的"手部瞬态错误耦合臂部导致采集中断"是真实问题，解决方案也正确。但对数据采集，即使耦合了，后果也只是采集暂停几秒，操作员注意到后重试 — 不是安全问题。

---

### 12.3 EtherCAT 管理类

#### 12.3.1 两阶段 EtherCAT 打开 (独立 XHandControl 实例)

**是什么:** 设备枚举和打开使用**不同的** XHandControl 实例。阶段 1：临时 control 发现设备后关闭。阶段 2：每轮重试创建全新 control。防止 fork 进程中 stale raw socket 导致 SDO 路由失败。

**代码位置:** `xhand.py:288-398`

**为什么五个参考项目没有:**
pi-r2-flow/DexScrew 都用 EtherCAT 但单实例 `enumerate_devices` 后同一实例 `open_ethercat`。不需要两阶段是因为它们没有进程隔离 — 简单的 `enumerate→open` 在主线程中完全正常。DexMani 之所以需要是因为 fork 后 raw socket 句柄状态不确定。

**必要性评级: 必要**

**诚实评估:** 和 estop 抢占一样，这是为解决进程隔离引入的问题而存在的。如果不用 fork，单实例 `enumerate→open` 完全够用（DexScrew 288 行就是这么干的）。它的"必要"是进程隔离的连锁代价。

#### 12.3.2 EtherCAT stale-OP 恢复 + 恢复后稳定等待

**是什么:** 首次 EtherCAT 重试失败后等待 3.0s 让从站 SM-watchdog 因不洁断开而过期。重试 >1 成功后额外 1.0s 稳定延迟让 SDO 毛刺恢复。

**代码位置:** `xhand.py:344-353,369-371`

**为什么五个参考项目没有:**
pi-r2-flow/DexScrew 虽有 EtherCAT 但从站卡 OP 后无法重连时直接崩溃或人工重启。它们不处理"上次进程被 kill -9 后从站还卡在 OP"的场景，因为它们的进程要么正常运行，要么被 Ctrl-C 后手动重启脚本（重启 SDK 会处理）。DexMani 的自动恢复需求又来自进程隔离 — fork 子进程可能被 kill 而主进程仍在运行，需要自动恢复。

**必要性评级: 必要**

**诚实评估:** 对数据采集场景，如果不用进程隔离，操作员 Ctrl-C 整个脚本再重启即可 — 不需要自动 stale-OP 恢复。进程隔离场景下这个特性是必要的，问题是你是否需要进程隔离。

#### 12.3.3 两阶段断开 (INIT 请求 + close + SM-watchdog 等待)

**是什么:** 断开：`set_firmware_state(INIT)` + `close_device()` + 2.0s watchdog 等待。防 post-fail double-close。从站可无断电重连。

**代码位置:** `xhand.py:676-699`

**为什么五个参考项目没有:**
LeFranX 无 SDK close。pi-r2-flow 被动模式无断开。DexScrew 无任何 disconnect — Ctrl-C 后手停在 Mode 3。

**必要性评级: 有益**

**诚实评估:** 断开前设 INIT 让从站从 OP 优雅退出是好习惯，可避免不洁断开后下一次打开需要 stale-OP 恢复。但数据采集场景下，99% 的断开就是 Ctrl-C 结束脚本 — 此时 Python 进程直接终止，这些 cleanup 代码根本跑不到。实际生效的场景只有正常退出（流程走完 `atexit`/`finally`）。投入产出比一般。

---

### 12.4 安全类

#### 12.4.1 Deadband 卡死防护 (init 后回读验证)

**是什么:** `init` 设置 `last_qpos_cmd` 后，立即回读硬件实际 qpos。如果 delta >0.05 rad，将 `last_qpos_cmd` 修正为实际位置。防止 deadband 冻结：每个 `send_action` 都因"delta 小于阈值"而跳过，永不对硬件写任何值。

**代码位置:** `xhand.py:425-445`

**为什么五个参考项目没有:**
五个项目**没有 deadband 节流机制**，每帧都发 `send_command`，所以不需要防 deadband 卡死。Dexora 的 epsilon 门是发送节流（避免重复发相同指令），不是 DexMani 的 deadband 跳过。DexMani 自己引入了 deadband（为了减少 EtherCAT 负载），又自己加了防卡死保护 — 这是解决自创问题的典型案例。

**必要性评级: 必要**

**诚实评估:** 必要性评级"必要"仅因为 DexMani 有 deadband — 如果去掉 deadband（像五个项目一样每帧无脑发），这个保护就不需要了。对 EtherCAT 而言，每帧 16Hz 发送完全不会造成总线瓶颈（100Mbps vs 12 个 float = 48 bytes），deadband 节省的带宽可以忽略不计。所以真正的问题不是"这个保护是否必要"，而是"deadband 本身是否必要"。

#### 12.4.2 逐步 delta 钳制 (max_delta_rad) — **已于 2026-08-02 移除**

**曾是什么:** 每步指令 delta 钳制到 max_delta_rad (0.3 rad)。Mode 3 PID 无固件轨迹规划，大跳变直接送电机 PID 以全扭矩执行。

**代码位置:** `xhand.py:184` (`max_delta_rad` 默认值改为 `None`，`send_action()` 中的钳制逻辑在 `max_delta_rad is None` 时跳过)

**移除原因:**
- 五个参考项目均无 delta 钳制，依赖遥操数据天然平滑 + 固件 tor_max 过流保护，采集数千 episode 无事故
- delta 钳制导致归位 (home) 无法收敛：`send_action(home_qpos)` 只调用一次时，钳制将首次指令限制为 0.3 rad 步进，后续轮询不再发送新指令，手部永远无法到达 home_qpos
- `max_delta_rad` 默认值已改为 `None`（禁用），保留配置项供未来按需启用

---

### 12.5 错误恢复类

#### 12.5.1 板级错误瞬态自动清除

**是什么:** `error_state` 每次 `get_state()` 从 commboard/jointboard/tipboard_err 寄存器重新计算。硬件恢复正常时自动清除。与持久性 send/read 错误（通过熔断器跟踪）形成对比。

**代码位置:** `xhand.py:827-838`

**为什么五个参考项目没有:**
五个项目不区分瞬态板错误和持久通信错误。LeFranX/DexUMI 只在单次 `read_state` 时报告错误 — 下一次读如果硬件恢复了，错误就消失了（但它们没有明确的设计意图利用这一点）。DexMani 显式区分瞬态（自动清除）和持久（熔断器累计），是更精细的错误分类。

**必要性评级: 有益**

**诚实评估:** 好的设计但不是数据采集的刚需。操作员看到错误日志后自然会判断是瞬态还是持久 — 不需要代码代劳。对 7x24 无人值守部署是必要特性。

#### 12.5.2 分级双重看门狗 (send 错误 + qpos 僵死)

**是什么:** 两套独立看门狗。(A) send-error: Tier 1=30 帧时 clear_error（`_hand_child_main` 旧架构），带冷却时间。(B) qpos-stale: 检测成功发送但 ESC 缓存冻结的静默驱动板锁死。Tier 1=0.5s clear_error。熔断器计数器跟踪连续错误。

> **2026-08-03 更新:** `reset_connection()` (Tier 2) 已移除。该方法是零调用者的死代码——没有一个看门狗路径实际调用它。参考项目均依赖操作员手动重启，运行时自动重连在采集场景中既无必要也存在风险（disconnect 期间 Mode 3 扭矩释放导致手指漂移）。

**代码位置:** `hand_process.py:953-1030; xhand.py:215,892,896`

**为什么五个参考项目没有:**
五个项目无分级多级恢复。Dexora CRC 重试仅每次发送层面。pi-r2-flow send 失败仅 print warning 继续。**最关键的缺失是 qpos-stale 检测** — 没有参考项目能检测到"send_command 返回成功但数据是旧的"这种静默驱动板锁死。这是一个真实硬件故障模式（ESC 缓存冻结），未被任何参考项目覆盖。

**必要性评级: 必要**

**诚实评估:** qpos-stale 检测是 DexMani 独有特性中**数据可靠性价值最高**的一个。静默驱动板锁死在真机运行中确实发生过（ESC 缓存冻结时 send 返回成功但 qpos 不更新）— 五个参考项目对此完全盲视，会静默录制损坏数据。send-error 分级恢复的价值略低（Dexora CRC 重试已覆盖大部分 RS485 噪声场景）。**保留 qpos-stale 检测，send-error 分级可简化为单级 clear_error + 人工介入。**

#### 12.5.3 板级错误逐帧升级跟踪 (10 帧阈值)

**是什么:** 三通道板错误 (commboard/jointboard/tipboard) 的逐帧计数器。10 帧升级：一次性告警含关节索引。恢复检测并日志记录。阈值以下：限流告警 (5s 间隔)。

**代码位置:** `hand_process.py:560-595`

**为什么五个参考项目没有:**
五个项目要么不读错误寄存器，要么读了不检查。LeFranX/DexUMI 有 per-joint 错误寄存器字段但不做累积跟踪 — 瞬态错误被当作每帧独立事件处理，无法区分"偶尔毛刺"和"趋势恶化"。

**必要性评级: 有益**

**诚实评估:** 对运维诊断有价值但不是数据采集的刚需。操作员如果看到频繁错误告警自然会注意。10 帧累积升级阈值是一个好的自动化诊断设计，但数据采集场景下操作员在场，自动化诊断的边际价值低于无人值守场景。

---

### 12.6 状态监控类

#### 12.6.1 Seqlock odd/even 防撕裂协议

**是什么:** 无锁 seqlock 覆盖 SeqlockRingBuffer 和 CameraRingBuffer。写前 odd (2\*seq-1) 标记，写后 even (2\*seq) 标记。读者前后采样 seq，仅当相等+非零+偶数时接受。重试+last-good 回退。CameraRingBuffer 增加 RGB/深度 size/shape 防撕裂守卫。create_or_replace 自动 unlink 残留环。

**代码位置:** `robot_ring.py:36-53,134-247; ring_buffer.py:369-545`

**为什么五个参考项目没有:**
五个项目无 SHM，都在进程内用 threading.Lock（pi-r2-flow）、deque+Lock（DexUMI）、或 ZMQ 消息（Dexora）传递状态。Seqlock 是为零拷贝 SHM 跨进程通信设计的，仅当你有 fork 进程隔离时才需要。

**必要性评级: 有益**

**诚实评估:** 又是进程隔离的连锁代价。Seqlock 协议本身设计精巧（odd/even 标记、重试+last-good 回退），但在数据采集场景下，如果不用进程隔离，threading.Lock + 共享 numpy 数组在 16Hz 下完全足够。Seqlock 的 odd/even 协议在极端竞态下仍有残余 torn-read 可能性（理论上需要 memory barrier，Python 无此保证），调试极端难复现。**不必要 — 如果去掉进程隔离。**

---

### 12.7 架构类

#### 12.7.1 宏 RPC 系统 (含 settle-to-target 诊断)

**是什么:** 4 个宏 (RESET/STOP/CLEAR_ERROR/SEND_TRAJECTORY)，通过专用 SHM ring 通信，带 seq 关联。RESET: 迭代 settle (容差 0.06 rad, 3s 超时)，幽灵通信守卫 (5 次连续 send 失败)，超时时 per-joint 收敛诊断。STOP: 仅显式触发，从不自动。

**代码位置:** `hand_process.py:752-829,478-508`

**为什么五个参考项目没有:**
五个项目直接调 SDK 函数 — `reset_sensors()`, `send_command()` 等。不需要 RPC 因为函数调用在同一进程/线程内。Dexora 的 ZMQ command dispatch (obs/action/ping) 最接近但仅传输数据而非生命周期宏含 settle 诊断和结果关联。

**必要性评级: 有益**

**诚实评估:** Settle-to-target 诊断在调试时极有价值（知道哪个关节没收敛），但这是开发/调试工具不是数据采集的运行时需求。生产级 settle 验证 + 幽灵通信守卫的设计质量很高，但 RPC 通信层本身（seq 关联、专用 ring、宏锁）是 SHM 跨进程的必要开销 — 如果不用进程隔离，直接函数调用即可，不需要 RPC。

#### 12.7.2 Producer-ID 门控

**是什么:** `hand_cmd` 和 `arm_target` ring 携带 producer_id (1=teleop/2=replay/3=policy)。子进程拒绝不匹配的 producer，限流告警。防止多控制源的交叉污染。

**代码位置:** `hand_process.py:903-911; robot_layouts.py:35-37,62-64,118`

**为什么五个参考项目没有:**
五个项目无多 producer SHM 架构。teleop 和 replay 是不同的脚本入口，不会同时运行。DexMani 的 producer-ID 门控再次是为多 producer 共享 SHM 场景设计的 — 这在数据采集中不太可能发生（操作员不会同时运行 teleop 和 policy 推理）。

**必要性评级: 有益**

**诚实评估:** 这是预防一个不太可能发生的场景（多控制源同时运行）。如果架构简化为单进程，这个门控整层消失。代码量小（~10 行），维护成本低，但解决的问题的优先级很低。

#### 12.7.3 类型化 numpy dtype SHM schema

**是什么:** 10 个类型化 dtype schema: ARM_STATE(7-DOF+flags), ARM_TARGET(hold+pid), ARM_CMD(2048 waypoints), HAND_STATE(12-DOF+tactile 5x120x3+3 board error channels+qpos_stale+echo), VR_FRAME, CAMERA_FRAME_HEADER。所有字段有显式 shape 和对齐。

**代码位置:** `robot_layouts.py:45-120; layouts.py:20-68`

**为什么五个参考项目没有:**
SHM 本身是 DexMani 独有的。五个项目用 dataclass/dict 在进程内传状态。

**必要性评级: 有益**

**诚实评估:** numpy dtype 作为 SHM schema 是自文档化且类型安全的 IPC 合约 — 这是好的工程实践。但同样，它仅存在于 SHM 跨进程通信的需求下。去掉进程隔离后，普通的 Python dataclass 足够。

#### 12.7.4 手部运动学 FK (世界系指尖位置)

**是什么:** 串联 T_world_eef \* T_eef_handbase \* tip_in_handbase 通过 URDF HandKinematics 计算 5 指世界系位置。每步 NaN 守卫。存入 RobotState.fingertip_pos 供录制。

**代码位置:** `interface.py:607-632`

**为什么五个参考项目没有:**
五个项目只返回原始关节角，不计算手部正运动学。pi-r2-flow 录制了 `fingertip_force` (触觉) 但不计算指尖空间位置。

**必要性评级: 有益**

**诚实评估:** 世界系指尖位置对策略学习有价值（空间推理，如"指尖离物体多远"），但对数据采集不是必选项 — 可以离线后处理计算（已知臂 EEF 姿态+手关节角+URDF）。在线计算的好处是录制即得，省去离线计算步骤。成本低（~25 行 NaN-guarded FK 链式乘法），纯收益无副作用。**保留，无论是否简化架构。**

---

### 12.8 其他

#### 12.8.1 C stdout 抑制 (reset_sensor 时)

**是什么:** C SDK 每次 `reset_sensor` 打印 "Unknow Cmd!" 到 stdout。dup2 重定向 fd1 到 /dev/null 在传感器复位期间，保留 fd2 给 Python logging。

**代码位置:** `xhand.py:480-502`

**为什么五个参考项目没有:**
DexUMI/pi-r2-flow 也调 `reset_sensor` 但不抑制 C stdout — 终端会刷 "Unknow Cmd!" 但不影响功能。DexScrew 完全无 reset_sensor。

**必要性评级: 鸡肋**

**诚实评估:** 纯 cosmetic。不影响功能，不影响数据质量，不影响安全性。唯一作用是终端干净一点。五个参考项目终端刷 "Unknow Cmd!" 照样正常工作。**可以删。**

---

### 12.9 如果重来：纯数据采集系统的设计

假设目标是构建一个**仅服务于数据采集**的手部控制系统（遥操录制 episode，不做 7x24 部署），以下是基于对五个参考项目的审查和 DexMani 的真机经验的"最优简化"设计。

#### 保留（12 项 → 保留 6 项）

| 保留特性 | 原因 |
|---------|------|
| **qpos-stale 检测** (12.5.2 的 B 部分) | 静默驱动板锁死会损坏录制数据且无法事后检测。保留 B 看门狗，A 看门狗简化为单级 |
| **板级错误瞬态自动清除** (12.5.1) | 不增加复杂度（每次 get_state 重新算即可），消除瞬态毛刺告警噪音 |
| **手部 FK 世界系指尖位置** (12.7.4) | 25 行，纯收益无副作用。即使不在线算也要离线算，不如录制时直接写入 |
| **手部错误与臂部解耦** (12.2.3) | 正确的工程决策，避免臂部因手部瞬态 glitch 频繁停顿 |
| **EtherCAT stale-OP 恢复** (12.3.2) | 用了 EtherCAT 就需要。即使主线程模式，脚本 Ctrl-C 重启也需要从站恢复 |
| **两阶段断开** (12.3.3) | 十几行 `set_firmware_state(INIT)` + wait，避免不洁断开。但如果在 `atexit` 中注册，正常退出和 Ctrl-C 都能走到 |

#### 删除/简化（12 项 → 移除 11 项）

| 移除特性 | 替代方案 | 省下代码量 |
|---------|---------|-----------|
| **逐步 delta 钳制** (12.4.2) | 五个参考项目均无，遥操数据天然平滑 + 固件 tor_max 足够 | ~15 行 |
| **Fork 进程隔离** (12.2.1) | 主线程直调 SDK | ~1700 行 |
| **Seqlock odd/even 协议** (12.6.1) | 无 SHM → 不需要 | ~400 行 |
| **两阶段 EtherCAT 打开** (12.3.1) | 无 fork → 单实例 enumerate→open 即可 | ~100 行 |
| **宏 RPC 系统** (12.7.1) | 直接函数调用 stop/reset/clear_error | ~300 行 |
| **Producer-ID 门控** (12.7.2) | 单 producer → 不需要 | ~10 行 |
| **类型化 numpy dtype SHM** (12.7.3) | Python dataclass | ~100 行 |
| **Estop 抢占** (12.2.4) | 无子进程 → 不需要 | ~40 行 |
| **Deadband 卡死防护** (12.4.1) | 去掉 deadband 本身（EtherCAT 不需要节省带宽） | ~30 行 |
| **Send-error 分级看门狗** (12.5.2 A 部分) | 单级 clear_error + 人工介入 | ~100 行 |
| **板级错误逐帧升级** (12.5.3) | 简单 per-frame warning，不累积升级 | ~40 行 |
| **C stdout 抑制** (12.8.1) | 忍受 "Unknow Cmd!" 输出 | ~25 行 |

**简化后预计代码量: ~800 行**（vs 当前 ~3000 行，五个参考项目平均 ~370 行）

800 行 ≈ DexScrew 的 288 行（最精简 EtherCAT 项目）+ DexMani 保留的 7 项安全/诊断特性。比参考项目平均多出的 ~430 行主要来自：qpos-stale 检测、delta 钳制、FK 计算、stale-OP 恢复。这些是五个参考项目的真实能力缺口，不是纯粹过度工程。

#### 核心教训

**"为了解决一个自创问题而引入另一个自创问题"是 DexMani 架构膨胀的根本模式:**

```
Deadband 节省带宽 → Deadband 卡死 → 需要防卡死保护
Fork 进程隔离 → 需要 SHM → 需要 seqlock → 需要 stale-OP 恢复 → 需要两阶段打开 → 需要宏 RPC → 需要 Producer-ID 门控
```

每一步单独看都合理，串联起来就是 ~2300 行额外代码。五个参考项目的简单架构证明这些都不是数据采集的刚需。

**"为 7x24 部署做数据采集"是过度设计的根源。** 数据采集和无人值守部署是两个不同的需求剖面。在数据采集场景中：操作员在场、采集时长有限（几分钟到几十分钟）、出问题可当场重启。五大参考项目的 ~370 行平均代码量是对这个场景的正确回答。

**区分"好的工程实践"和"场景刚需"。** numpy dtype SHM schema 是好的工程实践，但数据采集不需要。逐帧错误升级跟踪是好的诊断设计，但数据采集不需要。这些特性在 7x24 部署中都是正确的投入 — 但不应该在数据采集阶段做。

---

## 13. 已执行的简化 (2026-08-02)

基于 §12 的分析，已从 DexMani 代码中移除以下 5 项低风险冗余。全部通过 mypy 类型检查（83 文件零错误）。

### 13.1 删除清单

| # | 删除项 | 文件 | 原因 |
|---|--------|------|------|
| 1 | **Deadband 节流** (0.001 rad skip) | `xhand.py` | EtherCAT 100Mbps 不需要带宽节省。五个参考项目（含 DexScrew EtherCAT）都每帧发送，无 deadband |
| 2 | **D11 卡死防护** (init 后回读验证) | `xhand.py` | 随 deadband 删除 — 此防护仅因 deadband 存在而需要。去掉 deadband 后每帧必发，不存在卡死场景 |
| 3 | **C stdout 抑制** (dup2 /dev/null) | `xhand.py` | 纯 cosmetic。DexUMI/pi-r2-flow 都不抑制，终端打印 "Unknow Cmd!" 不影响功能 |
| 4 | **板级错误升级跟踪** (10 帧阈值) | `hand_process.py` | 保留 per-frame throttled warning，去掉累积升级逻辑。操作员看到频繁错误自然会注意，无需代码代劳 |
| 5 | **Producer-ID 门控** + 死 dtype | `hand_process.py` + `robot_layouts.py` | 当前永远 `PRODUCER_TELEOP=1`，无多 producer 场景。`ARM_CMD_DTYPE`/`ARM_CMD_RESULT_DTYPE` 从未用于 ring 分配 |

### 13.2 效果

| 指标 | 简化前 | 简化后 |
|------|--------|--------|
| XHand 相关总行数 | ~3100 | ~2980 |
| `xhand.py` | 1250 | 1204 |
| `hand_process.py` | 1298 | 1251 |
| `robot_layouts.py` | 126 | 101 |
| 净删除 | — | **-118 行** |

### 13.3 保留（经评估不删）

| 特性 | 保留理由 |
|------|---------|
| qpos-stale 检测 | 静默驱动板锁死会损坏录制数据且无法事后检测 |
| 分级看门狗 (send 错误 30/90 帧) | 真机运行中驱动板锁死确实发生过，双级恢复是正确的 |
| 板级错误瞬态自动清除 | 每次 get_state 重算，零额外复杂度 |
| 手部错误与臂部解耦 | 避免臂部因手部瞬态 glitch 频繁停顿 |
| EtherCAT stale-OP 恢复 + 两阶段断开 | 用了 EtherCAT 就需要 |

### 13.4 未来可选简化（未执行）

以下项目在上文 §12.9 中讨论过，因涉及架构变更（需去 fork 进程隔离），风险较高，本轮未执行：

- **Fork 进程隔离移除**（~1700 行）— 需恢复 XHand 在主进程直接实例化，影响所有入口点
- **Seqlock odd/even 协议**（~400 行）— 随 fork 移除而自然消失
- **宏 RPC 系统**（~300 行）— 随 fork 移除后可改为直接函数调用
- **两阶段 EtherCAT 打开**（~100 行）— 无 fork 后单实例 enumerate→open 即可

### 13.5 已执行：移除 delta 钳制 (2026-08-02)

基于用户决策：五个参考项目均无 per-step delta 钳制，且该钳制导致 hand homing 无法收敛（首次 `send_action` 只前进 0.3 rad，后续轮询不重发命令，手部永远到不了 home_qpos）。保持钳制需要所有归位代码路径重复发送命令 — 增加了不必要的复杂度。

**变更:**

| 文件 | 变更 |
|------|------|
| `xhand.py:184` | `max_delta_rad` 默认值 `0.3` → `None`（禁用）。保留配置项供未来按需启用 |
| `xhand.py:175-183` | 更新注释：说明默认关闭，引用五个参考项目均无此机制 |
| `hand_process.py` (2处) + `arm_process.py` (1处) | 更新 homing 收敛注释，移除 `max_delta_rad=0.3` 引用 |

**`send_action()` 中的钳制逻辑** (`xhand.py:833-837`) 保留不动 — `max_delta_rad is None` 时自动跳过，无需删除代码。

---


## 附录 A：EtherCAT vs RS485 详解

### 物理层差异

| 维度 | RS485 | EtherCAT |
|------|-------|----------|
| **物理介质** | USB-to-RS485 转换器 (FTDI 芯片) | 专用 Intel 以太网 NIC (如 i210/i225) |
| **带宽** | 3 Mbps | 100 Mbps |
| **延迟** | 毫秒级，非确定性 | 微秒级，硬实时确定性 |
| **总线拓扑** | 点对点 (每只手一个 USB 口) | 一线多从站 (臂+手+传感器可共享总线) |
| **所需内核模块** | 标准 CDC-ACM (内核自带) | IgH EtherCAT master 或 SOEM 栈 |
| **内核要求** | 无 | PREEMPT_RT 实时内核（否则 SM-watchdog 频繁超时） |
| **即插即用** | 换台机器就能用 | 需要配置 master + 专用 NIC |
| **固件** | XHand RS485 固件 | XHand EtherCAT 固件 (不同构建) |
| **典型故障** | CRC 噪声、总线竞争 | SDO 写入失败、从站状态卡死 |

### 五个参考项目选择 RS485 (或 EtherCAT) 的原因

**多数项目选 RS485 的根本原因：省事。** RS485 插上 USB 线就能用，对学术研究和数据采集完全够用。

具体因素：

1. **硬件门槛** — EtherCAT 需要 Intel 专用网卡 + 实时内核，笔记本电脑无法即插即用
2. **固件默认** — XHand 出厂固件大概率是 RS485，EtherCAT 固件需另行刷写
3. **SDK 支持不完善** — 五个项目对 EtherCAT 的支持程度呈光谱：
   - LeFranX: `_connect_ethercat()` → `NotImplementedError`（完全未实现）
   - DexUMI: `open_ethercat()` 存在但结果只 print 不验证（半成品）
   - Dexora: `use_ethercat: false`，vendor_id/product_code 全为占位零值（从未启用）
   - pi-r2-flow: **EtherCAT 路径完整可用**（`xhand_robot.py:86-98`），是唯一提供可用 EtherCAT 备选的 RS485 项目
   - **DexScrew: 默认 EtherCAT**（`xhand_deploy.py:203`），也支持 RS485 fallback（baud=115200）
4. **使用场景** — 学术数据采集不需要微秒级确定性，RS485 足够的

### DexScrew 与 DexMani 的 EtherCAT 使用差异

DexScrew 和 DexMani 是六个项目中**仅有的两个默认 EtherCAT 的项目**，但代表了两种截然不同的设计哲学：

| 维度 | DexScrew | DexMani |
|------|----------|---------|
| 手部控制代码量 | **288 行** | **~3000 行**（曾 ~3100，-118） |
| EtherCAT 状态管理 | 无（依赖 SDK 默认行为） | 完整的 AL state machine (INIT→PRE_OP→SAFE_OP→OP) |
| SM-watchdog 处理 | 无 | 等待 2.0s + stale-OP 额外 3.0s |
| SDO 重试 | 无 | `open_ethercat_retries=2` |
| 两阶段设备发现 | 无 (同一实例 enumerate+open) | 不同 XHandControl 实例避免 raw socket 冲突 |
| 断连恢复 | 无 | `_request_slave_init()` + `set_firmware_state()` |
| Cleanup (OP→INIT) | 无 (Ctrl-C 后手停在 Mode 3) | `close_device()` 前请求从站回 INIT |
| 数据采集可用性 | ✅ (策略部署已验证) | ✅ (数千 episode 已验证) |

**关键问题:** DexScrew 288 行 EtherCAT 代码完成了手部控制 + 策略部署（仅在 Ctrl-C cleanup 上有缺陷）。DexMani 多出的 ~2800 行主要用于处理优雅降级场景（进程崩溃、SM-watchdog 超时、stale-OP 卡死）。问题在于：**这些场景在数据采集中有多频繁？如果大部份是处理 99.9 百分位极端情况，额外的代码是否值得？** 这是一个没有标准答案的工程取舍问题。

### 对借鉴的影响

| 借鉴模式 | RS485 下有效 | EtherCAT 下有效 | 说明 |
|----------|-------------|----------------|------|
| CRC 分类+退避重试 (Dexora) | ✅ | ❌ | EtherCAT 硬件 CRC 不透传到应用层 |
| min_interval 发送节流 (Dexora) | ✅ | ❌ | 100Mbps 无需人为限速 |
| epsilon 门跳过冗余发送 (Dexora) | ✅ | ✅ | 任何总线减少冗余帧都有益 |
| 错误寄存器解码 (DexUMI) | ✅ | ✅ | 错误寄存器是固件层，与物理层无关 |
| 错误白名单+计数 (LeFranX) | ✅ | ✅ | 传感器/温度错误与物理层无关 |
| reset_sensors 偏置消除 (pi-r2-flow) | ✅ | ✅ | 触觉传感器校准，与物理层无关 |
| reset_sensors 迭代验证 (pi-r2-flow) | ✅ | ✅ | 触觉传感器校准，与物理层无关 |
| 端口级回退 (pi-r2-flow) | ✅ (RS485) | N/A (EtherCAT 只有一个从站) | RS485 有多个 `/dev/ttyUSB*` 候选 |
| send_command 响应缓存读 (pi-r2-flow) | ✅ | ✅ | 零额外往返，通用优化 |
| Proprioceptive history buffer (DexScrew) | ✅ | ✅ | 策略输入机制，与物理层无关 |
| 策略推理 decimation (DexScrew) | ✅ | ✅ | 计算优化，与物理层无关 |
| 归一化嵌入 JIT 导出 (DexScrew) | ✅ | ✅ | 模型导出机制，与物理层无关 |
