# DexMani Real Repository Map

本文件只记录稳定的运行拓扑、数据流和安全边界。运行行为以源码、schema 与配置为准；
本文件不维护逐文件职责清单。新增、删除或移动文件时，仅在改变下列边界时更新本文件。

## 仓库导航

| 文件 | 作用 |
|---|---|
| `AGENTS.md` | 仓库级工程、安全、范围与验收契约。 |
| `CLAUDE.md` | Claude Code 精简入口，具体规则委托给 `AGENTS.md`。 |
| `code_style.md` | 本研究代码库的具体编码与审查约定。 |
| `README.md` | 面向使用者的能力、环境、工作流与稳定架构。 |
| `repo_map.md` | 当前运行拓扑、核心数据流与边界索引。 |
| `.codex/config.toml` | 项目级 Codex 权限、联网与子智能体并发配置。 |

## Process topology

```text
hardware SDKs
    │
    ▼
arm / hand / VR / camera / point-cloud / recorder workers
    │ typed rings, queues, flags and lifecycle state
    ▼
RuntimeChannels
    ├── teleop session + control loop
    ├── policy inference worker → prediction ring → policy executor
    └── replay / calibration control owner
    │
    ▼
control safety gate → command publication → arm / hand workers
```

- 每个硬件 SDK 只在其 owning worker 或 driver 内创建和使用；父进程只负责生命周期与监督。
- `RuntimeChannels` 是跨进程状态的唯一 allocation owner；固定 wire shape、dtype 和持久化
  record layout 由 `dexmani_real/ipc/schema.py` 定义。
- learned-policy lifecycle 先等待 inference restore/warmup，再启动所需传感器和执行器 worker；
  supervisor 只监督实际运行的 process heartbeat，readiness 只负责有界启动等待。
- teleop lifecycle 按启用的 arm、hand、VR、camera、recorder worker 构造同一组 channels；
  replay 与 calibration 复用相同的 safety、publication 和 worker 边界。

## Policy flow

```text
causal observation history
    → Policy public runtime
    → Real NumPy adapter
    → flat Prediction IPC record [N, D] + provenance
    → one-slot latest-wins prediction_ring
    → PolicyExecutor (sync/async timing, endpoint decode, EE→IK)
    → physical SafetyGate
    → non-blocking coupled publication
    → arm / hand worker final checks
```

- `Prediction` 是唯一的内部策略输出对象；动作是拥有自身内存的 finite `float64[N, D]`，
  `run_generation`、source timestamp 和 logical-step timestamp 随对象传播。
- inference worker 只负责 observation、策略调用和 prediction 发布；PolicyExecutor 独占动作
  horizon 解码、control-grid 调度、EE→IK、候选校验和 command-progress watchdog。
- 正常策略 tick 的路径是 `validate → publish → continue`；`execute=False` 完成同样的候选
  校验但不产生 actuator side effect。需要确认 SDK 接受的 home、calibration 和 replay 操作，
  才显式调用 blocking acceptance。
- B 只在 ARMED、上一轮 S 已清理、physical home（物理运行）完成后进入 RUNNING；sync/async
  都不追赶过期 deadline，也不制造超过 `control_hz` 的 command burst。

## Teleop flow

```text
VR / keyboard input
    → teleop session lifecycle
    → control loop / causal fixed-grid tick
    → TeleopController + action proposal
    → SafetyGate + publication
    → arm / hand workers
    └→ fixed-grid record sample → RecorderIO
```

- `teleop/control_grid.py` 持有 proposal、EMA、hand ramp/retarget、IK hold 与上一 endpoint 等
  算法状态；loop 持有 pause、recording、keyboard 和退出编排状态。
- feedback 缺失、过期、断开或录制边界进入 pause boundary：先使 generation 失效并清除
  controller reference，恢复时必须收到 pause 之后的新鲜因果 arm/VR/hand feedback，再重锚并
  继续发布。静默期间不发布旧目标。
- teleop 和 replay 的动作决策都必须经过共享 publication/safety 边界；worker 侧再次检查
  state、generation、ticket、shape、finite、limits 和 freshness。

## Command safety invariants

- runtime safety state 只有 `DISARMED → ARMED → RUNNING` 与 `FAULT` 终态路径；共享的
  `motion_lock` 同时保护 state、`run_generation` 和 active command ticket。
- 每次开始、停止、暂停、故障或 supersede 都推进 `run_generation` 并清除 active ticket；
  stale、过期、被覆盖或不再拥有 latest-wins slot 的命令不得跨 SDK 边界。
- `SafetyGate` 校验 joint limits、workspace、collision 和 command delta；它不把 feedback
  缺失当作安全，也不以隐式 clip 代替 reject。arm/hand worker 保留最后一道硬件调用前守卫。
- STOP、e-stop、worker death、heartbeat timeout、command-progress stall 和 cleanup failure
  都 fail closed；home 仍执行 hand + arm 的显式碰撞检查路径。
- 命令发布与物理接受是两个边界：实时路径非阻塞发布，只有确实需要确认的 home、calibration
  和 replay 路径等待明确 acceptance result。

## Recording/Data boundaries

- teleop 只把已经选择并校验的 causal fixed-grid sample 交给 RecorderIO；RecorderIO 独占
  episode transaction、sidecar、sequence continuity、validation 和 atomic finalize，不决定
  机器人动作。
- raw episode 的 schema、字段语义和对齐保持单一来源：`recording/schema.py` 与
  [`docs/data_schema.md`](docs/data_schema.md)。当前链路为 raw v24 → processed HDF5 v11 →
  Policy Zarr v5；离线 `data/` 负责清洗、审计和导出，不改变 raw 字段含义。
- `recording/hdf5_writer.py` 独占单个 `data.h5` handle；camera sidecar 和 video writer 不
  反向拥有控制状态。缺口、失败或未完成 finalize 不伪装成完整 episode。
- 物理回放读取已发布的 raw command/provenance，并重新经过当前 runtime 的 preflight、safety、
  generation 与 worker 边界；processed 产物不能重新解释或替代 raw 命令事实。

源代码、schema 和 canonical config 是实现真相；本文件只帮助定位上述稳定边界。
