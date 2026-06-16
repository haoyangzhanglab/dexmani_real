# dexmani_real 项目总索引

> **Python 环境**：`source /home/zhy/anaconda3/etc/profile.d/conda.sh && conda activate real`（`/home/zhy/anaconda3/envs/real/bin/python`）
>
> **文件定位**：本文档是项目的入口索引。详细规范按关注点拆分到 `.claude/rules/` 目录；具体实施方案在 `docs/development_plan.md`。
>
> **规则优先级**：本文档与 `.claude/rules/` 冲突时以本文档为准。SDK 细节以 `.claude/rules/sdk-dependencies.md` 为准。

---

## 架构速览

```
VR 进程(80Hz) ──► ring["vr_frame"] ──┐
                                     ├── Controller 进程(50Hz)
Camera 进程(30Hz) ──► ring["camera"] ─┘     │
                                            ├─ _tick(): VR frame → IK(arm) + retarget(hand)
                                            │    arm: EMA(alpha=0.3)  hand: EMA(alpha=0.3)
                                            ├─ robot.send_action()
                                            └─ [RECORDING] recorder.add_frame()

Keyboard ──► multiprocessing.Queue ──► 控制信号(T/R/S/H/ESC)
```

**Episode 生命周期**: `robot.return_to_home()` → `arm_mapper.reset()` → `recorder.start_episode()` → [teleop] → `recorder.stop_episode()`

**状态机**: `IDLE ─T─► TELEOP ─R─► RECORDING ─S─► IDLE`（H=home, 超时→EMERGENCY_STOP）

---

## Rules 文件索引

| 关注点 | 文件 | 内容 |
|--------|------|------|
| **架构 + 模块职责** | [architecture.md](.claude/rules/architecture.md) | 数据流、生命周期、状态机、模块职责边界、坐标系约束 |
| **硬件驱动接口** | [hardware-interface.md](.claude/rules/hardware-interface.md) | Device/RobotInterface 契约、安全层、线程安全 |
| **编码约定** | [coding-conventions.md](.claude/rules/coding-conventions.md) | 命名、配置管理、模块结构、导入规范 |
| **错误处理** | [error-safety.md](.claude/rules/error-safety.md) | 异常粒度、stop/clear_error 语义、WorkspaceSafety |
| **真机安全** | [hardware-safety.md](.claude/rules/hardware-safety.md) | Pre-Flight 清单、E-Stop 条件、禁止操作 |
| **参考检索** | [reference-protocol.md](.claude/rules/reference-protocol.md) | 参考优先级、代码库位置、模块映射、Fact-Check、不采纳清单 |
| **SDK 依赖** | [sdk-dependencies.md](.claude/rules/sdk-dependencies.md) | xArm/XHand/HTS/dex-retargeting/MPlib/SAPIEN API + 陷阱 |
| **IPC 通信** | [ipc-spec.md](.claude/rules/ipc-spec.md) | RingBuffer、键盘事件、两阶段握手 |
| **控制器** | [controller-spec.md](.claude/rules/controller-spec.md) | _tick 逻辑、EMA 平滑、VR re-anchoring、RateLimiter |
| **录制层** | [recording-spec.md](.claude/rules/recording-spec.md) | HDF5 结构、EpisodeRecorder、QualityFlags |
| **数据层** | [data-spec.md](.claude/rules/data-spec.md) | EpisodeReader、DataValidator、归一化 |
| **部署层** | [deploy-spec.md](.claude/rules/deploy-spec.md) | PolicyLoader、PolicyRunner、SafetyMonitor |
| **检查清单** | [checklist.md](.claude/rules/checklist.md) | 硬件驱动/传感器/控制器/录制/数据模块验收清单 |

---

## 关键约束

### 接口命名统一

| 统一名称 | 禁止使用 |
|---------|---------|
| `get_state()` | `get_obs()`, `get_observation()` |
| `send_action()` | `move()`, `control_*()` |
| `connect()` / `disconnect()` | `start()` / `stop()`（传感器也统一） |

### 坐标系（重要）

Robot base 相对 world 绕 Z 轴 **+30° yaw**，零平移：

```python
base_pose_world = Pose(p=[0,0,0], q=[cos(15°), 0, 0, sin(15°)])
```

MPlib IK/planning 在 base frame 执行，VR 目标/相机外参/workspace 在 world frame。变换通过 `XArm7Kinematics.world_to_base_pose()` / `base_to_world_pose()` 完成。`XArm7PlannerConfig.base_pose_world` 默认为 identity，真机运行时必须覆盖。

### VR 连接模式（重要）

**PC 作为 TCP Server 监听，Quest 作为 TCP Client 主动连接。** 这是本项目的标准连接方式：

| 角色 | 传输模式 | 地址 | 说明 |
|------|---------|------|------|
| PC（本机） | `tcp_server` | `0.0.0.0:8000` | 监听所有网络接口 |
| Quest HTS App | TCP Client | `<PC 局域网 IP>:8000` | 主动连 PC |

```python
# 所有脚本/测试中 VR 连接默认值必须遵循此约定
VR_TRANSPORT = "tcp_server"
VR_HOST = "0.0.0.0"
VR_PORT = 8000
```

USB 有线模式（`tcp_client` + `adb reverse`）仅在 WiFi 不可用时作为备选，**不作为默认配置**。

### 设计原则

- 硬件驱动只依赖 SDK + numpy + 标准库（cv2/torch 局部 import）
- 控制器通过 RobotInterface 操作硬件，不直接调 XArm7/XHand
- send_action 内 joint limit + delta limit 裁剪，裁剪状态记录到 `last_*`，不污染返回值
- 录制时使用 send_action 返回的 post-clip cmd 值，而非 IK 原始输出
- 配置使用 `@dataclass` + `default_factory`，运行时验证在 `__post_init__`

### 外部参考

6 个参考代码库（P1: LeFranX + ManiUniCon, P2: BunnyVisionPro + DexUMI + Open-Teach, P3: Bidex_Manus_Teleop）。详细检索协议、代码库路径和模块映射见 `.claude/rules/reference-protocol.md`。

### 已禁用的技术

libfranka+Ruckig C++ server / 解析式 IK (geofik) / LeRobot Parquet+draccus / Hydra / ROS/ROS2 / Vision Pro API / Allegro Hand retargeting / Manus Core SDK / Pydantic / Zarr / ZMQ IPC / iPhone ARKit / 外骨骼编码器 / UR5 RTDE

---

## 参考模板

**XHand（`robot/xhand.py`）** — 执行器模板：完整 Config dataclass + `connect()→bool` + `send_action(np.ndarray)→bool`（含 range clip + delta limit）+ 状态变量 + `example()`

**RealSense（`sensor/realsense.py`）** — 传感器模板：frozen dataclass + 结构化 Frame 输出 + 点云解耦到 utils
