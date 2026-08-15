# XArm7 项目机制、代码实现与 SDK 调用技术文档

> 文档性质：基于仓库当前 `main` 分支的只读代码审计与本机环境核验
>
> 审计日期：2026-08-15
>
> 适用仓库：`Xarm7-`
>
> 代码版本：`74d510f`（`Add policy evaluation workflow`）
>
> SDK 核验版本：`xarm-python-sdk 1.18.4`

## 1. 文档目的与事实边界

本文档完整说明本项目中与 XArm7 有关的：

- 系统目标、模块边界与运行入口；
- 配置加载、HOME 来源、单位和坐标约定；
- `xarm.wrapper.XArmAPI` 的封装方式与实际调用顺序；
- SpaceMouse 遥操作的输入变换、目标积分与控制循环；
- RealSense 采集、episode 状态机与 HDF5 数据格式；
- 本地或 WebSocket policy 的 observation/action 协议；
- RTC、stop-go、轨迹插值与 125 Hz servo target 流；
- 线程、时钟、错误处理、停机与资源生命周期；
- dry-run、自动化测试、当前环境状态和能力缺口；
- 经过代码审计得到的安全风险与改进优先级。

本文严格区分以下三类内容：

1. **实现事实**：可以直接从当前仓库代码、配置或测试确认。
2. **SDK 事实**：由仓库文档和本机已安装的 SDK 1.18.4 源码/签名共同确认。
3. **审计判断**：根据实现推导的风险或建议，不代表项目已经提供相应保证。

本文不会把时间插值称为完整运动规划，也不会把仓库中的 UR RTDE 参考代码归入 XArm7 主链路。

---

## 2. 执行摘要

本项目是一个围绕 XArm7 真机应用构建的 Python 工具包，核心能力包括：

- SpaceMouse 遥操作；
- xArm 官方夹爪开合；
- 多路 RealSense 图像采集；
- 遥操作 episode 的 HDF5 录制；
- 本地 policy 或远程 WebSocket policy 推理；
- 将低频 policy action 插值为高频 XArm servo target。

系统的最终执行接口高度统一：

```text
上层输入或 policy action
  -> 生成绝对 TCP 目标位姿
  -> 应用层 workspace 检查/裁剪
  -> 可选时间插值
  -> XArm7Controller.servo_pose()
  -> XArmAPI.set_servo_cartesian()
```

项目不是一个带完整机器人模型的规划栈。当前仓库没有：

- URDF、SRDF、Xacro 或机器人 mesh；
- 本地正运动学/逆运动学求解器；
- 自碰撞模型或环境碰撞模型；
- 基于采样、优化或搜索的避障规划器；
- ROS、MoveIt 或其他规划框架集成。

因此，XArm7 的 IK、关节可达性、固件运动约束和底层运动执行主要交给控制器与 SDK。应用层额外提供的是轴对齐长方体 workspace、目标增量限制和 policy 轨迹的时间插值。

---

## 3. 仓库范围与目录职责

### 3.1 主程序目录

`xarm_teleop/` 是当前项目的实际主程序包，也是 `pyproject.toml` 中唯一被打包的代码范围：

```toml
[tool.setuptools.packages.find]
include = ["xarm_teleop*"]
```

主要子目录如下：

| 路径 | 职责 |
| --- | --- |
| `xarm_teleop/controllers/` | XArm7、夹爪和 dry-run 假设备封装 |
| `xarm_teleop/input/` | SpaceMouse 输入线程、坐标变换和假输入 |
| `xarm_teleop/teleop/` | 遥操作状态机和固定频率控制循环 |
| `xarm_teleop/cameras/` | RealSense 采集线程、设备管理和预览 |
| `xarm_teleop/recording/` | episode 状态机、HDF5 写入与时间戳工具 |
| `xarm_teleop/policy/` | policy 接口、动作转换、WebSocket、轨迹线程与评测循环 |
| `xarm_teleop/analysis/` | HDF5 完整性、shape、频率和 jitter 离线检查 |
| `xarm_teleop/config/` | 默认遥操作/录制配置和 policy eval 配置 |
| `xarm_teleop/cli.py` | 所有主要命令行入口的组装逻辑 |

### 3.2 启动脚本

`scripts/` 提供日常真机入口：

| 脚本 | 实际入口 | 用途 |
| --- | --- | --- |
| `scripts/start_xarm_teleop.sh` | `xarm-teleop` | 纯遥操作 |
| `scripts/start_xarm_teleop_record.sh` | `python -m xarm_teleop.record_cli` | 遥操作录制 |
| `scripts/start_xarm_policy_eval.sh` | `python -m xarm_teleop.policy_cli` | policy 真机评测 |
| `scripts/save_current_home.sh` | `xarm-save-home` | 保存当前关节角为纯遥操作 HOME |
| `scripts/visualize_xarm_hdf5.py` | Python 脚本 | HDF5 检查与视频可视化 |

`record_cli.py` 和 `policy_cli.py` 只是薄包装，分别调用 `cli.py` 中的 `teleop_record_main()` 与 `policy_eval_main()`。

`pyproject.toml` 注册的全部 console scripts 为：

| Console script | Python 入口 | 作用 |
| --- | --- | --- |
| `xarm-teleop` | `cli:teleop_main` | 纯遥操作 |
| `xarm-teleop-record` | `cli:teleop_record_main` | 遥操作录制 |
| `xarm-policy-eval` | `cli:policy_eval_main` | Policy eval |
| `xarm-save-home` | `cli:save_home_main` | 保存当前 HOME |
| `xarm-read-state` | `cli:read_state_main` | 单次读取机械臂状态 |
| `xarm-test-gripper` | `cli:test_gripper_main` | 依次打开、关闭夹爪 |
| `xarm-test-spacemouse` | `cli:test_spacemouse_main` | 持续打印 SpaceMouse action/buttons |

### 3.3 不属于 XArm7 主链路的目录

#### `real_world/`

该目录包含 `rtde_interpolation_controller.py`、`real_env.py` 等 UR/RTDE 风格参考实现。当前 `xarm_teleop`、启动脚本和测试均未导入该目录，打包配置也不包含它。

结论：它不是当前 XArm7 控制链的一部分，不能用来解释本项目的 XArm SDK 控制行为。

#### `spacemouse/`

该目录包含另一套 SpaceMouse 相关代码，但主程序实际使用的是：

```python
import pyspacemouse
```

或：

```python
import spnav
```

主程序的设备读取实现位于 `xarm_teleop/input/spacemouse.py`。顶层 `spacemouse/` 同样不在项目打包范围内。

---

## 4. 系统架构与数据流

### 4.1 纯遥操作链路

```text
start_xarm_teleop.sh
  -> xarm_teleop.cli.teleop_main
  -> load_config(default + local_home override)
  -> XArm7Controller
  -> SpaceMouseReader thread
  -> TeleopSession.prepare()
  -> HOME
  -> mode=1 servo
  -> TeleopSession.step() @ servo.frequency_hz
  -> absolute target pose integration
  -> controller.servo_pose()
  -> XArmAPI.set_servo_cartesian()
```

### 4.2 遥操作录制链路

```text
start_xarm_teleop_record.sh
  -> teleop_record_main
  -> RealSenseCameraManager
  -> one reader thread per camera
  -> TeleopSession
  -> ContinuousRecordingTeleopRunner
  -> right button starts/stops episode
  -> HDF5EpisodeWriter
  -> return HOME after each saved episode
```

### 4.3 Policy 评测链路

```text
start_xarm_policy_eval.sh
  -> policy_eval_main
  -> load robot config + policy_eval config
  -> RealSense cameras
  -> ObservationHistory
  -> local module policy / WebSocket policy / dry-run hold policy
  -> PolicyPrediction
  -> policy_actions_to_target_poses()
  -> workspace clip
  -> XArmServoTrajectoryController thread
  -> linear XYZ interpolation + rotation Slerp
  -> controller.servo_pose() @ trajectory_rate_hz
  -> XArmAPI.set_servo_cartesian()
```

### 4.4 进程与线程模型

主程序没有使用多进程或共享内存。活动组件均在同一 Python 进程中：

| 线程 | 创建条件 | 责任 |
| --- | --- | --- |
| 主线程 | 始终 | CLI、遥操作/录制/policy 状态机、键盘、HDF5 |
| `SpaceMouseReader` | 真机遥操作/录制 | 约 200 Hz 轮询输入并缓存最新状态 |
| `RealSense-<serial>` | 每台真相机一个 | 持续读取并缓存最新 RGB/深度帧 |
| `CameraPreview` | 录制预览开启 | 显示多相机画面和 episode 状态 |
| `XArmPolicyTrajectory` | policy rollout | 高频插值并发送 servo target |

共享状态主要使用 `threading.Lock`、`threading.Event` 和只保留最新值的缓存。Policy 模式下，主线程进行推理时，轨迹线程仍可继续发送上一段已排程的目标，从而实现推理与执行重叠。

---

## 5. 安装、依赖与当前环境状态

### 5.1 项目声明的环境

`environment.yml` 声明的 Conda 环境名为：

```yaml
name: xarm-mimic
```

关键依赖包括：

- Python 3.10；
- NumPy、SciPy、PyYAML、h5py；
- `xarm-python-sdk`；
- `pyspacemouse`；
- `pyrealsense2`；
- OpenCV；
- `websockets>=12.0`；
- `msgpack>=1.0.0`；
- pytest。

`pyproject.toml` 的基础依赖包含 NumPy、SciPy、PyYAML、h5py 和 OpenCV；`hardware` 可选依赖只列出 `xarm-python-sdk` 与 `pyspacemouse`。RealSense Python 包仅在 `environment.yml` 中声明，没有进入 `pyproject.toml` 的 optional dependencies。

### 5.2 当前机器核验结果

审计时当前机器存在以下 Conda 环境：

- `base`；
- `real_robot`；
- `sim`。

不存在启动脚本默认激活的 `xarm-mimic`。因此，在当前机器直接运行启动脚本会在 `conda activate xarm-mimic` 处失败，除非先创建环境或通过 `XARM_MIMIC_ENV`/`XARM_TELEOP_ENV` 指向现有环境。

本次使用 `real_robot` 环境进行只读核验，得到：

```text
xarm-python-sdk: 1.18.4
```

其关键方法签名与 `docs/xarm_python_sdk_api.md` 一致。

### 5.3 版本稳定性

仓库没有固定 `xarm-python-sdk` 的精确版本，只声明包名。仓库文档与当前实现按 1.18.4 核验，但重新创建环境时可能安装更新版本，因此生产部署应额外锁定已验证版本。

---

## 6. 配置系统

### 6.1 应用配置的加载方式

`xarm_teleop.config.schema.load_config()` 始终先读取打包的：

```text
xarm_teleop/config/tele_record.yaml
```

如果传入 `--config PATH`，再使用递归字典合并覆盖默认值。嵌套 mapping 会递归合并，非 mapping 值会整体替换。

这意味着 `configs/local_home.yaml` 即使只包含 `robot`、`teleop.home` 和 `gripper`，其他配置仍来自打包默认值。

### 6.2 默认机器人配置

当前默认值为：

| 配置项 | 默认值 | 含义 |
| --- | ---: | --- |
| `robot.ip` | `192.168.1.219` | 控制箱 IP |
| `robot.is_radian` | `false` | SDK 姿态与关节默认使用角度 |
| `robot.linear_speed` | `20` | `set_position` TCP 速度，通常 mm/s |
| `robot.linear_acc` | `100` | `set_position` TCP 加速度 |
| `robot.servo_speed` | `100` | 传给 servo SDK，同时被 policy 软件层用作位置限速 |
| `robot.servo_acc` | `2000` | 传给 servo SDK；SDK 文档标为 reserved |
| `robot.mode_settle_s` | `1.0` | 切换模式后等待时间 |

### 6.3 默认 workspace

workspace 是闭区间轴对齐长方体：

| 轴 | 最小值 | 最大值 | 单位 |
| --- | ---: | ---: | --- |
| X | 100 | 700 | mm |
| Y | -500 | 500 | mm |
| Z | 80 | 700 | mm |

边界值本身允许通过。配置加载时只验证每个最小值严格小于最大值。

### 6.4 默认 servo 配置

| 配置项 | 默认值 | 实际用途 |
| --- | ---: | --- |
| `frequency_hz` | 100 | 遥操作主循环频率 |
| `speed_scale` | 1.0 | 同时缩放每周期平移和旋转增量 |
| `max_translation_step_mm` | 2.0 | 满量程输入每周期最大平移 |
| `max_rotation_step_deg` | 0.35 | 满量程输入每周期最大旋转 |
| `command_timeout_s` | 0.25 | 当前仅被解析，未被运行代码使用 |
| `translation_frame` | `base` | 平移增量使用基坐标或 TCP 坐标 |
| `translation_axis_scale` | `[1,1,1]` | 三轴增益和方向 |
| `rotation_axis_scale` | `[1,1,1]` | 三旋转轴增益和方向 |

`frequency_hz`、`speed_scale`、最大步长会验证为正数；`translation_frame` 只允许 `base` 或 `tcp`。`command_timeout_s` 当前既没有正数校验，也没有被控制循环消费。

### 6.5 三套 HOME 的来源

HOME 被有意分离为三个使用场景：

| 场景 | HOME 来源 | 注入位置 |
| --- | --- | --- |
| 纯遥操作 | `configs/local_home.yaml -> teleop.home` | 直接由 `load_config()` 合并 |
| 遥操作录制 | `tele_record.yaml -> recording.home` | CLI 将其复制到 `teleop.home` |
| Policy eval | `policy_eval.yaml -> home` | CLI 将其复制到 `teleop.home` |

纯遥操作启动脚本在真机模式下会拒绝缺失的 `configs/local_home.yaml`，防止直接使用演示 HOME。

录制默认 HOME 为：

```yaml
joint_angles: [0, 0, 40, 0, 0, 0, 0]
speed: 30
acc: 180
```

Policy eval 使用 `policy_eval.yaml` 中另一组七关节角。录制和 policy 启动脚本不会强制要求先通过 `save_current_home.sh` 生成本地 HOME，因此这两套 HOME 必须由操作者在具体工位上人工验证。

当前仓库中三套 HOME 的实际值如下。数值相同不表示来源统一；任一配置文件都可以单独修改并与另外两套发生分歧。

| 场景 | 当前关节角 | speed/acc |
| --- | --- | --- |
| 纯遥操作 `configs/local_home.yaml` | `[-0.332144,-2.590686,-0.109779,74.35216,-0.000688,73.985467,-0.001547]` | `30 / 180` |
| 录制 `recording.home` | `[0,0,40,0,0,0,0]` | `30 / 180` |
| Policy eval `home` | `[-0.332144,-2.590686,-0.109779,74.35216,-0.000688,73.985467,-0.001547]` | `30 / 180` |

当前默认 `robot.is_radian=false`，所以上表是度。如果把机器人配置改为 rad，三套 HOME 的数值也必须同步改成弧度，代码不会根据数值自动识别单位。

### 6.6 保存 HOME

`xarm-save-home` 的行为是：

1. 加载机器人配置；
2. 创建控制器并连接；
3. 读取 SDK 缓存的当前 `angles` 与 `position`；
4. 断开连接；
5. 将七关节角写入 `teleop.home.joint_angles`；
6. 保留已有 HOME speed/acc，或使用配置/默认值；
7. 删除旧版 `robot.home_angles/home_speed/home_acc`；
8. 删除输出 YAML 中的 `recording` 段。

该命令只读取状态，不移动机械臂。

### 6.7 Policy eval 的独立配置

`load_policy_eval_config()` 以 `xarm_teleop/config/policy_eval.yaml` 为默认值，再递归合并可选覆盖文件。它与机器人/遥操作的 `AppConfig` 是两个独立 schema。

Policy 配置包括：

- policy 加载方式；
- pose 表示和 action mode；
- WebSocket endpoint；
- policy 专用 HOME；
- RTC/stop-go 控制参数；
- 相机、录制、预览和运行限制。

### 6.8 遥操作、录制和 policy 的关键默认值

#### 遥操作与夹爪

| 配置 | 默认值 |
| --- | --- |
| `teleop.startup_go_home` | true |
| `teleop.initial_mode` | `xyzrpy` |
| `teleop.keyboard_quit_key` | `q` |
| `teleop.keyboard_pause_key` | space |
| `gripper.enabled` | true；纯遥操作 shell 脚本默认覆盖为 false |
| `gripper.open_position` | 850 |
| `gripper.close_position` | 0 |
| `gripper.speed` | 2000 |
| `gripper.wait` | true |

#### 录制

| 配置 | 默认值 |
| --- | --- |
| `recording.dry_run` | false |
| cameras | 3 台、640×480、30 Hz、serial 均为 null |
| camera ready timeout | 10 s |
| output directory | `data/xarm_teleop_hdf5` |
| record rate | 10 Hz |
| compression | `lzf` |
| save depth | false |
| rotation representation | `axis_angle` |
| preview | 开启，15 Hz |
| duration/max steps | 均为 null，无配置限制 |

#### Policy eval

| 配置 | 默认值 |
| --- | --- |
| `policy.spec` | null；只允许 dry-run 回退 hold policy |
| rotation representation | `axis_angle` |
| action mode | `absolute_pose` |
| observation horizon | 1 |
| WebSocket URL/host | null |
| WebSocket port | 8000 |
| control model | `rtc` |
| action rate | 10 Hz |
| trajectory rate | 125 Hz |
| RTC steps per inference | 6 |
| RTC action offset | 0 |
| stop-go action steps | 6 |
| cameras | 3 台、640×480、30 Hz、serial 均为 null |
| output | 开启，`data/xarm_policy_eval_hdf5`，10 Hz，保存视频，LZF |
| preview | 开启，15 Hz |
| duration/max steps | 均为 null，无配置限制 |

---

## 7. 单位、位姿与坐标约定

### 7.1 XArm SDK 边界

XArm 控制器内部统一使用六维 TCP pose：

```text
[x, y, z, roll, pitch, yaw]
```

其中：

- `x/y/z` 始终是 mm；
- `roll/pitch/yaw` 由 `robot.is_radian` 决定；
- 七关节角同样由 `robot.is_radian` 决定；
- 当前默认 `is_radian=false`，所以 RPY 和关节角均为度。

代码通过 SciPy：

```python
Rotation.from_euler("xyz", ...)
```

在 RPY、旋转向量、四元数和旋转矩阵之间转换。

### 7.2 Policy/HDF5 边界

Policy 和 HDF5 使用米制位置，关节始终使用 rad。完整 pose 支持三种表示：

| 名称 | 维度 | 格式 |
| --- | ---: | --- |
| `axis_angle` | 6 | `[x_m,y_m,z_m,rx_rad,ry_rad,rz_rad]` |
| `quaternion` | 7 | `[x_m,y_m,z_m,qw,qx,qy,qz]` |
| `rotation_6d` | 9 | `[x_m,y_m,z_m,r00,r01,r02,r10,r11,r12]` |

四元数顺序固定为 `wxyz`。从旋转对象输出四元数时，代码会在 `w < 0` 时整体取反，使输出优先保持非负 w；输入四元数会先归一化，并转换为 SciPy 使用的 `xyzw`。

`rotation_6d` 使用旋转矩阵前两行。反向转换时通过 Gram-Schmidt 方式归一化第一行、正交化第二行，再用叉积生成第三行。

### 7.3 遥操作平移坐标系

默认 `translation_frame=base`：

```python
next_target[:3] += delta[:3]
```

若配置为 `tcp`：

```python
next_target[:3] += current_rotation.apply(delta[:3])
```

即先用当前目标姿态把局部平移增量旋转到基坐标系。

### 7.4 遥操作旋转复合

运行时遥操作使用：

```python
next_rotation = delta_rotation * current_rotation
```

Policy 的完整 delta pose 也使用相同的左乘顺序。位置 delta 则始终直接累加到基坐标 XYZ。

控制器文件中另有 `compute_tcp_delta_target()` 和 `servo_tcp_delta()`，其旋转顺序是：

```python
current_rotation * delta_rotation
```

并使用当前 SDK 反馈 pose 作为起点。但当前高层遥操作和 policy 主链路均不调用 `servo_tcp_delta()`；它主要由控制器单元测试覆盖。因此不能用该辅助函数解释当前 TeleopSession 的实际旋转语义。

---

## 8. XArm Python SDK 集成

### 8.1 SDK 导入与构造

默认 factory 在真正创建机械臂时才导入：

```python
from xarm.wrapper import XArmAPI
return XArmAPI(ip, is_radian=is_radian)
```

延迟导入使 dry-run 和不接真机的单元测试可以不实例化真实 SDK。

SDK 1.18.4 的 `XArmAPI` 构造签名是：

```python
XArmAPI(port=None, is_radian=False, do_not_open=False, **kwargs)
```

`do_not_open` 默认是 `False`，所以传入 IP 创建对象时 SDK 会自动连接。项目 `XArm7Controller.connect()` 仍会检查 `arm.connected`；若对象存在但未连接，才额外调用 `arm.connect()`。

### 8.2 返回码处理

项目统一通过：

```python
_require_code(code, command)
```

处理 SDK 调用结果：

- `0`：成功；
- `None`：也视作成功，用于兼容不返回值的 wrapper 方法；
- 其他值：抛出 `XArmCommandError(command, code)`。

项目没有在异常消息中进一步翻译 API code 或控制器 error code。

### 8.3 Mode 与 state

SDK 的 mode 表示控制方式，state 表示当前运行状态。项目显式使用的只有 mode 0/1 和 state 0/4。

| Mode | SDK 含义 | 本项目用途 |
| ---: | --- | --- |
| 0 | 普通位置/关节运动模式 | enable、HOME、`set_position`、停止 servo 后恢复 |
| 1 | servo target 流模式 | 高频 `set_servo_cartesian` |
| 2 | 关节拖动示教模式 | 未使用 |
| 3 | SDK 文档中的笛卡尔示教相关模式 | 未使用 |
| 4 | 关节速度控制模式 | 未使用 |
| 5 | 笛卡尔速度控制模式 | 未使用 |

| State | SDK 报告含义 | 本项目用途 |
| ---: | --- | --- |
| 0 | ready | 每次普通运动/servo 前设置 |
| 1 | moving | 只可能从状态属性观察，代码不主动设置 |
| 2 | sleeping | 只可能观察，代码不主动设置 |
| 3 | suspended | 只可能观察，代码不主动设置 |
| 4 | stopping/stopped | stop servo 和 emergency stop 使用 |

项目没有持续验证报告的 mode/state 是否仍与应用层 `_servo_started` 标志一致。

### 8.4 连接与断开

| 控制器方法 | SDK 行为 |
| --- | --- |
| `connect()` | 创建 `XArmAPI`；必要时调用 `connect()` |
| `disconnect()` | 调用 SDK `disconnect()`，清空 `self.arm` 和 servo 标志 |

长时间运行的遥操作、录制和 policy 主流程在 shutdown 中停止 servo，但没有显式调用 `controller.disconnect()`。进程退出后连接通常由进程资源回收；一次性 `read-state`、`save-home` 和 `test-gripper` 命令会显式断开。

### 8.5 清错与使能

`clear_errors()` 先读取 SDK 缓存的 `warn_code` 和 `error_code`：

1. warning 非零时调用 `clean_warn()`；
2. error 非零时调用 `clean_error()`。

`enable()` 的顺序为：

```text
clear_errors()
motion_enable(enable=True)
set_mode(0)
set_state(0)
sleep(mode_settle_s)
```

这会自动清理当前缓存错误再使能，没有要求操作者先确认错误原因。

### 8.6 状态读取

`ControllerState` 包含：

- `connected`；
- `mode`；
- `state`；
- `error_code`；
- `warn_code`；
- `position`；
- `angles`；
- 本机 `time.time()` 时间戳。

项目读取的是 SDK 属性，而非每次显式调用 `get_position()`、`get_servo_angle()` 或 `get_state()`。在 socket report 开启时，这些属性由 SDK 报告线程异步更新。因此单次 `ControllerState` 是应用层按顺序读取多个缓存字段得到的快照，不是控制器提供的原子同步采样。

### 8.7 HOME：普通关节运动

`go_home()` 的调用顺序是：

```text
set_mode(0)
set_state(0)
sleep(mode_settle_s)
set_servo_angle(
    angle=<7 joints>,
    speed=<home speed>,
    mvacc=<home acc>,
    wait=True,
    is_radian=<robot.is_radian>
)
check cached robot error_code
```

虽然 SDK 方法名含有 `servo`，`set_servo_angle()` 在这里是普通关节空间点到点运动，不是 mode 1 target 流接口。

应用层不会预先检查 HOME 关节角对应的 TCP workspace，也不计算 HOME 运动路径上的碰撞。SDK 默认具有关节范围检查和控制器自身安全机制，但本项目没有建立独立的路径安全证明。

### 8.8 普通 TCP 点到点运动

`move_pose()` 封装 `set_position()`：

1. 验证目标 shape 为 6；
2. 检查目标 XYZ 是否在 workspace；
3. 切换到 mode 0/state 0；
4. 调用 `set_position(x,y,z,roll,pitch,yaw,speed,mvacc,wait)`；
5. 检查缓存 error code。

当前三个主要 CLI 工作流没有调用 `move_pose()`；HOME 使用关节运动，实时控制使用 servo target。

### 8.9 进入 servo target 流

`start_servo()` 的顺序为：

```text
clear_errors()
motion_enable(enable=True)
set_mode(1)
set_state(0)
sleep(mode_settle_s)
_servo_started = True
```

只有 `_servo_started=True` 时，`servo_pose()` 和 `servo_tcp_delta()` 才允许发送目标。

### 8.10 发送笛卡尔目标

`servo_pose()` 执行：

1. 检查 servo 已启动；
2. 转换为 NumPy float64；
3. 检查 workspace；
4. 调用：

```python
arm.set_servo_cartesian(
    target.tolist(),
    speed=config.robot.servo_speed,
    mvacc=config.robot.servo_acc,
)
```

5. 校验本次 SDK 返回码；
6. 返回目标 pose。

SDK 1.18.4 明确说明 `set_servo_cartesian()`：

- 需要 mode 1；
- 只执行最新目标；
- 没有 `wait` 参数；
- `speed/mvacc/mvtime` 是 reserved；
- 实际运动主要由目标流频率和相邻目标距离决定。

`servo_pose()` 不像 `servo_tcp_delta()` 那样在发送后再次读取缓存 `error_code`。不过 SDK 本次命令若直接返回非零，仍会立即抛出异常。

### 8.11 停止 servo

默认 `stop_servo(keep_enabled=True)`：

```text
set_state(4)
set_mode(0)
set_state(0)
_servo_started = False
```

该操作停止 servo 流并恢复普通 ready 状态，但不会 `motion_enable(False)`，所以不等同于电机失能或急停。

### 8.12 应用层急停封装

`emergency_stop()` 优先调用 SDK 同名方法。SDK 1.18.4 的实现主要进入 `state=4` 并同步状态。若底层对象没有该方法，项目 fallback 为：

```text
set_state(4)
motion_enable(False)
```

当前 CLI 没有把该方法绑定到键盘按键；`q`、`c` 和 `space` 都不是应用层 emergency stop。Policy UI 也明确提示操作者保持物理急停可触及。

### 8.13 官方夹爪调用

`XArmGripper` 使用：

```text
clean_gripper_error()
set_gripper_enable(True)
set_gripper_speed(config.speed)
get_gripper_position()
set_gripper_position(position, wait=config.wait)
```

位置范围在应用层限制为 `[-10, 850]`。默认：

- open = 850；
- close = 0；
- speed = 2000；
- wait = true。

开合状态根据 open/close 中点判断。若内部 `_is_open` 未初始化，首次 toggle 会读取真实夹爪位置。

### 8.14 项目实际使用的 SDK API 总表

| API/属性 | 参数或返回 | 调用位置/用途 |
| --- | --- | --- |
| `XArmAPI(ip,is_radian)` | IP、默认角度单位 | 默认 arm factory；构造时自动连接 |
| `connected` | bool 属性 | 连接状态检查 |
| `connect()` | wrapper 无稳定返回值 | 仅对象尚未连接时调用 |
| `disconnect()` | wrapper 无稳定返回值 | 一次性命令和显式 controller disconnect |
| `position` | `[x,y,z,r,p,y]` | 反馈状态和首次 target 同步 |
| `angles` | 七关节角 | 状态、HOME 保存、observation/HDF5 |
| `mode/state` | 整数属性 | ControllerState |
| `error_code/warn_code` | 整数属性 | 清错和健康状态 |
| `clean_error/clean_warn()` | API code | enable/start servo 前按缓存码调用 |
| `motion_enable(True)` | API code | enable/start servo |
| `motion_enable(False)` | API code | 仅 emergency fallback |
| `set_mode(0/1)` | API code | 普通运动与 servo 模式切换 |
| `set_state(0/4)` | API code | ready 与停止 |
| `set_servo_angle(...)` | 7 joints、speed、acc、wait、unit | HOME |
| `set_position(...)` | 绝对 TCP、speed、acc、wait | 已封装的普通 TCP 运动 |
| `set_servo_cartesian(...)` | 绝对 TCP target | 遥操作与 policy 的最终执行接口 |
| `emergency_stop()` | wrapper 返回值可为 None | 应用层急停封装，CLI 未绑定 |
| `clean_gripper_error()` | API code | 夹爪初始化 |
| `set_gripper_enable(True)` | API code | 夹爪使能 |
| `set_gripper_speed(speed)` | API code | 夹爪速度 |
| `set_gripper_position(pos,wait)` | API code | 开、合、toggle |
| `get_gripper_position()` | `(code,position)` 或 position | 初始状态与反馈宽度 |

仓库 SDK 速查还记录了 velocity、IK/FK、limit、TCP load/offset 和 collision sensitivity 等 API，但上表之外的这些能力没有被 XArm7 主控制器实际调用。

---

## 9. XArm7Controller 的应用层安全模型

### 9.1 已实现的检查

应用层对所有 `servo_pose()` 和 `move_pose()` 目标执行：

- pose 必须是六维；
- X、Y、Z 必须位于配置 workspace；
- SDK 返回码必须为 0 或 None；
- 部分方法在命令后检查缓存 `error_code`。

越界时抛出 `SafetyError`，不会把该目标发送给 SDK。

### 9.2 未实现的检查

项目当前没有实现或调用：

- 姿态范围限制；
- `XArmAPI.is_tcp_limit()`；
- `XArmAPI.is_joint_limit()`；
- `get_inverse_kinematics()` 预检；
- `get_forward_kinematics()`；
- 自碰撞检测；
- 与桌面、支架、相机、人员或其他机械臂的环境碰撞检测；
- 奇异位形或可操作度检查；
- 实际关节速度、加速度或力矩监测；
- TCP load、TCP offset、重力方向或碰撞灵敏度的启动配置；
- 控制器 command queue/cmd_num 监测。

这不表示 XArm 控制器固件完全没有这些保护，而是表示本项目本身没有配置、调用或独立验证这些机制。

### 9.3 Teleop 与 policy 的边界策略不同

- Teleop：目标越界时拒绝本周期命令，保留之前的内部目标。
- Policy：先把整段目标轨迹的 XYZ 分量裁剪到边界，再排程执行。

Policy 的裁剪会改变 policy 原始动作语义，例如多个越界点可能被压到同一边界位置，而不是终止 rollout。

---

## 10. SpaceMouse 输入机制

### 10.1 支持的 backend

`SpaceMouseReader` 支持：

- `pyspacemouse`，默认配置；
- `spnav`，备用方式。

读取线程是 daemon thread，默认每 `1/200 s` 轮询一次。

默认 SpaceMouse 配置为：

| 配置 | 默认值 |
| --- | --- |
| backend | `pyspacemouse` |
| `max_value` | 500；只用于 spnav |
| translation deadzone | `[0.08,0.08,0.08]` |
| rotation deadzone | `[0.12,0.12,0.12]` |
| buttons | 2 |
| spnav right-handed transform | true |

### 10.2 pyspacemouse 轴映射

原始状态映射为：

```text
[-y, +x, +z, -roll, -pitch, -yaw]
```

之后：

1. 裁剪到 `[-1, 1]`；
2. 按六轴 deadzone 把小输入置零；
3. 缓存最新 action、按钮和本机时间戳。

`spacemouse.max_value` 不参与 pyspacemouse backend，它只用于 spnav 原始整数值归一化。

### 10.3 spnav 轴映射

spnav 将原始 translation 和 rotation 分别除以 `max_value`，应用 deadzone，并可使用固定矩阵：

```text
[[ 0, 0,-1],
 [ 1, 0, 0],
 [ 0, 1, 0]]
```

转换到项目使用的右手 Z-up 坐标约定。

### 10.4 按钮

主运行逻辑使用固定索引：

- button 0：夹爪 toggle；
- button 1：纯遥操作中切换 `xyzrpy/xyz`，录制中由 runner 用于开始/结束 episode。

`teleop.left_button` 和 `teleop.right_button` 字符串配置会被解析和保存，但当前运行逻辑没有根据它们动态映射行为。

### 10.5 输入缓存与超时

读取线程只缓存最近一次 action，没有根据 `SpaceMouseState.timestamp` 实现陈旧输入归零。配置中的 `command_timeout_s=0.25` 没有接入。

相机线程会保存异常并在下一次读取时抛出；SpaceMouse 线程没有同类 `_error` 传播通道。如果设备读取抛出未处理异常，线程可能结束，而主线程仍能读取之前缓存的 action。

这是后文安全风险中输入 watchdog 的主要依据。

---

## 11. 遥操作实现

### 11.1 TeleopSession.prepare()

准备顺序为：

```text
controller.connect()
controller.enable()
optional gripper construction and enable
optional go_home()
controller.start_servo()
sync internal target from controller.position
spacemouse.start()
running = True
```

夹爪初始化发生在 HOME 之前。默认录制配置启用夹爪；纯遥操作启动脚本默认通过 `--no-gripper` 将其关闭，只有显式 `--with-gripper` 才启用。

### 11.2 每周期增量

平移增益：

```text
translation_axis_scale
  * max_translation_step_mm
  * speed_scale
```

旋转增益：

```text
rotation_axis_scale
  * max_rotation_step_deg
  * speed_scale
```

若机器人单位为 rad，旋转增益会转换为 rad。

SpaceMouse action 再次裁剪到 `[-1,1]` 后与增益相乘。默认满量程每周期最多：

- 平移 2 mm；
- 每轴旋转 0.35°。

默认 100 Hz 下，名义上限约为 200 mm/s 和 35°/s；这是每周期步长乘名义频率得到的应用层估计，不是 SDK 对实时速度的硬保证。

### 11.3 两种遥操作模式

| 模式 | 平移 | 旋转 |
| --- | --- | --- |
| `xyzrpy` | 启用 | 启用 |
| `xyz` | 启用 | 强制 delta 为零，保持目标姿态 |

右键在纯遥操作中对两种模式做边沿触发切换。

### 11.4 命令目标积分

TeleopSession 维护独立的 `_target_pose`。每周期增量基于上一次成功发送的命令目标积分，而不是基于当前机器人反馈。

这样做的意图是避免机器人反馈滞后导致同一输入增量被反复应用在滞后的起点上。测试中的 `LaggyArm` 明确验证了连续两个 2 mm 输入会得到 502 mm、504 mm 的命令目标，即使反馈仍停留在 500 mm。

代价是命令目标可能暂时领先于真实机器人状态，因此应用层没有自动的 command-vs-feedback 跟踪误差限制。

### 11.5 每周期发送行为

非暂停状态下，即使六维 delta 全为零，代码仍调用一次 `controller.servo_pose(target)`。这会在正常空闲时持续刷新相同的 servo target。

如果目标越界：

- `servo_pose()` 抛出 `SafetyError`；
- step 返回 `safety_blocked=True`；
- `_target_pose` 不更新；
- warning 包含具体越界轴和值。

### 11.6 暂停

按 space 切换 `paused`。暂停时：

- 仍然读取 SpaceMouse；
- 仍然处理按钮边沿和夹爪 toggle；
- 不计算/发送新的 servo pose；
- 内部 target 保持不变。

因此暂停期间应用层不持续刷新 `set_servo_cartesian()`。

### 11.7 夹爪对实时循环的影响

按钮处理发生在机械臂 pose 命令之前。默认 `gripper.wait=true`，因此 toggle 可能阻塞到夹爪动作完成，期间主线程不会发送新的 servo target。

### 11.8 循环调度

遥操作周期为：

```python
period = 1 / servo.frequency_hz
```

循环使用单调时钟累计 deadline。若本周期超时：

- 不补发多个控制 step；
- 把下一基准重置为当前单调时间。

这避免持续追赶造成突发命令，但实际频率会在慢周期后下降。

### 11.9 HOME 返回

录制结束后：

```text
stop_servo(keep_enabled=True)
go_home(wait=True)
wait until right button is released
start_servo()
sync target from feedback
reset mode and pause state
```

等待按钮释放避免同一次长按立即启动下一 episode。

### 11.10 Shutdown

```text
spacemouse.stop()
controller.stop_servo(keep_enabled=True)
running = False
clear internal target
```

不会自动断开 SDK，也不会电机失能。

---

## 12. RealSense 相机机制

### 12.1 一相机一线程

每个 `RealSenseCameraReader`：

1. 绑定明确序列号；
2. 启动 RGB stream；
3. 可选启动 depth stream；
4. 可选把 depth 对齐到 color；
5. 尝试开启 `global_time_enabled`；
6. 循环 `wait_for_frames(timeout_ms=1000)`；
7. 缓存最新一帧。

项目请求 SDK 输出 `bgr8`，若逻辑格式配置为 `rgb8`，再通过 `[..., ::-1]` 转成 RGB。

### 12.2 帧元数据

每个缓存帧包含：

- `color`；
- 可选 `depth`；
- `receive_timestamp = time.time()`；
- RealSense `capture_timestamp`，从毫秒换算为秒；
- `frame_number`。

读取 `get_latest()` 时会复制图像，避免其他线程观察到正在变化的数组。

### 12.3 设备发现与 key 绑定

若 YAML 未指定 serial：

1. 查询全部连接设备；
2. 按 serial 字符串排序；
3. 依次绑定配置的 camera key。

设备数量必须精确等于 `expected_count`，默认要求三台。若 YAML 中任何一个设备提供 serial，则 schema 要求全部设备都提供 serial。

### 12.4 相机同步边界

`RealSenseCameraManager.get_latest()` 分别读取每个线程的最新帧。它不保证：

- 三台相机来自同一硬件触发时刻；
- frame number 对齐；
- capture timestamp 对齐；
- 机器人状态和相机曝光时刻一致。

开启 global time 只改善时钟可比性，不等同于硬件同步。

### 12.5 相机异常

相机线程捕获异常到 `_error`。等待 ready 或后续 `get_latest()` 会把它包装为 `DeviceError` 抛给主线程。

### 12.6 预览行为

录制预览在独立 `CameraPreview` 线程中读取相同的最新帧缓存。关闭预览窗口、按 Esc 或按 `q` 只会停止预览线程，不会通知录制 runner 退出。Policy 预览则由主线程处理按键，其 Esc/`q` 会进入 policy 的退出状态机。

---

## 13. 遥操作录制状态机

### 13.1 录制入口

`teleop_record_main()`：

1. 加载 AppConfig；
2. 把 `recording.home` 复制为本次 `teleop.home`；
3. 创建真实或 fake XArm；
4. 创建真实或 fake 相机；
5. 先启动相机并等待 ready；
6. 可选启动相机预览；
7. 创建 TeleopSession；
8. 创建 `ContinuousRecordingTeleopRunner`。

相机比机械臂先启动。退出时先停止预览，再停止相机。

### 13.2 频率约束

runner 构造时要求：

```text
record_frequency_hz <= servo.frequency_hz
record_frequency_hz <= 所有相机中最低 fps
```

默认是：

- 控制 100 Hz；
- 相机 30 Hz；
- 记录 10 Hz。

### 13.3 Episode 状态

状态机可概括为：

```text
prepare + HOME + start servo
  -> WAITING_START
  -> right button rising edge
  -> open HDF5 writer
  -> RECORDING
  -> right button rising edge
  -> close/save episode
  -> stop servo
  -> HOME
  -> restart servo
  -> WAITING_START
```

等待开始时 runner 只读取 SpaceMouse 按钮，不调用 `TeleopSession.step()`；此时机械臂已经进入 mode 1，但没有应用层持续 target 流。

### 13.4 采样时刻

episode 开始时记录：

```python
episode_start_time = time.time()
```

第 k 个成功样本的落盘时间戳是：

```text
episode_start_time + k * record_period
```

这是一条理想固定频率时间网格，而不是该样本相机曝光或机器人报告的真实时间。

当主循环落后多个记录周期时，会计算 `due_count` 并循环补齐多个样本。每次补样都会重新读取最新相机与机器人状态，但短时间内可能读到相同相机帧。对应时间戳仍按理想网格递增。

### 13.5 Episode 保存与丢弃

- 正常右键结束且至少一个样本：保存文件；
- 样本数为 0：关闭后删除文件；
- 程序退出时仍在录制：关闭后删除该未完成 episode；
- 文件删除使用 `Path.unlink(missing_ok=True)`。

这与 policy 录制不同：policy writer 在 rollout finally 中通常会保留已经打开的部分文件。

### 13.6 文件命名

默认：

```text
episode_YYYYMMDD_HHMMSS.hdf5
```

同一秒重名时添加 `_01`、`_02` 等后缀。

### 13.7 辅助单文件 runner

`RecordingTeleopRunner` 是另一个已实现的录制类：启动后立即进行普通遥操作并写一个固定 writer，没有右键切分和自动 HOME episode 状态机。它使用采样当下的 `wall_clock()` 作为 timestamp。当前 `teleop_record_main()` 使用的是 `ContinuousRecordingTeleopRunner`，不是该辅助类。

---

## 14. HDF5 数据模型

### 14.1 文件层级

```text
/
  attrs
  data/
    demo_000000/
      datasets...
      attrs...
```

每个文件固定声明 `n_episodes=1`，即一文件一 episode。

### 14.2 主要数据集

| 数据集 | 内容 |
| --- | --- |
| `timestamp` | 固定频率 episode 时间网格 |
| `action` | 命令目标 TCP pose + 夹爪目标宽度 |
| `delta_action` | 当前 SpaceMouse 控制增量 + 夹爪目标/反馈差值 |
| `stage` | 当前固定写入整数 0 |
| `robot_eef_pose` | SDK 反馈 TCP pose |
| `robot_joint` | SDK 反馈七关节角，统一 rad |
| `gripper_width` | 采样时实际夹爪位置；无夹爪时 NaN |
| `<camera_key>` | RGB uint8 图像，可选 |
| `depth_*` | uint16 对齐深度，可选 |

### 14.3 Action 维度

| 旋转表示 | pose 维度 | action 维度 | action 末维 |
| --- | ---: | ---: | --- |
| axis-angle | 6 | 7 | gripper action width |
| quaternion | 7 | 8 | gripper action width |
| rotation-6D | 9 | 10 | gripper action width |

### 14.4 Action 的来源

`action` 优先使用 `TeleopStepResult.target_pose`，也就是本周期成功发送的命令目标。只有 target 缺失时才回退到机器人反馈 pose。

因此数据明确区分：

- `action`：操作者/控制器要求机械臂到达的位置；
- `robot_eef_pose`：机械臂报告的实际位置。

### 14.5 Delta action 的语义

`delta_action` 的机械臂部分来自本周期 SpaceMouse delta，不是相邻两个 `action` 直接做数值相减。

- XYZ 从 mm 转 m；
- axis-angle 由 RPY delta 转旋转向量；
- quaternion/rotation-6D 表示对应的 delta rotation；
- 无 delta 时填 NaN。

夹爪 delta 是：

```text
gripper_action_width - sampled_gripper_width
```

若任一值缺失则为 NaN。默认夹爪命令是阻塞等待，采样时该差值可能已经接近零。

### 14.6 压缩与 chunk

每个数据集按第一维无限扩展，chunk 大小是一个样本。压缩只对 image/depth dataset 启用：

- `lzf`；
- `gzip`，level 4 并开启 shuffle；
- `none`。

机器人数值数据不压缩。

### 14.7 元数据

文件和 demo attrs 会记录：

- 格式名和版本；
- 记录/控制频率；
- action/delta action 格式；
- pose 单位和来源格式；
- 机器人是否使用 rad；
- 关节单位；
- 四元数或 rotation-6D 约定；
- 相机 key、serial、分辨率、fps、颜色格式；
- depth key、格式、scale 与对齐目标；
- 样本总数。

### 14.8 未落盘的时间信息

尽管运行时对象包含以下字段，当前 HDF5 writer 没有保存它们：

- SpaceMouse event timestamp；
- 相机 receive timestamp；
- 相机 capture timestamp；
- frame number；
- ControllerState 自身 timestamp；
- error code、warn code、mode、state。

因此仅凭当前 HDF5 文件不能重建每个传感器的真实采样时刻或判断相机是否重复帧。

### 14.9 时间戳对齐工具

`recording/timestamps.py` 提供 nearest-neighbor 对齐函数，但主录制流程没有调用它们。它们目前只是导出的辅助 API。

---

## 15. Policy 接口与加载

### 15.1 统一接口

Policy 需要实现：

```python
reset()
predict_action(obs)
```

`predict_action()` 可以返回：

- `PolicyPrediction`；
- NumPy 数组；
- 包含 `actions` 和可选 `mode` 的 dict。

一维 action 会自动扩展成单步 batch；最终必须是非空二维数组。

### 15.2 Policy 加载方式

支持：

1. `module:factory` 动态导入；
2. `websocket`/`web_policy`；
3. 内置 `hold` policy。

`hold` 重复 observation 中最新末端 pose，只允许 dry-run 或显式选择 hold 时使用。真实运行若没有 policy spec，会拒绝启动。

动态 factory 接收：

```python
factory(runtime=PolicyRuntimeConfig(...), **policy.args)
```

若对象有 `predict_action()` 但没有 `reset()`，会由一个 no-op reset wrapper 包装。

### 15.3 PolicyRuntimeConfig

传给 policy 的运行元数据包括：

- `n_obs_steps`；
- `action_horizon`；
- `inference_frequency_hz`；
- `robot_is_radian`；
- `pose_format`；
- 相机信息；
- action mode、旋转表示、控制模型、相机 key/尺寸/fps 等 extra 字段。

---

## 16. Observation 采集

### 16.1 单个 ObservationSample

每次 capture：

1. 读取所有相机最新帧；
2. 若任一相机尚无帧，返回 None；
3. 读取一次 ControllerState；
4. 取机器人状态时间和各相机 receive time 的最大值作为 sample timestamp；
5. 把样本加入固定长度 deque。

这个 timestamp 表示“构成本次 observation 的各最新数据中最晚的本机时间”，不是统一传感器曝光时刻。

### 16.2 Observation dict

```text
timestamp:       (N,)
robot_eef_pose:  (N,6) / (N,7) / (N,9)
robot_joint:     (N,7), rad
<camera_key>:    (N,H,W,3), RGB uint8
```

历史不足 N 帧时，会在最前面复制最早已有样本，直到达到固定 horizon。

### 16.3 多传感器一致性

每个 capture 只是读取各自缓存的最新值：

- 不做相机之间 nearest timestamp matching；
- 不按机器人状态时间插值；
- 不检查 frame 是否比上次 capture 更新；
- 不检测 observation 中重复图像。

---

## 17. WebSocket Policy

### 17.1 Client 责任

`WebSocketPolicyAdapter` 本身不做推理。它执行：

```text
format observation
  -> client.infer(obs)
  -> validate result mapping
  -> require "actions"
  -> return PolicyPrediction
```

远端 server 是实际执行模型推理的一方。

### 17.2 Endpoint

支持：

- 完整 `url`；
- 或 `host + port`。

`http://` 和 `https://` 会分别转换为 `ws://` 与 `wss://`。

### 17.3 Client 加载

优先：

```python
from web_policy import WebSocketClientPolicy
```

失败后尝试仓库同级的 `web_policy/src`。再失败会给出安装提示，其中包含开发机器路径 `/home/haoce/xarm-sdk/web_policy`；该路径不是当前仓库内资源。

### 17.4 Payload 验证

Adapter 要求：

- timestamp 是 1D；
- eef pose 是 2D 且末维符合当前 pose format；
- joint 是 2D、末维 7；
- 至少一组 camera 是 4D、末维 3。

它不验证所有数组第一维 N 是否一致，也不检查图像 dtype 必须是 uint8。

### 17.5 协议字段

配置中的 `web_policy.protocol` 只允许 `xarm_obs_v1`，但当前 adapter 不在 observation dict 中显式加入 `protocol` key，也没有把 protocol 参数传给 client factory。它目前主要用于配置约束、日志和文档约定。

### 17.6 Server 返回

必须至少包含：

```python
{"actions": actions}
```

可选：

```python
{"mode": "absolute_pose"}
```

缺少 mode 时使用 YAML 的默认 action mode。

---

## 18. Policy action 语义与转换

### 18.1 支持的 action mode

| Mode | 输入维度 | 语义 |
| --- | ---: | --- |
| `absolute_pose` | 6/7/9 | 完整绝对 pose |
| `absolute_xyz` | 3 | 绝对 XYZ，保持当前目标姿态 |
| `absolute_xy` | 2 | 绝对 XY，保持 Z 和姿态 |
| `delta_pose` | 6/7/9 | 逐步累计完整 pose delta |
| `delta_xyz` | 3 | 逐步累计 XYZ delta |
| `delta_xy` | 2 | 逐步累计 XY delta |

### 18.2 当前目标基准

Delta 和部分绝对 action 使用 `_active_target_pose` 作为基准。该值主要来自：

- rollout 开始时的机器人反馈；
- 已发送/插值的最新 command pose；
- 最近排程的 policy target。

它不是每次都重新使用实时反馈，因此同样采用命令目标连续性而非反馈闭环修正。

### 18.3 Delta 累计

- XYZ 乘 1000 从 m 转 mm 后逐步累加；
- 完整 rotation delta 与当前目标 rotation 左乘；
- action chunk 中每一步都基于前一步结果累计。

### 18.4 Workspace 裁剪

action chunk 转换成 XArm pose 后，所有目标的 XYZ 会独立裁剪到 workspace。姿态不裁剪。

### 18.5 夹爪

Policy action schema 不包含夹爪，policy runner 也不初始化或控制夹爪。因此：

- policy 无法通过当前统一 action 接口开合夹爪；
- policy HDF5 中 gripper action/width 通常是 NaN；
- 遥操作数据若含夹爪维度，不能直接假设 policy runner 会执行该维。

---

## 19. Policy 轨迹插值

### 19.1 插值器

`XArmPoseTrajectoryInterpolator` 对：

- XYZ 使用 `scipy.interpolate.interp1d` 线性插值；
- 姿态先从 RPY 转 Rotation，再使用 SciPy `Slerp`。

查询超出时间范围时会裁剪到首尾时间，因此轨迹结束后继续查询会保持最后目标。

### 19.2 Waypoint 重排

加入 waypoint 时：

1. 忽略 `target_time <= curr_time` 的点；
2. 根据当前时间和上一个请求 waypoint 时间裁剪现有轨迹；
3. 计算末端位置距离和姿态角距离；
4. 根据 `max_pos_speed`、`max_rot_speed` 延长最小执行时间；
5. 添加新终点。

如果请求的 waypoint 太快，实际插值器终点会晚于请求时间。代码的 `_last_waypoint_time` 仍保存调用者请求时间，而不是限速后延长的最终时间；测试明确覆盖了这一行为。

### 19.3 软件限速来源

Policy 轨迹位置限速：

```text
max_pos_speed = robot.servo_speed
```

默认 100 mm/s。

旋转限速：

```text
max_rotation_step_deg * servo.frequency_hz
```

默认 `0.35 * 100 = 35°/s`。注意这里使用遥操作的 `servo.frequency_hz=100` 计算旋转速度，即使 policy 实际 trajectory rate 默认是 125 Hz。

### 19.4 轨迹线程

默认以 125 Hz 运行：

1. 使用固定 `loop_start + tick_idx * period` deadline；
2. 距 deadline 超过约 1 ms 时 sleep；
3. 最后约 1 ms busy-wait；
4. 查询当前插值 pose；
5. 调用 `controller.servo_pose()`；
6. 更新 command pose、计数、频率与迟到统计。

若发送命令或插值发生异常，线程捕获异常并写入 status，主 policy 循环检测到后停止 rollout。

### 19.5 Missed cycle

如果命令结束时间已经超过下一 deadline 一个完整周期以上，线程跳过相应 tick，并累计 `missed_cycles`。它不会突发补发所有漏掉的 target。

### 19.6 Stop 行为

`trajectory.stop()` 设置 event，并最多等待线程 2 秒，然后把内部 thread 引用置空。若线程因某种原因在 2 秒后仍存活，当前实现不会继续跟踪它。

---

## 20. Policy rollout 状态机

### 20.1 启动

Policy runner 不调用完整 `TeleopSession.prepare()`，而是只执行：

```text
start cameras
controller.connect()
controller.enable()
```

因此真机 policy 模式：

- 不启动 SpaceMouse 线程；
- 不初始化夹爪；
- 使用 FakeSpaceMouse 作为 session 占位对象。

### 20.2 每轮 rollout 前

```text
if previous servo active: stop_servo()
go_home(wait=True)
sync active target from feedback
clear observation history
wait for camera frame
warm up policy once per process
wait for key 's'
```

Policy warm-up 只在第一次 rollout 前执行；每次正式 rollout 都调用 `policy.reset()`。

### 20.3 Rollout 启动延迟

默认设置：

```text
POLICY_START_DELAY_S = 1.0
RTC_ACTION_EXEC_LATENCY_S = 0.01
```

代码建立 wall clock 与 monotonic clock 的偏移，把 observation/action 的 wall timestamp 转换为本机单调时钟排程。

### 20.4 RTC 模式

默认：

- action rate = 10 Hz；
- 每次推理处理 6 step；
- 每轮推理窗口约 0.6 s；
- trajectory stream = 125 Hz。

流程：

1. capture observation；
2. 执行 policy inference；
3. 转换 action chunk；
4. 每个 action 的 wall timestamp 为 `obs_timestamp + (index + offset) * action_dt`；
5. 丢弃不晚于 `now + 10 ms` 的 action；
6. 把剩余 target 转 monotonic 时间并排程；
7. 轨迹线程继续执行；
8. 主线程采集 observation、记录数据、检查按键与错误；
9. 到下一个 inference window 后重复。

若整段 action 都过期：

- 只保留最后一个 target；
- 把它排到下一个 eval 时间网格；
- 输出 over-budget 状态提示。

### 20.5 Stop-go 模式

流程：

1. capture observation；
2. inference；
3. 只取前 `steps_per_inference` 个 action；
4. 从当前 monotonic 时间开始，以 `(1..N)*action_dt` 排程；
5. 等到最后一个 action 时间；
6. 再进行下一次 inference。

虽然名为 stop-go，执行期间仍由轨迹线程持续插值和发送 target；“停”主要指推理和 action chunk 执行不重叠，而不是每个 action 之间停止机械臂。

### 20.6 人机交互

| 状态 | 按键 | 行为 |
| --- | --- | --- |
| HOME | `s` | 开始 rollout |
| HOME | `q` | 退出 |
| rollout | `c` | 停止本轮 |
| rollout | `q` | 停止并退出 |
| rollout 后 | space | 返回 HOME，进入下一轮 |
| rollout 后 | `q` | 退出 |

OpenCV 窗口 Esc 被映射为 `q`。

### 20.7 Rollout 结束

无论正常、按键停止还是异常进入 `_policy_control_loop` 的 finally：

1. 读取轨迹 status 的最新 command pose；
2. 停止轨迹线程；
3. 若 writer 存在则关闭并把路径加入 saved list；
4. 同步 TeleopSession 内部 target。

这里不会立即调用 `controller.stop_servo()`。`_servo_active` 保持 True，直到：

- 操作者按 space 进入下一轮 HOME，`_move_home_before_policy()` 先 stop servo；或
- 整个 runner 退出，`session.shutdown()` stop servo。

因此 rollout 停止后的等待阶段处于 mode 1、无轨迹线程持续喂 target 的状态。

### 20.8 Policy 录制

若 output 开启，每个 rollout 创建：

```text
policy_episode_YYYYMMDD_HHMMSS.hdf5
```

控制频率 metadata 使用 trajectory rate。记录的 action 是 `_active_target_pose`，robot state 是 SDK feedback。Policy 没有 Teleop delta，因此 `delta_action` 中对应部分为 NaN。

与遥操作连续录制不同，policy writer 在 rollout finally 中关闭并保存；若推理中途抛出异常，只要 writer 已创建，部分 episode 通常仍会保留。

---

## 21. 时间与同步模型

项目同时使用两种时钟：

| 时钟 | 典型用途 |
| --- | --- |
| `time.time()` | HDF5 时间戳、机器人状态时间、相机 receive time、RTC wall timestamp |
| `time.monotonic()` | 控制 deadline、睡眠、轨迹排程、超时 |

关键边界：

1. XArm 状态 timestamp 是读取完缓存字段后的本机 wall time。
2. 相机 receive time 是取到 frameset 后的本机 wall time。
3. RealSense capture time 来自设备，但不进入 HDF5 或 policy timestamp。
4. Observation timestamp 是各 receive/state 时间的最大值。
5. 录制 episode timestamp 是固定理想网格。
6. RTC 通过一次性 `wall_to_mono` 偏移转换排程，没有持续校正 wall clock 跳变。

因此项目提供的是“应用层近似时间一致性”，不是硬实时或硬件同步系统。

---

## 22. 错误处理与资源生命周期

### 22.1 自定义异常

| 异常 | 含义 |
| --- | --- |
| `XArmTeleopError` | 项目异常基类 |
| `ConfigError` | 配置缺失或非法 |
| `DeviceError` | SpaceMouse/RealSense 设备错误 |
| `XArmNotConnected` | 未连接时调用控制器 |
| `SafetyError` | 应用层 workspace 越界 |
| `XArmCommandError` | SDK 返回非零 code 或缓存 error code |

### 22.2 主循环 finally

- Teleop：停止 SpaceMouse，再 stop servo。
- 录制：关闭/丢弃 writer，再 shutdown session；CLI finally 停预览和相机。
- Policy：shutdown session、停相机、关 OpenCV 窗口。
- Trajectory：异常保存到 status，由主循环发现。

### 22.3 没有统一处理的状态

项目没有统一 watchdog 检查：

- `connected` 变为 false；
- SDK `state` 不再是预期值；
- SDK `mode` 被外部改变；
- warning code 非零；
- cmd_num 持续堆积；
- SpaceMouse timestamp 过期；
- command 与 feedback 偏差持续扩大；
- 轨迹线程 stop join 超时后仍存活。

### 22.4 清错策略

每次 enable/start servo 都会自动清除当前 error/warn。代码不记录错误发生原因，也不区分可恢复错误、碰撞、急停、通信故障或硬件故障。

---

## 23. Dry-run 与测试替身

### 23.1 DryRunArm

`DryRunArm` 模拟 XArmAPI：

- 所有方法通常返回 0；
- mode/state/error/warn 是普通字段；
- `set_position` 和 `set_servo_cartesian` 会直接把目标写入 `position`；
- `set_servo_angle` 直接更新 `angles`；
- 夹爪位置立即变化；
- 所有调用记录在 `calls`。

它适合验证调用顺序、参数、状态机和数据格式，但没有：

- 运动学；
- 关节限制；
- 延迟；
- 网络错误；
- 控制器状态转换延迟；
- 碰撞或跟踪误差。

三个主工作流的 `--dry-run` 范围不同：

| 工作流 | Fake XArm | Fake cameras | Fake SpaceMouse | Hold policy fallback |
| --- | --- | --- | --- | --- |
| 纯遥操作 | 是 | 不适用 | **否，仍创建真实 SpaceMouseReader** | 不适用 |
| 遥操作录制 | 是 | 是 | 是 | 不适用 |
| Policy eval | 是 | 是 | 是 | policy spec 为 null 时允许 |

因此 `xarm-teleop --dry-run` 不是完全隔离硬件的启动检查：CLI 仍会尝试启动真实 SpaceMouseReader，而不会自动替换 FakeSpaceMouse。若 reader 线程启动失败，当前实现也缺少向主线程传播该异常的通道。录制和 policy 的 dry-run 才会从 CLI 层替换相应输入设备。

### 23.2 FakeSpaceMouse

始终返回零 action 和未按下按钮，供 policy 或 dry-run 使用。

### 23.3 FakeRealSenseManager

生成固定尺寸彩色图案和可选深度图，timestamp 由注入 clock 提供。它不启动线程。

---

## 24. 自动化测试现状

本次审计在 `real_robot` 环境运行：

```text
75 passed in 0.87s
```

测试覆盖：

- 配置默认值、递归覆盖和校验；
- HOME 分离与保存语义；
- SpaceMouse 变换和 deadzone；
- TCP-local delta 辅助函数；
- workspace 越界拒绝；
- 夹爪 enable/toggle；
- 遥操作按钮边沿、模式、增量和目标积分；
- 连续 episode 开始、保存、丢弃、HOME 返回；
- 记录频率约束；
- HDF5 字段、维度、attrs 和三种旋转表示；
- WebSocket payload 与返回值；
- policy action mode 和 pose 转换；
- observation history padding；
- RTC 过期 action 过滤；
- stop-go action slice；
- XYZ 线性插值、姿态 Slerp；
- 轨迹 target 流频率和 missed cycle 状态；
- 相机预览和 HDF5 inspector。

未覆盖或不能由 fake 可靠覆盖：

- 真机 SDK 网络重连；
- 固件 mode/state 的真实切换时序；
- 真实 `set_servo_cartesian` 稳定频率；
- 设备断开时 SpaceMouse 缓存行为；
- 控制器 collision/error code 恢复；
- 真实夹爪阻塞对 servo 流的影响；
- 三相机硬件同步和重复帧；
- IK 不可达、奇异位形、关节极限和实际碰撞；
- CPU 负载、GIL、OpenCV 与 HDF5 对 125 Hz 线程的影响；
- 长时间运行的时钟漂移和文件完整性。

---

## 25. 已确认但未被主流程使用的代码或配置

| 项目 | 当前状态 |
| --- | --- |
| `controller.move_pose()` | 已实现，三个主 CLI 不调用 |
| `controller.servo_tcp_delta()` | 已实现并测试，主遥操作使用 `servo_pose()` |
| `compute_tcp_delta_target()` | 仅由辅助方法/测试使用 |
| `recording.timestamps.*` | 导出但主录制不调用 |
| `RecordingTeleopRunner` | 已实现，但主录制 CLI 使用 continuous runner |
| `servo.command_timeout_s` | 解析但未使用 |
| `teleop.left_button/right_button` | 解析但行为仍硬编码到索引 0/1 |
| `web_policy.protocol` | 校验与日志使用，未显式加入 adapter payload |
| SDK `set_timeout()` | 文档列出，控制器未调用 |
| SDK state/error callbacks | 未注册 |
| SDK limit/IK/FK API | 文档列出，主控制器未调用 |
| `real_world/` RTDE 代码 | 不属于 XArm7 主链路 |
| 顶层 `spacemouse/` | 不属于打包后的主输入实现 |

---

## 26. 安全与可靠性审计结论

以下优先级是本文审计判断，不是仓库原有分级。

### 26.1 P0：输入和命令 watchdog 缺失

**事实：**

- `command_timeout_s` 未使用；
- SpaceMouse 缓存最后 action；
- 主循环不检查输入 timestamp；
- SpaceMouse 线程异常不会像相机一样传播。

**可能后果：**

设备或读取线程在非零输入时异常，应用层没有明确机制立即把输入归零并停机。

**建议：**

- 每周期检查 `now - spacemouse_timestamp`；
- 超时立即发送 hold/stop，并进入锁定状态；
- 给 SpaceMouseReader 增加 error/alive 状态；
- 输入恢复后要求人工确认再继续。

### 26.2 P0：应用层安全模型不足以证明真机路径安全

**事实：**

只有 XYZ workspace；没有本地 IK、关节限位预检、碰撞、奇异性和环境模型。

**可能后果：**

workspace 内的 TCP 点仍可能对应不可达姿态、危险关节构型或穿越环境障碍的路径。

**建议：**

- 至少接入 SDK `is_tcp_limit/is_joint_limit/get_inverse_kinematics`；
- 对 HOME 和 policy waypoint 做预检；
- 若任务环境复杂，引入真实机器人模型和碰撞场景；
- 明确控制器碰撞灵敏度、TCP offset、load 和重力配置由谁负责。

### 26.3 P0：录制与 policy HOME 缺少本地校准强制门槛

**事实：**

- 纯遥操作脚本要求本地 HOME；
- 录制默认直接使用打包的 `[0,0,40,0,0,0,0]`；
- policy 使用另一组打包 HOME；
- 两者都没有检查 HOME 是否由当前机器保存。

**建议：**

- 录制和 policy 也要求显式本地 HOME 文件；
- 文件中绑定机器人 serial、工具配置和工位版本；
- 启动时先打印并要求人工确认；
- 首次移动使用更低速度并提供预检模式。

### 26.4 P1：servo target 流存在主动断流窗口

已确认的断流窗口：

- Teleop pause；
- 录制 HOME 后等待开始；
- 阻塞式夹爪动作；
- policy rollout 停止后等待 space；
- 主线程中其他意外长耗时操作。

SDK 文档要求 mode 1 下持续喂 target，但具体断流效果取决于控制器固件。当前应用层没有独立 watchdog 或专门的 hold streaming thread。

**建议：**

- 进入 mode 1 后由独立固定频率线程始终发送最新安全 target；
- 业务线程只更新 target；
- pause/等待状态发送明确 hold；
- 夹爪默认改为非阻塞并单独管理完成状态。

### 26.5 P1：错误恢复和状态监控不足

**事实：**

- 启动时自动清错；
- 没有 error/state/connect callback；
- 不监控 warn、cmd_num 或 mode 漂移；
- `servo_pose()` 无命令后缓存健康检查；
- 错误原因不分类。

**建议：**

- 建立统一健康状态机；
- 对碰撞、急停、通信、超速和硬件故障区别处理；
- 清错前记录并显示完整错误信息；
- 需要人工介入的错误禁止自动恢复。

### 26.6 P1：应用层急停未接入操作界面

`emergency_stop()` 已封装，但没有键盘/SpaceMouse 绑定。正常退出会恢复 mode 0/state 0 并保持使能。

**建议：**

- 继续以物理急停为最终安全手段；
- 增加独立软件 stop 键，语义和 UI 明确区别于 pause/quit；
- 软件 stop 后不要自动回 ready，要求人工复位。

### 26.7 P1：传感器与动作时间对齐不可追溯

当前 HDF5 不保存相机真实时间、frame number、SpaceMouse 时间和机器人状态时间。

**可能后果：**

- 无法离线识别重复帧；
- 无法精确估计 observation-action 延迟；
- 无法验证多相机同步质量；
- RTC 训练/评测时间语义难以复现。

**建议：**

- 为每路相机保存 capture/receive timestamp 和 frame number；
- 保存 robot/spacemouse timestamp；
- 明确 action timestamp 是生成、计划还是执行时间；
- 使用已存在的 nearest timestamp 工具或更严格的插值/同步流程。

### 26.8 P2：配置表面语义与实际语义不一致

- `command_timeout_s` 不生效；
- left/right button 文本不控制行为；
- servo speed/acc 被传给 reserved 参数；
- 同一 `servo_speed` 又被 policy 当作软件 mm/s 限速；
- protocol 字段没有显式进入 adapter payload。

建议删除无效项、接入真实行为，或在 schema/documentation 中明确标记。

### 26.9 P2：环境与依赖可复现性

- 当前机器缺少脚本默认环境；
- SDK 未锁版本；
- pyproject hardware extra 未包含 RealSense；
- WebSocket fallback 错误信息包含外部开发路径。

建议提供锁定文件、环境自检命令和不依赖个人路径的安装说明。

### 26.10 P2：连接与线程资源收尾

- 长运行 CLI 不显式 disconnect；
- trajectory stop 超时后丢失 thread 引用；
- preview/camera thread stop 使用有限 join timeout。

建议建立统一 context manager/lifecycle manager，收尾时检查每个线程和 SDK 连接都已结束。

---

## 27. 真机运行前检查清单

### 27.1 环境

- [ ] 已创建并激活预期 Conda 环境；
- [ ] 已确认 `xarm-python-sdk` 版本；
- [ ] `pip install --no-build-isolation -e .` 成功；
- [ ] SpaceMouse udev 权限正确；
- [ ] RealSense 数量、serial、分辨率和 fps 与 YAML 一致；
- [ ] 不存在 ROS `PYTHONPATH` 污染。

### 27.2 机器人

- [ ] IP 与控制箱一致；
- [ ] 当前工具/TCP offset 已在控制器或外部流程正确配置；
- [ ] TCP load、重心与安装方向正确；
- [ ] 控制器碰撞灵敏度符合任务；
- [ ] 物理急停可立即触达；
- [ ] workspace 与真实桌面、支架、相机位置一致；
- [ ] HOME 在当前工位安全，且回 HOME 路径无遮挡；
- [ ] 夹爪 open/close 脉冲范围与实际硬件一致。

### 27.3 低风险软件检查

```bash
xarm-read-state --dry-run
xarm-test-gripper --dry-run --pause 0
xarm-teleop --dry-run
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

### 27.4 真机首次运行

- [ ] 先用 `xarm-read-state` 确认 mode/state/error/warn；
- [ ] 先保存并人工检查 HOME；
- [ ] 降低 `speed_scale`；
- [ ] 暂不启用夹爪与相机预览，先验证机械臂；
- [ ] 确认 SpaceMouse 静止时六轴均为零；
- [ ] 分别测试 X/Y/Z 正方向和姿态方向；
- [ ] 在 workspace 边界前留出物理安全余量；
- [ ] 再逐步启用录制和 policy。

---

## 28. 推荐改造顺序

### 第一阶段：安全闭环

1. 实现 SpaceMouse freshness watchdog；
2. 建立独立持续 servo/hold 线程；
3. 增加软件 stop 与人工复位状态；
4. 接入 SDK connection/state/error callback；
5. 对 HOME、teleop、policy 使用统一健康门禁。

### 第二阶段：运动可行性

1. SDK TCP/joint limit 检查；
2. IK 预检；
3. 关节速度和 command-feedback error 限制；
4. 明确 TCP/load/collision 参数；
5. 根据工位需要增加机器人模型和碰撞场景。

### 第三阶段：数据可靠性

1. 保存所有传感器真实时间戳；
2. 保存 frame number、error/warn/mode/state；
3. 检测重复帧与相机掉帧；
4. 定义 action 生成/计划/执行时间；
5. 增加 episode 原子写入和异常恢复标记。

### 第四阶段：Policy 能力

1. 增加夹爪 action schema；
2. 明确 delta pose 坐标系；
3. 对 workspace clip 提供 reject/terminate 选项；
4. 增加 inference timeout；
5. 对远程协议加入版本握手与 shape/dtype 全量校验。

### 第五阶段：工程化

1. 锁定依赖版本；
2. 修正环境启动与自检；
3. 清理未生效配置和个人路径；
4. 增加真机集成测试；
5. 建立长期运行、断联和错误注入测试。

---

## 29. 建议阅读顺序

理解项目时建议依次阅读：

1. `README.md`：使用场景和操作入口；
2. `xarm_teleop/config/tele_record.yaml`：机器人与录制默认值；
3. `xarm_teleop/controllers/xarm7_controller.py`：SDK 封装；
4. `xarm_teleop/teleop/session.py`：遥操作目标生成；
5. `xarm_teleop/input/spacemouse.py`：输入坐标与线程；
6. `xarm_teleop/recording/session.py`：episode 状态机；
7. `xarm_teleop/recording/hdf5.py`：数据语义；
8. `xarm_teleop/config/policy_eval.yaml`：policy 参数；
9. `xarm_teleop/policy/observation.py`：observation；
10. `xarm_teleop/policy/actions.py` 和 `pose.py`：动作转换；
11. `xarm_teleop/policy/trajectory.py`：插值 target 流；
12. `xarm_teleop/policy/runner.py`：RTC/stop-go 状态机；
13. `docs/xarm_python_sdk_api.md`：SDK 1.18.4 速查；
14. `tests/`：预期行为和边界条件。

---

## 30. 核心事实速查

| 项目 | 当前事实 |
| --- | --- |
| 机械臂 | XArm7，七关节 |
| Python | 3.10 |
| 已核验 SDK | xarm-python-sdk 1.18.4 |
| 实时执行接口 | `set_servo_cartesian()` |
| 普通 HOME 接口 | `set_servo_angle(..., wait=True)` |
| 遥操作频率 | 默认 100 Hz |
| Policy 轨迹频率 | 默认 125 Hz |
| Policy action rate | 默认 10 Hz |
| 录制频率 | 默认 10 Hz |
| 相机 | 默认三路 RealSense，640×480@30 Hz |
| XArm TCP 单位 | mm + RPY deg（默认） |
| Policy/HDF5 位置 | m |
| Policy/HDF5 关节 | rad |
| Workspace | XYZ 轴对齐长方体 |
| 运动规划 | 无；只有 waypoint 时间插值 |
| 本地 IK/FK | 无 |
| 碰撞模型 | 无 |
| Policy 夹爪 | 未实现 |
| 多进程/共享内存 | 主链路未使用 |
| 自动化测试 | 75 passed（审计时） |
| 当前机器默认环境 | `xarm-mimic` 不存在，需创建或覆盖环境名 |

---

## 31. 最终结论

当前项目已经形成了一条清晰、可测试的 XArm7 应用链：统一 SDK 封装、命令目标积分、固定频率 servo、RealSense 最新帧采集、HDF5 数据转换，以及可重叠推理与执行的 policy 轨迹线程。代码结构相对直接，dry-run 和单元测试对上层语义覆盖良好。

但其可靠性边界同样明确：它依赖 XArm 控制器完成绝大多数底层运动学和安全约束，应用层只检查 XYZ workspace；输入 freshness、持续 servo watchdog、错误状态机、HOME 校准门槛、传感器时间追溯和真机故障测试尚未形成闭环。

因此，当前代码适合作为受控实验环境中的遥操作、数据采集和 policy 执行基础，但不能仅凭现有 workspace 检查和 dry-run 测试就宣称具备完整的真机避障、安全认证或硬实时能力。真机使用必须依赖经过验证的工位配置、物理急停、低速调试和逐步放开的运行流程。
