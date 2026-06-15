# dexmani_real 项目代码风格约束

> **Python 环境**：`conda activate real`（`/home/zhy/anaconda3/envs/real/bin/python`）。运行/测试任何 Python 代码前必须先激活此环境：`source /home/zhy/anaconda3/etc/profile.d/conda.sh && conda activate real`。
>
> **文件定位**：本文档是项目的**活代码风格与接口约束**——定义各模块的接口签名、行为契约和设计原则。它不包含实现细节或工期安排。
> 
> **与 development_plan.md 的关系**：`docs/development_plan.md` 是本约束的**具体实施方案**（含工期、文件清单、验收标准），实施时必须遵循本文档的接口规范。两个文件中的架构图有重复是故意的——CLAUDE.md 供 AI 编码时快速参考，development_plan.md 供开发者通读。
> 
> **参考代码库**：开发前按 P1→P2→P3 优先级检索 6 个参考项目。P1: LeFranX + ManiUniCon, P2: BunnyVisionPro + Open-Teach + DexUMI, P3: Bidex_Manus_Teleop。详见 Section 0.5。
>
> **编码规则文件**：`.claude/rules/` 目录包含按关注点拆分的精简规则文件，供 AI 编码时快速查阅：
> - `coding-conventions.md` — 命名规范、配置管理、模块结构、导入规范
> - `hardware-interface.md` — 硬件驱动接口契约、安全层、线程安全
> - `error-safety.md` — 错误处理、异常粒度、安全监控
> - `reference-protocol.md` — 参考检索协议、Fact-Check 规则、不采纳清单、SDK 参考资源
> - `hardware-safety.md` — 真机操作安全（Pre-Flight 清单、E-Stop 条件、禁止操作）
> - `sdk-dependencies.md` — 外部 SDK 版本/API/陷阱速查（xArm/XHand/HTS/dex-retargeting/MPlib/SAPIEN）
>
> 本文档是权威参考（含完整的接口签名和数据流），规则文件是精简的行动指南。两者冲突时以本文档为准。
>
> **用户文档**：`docs/user/` 目录包含硬件安装、SDK 编译、驱动 API 及遥操作模块的使用说明（xArm7、XHand、RealSense、L515、Quest VR）。

## 0. 项目架构速览

### 进程与数据流

```
VR 进程(80Hz) ──► ring["vr_frame"] ──┐
                                     ├── Controller 进程(50Hz)
Camera 进程(30Hz) ──► ring["camera"] ─┘     │
                                            ├─ _tick(): 同一份 VR frame 同时驱动 arm + hand
                                            │    arm: IK(pose, qpos, prev) → EMA(alpha=0.3) → arm_cmd
                                            │    hand: retarget(landmarks) → EMA(alpha=0.3) → hand_cmd
                                            ├─ robot.send_action()
                                            └─ [RECORDING] recorder.add_frame()

Keyboard ──► multiprocessing.Queue ──► 控制信号(T/R/S/H/ESC)
--no-ipc 回退到单进程
```

### Episode 生命周期（LeFranX 风格）

```
for each episode:
  1. robot.return_to_home()           # 路径规划回 home + hand 复位
  2. arm_mapper.reset(vr_frame, eef)  # re-anchor VR reference
  3. recorder.start_episode()         # 创建 .h5
  4. [teleop 录制...]
  5. recorder.stop_episode()          # 写入 .h5，存入 norm_stats
```

### 控制器状态机

```
启动 → IDLE ──T──► TELEOP ──R──► RECORDING ──S──► IDLE
         │            │               │
         H            S               │ 追踪丢失>1s / IK连败>10
         │            │               │
         ▼            ▼               ▼
    return_to_home   IDLE         EMERGENCY_STOP → 退出
    (规划路径)
```

---

## 0.5 开发参考约束（Reference-First Principle）

### 0.5.1 核心原则

开发新功能/模块前，必须先按优先级检索参考项目中对应模块的实现，理解其设计意图和边界条件，再结合本项目实际情况（硬件差异、依赖差异）决定采纳方式。禁止不参考现有实现就凭空编写代码。

**检索优先级：**
1. **P1（最高）** — LeFranX、ManiUniCon: 本项目的直接方法论来源，决定架构设计和接口风格
2. **P2（中等）** — BunnyVisionPro、Open-Teach、DexUMI: 与本项目硬件高度重叠（xArm7 + XHand + Quest VR），提供已验证的硬件控制模式、遥操作流程、数据采集 pipeline 和策略部署架构
3. **P3（补充）** — Bidex_Manus_Teleop: 手部数据手套和 IK retargeting 的补充参考，仅在特定场景查阅

**逐级检索规则：** P1 决定"怎么做架构"，P2 决定"怎么写硬件代码"，P3 补充特殊场景。当高优先级参考没有对应模块时，逐级向低优先级查找。同优先级内有冲突时，结合本项目硬件选最合适的方案。

### 0.5.2 参考代码库位置

| 优先级 | 系统 | 本地路径 | 远程 |
|--------|------|---------|------|
| P1 | LeFranX | /home/zhy/Desktop/LeFranX/ | https://github.com/wengmister/LeFranX |
| P1 | ManiUniCon | /home/zhy/Desktop/ManiUniCon/ | — |
| P2 | BunnyVisionPro | /home/zhy/Desktop/BunnyVisionPro/ | https://github.com/Dingry/BunnyVisionPro |
| P2 | Open-Teach | /home/zhy/Desktop/Open-Teach/ | https://github.com/NYU-robot-learning/Open-Teach |
| P2 | DexUMI | /home/zhy/Desktop/DexUMI/ | https://github.com/real-stanford/DexUMI |
| P3 | Bidex_Manus_Teleop | /home/zhy/Desktop/Bidex_Manus_Teleop/ | — |

### 0.5.3 模块参考映射

#### robot/ 硬件驱动

| 优先级 | 参考来源 | 路径 | 说明 |
|--------|---------|------|------|
| P1 | ManiUniCon | maniunicon/robot_interface/xarm6_robotiq.py | xArm6 接口，与 xArm7 SDK 最接近 |
| P1 | ManiUniCon | maniunicon/robot_interface/base.py | RobotInterface 基类设计 |
| P1 | LeFranX | src/lerobot/robots/franka_fer_xhand/ | Franka+XHand 复合接口设计 |
| P1 | LeFranX | src/lerobot/robots/xhand/xhand.py | XHand 接口参考 |
| P2 | BunnyVisionPro | real_control/xarm7_ability.py | **xArm7 真机控制（PID、servo 模式、Ability Hand 集成）** |
| P2 | Open-Teach | openteach/robot/robot.py | RobotWrapper ABC 抽象模式（多机器人接口） |
| P2 | Open-Teach | openteach/robot/franka.py | Franka 驱动实现参考 |
| P2 | DexUMI | dexumi/hand_sdk/dexhand.py | **DexterousHand/ExoDexterousHand ABC 设计（XHand + Inspire 双实现）** |
| P2 | DexUMI | dexumi/hand_sdk/xhand/hand_api_cls.py | **XHand SDK 封装（后台读取线程 + joint clip + 触觉读取）** |
| P2 | DexUMI | dexumi/real_env/common/ur5.py | UR5 servoL 伺服控制（ZMQ Server/Client 分离模式） |
| * | 本项目 | robot/xhand.py | XHand 模板（遵循 CLAUDE.md 接口规范） |

#### controller/ 遥操作控制器

| 优先级 | 参考来源 | 路径 | 说明 |
|--------|---------|------|------|
| P1 | LeFranX | src/lerobot/teleoperators/franka_fer_xhand_vr/ | Franka+XHand+VR 遥操作主循环 |
| P1 | LeFranX | src/lerobot/teleoperators/franka_fer_vr/arm_ik_processor.py | Arm IK 处理流程 |
| P1 | ManiUniCon | maniunicon/policies/quest.py | Quest VR 遥操作策略 |
| P1 | ManiUniCon | maniunicon/utils/quest_controller.py | Quest 控制器工具 |
| P2 | BunnyVisionPro | real_control/teleop_bimanual_xarm7_ability.py | **xArm7 + Ability Hand 双手遥操作主循环（含录制、状态机、键盘控制）** |
| P2 | Open-Teach | openteach/components/operators/operator.py | Operator 模式：retargeting → streaming 控制循环 |
| P2 | Open-Teach | openteach/components/operators/allegro.py | 手部 operator 实现（移动平均滤波 + 关节限位） |
| P2 | DexUMI | real_script/teleoperation/teleoperation.py | **外骨骼遥操作主循环（iPhone 手腕追踪 + 外骨骼编码器手指追踪 + UR5 servoL）** |
| P2 | DexUMI | dexumi/real_env/common/pose_trajectory_interpolator.py | **位姿轨迹插值器（速度限制 + schedule_waypoint）** |

#### teleop/ VR 追踪 + 手部重定向

| 优先级 | 参考来源 | 路径 | 说明 |
|--------|---------|------|------|
| P1 | LeFranX | src/lerobot/teleoperators/xhand_vr/ | XHand VR 重定向（vr_hand_detector_adapter.py） |
| P1 | LeFranX | src/lerobot/teleoperators/vr_router_manager.py | VR 路由管理 |
| P1 | ManiUniCon | third_party/oculus_reader/ | Oculus/Quest 数据读取 |
| P2 | BunnyVisionPro | examples/retargeting/retargeting.py | dex-retargeting 手部重定向（OPERATOR2MANO/AVP 坐标变换） |
| P2 | BunnyVisionPro | bunny_teleop/bimanual_teleop_client.py | ZMQ-based 双手遥操作 client/server 模式 |
| P2 | BunnyVisionPro | bunny_teleop/init_config.py | 双手对齐模式配置（BimanualAlignmentMode） |
| P2 | Open-Teach | openteach/components/detector/oculus.py | Oculus/Quest 手部关键点检测 |
| P2 | Open-Teach | openteach/robot/allegro/allegro_retargeters.py | Kinematic retargeting（关节限位 + 移动平均滤波） |
| P2 | DexUMI | dexumi/encoder/encoder.py | **外骨骼关节编码器（UART 读取 + 电压→角度转换 + XHand/Inspire 双实现）** |
| P2 | DexUMI | dexumi/encoder/UARTReader.py | UART 串口读取基类（后台线程 + 环形缓冲区） |
| P2 | DexUMI | dexumi/hand_sdk/xhand/hand_api_cls.py | **ExoXhandSDK：外骨骼编码器角度 → 电机值预测（sklearn 回归模型）** |
| P2 | DexUMI | dexumi/camera/iphone_camera.py | **iPhone ARKit (Record3D) 6-DoF 手腕位姿追踪** |
| P3 | Bidex_Manus_Teleop | python/minimal_example.py | **Manus 数据手套 ZMQ 数据解析（skeleton + ergonomics）** |
| P3 | Bidex_Manus_Teleop | ros2/telekinesis/telekinesis/leap_ik.py | **PyBullet IK 指尖重定向（glove fingertip → LEAP hand joint）** |
| P3 | Bidex_Manus_Teleop | ros2/glove/glove/read_and_send_zmq.py | Manus glove ZMQ → ROS2 桥接模式 |

#### ipc/ 进程间通信

| 优先级 | 参考来源 | 路径 | 说明 |
|--------|---------|------|------|
| P1 | ManiUniCon | maniunicon/utils/shared_memory/ | 完整 shared memory 实现（ring buffer、queue、ndarray） |
| P1 | LeFranX | src/lerobot/teleoperators/vr_router_manager.py | VR 数据路由模式 |
| P2 | BunnyVisionPro | bunny_teleop/bimanual_teleop_client.py | ZMQ + threading 多进程遥操作数据传递 |
| P2 | DexUMI | dexumi/real_env/common/base.py | **ZMQServerBase/ZMQClientBase 通用 IPC 框架（PUB/SUB + ROUTER/DEALER）** |
| P2 | DexUMI | dexumi/real_env/ring_buffer.py | **线程安全 RingBuffer（deque + RLock）** |

#### recording/ 数据录制

| 优先级 | 参考来源 | 路径 | 说明 |
|--------|---------|------|------|
| P1 | LeFranX | scripts/dual_robot/dual_vr_record.py | VR 录制流程 |
| P1 | ManiUniCon | maniunicon/utils/replay_buffer.py | Replay buffer 设计 |
| P1 | ManiUniCon | maniunicon/sensors/replay.py | 传感器数据回放/录制 |
| P2 | BunnyVisionPro | real_control/teleop_bimanual_xarm7_ability.py | HDF5 录制集成在遥操作主循环中的完整模式 |
| P2 | Open-Teach | data_collect.py | 多进程组件式数据采集 pipeline |
| P2 | DexUMI | dexumi/data_recording/record_manager.py | **RecorderManager 模式（多 Recorder 编排 + episode 生命周期管理）** |
| P2 | DexUMI | dexumi/data_recording/video_recorder.py | **视频录制器（多相机源 + 帧队列 + 录制/流分离线程）** |
| P2 | DexUMI | dexumi/data_recording/numeric_recorder.py | **数值数据录制器（Zarr 存储 + 后台录制线程 + episode 管理）** |
| P2 | DexUMI | real_script/data_collection/record_exoskeleton.py | **外骨骼数据采集主脚本（多传感器同步录制完整流程）** |

#### deploy/ 策略部署

| 优先级 | 参考来源 | 路径 | 说明 |
|--------|---------|------|------|
| P1 | LeFranX | src/lerobot/scripts/server/robot_client.py | 策略客户端 |
| P1 | LeFranX | src/lerobot/scripts/server/policy_server.py | 策略服务端 |
| P1 | LeFranX | scripts/dual_robot/dual_robot_deploy_act.py | ACT 策略部署 |
| P1 | ManiUniCon | maniunicon/customize/act_wrapper/chunk_wrapper.py | Action chunk 包装 |
| P1 | ManiUniCon | maniunicon/customize/obs_wrapper/ | 观测构建包装 |
| P1 | ManiUniCon | maniunicon/policies/torch_model.py | Torch 模型加载与推理 |
| P2 | Open-Teach | deploy_server.py | 策略部署服务端（多机器人支持） |
| P2 | DexUMI | dexumi/real_env/common/policy.py | **PolicyServer/PolicyClient 通用框架（ZMQ IPC + obs_config 验证 + preprocess/inference 分离）** |
| P2 | DexUMI | dexumi/real_env/dexumi_policy.py | **DexUMIPolicySever（Diffusion Policy 模型加载 + 推理 + 归一化/反归一化）** |
| P2 | DexUMI | dexumi/real_env/real_policy.py | **RealPolicy：Diffusion Policy 推理（visual obs 预处理 + action chunk 输出）** |
| P2 | DexUMI | real_script/eval_policy/eval_xhand.py | **XHand 策略评估脚本（真实硬件部署）** |

#### sensor/ 传感器

| 优先级 | 参考来源 | 路径 | 说明 |
|--------|---------|------|------|
| P1 | ManiUniCon | maniunicon/sensors/realsense.py | RealSense 驱动 |
| P2 | Open-Teach | robot_camera.py / fish_eye_camera.py | 相机传感器抽象与 fish-eye 处理 |
| P2 | DexUMI | dexumi/camera/camera.py | **Camera ABC + FrameData/FrameNumericData dataclass（含位姿/内参/时间戳）** |
| P2 | DexUMI | dexumi/camera/realsense_camera.py | RealSense 相机实现 |
| P2 | DexUMI | dexumi/camera/oak_camera.py | OAK-D 立体相机实现 |
| P2 | DexUMI | dexumi/encoder/fsr.py | **FSR 力传感器驱动** |
| P2 | DexUMI | dexumi/encoder/xhand_tactile.py | XHand 指尖触觉传感器读取 |
| * | 本项目 | sensor/realsense.py | 本项目的 RealSense 模板 |

#### planner/ 运动规划

| 优先级 | 参考来源 | 路径 | 说明 |
|--------|---------|------|------|
| P1 | ManiUniCon | maniunicon/utils/ik_solver.py | IK solver 实现 |
| P1 | ManiUniCon | maniunicon/utils/pose_trajectory_interpolator.py | 位姿轨迹插值 |
| P1 | ManiUniCon | maniunicon/utils/ruckig_utils.py | Ruckig 轨迹生成（可选参考） |
| P1 | LeFranX | src/lerobot/teleoperators/franka_fer_vr/arm_ik_processor.py | IK 处理流程 |
| P2 | DexUMI | dexumi/real_env/common/pose_trajectory_interpolator.py | **PoseTrajectoryInterpolator（scipy Slerp + 速度限制 + schedule_waypoint）** |
| P2 | DexUMI | dexumi/real_env/common/motor_trajectory_interpolator.py | 电机轨迹插值（外骨骼手部回放用） |
| P3 | Bidex_Manus_Teleop | ros2/telekinesis/telekinesis/leap_ik.py | PyBullet SDLS IK（glove fingertip → robot hand joint） |

#### utils/ 工具

| 优先级 | 参考来源 | 路径 | 说明 |
|--------|---------|------|------|
| P1 | ManiUniCon | maniunicon/utils/filter.py | 滤波器实现 |
| P1 | ManiUniCon | maniunicon/utils/math_utils.py | 数学工具 |
| P1 | ManiUniCon | maniunicon/utils/pcd_utils.py | 点云工具 |
| P1 | ManiUniCon | maniunicon/utils/timestamp_accumulator.py | 时间戳对齐 |
| P2 | Open-Teach | openteach/utils/vectorops.py | 向量运算工具 |
| P2 | Open-Teach | openteach/utils/network.py | ZMQ publisher/subscriber 网络工具 |
| P2 | Open-Teach | openteach/utils/timer.py | 控制循环计时器 |
| P2 | DexUMI | dexumi/common/precise_sleep.py | **高精度 sleep/wait（hybrid sleep+spin 最小化 jitter）** |
| P2 | DexUMI | dexumi/common/frame_manager.py | **FrameRateContext：速率限制器上下文管理器** |
| P2 | DexUMI | dexumi/common/utility/matrix.py | 矩阵/位姿变换工具（relative/invert transformation） |
| P2 | DexUMI | dexumi/common/data.py | 通用数据结构定义 |
| P3 | Bidex_Manus_Teleop | steamvr/triad_openvr-master/triad_openvr.py | SteamVR/OpenVR 追踪器数据读取 |

### 0.5.4 参考检索协议（Reference Retrieval Protocol）

开发某模块时，执行以下检索流程：

```
1. 查 Section 0.5.3 映射表，按优先级 P1→P2→P3 列出所有相关参考文件
2. 必须实际 Read P1 参考文件（LeFranX + ManiUniCon），理解核心设计意图
3. 如果 P1 参考未覆盖当前子问题:
   a. 向下查 P2 参考 — BunnyVisionPro 优先于 Open-Teach（硬件更接近）
   b. P2 未覆盖再查 P3 — Bidex_Manus_Teleop（仅手部数据手套相关场景）
4. 如果所有参考均无对应实现:
   a. 标记为「无参考实现」，需自行设计
   b. 设计时仍需遵循 CLAUDE.md 接口规范
5. 在实现注释中标注参考来源，格式:
   "# ref: [P1] LeFranX arm_ik_processor.py L120-150"  或
   "# ref: [P2] BunnyVisionPro xarm7_ability.py L80-100"
```

**P2 项目选择规则：**
- **BunnyVisionPro**: xArm7 硬件控制、双手遥操作、手部 retargeting 时优先查（硬件 xArm7 + Ability Hand 最接近本项目）
- **Open-Teach**: 多机器人抽象模式、组件生命周期、VR 检测器模式时优先查
- **DexUMI**: XHand 硬件封装、数据录制 pipeline（RecorderManager 模式）、策略部署架构（PolicyServer/Client）、外骨骼编码器、位姿轨迹插值、高精度计时工具时优先查

### 0.5.5 Fact-Check 规则

1. 必须实际 Read 参考代码文件，确认其行为，不基于猜测或记忆
2. 验证参考实现是否适用于本项目硬件：
   - Franka → xArm7: 运动学模型不同，力矩控制 vs 位置控制
   - ManiUniCon xArm6 → xArm7: SDK API 可能有版本差异
   - BunnyVisionPro XArm7 PID 控制: 可直接参考硬件调用方式（同型号）
   - Open-Teach Allegro → XHand: 手型结构不同，retargeting 逻辑需适配，仅参考滤波和限位模式
   - DexUMI UR5 → xArm7: 运动学模型不同（6-DOF vs 7-DOF），servoL 控制模式 vs set_servo_angle_j。仅参考 ZMQ Server/Client 分离架构和位姿插值模式，不直接复用 UR5 控制代码
   - DexUMI iPhone ARKit → Quest VR: 手腕追踪方式不同，仅参考坐标变换链（T_ET offset）的设计模式
   - DexUMI 外骨骼编码器 → VR hand tracking: 手指追踪输入源完全不同，仅参考 encoder→motor 的校准模型加载模式（pickle sklearn 模型）
   - DexUMI Zarr → HDF5: 存储格式不同，仅参考 RecorderManager 编排模式和 episode 生命周期管理
   - DexUMI ZMQ IPC → Shared Memory: 通信机制不同，仅参考 Server/Client 分离架构和请求类型枚举模式
   - Bidex LEAP Hand → XHand: 手型结构不同，仅参考 IK 方法（SDLS）
3. 如果参考代码与 CLAUDE.md 项目约束冲突，以项目约束为准并记录差异原因
4. 参考前确认代码许可协议兼容
5. 多参考库实现有冲突时，按优先级解决：P1 > P2 > P3，同优先级按硬件接近度选择

### 0.5.6 已确认不采纳的外部设计（避免重复评估）

| 不采纳 | 来源 | 原因 |
|--------|------|------|
| C++ 实时 server (libfranka + Ruckig) | LeFranX | xArm7 内置伺服控制 |
| 解析式 IK (geofik + Brent) | LeFranX | 已有 MPlib 数值 IK + 碰撞检测 |
| LeRobot Parquet 数据集 | LeFranX | 使用 HDF5 |
| LeRobot draccus CLI | LeFranX | 不引入该框架依赖 |
| Hydra 配置管理 | Open-Teach | 不引入 Hydra 依赖，本项目使用 @dataclass |
| ROS/ROS2 通信层 | Open-Teach / Bidex | 本项目不使用 ROS，使用 multiprocessing + shared memory |
| Vision Pro 专用 API | BunnyVisionPro | 本项目使用 Quest 3，仅参考 retargeting 逻辑和 ZMQ 通信模式 |
| Allegro Hand 专用 retargeting | Open-Teach | 手型差异，仅参考滤波和限位模式 |
| Manus Core C++ SDK 直连 | Bidex | 本项目不依赖 Manus 数据手套，仅参考 IK 方法 |
| Pydantic BaseModel（状态/动作类型） | ManiUniCon | 使用 @dataclass + `__post_init__` 验证，避免 Pydantic 依赖 |
| LeFranX 扁平 dict 风格 action（`"joint_0.pos"`） | LeFranX | 使用结构化 RobotAction dataclass |
| `connect()` 抛异常（DeviceAlreadyConnectedError） | LeFranX | 返回 bool，connect() 幂等 |
| `get_observation()` 命名 | LeFranX | 统一使用 `get_state()` |
| Zarr 存储格式 | DexUMI | 本项目使用 HDF5。仅参考 RecorderManager 的编排模式和 episode 生命周期设计 |
| ZMQ IPC 通信 | DexUMI | 本项目使用 multiprocessing + shared memory。仅参考 Server/Client 分离架构和 Request/Response 类型化模式 |
| `DexterousHand` ABC 接口 (`write_hand_angle`/`send_command` 分离) | DexUMI | 本项目统一为 `send_action(np.ndarray) → bool`，单一步骤完成命令构建+发送 |
| `get_current_position()` 命名 | DexUMI | 统一使用 `get_state()` |
| iPhone ARKit (Record3D) 手腕追踪 | DexUMI | 本项目使用 Quest 3 VR，仅参考坐标变换链（T_ET offset）的设计思路 |
| 外骨骼编码器 (UART) 手指追踪 | DexUMI | 本项目使用 VR hand tracking + dex-retargeting，仅参考 encoder→motor 校准模型的加载模式 |
| `ExoDexterousHand.predict_motor_value()` 模式 | DexUMI | 外骨骼专用（encoder → motor 映射），VR 遥操作不需要此步骤 |
| UR5 RTDE servoL 控制 | DexUMI | xArm7 使用 `set_servo_angle_j()`，控制接口不同。仅参考 Server/Client 分离架构 |
| `Recorder` Protocol 类 | DexUMI | 参考其接口设计思想（episode 生命周期），但本项目使用 HDF5 + 具体类 |

### 0.5.7 参考流程

开发新模块时:
1. 查 Section 0.5.3 映射表，按 P1→P2→P3 列出所有相关参考文件
2. 先 Read P1 参考（LeFranX + ManiUniCon），对比异同，确认核心逻辑和架构设计
3. 若 P1 未覆盖当前子问题，Read P2 参考（BunnyVisionPro 优先于 Open-Teach），提取可复用的硬件交互模式
4. 若 P2 仍未覆盖，Read P3 参考（Bidex_Manus_Teleop），仅用于手部数据手套/IK 相关场景
5. 确认参考实现的边界条件
6. 对比本项目硬件/依赖差异，决定采纳/改造/跳过
7. 在实现注释中标注优先级和参考来源（如 "# ref: [P2] BunnyVisionPro xarm7_ability.py L120-150"）

---

## 1. 模块职责边界

| 模块 | 职责 | 不负责 |
|------|------|--------|
| `robot/` | 硬件驱动层，只操作硬件 SDK | 策略推理、训练、数据记录、可视化 |
| `controller/` | 遥操作控制逻辑（_tick、状态机、错误恢复） | 硬件 SDK 调用、数据写入 |
| `recording/` | 数据录制（HDF5 写入、质量标记） | 控制机器人、策略推理 |
| `deploy/` | 策略部署（通过 robot 接口操作硬件） | 直接调用硬件 SDK |
| `sensor/` | 传感器驱动（相机、IMU 等） | 点云观测构造、策略输入构建 |
| `teleop/` | VR 追踪、手部重定向（含 pinky 自适应）、可视化 | 机器人控制、数据录制 |
| `data/` | 离线数据读写（EpisodeReader 等） | 在线采集、硬件操作 |
| `planner/` | 运动规划（纯几何，不依赖硬件） | 硬件 SDK 调用 |
| `utils/` | 小型纯函数工具 | 有状态的业务逻辑 |

---

## 2. 硬件驱动接口规范（robot/*.py）

### 2.1 执行器类设备（Arm、Hand）

**`XHand` 是现有代码中最符合约束的实现，新硬件驱动必须以此模板为参考。**

```python
@dataclass
class DeviceConfig:
    """配置 dataclass，可变默认值使用 default_factory。"""
    dt: float = 1.0 / 50.0
    home_qpos: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))
    qpos_min: np.ndarray = field(default_factory=lambda: ...)
    qpos_max: np.ndarray = field(default_factory=lambda: ...)
    max_qvel: np.ndarray = field(default_factory=lambda: ...)
    use_delta_limit: bool = True
    clip_joint_limit: bool = True

class Device:
    def __init__(self, config: DeviceConfig):
        self.config = config
        # 必须包含的状态变量：
        self.connected_flag = False
        self.error_state = False
        self.last_error_message = ""
        self.last_action_code = None
        self.last_qpos_cmd: np.ndarray | None = None
        self.last_cmd_time: float | None = None
        self.last_joint_limit_clipped = False
        self.last_delta_limited = False

    def connect(self) -> bool: ...
        """最小可用初始化：创建 SDK 对象、打开连接、设控制模式、
        读取初始状态、设 connected_flag=True。失败返回 False 并设 error_state。
        connect() 必须幂等：重复调用已连接设备应直接返回 True，不抛异常。
        （ref: P1 ManiUniCon 的 bool 返回模式；LeFranX 抛 DeviceAlreadyConnectedError 不采纳）"""

    def disconnect(self) -> None: ...

    def get_state(self, full: bool = False) -> dict[str, Any]: ...
        """默认返回 {"qpos", "qvel", "timestamp"}。
        full=True 返回 SDK 原始字段、错误码、内部 flags 等。"""

    def send_action(self, action: np.ndarray) -> bool: ...
        """只接收 np.ndarray 一种动作类型。返回 bool。
        内部做 range clip + delta limit，但不污染返回值。"""

    def reset(self, target: np.ndarray | None = None) -> bool: ...

    def stop(self) -> bool: ...
        """急停/软停。语义必须在文档中写明。"""

    def is_connected(self) -> bool: ...
    def is_error(self) -> bool: ...
    def clear_error(self) -> bool: ...
```

**关于 `__post_init__` 验证**：本项目使用 `@dataclass` 而非 Pydantic（ManiUniCon 的方案，我们不采纳其 Hydra+Pydantic 依赖）。需要在 `__post_init__` 中实现关键验证（shape 一致性、取值范围等），替代 Pydantic 的自动验证功能。

**关于 `RobotState` / `RobotAction`**：复合层的状态和动作类型使用 `@dataclass` 定义，字段类型标注 + `__post_init__` 验证。不引入 Pydantic 依赖。

### 2.2 XArm7 接口说明

`robot/xarm7.py` 已按本约束重构，与 XHand 保持接口一致：
- `connect() → bool`（幂等）、`get_state(full=False) → dict`
- `send_action(np.ndarray) → bool`（含 range clip + delta limit，记录裁剪状态）
- `is_connected() / is_error() / clear_error() / stop() / reset()`
- 完整状态变量（`connected_flag`, `error_state`, `last_*`）
- 简单 `example()` 函数

### 2.3 传感器类设备（Camera 等）

```python
@dataclass
class SensorConfig:
    ...

class Sensor:
    def connect(self) -> bool: ...
    def disconnect(self) -> None: ...
    def get_state(self, full: bool = False) -> dict: ...
    def stop(self) -> bool: ...
    def is_connected(self) -> bool: ...
    def is_error(self) -> bool: ...
    def clear_error(self) -> bool: ...
```

传感器没有 `send_action()`，除非设备确实有主动控制命令。

**注意**：当前 `sensor/realsense.py` 使用 `start()/stop()` 命名。新传感器代码请使用 `connect()/disconnect()` 与硬件驱动接口统一。

### 2.4 动作安全层

真机执行器 `send_action()` 内部至少包含：

```
shape 规整 → 数值类型转换 → 物理范围裁剪(joint limit) → 单步变化量限制(delta limit) → 错误状态检查
```

记录裁剪状态到 `self.last_joint_limit_clipped` / `self.last_delta_limited`，但不污染返回值。

### 2.5 stop() 语义

- 执行器：急停/软停，进入无力模式或停止控制循环
- 传感器：停止数据流，关闭 pipeline

在文档/注释中明确是 `soft stop` 还是硬件急停。

### 2.6 clear_error() 原则

可以清除本地错误状态（`error_state=False`），如果 SDK 有清错接口可调用。但如果无法清除硬件故障，必须在文档中说明需人工处理。

### 2.7 危险操作

标定、固件更新、写 flash 等危险操作不应放入核心驱动默认路径。如需支持，做成独立工具脚本（`tools/calibrate_*.py`），不在 `example()` 中自动执行。

### 2.8 RobotInterface 复合接口（Day 1）

`RobotInterface` 是 arm + hand 的统一上层接口，控制器和部署模块只通过它操作硬件，不直接调 `XArm7`/`XHand`。

```python
@dataclass
class RobotState:
    arm_qpos: np.ndarray       # (7,) rad
    arm_qvel: np.ndarray       # (7,) rad/s
    arm_tau: np.ndarray        # (7,) N·m
    eef_pos: np.ndarray        # (3,) m
    eef_quat_wxyz: np.ndarray  # (4,)
    hand_qpos: np.ndarray      # (12,) rad
    hand_current: np.ndarray   # (12,) mA
    hand_tactile_sum: np.ndarray  # (5,3) N
    hand_temperature: np.ndarray  # (12,) °C
    arm_connected: bool
    hand_connected: bool
    hand_error: bool
    timestamp: float           # seconds

@dataclass
class RobotAction:
    arm_qpos_cmd: np.ndarray       # (7,) rad
    hand_qpos_cmd: np.ndarray      # (12,) rad
    target_eef_pose: np.ndarray | None  # (7,) pos+quat_wxyz，可选

class RobotInterface:
    def connect(self) -> dict[str, bool]: ...
        """连接 arm + hand。返回 {"arm": True, "hand": False} 表示部分连接。
        hand 断连时降级运行（arm 仍可工作）。"""

    def return_to_home(self, use_planning: bool = True,
                       cancel_event = None) -> bool: ...
        """路径规划回 home + hand 复位。
        use_planning=True: planner.plan_path(target_home, current) → 逐点执行
        规划失败时 fallback 直线 reset()
        后跟 reset_hand()"""

    def reset_hand(self) -> bool: ...

    def get_state(self) -> RobotState: ...
        """读取 arm + hand 状态，含 FK 计算 eef_pos/eef_quat。"""

    def send_action(self, action: RobotAction) -> dict: ...
        """发送 arm + hand 动作。
        Returns:
            {"arm_ok": bool, "hand_ok": bool,
             "arm_cmd": ndarray | None,   # (7,) post-clip 实际发送值
             "hand_cmd": ndarray | None}  # (12,) post-clip 实际发送值
        arm_cmd/hand_cmd 经过 joint limit + delta limit 裁剪。
        发送失败时为 None。录制时使用 post-clip 值而非 IK 原始输出。"""

    def emergency_stop(self) -> None: ...
        """arm + hand 同时急停。"""

    def is_connected(self) -> bool: ...
    def is_error(self) -> bool: ...
    def clear_error(self) -> bool: ...
```

**注意**：`RobotInterface` 是复合层，`send_action()` 返回 `{"arm_ok", "hand_ok", "arm_cmd", "hand_cmd"}`。`arm_cmd`/`hand_cmd` 是经过 joint limit + delta limit 裁剪后的实际发送值（post-clip），录制时必须使用这些值而非 IK 原始输出。底层 `XArm7.send_action()` / `XHand.send_action()` 仍只返回 `bool`。

### 2.9 Workspace 安全（Day 1）

```python
class WorkspaceSafety:
    def __init__(self, workspace_bounds: np.ndarray): ...
        """workspace_bounds: (3,2) [[x_min,x_max],[y_min,y_max],[z_min,z_max]] in meters."""

    def check(self, eef_pos: np.ndarray) -> bool: ...
        """检查 EEF 是否在 workspace 内。"""

    def clamp(self, target_pos: np.ndarray) -> np.ndarray: ...
        """将目标位置裁剪到 workspace 内。"""
```

### 2.10 线程安全（后台控制线程）

如使用后台线程进行实时 arm 控制（如 PID 控制线程、servo 模式线程），必须遵循：

```python
class XArm7:
    def __init__(self, ...):
        self.arm_lock = threading.Lock()
        self.arm_target: np.ndarray | None = None
        self.control_thread: threading.Thread | None = None

    def control_arm_qpos(self, target: np.ndarray):
        """线程安全地更新控制目标。"""
        with self.arm_lock:
            self.arm_target = target.copy()

    def _control_loop(self):
        """后台控制线程。"""
        while self.running:
            with self.arm_lock:
                target = self.arm_target.copy() if self.arm_target is not None else None
            if target is not None:
                try:
                    self._send_servo_command(target)
                except Exception:
                    self.error_state = True
                    self.last_error_message = "control loop error"
            self._rate_limiter.wait()
```

- 共享目标变量通过 `threading.Lock` 保护
- 锁命名包含被保护变量名（如 `arm_lock`、`hand_lock`）
- 后台线程捕获异常后设置 `error_state=True`，不静默退出
- 控制循环内不做耗时操作（如 print、日志写入），保持实时性

> **ref:** [P2] BunnyVisionPro xarm7_ability.py L196-230 — `_arm_lock` + `_internal_control_arm_qpos()` 后台线程模式。本项目 XHand 无后台线程（同步发送），仅在需要 PID/伺服速率解耦时引入。

---

## 3. IPC 进程间通信规范

### 3.1 SharedRingBuffer

```python
@dataclass
class RingBufferConfig:
    slot_count: int = 64
    slot_size: int = 1024 * 1024  # 每槽最大字节数
    create: bool = True

class SharedRingBuffer:
    def __init__(self, name: str, config: RingBufferConfig): ...
    def write(self, data: bytes, seq: int | None = None) -> int: ...
        """写入一帧，返回写入的 seq_num。"""
    def read(self, last_seq: int = -1) -> tuple[bytes | None, int]: ...
        """读取最新帧。返回 (data, seq_num)。若无新数据，data 为 None。"""
    def close(self) -> None: ...
```

### 3.2 键盘事件

键盘事件通过 `multiprocessing.Queue` 传递控制信号（T/R/S/H/ESC），不通过 ring buffer。

### 3.3 两阶段握手协议（多进程同步）

Controller 进程和 Robot 进程间采用 Event-based 两阶段握手：

```
Robot: 执行动作完毕 → robot_ready.set()
Controller: 等 robot_ready → clear → 推理 → policy_ready.set()
Robot: 等 policy_ready → clear → 执行动作 → robot_ready.set()
```

```python
robot_ready = multiprocessing.Event()
policy_ready = multiprocessing.Event()

# Robot 进程
while running:
    robot_ready.wait()
    robot_ready.clear()
    action = action_queue.get_latest()
    robot.send_action(action)
    policy_ready.set()

# Controller 进程
while running:
    state = robot.get_state()
    action = compute_action(state)
    action_queue.put(action)
    policy_ready.wait()
    policy_ready.clear()
    robot_ready.set()
```

不使用 busy-wait 轮询，两阶段握手保证数据同步无竞争。

> **ref:** [P1] ManiUniCon shared_storage.py L185-196 — `robot_ready` / `policy_ready` Event 对象。LeFranX 无 IPC（单进程直接调用），不适用。

---

## 4. 控制器与数据流规范

### 4.1 TeleopController._tick() 核心逻辑

```python
def _tick(self, vr_frame):
    # 0. 追踪质量门控
    # 1. Arm: VR wrist → EEF → IK → EMA 平滑（alpha=0.3）
    # 2. Hand: landmarks → retarget → EMA 平滑（alpha=0.3）
    # 3. 关节跳变 clamp（arm + hand 各自独立限速）
    return RobotAction(arm_qpos_cmd=..., hand_qpos_cmd=...)
```

### 4.2 EMA 平滑规则及原因

- **遥操作录制时 arm+hand 都做 EMA 平滑（alpha=0.3）**
  - Arm IK 使用数值方法（MPlib），seed 随机性会导致帧间关节跳变，需要轻度 EMA 抑制。ref: LeFranX `arm_ik_processor.py:360-363` 对 arm IK 输出使用相同 alpha（`smoothing_factor=0.7`，等价于 EMA alpha=0.3）
  - Hand retargeting 从 21 个稀疏 landmark 映射到 12 DOF，输入噪声大，需要 EMA 过滤手指抖动
  - Alpha=0.3 是轻度平滑，在抑制抖动的同时保持响应性（手部动作延迟 < 50ms）
- **策略部署时 arm+hand 都做 EMA 平滑（alpha=0.5）**
  - 原因：策略推理可能有帧间抖动，需要更强的平滑来抑制高频噪声保护真实机器人
  - 部署时 alpha 更高（0.5 vs 0.3），因为策略输出比人类遥操作更不稳定

### 4.3 VR Re-anchoring 规则

- 每次进入 RECORDING 状态前，调用 `arm_mapper.reset(vr_frame, eef)` 重置 VR 参考原点
- 原因：操作员在 episode 之间会自然移动 VR 手的位置，不重置会导致机器人初始位姿与 VR 参考产生大跳变

### 4.4 控制频率约束

控制循环必须使用速率限制器（rate limiter）而非单纯 `time.sleep()`：

```python
class RateLimiter:
    def __init__(self, target_hz: float):
        self.dt = 1.0 / target_hz
        self.last_wake = time.perf_counter()

    def wait(self):
        elapsed = time.perf_counter() - self.last_wake
        sleep_time = self.dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.last_wake = time.perf_counter()
```

- `RateLimiter.wait()` 而非 `time.sleep(dt)`：补偿计算耗时，保证长期频率准确
- 验收标准：端到端延迟 < 30ms，控制频率 > 40Hz，帧间抖动 < 5ms

> **ref:** [P1] ManiUniCon quest.py 使用 `RateLimiter`；[P2] BunnyVisionPro xarm7_ability.py 使用 `wait_until_next_control_signal()`。

---

## 5. 录制层接口规范

### 5.1 HDF5 数据结构

```
episode_000.h5
  /meta: task_label, operator, tags, duration, fps,
         num_frames, num_valid_frames, success,
         camera_serial, camera_type,          # "eye_to_hand" | "eye_in_hand"
         camera_K,                            # [fx, fy, cx, cy]
         camera_T_base_camera | camera_T_eef_camera,  # 4x4 flat，按 camera_type 二选一
         retargeting_config, pipeline_snapshot

  /obs/arm_qpos(7)  arm_qvel(7)  arm_tau(7)  eef_pos(3)  eef_quat(4)
  /obs/hand_qpos(12)  hand_current(12)  hand_tactile_sum(5,3)  hand_temperature(12)
  /action/arm_qpos(7)  hand_qpos(12)
  /vr/wrist_pos(3)  wrist_quat(4)  landmarks(21,3)
  /quality_flags(T,) uint16
  /camera/rgb(T,H,W,3)  depth(T,H,W)  timestamps(T)
  /camera/K(3,3)                        # 内参矩阵
  /camera/extrinsics(T,4,4)             # T_base_camera，逐帧外参
```

**相机内外参存储说明：**

- `/meta/camera_K`：内参标量值 `[fx, fy, cx, cy]`，录制时从 RealSense 硬件读取，方便快速读取
- `/meta/camera_T_base_camera`（eye-to-hand）或 `/meta/camera_T_eef_camera`（eye-in-hand）：4x4 矩阵以 16 个 float 存储，原始标定值
- `/camera/K(3,3)`：3x3 内参矩阵，与图像数据放一起便于预处理
- `/camera/extrinsics(T,4,4)`：逐帧的外参，已在录制时计算为 T_base_camera
  - eye-to-hand：每帧相同，等于 `/meta/camera_T_base_camera`
  - eye-in-hand：逐帧 `FK(arm_qpos) @ T_eef_camera`，随 arm 运动变化
- Episode 自包含，不依赖外部标定文件

### 5.2 EpisodeRecorder

```python
class EpisodeRecorder:
    def __init__(self, data_dir: str, camera_recorder=None): ...

    def start_episode(self, task_label: str = "", operator: str = "",
                      tags: list[str] | None = None) -> bool: ...
        """创建 .h5 文件，写入初始 meta。返回是否成功创建。"""

    def add_frame(self, state: RobotState, action: RobotAction,
                  vr_frame: dict, quality_flags: int) -> bool: ...
        """追加一帧数据到 .h5。返回是否写入成功。"""

    def stop_episode(self, success: bool = True) -> str | None: ...
        """关闭 .h5，写入最终 meta。返回文件路径。"""

    @property
    def is_recording(self) -> bool: ...
    @property
    def frame_count(self) -> int: ...
```

### 5.3 QualityFlags（11-bit）

```python
TRACKING_OK      = 1 << 0   # bit 0
IK_SUCCESS       = 1 << 1   # bit 1
RETARGET_OK      = 1 << 2   # bit 2
RETARGET_VALID   = 1 << 3   # bit 3
JOINT_JUMP_OK    = 1 << 4   # bit 4
IN_WORKSPACE     = 1 << 5   # bit 5
CAMERA_OK        = 1 << 6   # bit 6
ARM_TORQUE_OK    = 1 << 7   # bit 7
HAND_CURRENT_OK  = 1 << 8   # bit 8
HAND_TEMP_OK     = 1 << 9   # bit 9
HAND_COMM_OK     = 1 << 10  # bit 10
```

---

## 6. 数据层接口规范

### 6.1 EpisodeReader

```python
class EpisodeReader:
    def __init__(self, path: str): ...
        """懒加载 HDF5，不一次性读入内存。"""

    def read(self, key: str) -> np.ndarray: ...
        """读取任意数据集，如 read("obs/arm_qpos")。"""

    def iter_frames(self, skip_rejected: bool = True) -> Iterator[dict]: ...
        """逐帧迭代，skip_rejected=True 跳过 quality_flags 不全为 1 的帧。"""

    def get_valid_mask(self) -> np.ndarray: ...
        """返回 (T,) bool 数组，标记每帧是否全部 quality flags 通过。"""

    @property
    def num_frames(self) -> int: ...
    @property
    def num_valid_frames(self) -> int: ...
    @property
    def metadata(self) -> dict: ...
```

### 6.2 DataValidator

```python
@dataclass
class ValidationReport:
    passed: bool
    errors: list[str]      # 致命问题
    warnings: list[str]    # 非致命问题

class DataValidator:
    def validate(self, episode_path: str) -> ValidationReport: ...
        """检查: nan/inf, shape 一致, 时间戳单调, 关节范围, 电流异常。"""
```

### 6.3 数据转换（scripts/convert_data.py）

```
--norm-stats 输出 per-joint 归一化:
  arm_qpos: {mean: [j0..j6], std: [j0..j6]}
  hand_qpos: {mean: [j0..j11], std: [j0..j11]}
```

---

## 7. 部署层接口规范

### 7.1 PolicyLoader（LeFranX 模式：手动加载，不依赖训练框架）

```python
class PolicyLoader:
    """从 checkpoint 目录加载策略模型。

    目录结构:
      checkpoint/
        config.json          # policy 配置（obs dims, action dims, chunk 等）
        model.safetensors    # 模型权重
        stats.json           # 归一化统计量

    加载后返回 (model, norm_stats, policy_config) 三元组。
    model 需实现 predict(obs: dict) -> np.ndarray。
    """
    @staticmethod
    def load(checkpoint_dir: str) -> tuple[Any, dict, dict]: ...
```

### 7.2 PolicyRunner + Action Chunk

```python
class PolicyRunner:
    def __init__(self,
                 robot: RobotInterface,
                 model,                              # 需实现 predict(obs) -> np.ndarray
                 norm_stats: dict,
                 *,
                 chunk_size: int = 1,                # 模型输出的动作长度
                 n_action_steps: int = 1,            # 每次执行几步
                 query_freq: int = 1,                # 每 N 步重新推理一次
                 action_mode: str = "full",          # "full" | "arm_only" | "hand_only"
                 hand_smooth_alpha: float = 0.5,     # 部署时 arm+hand 都做 EMA
                 arm_smooth_alpha: float = 0.5,
                 safety_monitor: SafetyMonitor | None = None,
                 max_steps: int = 1000): ...

    def run(self) -> None:
        """主循环:
        for step in range(max_steps):
            if step % query_freq == 0 or action_buffer is None:
                model_output = model.predict(obs)
                action_buffer = extract_chunk(model_output)
            action_raw = action_buffer[chunk_idx]
            action = smooth(action_raw, prev_action)     # EMA alpha=0.5
            safety.check(state, action) → stop if unsafe
            robot.send_action(action)
        """
```

**为什么部署时 arm 也做平滑？** 遥操作时不加 arm 平滑是为了录原始数据。部署时策略推理可能有帧间抖动，EMA 平滑抑制高频噪声保护真实机器人。

### 7.3 SafetyMonitor

```python
@dataclass
class SafetyStatus:
    ok: bool
    arm_ok: bool
    hand_ok: bool
    message: str = ""

class SafetyMonitor:
    def check(self, state: RobotState, action: RobotAction) -> SafetyStatus: ...
        """Arm: workspace / 关节限位 / 力矩
           Hand: 关节限位 / 电流(堵转) / 温度 / 通信"""
```

### 7.4 ObservationBuilder

```python
class ObservationBuilder:
    """将 RobotState + 传感器数据构建为策略输入 obs dict。
    独立于模型，不包含模型 forward 逻辑。"""

    def build(self, state: RobotState, camera_frame: dict | None = None) -> dict: ...
        """返回归一化后的 obs dict，字段稳定不随模式变化。"""
```

### 7.5 ActionParser

```python
class ActionParser:
    """将策略输出解析为 RobotAction，支持 arm_only / hand_only / full 模式。

    action_mode="arm_only": hand_cmd 填充当前手部状态
    action_mode="hand_only": arm_cmd 填充当前 arm 状态
    action_mode="full": arm+hand 均由策略输出驱动
    """
    def parse(self, policy_output: np.ndarray, state: RobotState,
              action_mode: str = "full") -> RobotAction: ...
```

---

## 8. 工具层接口规范

### 8.1 hand_utils

```python
def estimate_frame_from_hand_points(landmarks: np.ndarray) -> np.ndarray: ...
    """从 21 个手部 landmark 估算手部坐标系（3x3 旋转矩阵）。"""

# 操作员坐标系到 MANO 右手坐标系的变换
OPERATOR2MANO_RIGHT: np.ndarray  # (3,3)
```

### 8.2 latency_bench

```python
@dataclass
class LatencyStats:
    vr_to_obs_ms: float       # VR 帧到达 → 观测构建完成
    obs_to_action_ms: float   # 观测完成 → 动作发送
    action_to_hw_ms: float    # 动作发送 → 硬件响应
    total_ms: float           # 端到端延迟
    fps: float                # 实际控制频率

def measure_latency(robot: RobotInterface, controller,
                    num_samples: int = 1000) -> LatencyStats: ...
    """分阶段统计延迟，验收: 端到端 < 30ms, > 40Hz, 抖动 < 5ms。"""
```

---

## 9. 配置管理规范

- 所有配置使用 `@dataclass` 定义
- 可变默认值（`np.ndarray`、`list`、`dict`）必须使用 `field(default_factory=...)`
- 每个模块独立配置类，命名以 `Config` 结尾（如 `XHandConfig`、`PipelineConfig`）
- 物理量单位必须在字段名或注释中显式标注

---

## 10. 命名与代码风格

| 规则 | 示例 |
|------|------|
| 类名 PascalCase | `XHand`, `TeleopController` |
| 函数/变量 snake_case | `send_action()`, `connected_flag` |
| 常量 UPPER_CASE | `JOINT_NAMES`, `SENSOR_IDS` |
| 配置类 XxxConfig 后缀 | `XHandConfig`, `PipelineConfig` |
| 不加前导下划线 | `connected_flag` 而非 `_connected_flag` |
| 物理量单位显式标注 | `qpos: np.ndarray  # rad` |
| 方法命名统一 | `get_state()` 而非 `get_obs()` |

---

## 11. 依赖管理

- 核心模块（robot/sensor/controller）不强依赖可视化、训练框架或重型库
- 可视化依赖（cv2/open3d）放到 viewer 或函数内部局部 import
- 策略训练依赖（torch/wandb/hydra）只在 deploy/ 代码中使用
- 硬件驱动只依赖硬件 SDK、numpy、标准库

```python
# 推荐：局部 import
def visualize(...):
    import cv2
    ...

# 避免：硬件驱动顶层 import cv2
```

---

## 12. 错误处理与安全

- 硬件驱动控制类函数优先返回 `bool`
- 失败时设置 `self.error_state = True` 和 `self.last_error_message`
- 捕获具体异常类型，禁止 bare `except`（除非在顶层主循环 `while True` 中兜底，此时必须记录 traceback）
- 真机安全裁剪必须在 `send_action()` 中执行，不可跳过
- `get_state()` 读失败时返回含 NaN 的默认结构，不要抛异常

```python
# 正确：捕获具体类型
try:
    self.arm.set_servo_angle_j(angles=target)
except XArmAPIError as e:
    self.error_state = True
    self.last_error_message = f"servo failed: {e}"
    return False

# 正确：顶层兜底（仅主循环允许）
while self.running:
    try:
        self._tick()
    except Exception:
        logging.exception("unhandled error in control loop")
        self.error_state = True

# 错误：吞异常（Open-Teach oculus.py L103 的反面案例）
except:
    break
```

> **ref:** [P1] LeFranX 使用类型化异常 `DeviceAlreadyConnectedError`、`DeviceNotConnectedError`；[P2] Open-Teach 的 `except: break`（bare except 吞所有异常）是反面案例，本项目禁止在非顶层使用。

---

## 13. 模块文件结构

每个模块提供简单 `example()` 函数，不默认提供复杂 CLI。如需批量参数/实验管理，单独写 `scripts/` 下的脚本。

```python
def example():
    config = DeviceConfig(...)
    device = Device(config)
    ...

if __name__ == "__main__":
    example()
```

---

## 14. 开发检查清单

### 新增硬件驱动（robot/*.py）

- [ ] 是否有 Config dataclass（可变默认值使用 default_factory）
- [ ] 是否有 connect() → bool / disconnect() / get_state(full=False) → dict
- [ ] 执行器是否有 send_action(action: np.ndarray) → bool
- [ ] send_action 是否只有一种动作语义（只接受 np.ndarray）
- [ ] 默认 get_state 是否足够轻量（只返回控制循环核心字段）
- [ ] full=True 是否包含必要调试信息
- [ ] 单位是否明确（rad, m, N, A 等）
- [ ] 是否记录 connected_flag / error_state / last_error_message
- [ ] 是否有 is_connected() / is_error() / clear_error() / stop()
- [ ] send_action 是否包含安全裁剪（joint limit + delta limit）
- [ ] 是否记录 last_joint_limit_clipped / last_delta_limited
- [ ] 是否避免在 connect() 中自动执行危险操作
- [ ] 是否没有混入策略推理、可视化、数据记录
- [ ] 是否没有强依赖不必要的重型库（cv2/torch）
- [ ] 是否提供简单 example() 而非 argparse CLI

### 新增传感器驱动（sensor/*.py）

- [ ] Config dataclass
- [ ] connect() → bool / disconnect() / get_state(full=False) → dict
- [ ] stop() / is_connected() / is_error() / clear_error()
- [ ] 默认 get_state 轻量，full=True 含调试信息
- [ ] 单位明确（深度: m, 图像: uint8 RGB/BGR）
- [ ] 传感器数据与派生观测分离（点云生成放 utils，不放入传感器驱动）

### 新增控制器/部署模块

- [ ] 通过 hardware.get_state() / hardware.send_action() 与硬件交互，不直接调 SDK
- [ ] 配置可保存和复现
- [ ] 策略输出到硬件动作有显式转换层（action adapter）
- [ ] 在线 runner 不把策略推理逻辑写入硬件驱动

### 新增录制模块（recording/*.py）

- [ ] EpisodeRecorder 与 Controller 解耦，只负责数据写入
- [ ] quality_flags 完整覆盖 arm+hand 的 11 个 bit
- [ ] start_episode → stop_episode 生命周期清晰
- [ ] HDF5 结构与 plan 一致（/obs, /action, /vr, /camera, /meta）

### 新增数据模块（data/*.py）

- [ ] EpisodeReader 懒加载，不一次性读入内存
- [ ] iter_frames 支持 skip_rejected
- [ ] DataValidator 检查 nan/inf/shape/时间戳/关节范围/电流异常
- [ ] convert_data.py 输出 per-joint 归一化统计量

---

## 15. 参考模板

> **内部模板 vs 外部参考**：本节列出本项目内部的最佳实践范例。外部参考库的完整模块映射见 Section 0.5.3（5 个项目，P1/P2/P3 三级优先级）。开发时先按优先级检索外部参考理解设计意图，再对照内部模板确保符合本项目接口规范。

**XHand（`robot/xhand.py`）** 是当前代码库中最符合风格约束的实现，包含：

- 完整的 Config dataclass（含 default_factory）
- `connect() → bool` 含错误状态设置和初始状态初始化
- `get_state(full=False)` 默认轻量，full=True 含触觉/温度/错误码
- `send_action(np.ndarray) → bool` 含 range clip + delta limit
- `is_connected() / is_error() / clear_error() / stop() / reset()`
- 完整的状态变量（`connected_flag`, `error_state`, `last_*`）
- 简单 `example()` 函数

**RealSense（`sensor/realsense.py`）** 展示了传感器模块的良好实践：

- frozen dataclass 配置
- 结构化的 `CameraFrame` 输出
- `read()` 返回强类型 frame 对象
- 点云生成通过独立工具函数（`pointcloud_utils.py`），不耦合在传感器类中
- 简单 `example()` 函数

**新代码请以 XHand 为执行器模板，以 RealSense 为传感器模板。**
