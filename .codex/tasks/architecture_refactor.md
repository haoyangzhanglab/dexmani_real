# Codex CLI 任务书：DexMani Real 目录与模块边界重构

> 历史记录，非当前实施说明。根目录 `tests` 已按用户要求删除；下文路径、测试命令及数量仅记录当时状态，不再作为当前入口或验证指引。

## 0. 任务身份与执行方式

- 仓库：`haoyangzhanglab/dexmani_real`。
- 编制日期：2026-10-25。
- 已核对的代码基线：`43ffe6a04916d7e8413f4c488efdef6c0d2f5046`。
- 任务状态：P0–P6 离线实施及后续代码审查已完成，证据见第 10 节；真机验收未执行。
- 本文件是前期架构讨论的统一实施规格；执行不依赖此前聊天记录。先阅读根目录及相关目录中适用的 `AGENTS.md` / `AGENTS.override.md`，再按本任务书实施。
- 开始时以本地实际源码和工作区为准，记录实际 HEAD 与本基线的差异；不要为了匹配本文件而回退仓库。路径已有变化时跟踪实际职责，避免重复重构。
- 用户启动本任务后，目标是完成代码、测试与必要文档修改，不是再次输出一份设计方案。按下文阶段连续推进，每阶段形成可验证的闭环；不得只完成目录图或阶段 P0 就宣称完成。

**目标：让项目简洁高效、正确好用、清晰易维护，并能以机器人学和机器人学习论文中的概念直接导航代码。优先级为：行为正确与证据可信 > 职责与依赖清晰 > 易用性 > 命名与目录美观。**

## 1. 范围与设计取舍

### 1.1 本次要完成

1. 将 `planning/` 中的共享几何、观测表示、标定和动作控制归入各自领域，消除不合理的跨领域引用。
2. 将传感设备采集留在 `sensor/`，将点云与手部几何表示构建集中到 `perception/`，在线与离线继续共享唯一实现。
3. 将相机、桌面和 VR 标定集中到 `calibration/`，明确默认参数文件位置，并同步路径和打包声明。
4. 将共享运行机制收敛为 `runtime/` 加 `motion/`；将策略部署收敛为 `deployment/` 加 `rollout/`。
5. 仅执行下文明确列出的必要命名修正，迁移相关测试并建立针对实际风险的离线回归检查。
6. 更新受影响的 CLI 导入、字符串 patch 路径、进程入口、仓内集成适配器和 README 导航。

### 1.2 明确不做

- 不引入 `src/` 布局，不重命名仓库或 Python 包，不移动实验输出目录。
- 不新增空的 `configs/`、`evaluation/`、`common/`、`core/`、`interfaces/` 等占位层；确有现存配置时只迁移受影响引用。
- 不建立通用 `BaseSession`、manager、registry、插件系统、事件总线、全局服务容器或多套运行状态。
- 不因文件较长而拆碎 `PolicyEpisode`、`TeleopRunner`、设备 worker 或数据写入事务。
- 不调整 IK、点云采样、滤波、HOME、控制频率、设备限位或策略调度算法；不合并不同 FK 后端，不更换依赖版本。
- 不修改 Raw/Canonical 字段、模型输入输出、已有 YAML 配置键、结果 JSON 字段或终止原因枚举。默认标定文件路径是本次明确批准的布局变化，按第 5.4 节处理。
- 不把 `RunEndReason.POLICY_FAILURE` 改成其他枚举值或字符串；此类语义清理另行立项。
- 不改写第三方算法、训练仓库或仿真仓库；仅允许修改本仓集成适配器中受本次迁移直接影响的引用。
- 不为了测试通过添加静默 fallback、自动恢复、放宽检查、伪造数据或永久旧路径兼容层。
- 不修改 `AGENTS.md`、Codex 权限配置或审批设置来绕过约束。

### 1.3 最终取舍

- `control/joint_angles.py` 保留 xArm7 专属的等价轴与关节范围语义，不伪装为通用数学工具。
- `geometry/kinematics/` 可以依赖 `robot/model.py` 的静态模型信息；“没有硬件 I/O”不等于“完全不依赖机器人模型”。
- `runtime/motion/` 与 `deployment/rollout/` 各只增加一层子目录，不再嵌套。
- 保留 `teleop/controller.py`、`teleop/vr_mapping.py`、`ipc/ring.py`、`ipc/schema.py` 和 `robot/xarm7/reference.py` 的名称。
- 保留 `ActionPlan`、`PolicySession`、`PolicyEpisode`、`InferenceWorker` 等已有类型名；本次不追求所有标识符统一改名。
- 两个明确的类型修正为 `SafetyParams -> RuntimeTimeouts`、`EpisodeResults -> SessionResults`，但对应配置键、结果文件名和结果内容不变。
- 文档使用中文，Python 标识符使用准确的英文领域术语。“论文风格”是概念与实现能对应，不是加入花哨缩写、把近似检查写成安全保证，或扩大算法能力的表述。

## 2. 目标目录

下图是目标而非当前状态。省略各包的必要 `__init__.py`、未变更的资源细节及仓库元文件；未列出不代表允许删除。包初始化应轻量，不在 `__init__.py` 中批量导入算法、驱动或会话。

```text
repository/
├── README.md
├── AGENTS.md
├── pyproject.toml
├── .codex/
│   ├── config.toml                         # 保持既有配置
│   └── tasks/architecture_refactor.md      # 本任务书与简短执行记录
├── assets/                                # 保持现有布局和内容
├── third_party/                           # 保持独立，限集成引用更新
├── scripts/                               # 保持现有实验命令名称
│   ├── collect_teleop.py
│   ├── keyboard_teleop.py
│   ├── run_policy.py
│   ├── replay_episode.py
│   ├── calibrate_camera.py
│   ├── calibrate_vr_heading.py
│   ├── export_zarr.py
│   ├── visualize_episode.py
│   ├── tools/
│   └── cordvip/
├── tests/                                 # 面向行为的离线测试，不机械镜像所有目录
│   └── test_deployment_contract.py         # 由已有测试入口迁入
└── dexmani_real/
    ├── __init__.py
    ├── config.py                          # 配置装配，CLI > YAML > 默认值
    ├── geometry/
    │   ├── pose.py                        # Pose、旋转表示转换、位姿误差
    │   ├── transforms.py                  # SO(3)/SE(3)/四元数验证基础
    │   ├── camera.py                      # 相机几何数据类型和表示转换
    │   ├── plane.py                       # 共享平面几何验证
    │   └── kinematics/
    │       ├── arm_fk.py
    │       └── hand_fk.py
    ├── perception/
    │   ├── hand_config.py
    │   ├── hand_surface.py                 # URDF 表面采样模板
    │   ├── hand_geometry.py                # FK 派生的指尖与手点云
    │   └── pointcloud/
    │       ├── config.py
    │       ├── processing.py
    │       └── worker.py
    ├── control/
    │   ├── config.py
    │   ├── action_mapping.py
    │   ├── online_ik.py
    │   ├── target_validation.py
    │   ├── collision.py
    │   ├── joint_angles.py
    │   └── homing.py
    ├── calibration/
    │   ├── paths.py                        # 仅定位资源，不读取标定或导入求解器
    │   ├── table_plane.py                  # 拟合、质量评估、读取和发布
    │   ├── vr_alignment.py
    │   ├── camera/
    │   │   ├── aruco.py
    │   │   ├── hand_eye.py
    │   │   ├── extrinsics.py
    │   │   └── session.py                  # 交互应用，不是底层标定数学
    │   └── parameters/
    │       ├── cameras.json
    │       ├── table_plane.json
    │       └── vr_transform.json
    ├── robot/
    │   ├── system.py                      # 原 robot.py，保留 DexManiRobot 类型
    │   ├── model.py
    │   ├── commands.py
    │   ├── feedback.py
    │   ├── xarm7/
    │   │   ├── config.py
    │   │   ├── driver.py
    │   │   ├── executor.py
    │   │   ├── worker.py
    │   │   ├── reference.py
    │   │   └── protocol.py
    │   └── xhand/
    │       ├── config.py
    │       ├── driver.py
    │       ├── feedback.py
    │       └── limits.py
    ├── sensor/
    │   ├── camera/
    │   │   ├── config.py
    │   │   ├── realsense.py
    │   │   └── worker.py
    │   └── vr/
    │       ├── config.py
    │       └── worker.py
    ├── ipc/
    │   ├── channels.py
    │   ├── ring.py
    │   └── schema.py
    ├── runtime/
    │   ├── config.py
    │   ├── state.py
    │   ├── observation.py
    │   ├── operator_input.py
    │   ├── preflight.py
    │   ├── sensor_processes.py
    │   ├── shutdown.py
    │   └── motion/
    │       ├── home.py
    │       └── jog.py
    ├── teleop/
    │   ├── config.py
    │   ├── session.py
    │   ├── runner.py
    │   ├── controller.py
    │   ├── vr_mapping.py
    │   ├── keyboard_session.py
    │   ├── audio_feedback.py
    │   └── retargeting/
    │       ├── tag.py
    │       ├── dexpilot.py
    │       ├── landmarks.py
    │       └── fingertip_objective.py
    ├── deployment/
    │   ├── config.py
    │   ├── preflight.py
    │   ├── run.py
    │   ├── session.py
    │   ├── policy_observation.py
    │   ├── inference.py
    │   └── rollout/
    │       ├── episode.py
    │       └── action_schedule.py
    ├── replay/
    │   ├── session.py
    │   ├── replayer.py
    │   ├── trajectory.py
    │   ├── capture.py
    │   └── consistency.py
    ├── data/
    │   ├── recorder.py
    │   ├── session_results.py
    │   ├── raw/
    │   │   ├── schema.py
    │   │   ├── reader.py
    │   │   ├── writer.py
    │   │   ├── video.py
    │   │   └── camera_geometry.py
    │   └── canonical/
    │       ├── schema.py
    │       ├── processing.py
    │       └── export.py
    └── utils/
        ├── atomic_io.py
        ├── timing.py
        ├── log.py
        └── visualization.py
```

不为实现目录图机械增加文件。上图中的新文件都有下文指定的现有职责来源；不新增与它们重叠的通用工具。若当前源码变化使某一局部合并更合理，可在不改变一级领域及行为契约的前提下采用更小实现，并记录具体依据。

## 3. 依赖与状态所有权

### 3.1 允许的主要依赖方向

- `teleop`、`deployment`、`replay` 和交互式标定会话调用 `runtime`、`control`、`data` 及设备接口。
- `runtime/motion` 调用 `control` 计算目标、路径及检查，使用 `robot` 执行授权操作；`runtime` 不导入具体 Policy 模型或部署调度。
- `control` 调用共享几何、静态机器人模型；不导入 `runtime`、实验会话或设备 SDK，不直接连接设备或启动进程。
- `perception` 的数值模块调用 `geometry` 和静态模型；`pointcloud/worker.py` 是进程适配边界，可使用 IPC，但不另开相机或机器人连接。
- `data/raw` 与 `data/canonical` 不依赖 `teleop`、`deployment`、运行会话或设备 I/O；`data/recorder.py` 作为在线记录适配器可以消费既有观测与命令，不为追求形式上的分层再造 DTO。
- 标定数学使用几何；`calibration/camera/session.py` 可以使用运行机制。`calibration/paths.py` 必须是轻量叶子模块，供配置引用；基础几何和碰撞检查不通过标定求解模块取得简单验证函数。
- `geometry` 可以使用 NumPy、SciPy、FK 后端及 `robot/model.py` 的静态定义，但不依赖设备 driver、runtime、teleop 或 deployment。

这是子模块级的实际职责约束，不是要求每个一级包都形成绝对单向的机械层级。保留必要的轻量配置和类型引用，避免通过新抽象“修复”不存在的依赖问题。

### 3.2 保持唯一所有者

| 所有者 | 唯一负责的状态/资源 | 不得顺手接管 |
| --- | --- | --- |
| `RuntimeState` | 软件运动身份、授权撤销、停止请求 | 物理停稳证明、策略动作块 |
| `DexManiRobot` 与设备 owner | 设备连接、提交边界、反馈与关闭 | 模型推理、实验轮次策略 |
| `SensorProcesses` | 已启动的传感进程及退出结果 | 机器人 SDK owner 的强制终止 |
| `PolicySession` | 多轮会话、设备和模型资源寿命 | 单轮每个控制槽的决定 |
| `PolicyEpisode` | 单轮准备、历史、动作提交、首次终止和录制协调 | 新增另一套全局授权状态 |
| `InferenceWorker` | 模型及观测 FK 工作区的串行执行 | 硬件发送、SDK 调用 |
| `action_schedule.py` | 给定输入和时钟值的调度决定 | 读墙钟、sleep、设备 I/O、停止副作用 |
| `SessionResults` | 会话及 episode 摘要落盘 | 设备、模型或 recorder 的所有权 |

保留 `run -> session -> episode`。不要将同一 owner 的事务拆到多个 manager，也不要引入进程间共享的可变 FK 工作区。

## 4. 精确迁移表

表内路径均相对仓库根的 `dexmani_real/`，另有标注者除外。修改定义时同时更新全部调用方，包括函数内延迟导入、相对导入、类型提示、字符串 patch 及进程 target。

| 现有位置 | 目标位置/处理 |
| --- | --- |
| `planning/kinematics/pose.py` | `geometry/pose.py` |
| `utils/geometry.py` | `geometry/transforms.py`，删除旧实现，不维护双份函数 |
| `planning/kinematics/arm_fk.py`、`hand_fk.py` | `geometry/kinematics/` 对应文件 |
| `sensor/camera/geometry.py` | `geometry/camera.py`；暂不拆散现有轻量相机类型 |
| `planning/table_plane.py` 的共享 `validate_table_plane()` | `geometry/plane.py`；拟合算法专属 helper 仍随标定模块 |
| `sensor/pointcloud/config.py`、`processing.py`、`worker.py` | `perception/pointcloud/` 对应文件 |
| `sensor/pointcloud/hand_config.py` | `perception/hand_config.py`；保持不导入 mesh/FK 后端 |
| `planning/kinematics/hand_pointcloud.py` | `perception/hand_surface.py` |
| `planning/kinematics/fingertip.py` | `perception/hand_geometry.py` |
| `planning/config.py` | `control/config.py`；标定默认路径从轻量 `calibration/paths.py` 引用 |
| `planning/action.py` | `control/action_mapping.py` |
| `planning/target.py` | `control/target_validation.py` |
| `planning/kinematics/ik.py` | `control/online_ik.py`，保留 MPlib/CLIK 与现有结果类型 |
| `planning/collision.py` | `control/collision.py` |
| `planning/paths.py` 中两项等价角函数 | `control/joint_angles.py`，保留 xArm7 约束 |
| `planning/home_planner.py` + `planning/paths.py` 其余 HOME 专属逻辑 | 合并为 `control/homing.py`；消除模块间回引，保留 `HomePlanner` 与现有函数语义 |
| `planning/table_plane.py` 其余内容 | `calibration/table_plane.py`，导入共享平面验证 |
| `sensor/camera/aruco.py`、`hand_eye.py`、`extrinsics.py` | `calibration/camera/` 对应文件 |
| `runtime/camera_calibration.py` | `calibration/camera/session.py` |
| `teleop/vr_alignment.py` | `calibration/vr_alignment.py` |
| `state/__init__.py` 的路径常量 | `calibration/paths.py`；导入时不读取文件 |
| `state/*.json` | `calibration/parameters/`，仅迁移路径，保留实际内容 |
| `runtime/processes.py` 的 `SensorProcesses` | `runtime/sensor_processes.py` |
| 同文件的 `shutdown_local_runtime()` | `runtime/shutdown.py`；函数名、参数和退出顺序不变 |
| `runtime/home.py`、`runtime/jog.py` | `runtime/motion/` 对应文件 |
| `runtime/preflight.py` 的 `validate_keyboard_workspace()` | `runtime/motion/jog.py`，复用共享频率验证 |
| `runtime/jog.py` 的 `CARTESIAN_JOG_KEYS` | `runtime/operator_input.py`；jog 导入它，输入层不反向导入 jog/IK |
| `deployment/episode.py` | `deployment/rollout/episode.py` |
| `deployment/schedule.py` | `deployment/rollout/action_schedule.py` |
| `deployment/observation.py` | `deployment/policy_observation.py` |
| `robot/robot.py` | `robot/system.py`；保留 `DexManiRobot` 类型名 |
| `data/results.py` | `data/session_results.py`；`EpisodeResults -> SessionResults` |
| `replay/evaluation.py` | `replay/consistency.py`；保留指标、保存流程和函数行为 |
| `runtime/config.py` 的 `SafetyParams` | 同文件改为 `RuntimeTimeouts`；保留 `ExperimentConfig.safety` 和 YAML 键 |
| 仓库根 `scripts/check_deployment.py` | `tests/test_deployment_contract.py`，记录测试命令变化，删除旧测试入口，不加转发脚本 |

`geometry/pose.py` 负责表示及位姿计算，`geometry/transforms.py` 保留当前底层验证/归一化原语；保持前者单向使用后者，不复制四元数实现。函数只移动到一个权威位置。

完整迁移后删除已无职责的 `planning/`、`state/`、`sensor/pointcloud/` 及表中旧模块。只有全部消费者迁完后才能删除；中间阶段允许旧领域仍存在，但不能留下失效导入。

## 5. 不可改变的行为契约

### 5.1 运动权限、线程与进程

- 保留 `run_id`、输入 generation、停止信号、健康期限、提交期限和执行期限的独立含义与所有检查点。
- 保留输入/状态锁的既有顺序；不把 IPC 等待或 SDK I/O 移入短临界区。不把检查与实际 SDK 调用夸大为不可竞态的原子操作。
- 保留 xArm SDK 独占 owner 和父进程手部 I/O 的边界；默认 `spawn` 的目标函数必须在新路径可导入。
- 传感进程的 terminate/kill 规则不得被泛化套用到机器人执行 owner。设备停止未确认、退出失败和 cleanup 错误仍需可见。
- 软件撤销权限、SDK 接受停止请求、实测停稳是不同事实；不得合并状态或改写日志制造已经停稳的假象。
- 已完成但尚未回收的模型 Future 仍算在途；load、warmup、predict、reset 和 close 的串行与清理语义不变。

### 5.2 几何、动作与控制

- 维持米/弧度、WXYZ、列顺序 rot6d、坐标变换方向、SDK/URDF 关节顺序及 dtype/shape 契约。
- 保留名义碰撞模型与实测手安装补偿的区别，不把两种几何“统一”为同一偏移。
- 普通在线目标检查不等于扫掠路径无碰撞证明；HOME 的密集采样检查也不宣称连续轨迹安全证明。
- 在线场景允许桌面接触与 HOME 的桌面间隙检查保持分离；不得为了共享工厂而统一成一套约束。
- 保留关节等价分支、绝对 SDK 角差、操作限位/机械限位的区别；冻结后的物理命令不能重新投影成另一分支。
- 策略动作分块调度与机械臂的连续伺服参考仍独立。不得加入新的平滑器、限幅、反馈重置或追赶补发。
- Teleop/jog 的已提交参考只在原有成功提交条件成立后更新；求解拒绝不得偷偷推进参考。

### 5.3 观测、数据和策略调度

- 物理采样时间、host 接收时间、处理时间、提交尝试时间不互相替代；复制、缓存或点云处理不能刷新来源时间戳。
- 保留相机与点云的来源 sequence 关联、只读快照及触觉缓存时效；不把多个新鲜来源描述成严格同步采样。
- 在线 Policy 观测和离线 Canonical 使用同一套数值算法。保留训练 artifact 提供的表示配方，不能悄悄套用当前默认值。
- 区分实测状态、动作候选、提交尝试、SDK 接受和实际到位；不得重新解释 Raw action 字段。
- 保留 sync/async/RTC 的不同交接条件：普通 async 可在期限与剩余预测允许时消费旧尾段；RTC 的冻结前缀、prefetch 槽和输入身份检查不能被合并成普通 async。
- 保留所有时间边界的 `<`、`<=` 和整数纳秒 `+1` 约定，不以“统一风格”为由修改。
- 保留准备取消、未进入 RUNNING、正常完成、提前终止与部分录制的结果语义；保持首次终止原因及后续 cleanup 错误的记录规则。
- 回放的跨运行帧索引一致性不等于硬件响应延迟；模块改名不能改变指标解释。

### 5.4 配置、标定、资源和命令

- 现有实验 CLI 名称、参数、退出状态与配置覆盖优先级不变；唯一明确迁走的脚本是离线部署契约测试。
- `--print-config` 仍不能连接设备、读取标定或加载模型。类名修改不得改变 YAML `safety` 等键或默认数值。
- 默认标定文件改到 `calibration/parameters/`；路径常量、默认 `environment.table.plane_path`、帮助文本和仓内受影响引用同时更新。
- 用户显式指定的标定路径仍按既有规则解析；相对桌面路径的基准不变。不要扫描、覆盖、重建或重新发布用户的标定数据。
- 仓内受版本控制的已知默认路径引用可以更新；历史外部 YAML 明确指向旧位置时，要求用户显式迁移，文件缺失需透明失败，不能自动改用另一份标定。
- 标定 JSON 迁移前后核对内容/hash；不得把当前参数换成示例值。修改前发现本地标定有用户改动时必须保留其内容。
- 同步 setuptools package-data，实际检查构建产物中三份 JSON 的路径与内容。包初始化不导入标定会话或求解器。
- 维持现有 `assets/` 定位方式；不要顺手扩展为完整资源分发改造。若 wheel 中机器人资源原本就不完整，区分既有打包限制与本次 JSON 迁移的回归，禁止声称整个独立安装已经验证。
- 结果仍写入原文件名和输出目录，原子写入、禁止覆盖和失败产物保留规则不变。

## 6. 实施顺序与每阶段出口

阶段是逻辑检查点，不要求用户逐阶段确认，不要求自动创建 commit/PR。每阶段先核查相关调用链、做完整迁移，再做针对性检查；顺序允许为保证可导入性小幅调整，不允许把整轮改动揉成无法区分的算法重写。

### P0 — 核实工作区并建立最小行为基线

1. 读取适用指令，检查 `git status --short`、当前 HEAD、已有 diff 和实际 Python/依赖环境；保留他人改动，不 reset、clean、stash 或覆盖。
2. 用定向搜索核对迁移表，覆盖 `dexmani_real/`、`scripts/`、测试、README、打包声明及仓内集成适配器；不无目的逐文件通读整个第三方目录。
3. 移入已有部署契约测试，明确它需要 `dexmani_policy` 与相关依赖；选取第 7 节的纯计算和边界用例建立重构前基线。
4. 记录基线失败与环境缺失，不把现存故障误记为本次回归，也不将错误行为作为应永久保留的“黄金答案”。

出口：能区分原有失败、缺失能力和迁移引入的失败；已选定针对性比较输入，开始进入代码迁移而非无限补测试。

### P1 — 抽出共享几何

迁移 pose、transforms、FK、相机类型及共享平面验证；更新所有消费者。保留各 backend 的独立工作区和延迟导入，不进行数值算法改写。

出口：基础数值和表示回归通过或明确标识 backend 缺失；配置导入不新引入重型模块；旧 `utils/geometry.py` 不再是第二实现。

### P2 — 统一感知表示

迁移点云配置/计算/worker 与手几何配置/模板/历史构建；同步在线观测、离线导出、可视化及仓内 CordViP 受影响引用。相机和 VR 采集保留在 `sensor/`。

出口：相同输入、配方、种子产生相同含义的结果；线上线下继续调用共享实现；来源时间与 dtype/shape 不变。

### P3 — 归位控制与标定

迁移动作映射、IK、目标检查、碰撞、关节等价角与 HOME；聚合 `homing.py`。迁移标定算法、默认路径模块和 JSON，更新打包声明。交互式相机标定 session 可在此迁移或与 P4 合并，但调用方必须同步。

注意：`planning/` 含有尚未迁移的桌面代码时不能提前宣布删除完成；标定文件、默认路径和 package-data 必须作为一个完整变更处理。

出口：控制拒绝/接受行为、HOME 场景与静态资源一致；标定路径迁移经过验证；所有职责迁走后移除旧 `planning/` 与 `state/`。

### P4 — 收敛 runtime

拆分 `sensor_processes.py` 与 `shutdown.py`；移动 `home/jog`；归位键集合和 jog 工作空间验证；完成相机标定交互会话迁移；将 `SafetyParams` 精确改名但保留配置键。

出口：没有 `operator_input -> jog/IK` 的反向依赖；fake-device 的授权撤销、取消、异常和清理轨迹不变；spawn 在无硬件测试替身下能够解析所需入口。

### P5 — 整理 deployment 与必要命名

建立 `rollout/`，迁移 `episode` 和 `action_schedule`，改名 `policy_observation`；保留顶层的会话、模型 worker、入口、配置和准入。完成 `robot/system.py`、`data/session_results.py`、`replay/consistency.py` 的精确迁移和两个指定类型改名。

出口：所有三种执行模式的关键边界经过离线检查；停止、录制、结果结算仍由既有 owner 协调；无新增 manager 或额外状态副本。

### P6 — 全链路离线验收与交付

核对旧引用、资源、CLI 和文档；复查 diff 是否夹带数值/行为变化；移除无职责的旧文件。README 更新模块导航、关键入口、测试方法和默认标定路径迁移说明，不再生成重复的架构长文。

出口：完成第 9 节验收表；提交最终报告，清晰区分“离线已验证”“环境受阻”“真机待人工验收”。

## 7. 验证矩阵

优先沿用 `unittest`、现有可注入 factory 和小型临时 fixture，不为本次引入新的测试框架。测试按行为组织在 `tests/`，不要求每个源码文件一份测试；小型数组/合成场景优先，不上传真实实验录像或大型模型。

| 范围 | 最少应覆盖的证据 |
| --- | --- |
| 表示与 FK | WXYZ/rot6d 转换、非法输入拒绝、等价角边界、SDK/URDF 顺序；有真实 backend 时比较固定关节样本的位姿 |
| 点云与手几何 | 空输入/有效输入、固定种子、输出数量/dtype/坐标系、颜色范围；同一输入经线上线下共享路径结果一致 |
| 动作映射 | joint/eef 输入、限位、IK 拒绝类别与候选不等于已发送；共享几何不改变冻结分支 |
| HOME | 原有候选顺序、等价分支对齐、桌面软间隙脱离与网格碰撞仍分开；执行取消及实测收敛检查 |
| 观测 | 来源超时端点、重复帧不刷新时间、点云/相机序列关联、只读快照、触觉缓存时钟 |
| 运行状态与输入 | 老 run_id 拒绝、停止/退出/急停请求、jog generation 失效、只有有效提交更新参考；锁外 I/O 边界保留 |
| 进程与退出 | 子进程启动失败、ready 超时、退出异常与 cleanup 错误仍可见；不对机器人 owner 套用传感器 kill |
| 推理 worker | 单线程所有权、已完成未回收 Future 仍占用、reset/close 顺序、错误传播；fake model 不加载 GPU 模型 |
| 策略调度 | sync bootstrap/等待、async 晚交接与尾段耗尽、RTC 前缀冻结及预取槽、过期结果/错误 run_id、无追赶补发 |
| 单轮与结果 | 未进入 RUNNING 不算已执行轮次、准备取消、部分录制、首次终止与 cleanup 错误、记录关闭顺序、JSON 字段保持 |
| 配置与 CLI | 覆盖优先级和未知字段拒绝、`safety` 键不变；安全的 print-config 无设备/标定/模型副作用 |
| 标定与打包 | 迁移前后 JSON 内容一致、绝对/相对显式路径、默认路径、旧路径不静默 fallback、wheel 中资源存在 |
| 外部契约 | 已有 Policy/Real 契约测试；仓内 CordViP 引用可解析；无环境则明确说明未覆盖范围 |

### 7.1 测试证据要求

- 行为基线同时参考现有实现与明确的数学/时序契约。固定输入的数值结果应先在迁移前记录，再在相同环境验证迁移后结果。
- 对纯重定位且确定性的计算优先精确比较；确有 backend 浮点差异时事先说明容差、单位和依据，不得观察到失败后任意放宽。
- 设备替身只验证软件调用、状态和错误边界，不能证明真实碰撞、到位或停止。后端替身不能代替真实 FK/IK 数值回归。
- 缺少可选包时仅对明确缺失的外部依赖显式跳过/标记 BLOCKED。不得用宽泛 `except ImportError` 将自己的错误导入和循环依赖隐藏成 skip。
- 不通过修改返回码、删除断言、无条件 skip、伪造 checkpoint 权重或卸掉校验来使测试变绿。
- 仅为重构引入的问题修改实现；发现独立既有 bug 时单列证据和影响。涉及安全/数据含义的既有缺陷不能被静默“顺手修复”或掩盖，应标记对应验收尚未成立。

### 7.2 安全验证命令示例

仅在 P0 已核实环境和入口副作用后使用；按实际新增测试名称选择针对性测试，再做整体发现：

```bash
python -m compileall -q dexmani_real scripts tests
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/run_policy.py --print-config
python scripts/collect_teleop.py --print-config
git diff --check
git diff --stat
```

`compileall` 不验证导入是否成功，也不验证运行语义；测试发现成功但全部被 skip 也不算功能验收。

用 `rg` 或等价搜索核查旧全路径、相对导入、延迟导入、字符串 patch、README 与配置中的路径。搜索范围包含源码、脚本、测试、pyproject 和被集成的第三方适配文件。本任务书的历史迁移表以及必要的迁移说明可以保留旧路径，不能机械地以全仓零命中为标准。

构建工具已可用时，在临时目录执行不安装新依赖的 wheel 构建，例如 `python -m pip wheel --no-deps --no-build-isolation --wheel-dir <临时目录> .`，用 Python `zipfile` 检查 JSON 位置与内容。需要安装验证时使用临时 target，并从仓库外测试导入与资源定位，避免 editable checkout 掩盖缺失资源。工具缺失时记录具体阻塞，不改全局环境或依赖锁定来制造通过结果。

### 7.3 效率与性能

本轮优化以消除职责混杂和重复路径为主，不承诺速度提升。禁止无依据增加控制热路径的数组复制、I/O、锁、逐点循环、重复 FK 或每帧配置解析。需要比较性能时仅用固定离线输入和相同环境，说明预热、重复次数与统计口径，不为了跑分改算法或降低精度。

## 8. 操作与风险边界

- 默认只在本地工作区完成代码修改和离线验证，不自动 `git commit`、push、合并 PR、改分支或重写历史；另有明确授权时再执行相应 Git 操作。
- 不自动 pull/rebase/stash，不删除或覆盖用户已修改的文件。非重叠改动可继续；冲突只阻塞受影响部分，报告具体路径及原因。
- 不连接机器人、手、相机或 VR；不发送动作、HOME、停止、复位或标定发布请求。连只读设备诊断也不在默认授权内。
- 不运行可能连接硬件的脚本本体来做“冒烟测试”；先检查参数解析，限定 print-config/help 或显式注入的测试替身。
- 不覆盖现有 episodes、datasets、rollouts、replay_results、checkpoints 或本地标定。测试只能写入自有临时目录并只清理自有产物。
- 不下载大型模型，不擅自安装/升级系统、CUDA、SDK 或项目依赖，不降低沙箱/审批级别。
- 缺硬件不阻塞纯代码重构与可完成的离线测试；缺关键数值 backend 时保留验证缺口，不将整个重构宣称为已验证可上机。
- 每阶段简短汇报实际进展和问题；普通结构问题按本规格自主决策，不反复要求用户批准。遇到真实权限、数据冲突或行为契约冲突时明确说明，不擅自越界。
- 若执行被上下文或环境限制打断，留下可导入的稳定检查点，在本文件末尾更新简短进度和下一步；未完成阶段不能打勾，不生成多个相互重复的计划文件。

## 9. 完成定义与交付

只有对应证据实际成立时才标记完成：

- [x] 目标领域归属落实，旧模块及所有实际调用方同步迁移，无无效导入、循环导入或双份权威实现。
- [x] runtime/motion 与 deployment/rollout 保持浅层结构，无额外 manager、注册系统或通用框架。
- [x] 状态/设备/模型/记录 owner 清晰，未改变锁、进程所有权、期限、停止与清理的关键边界。
- [x] 在线/离线表示共享、数值与数据契约保持；真实 backend 未验证部分被单独列出。
- [x] CLI、配置键、结果字段和实验输出位置保持；批准的默认标定路径变化有明确迁移说明。
- [x] 三份默认标定 JSON 内容保留、路径解析与 wheel package-data 已检查。
- [x] 测试、脚本、仓内集成适配器及 README 已同步；不需要同时维护新旧导入路径。
- [x] 相关离线检查给出具体命令和真实结果；失败、跳过、环境阻塞与真机待验收相互区分。
- [x] diff 已检查，无无关格式化、依赖升级、算法优化、用户数据改动或未获授权的外部操作。

最终用中文报告：

1. 实际修改的模块与最终职责，说明相对任务书的必要偏差及源码依据。
2. 测试命令、结果和覆盖边界，区分本次回归、既有问题、依赖缺失和未执行检查。
3. 配置/标定路径与旧导入路径的迁移事项，以及需要人工执行的受控真机验收项目。
4. 尚未完成的内容与精确下一步；只有全部阶段完成才称“重构完成”，只有实际验证过的能力才称“验证通过”。

离线通过后的真机验收由操作者另行安排：核查设备启停、取消/急停、HOME/jog、遥操作录制、三种策略执行模式及回放，并在实际工作站验证时延与状态。任务书本身不授权执行这些物理操作。

## 10. 基线定位与执行记录

基线依据均来自上述固定提交中的实际文件：`AGENTS.md`、`pyproject.toml`、`dexmani_real/config.py`、`planning/`、`runtime/`、`deployment/`、`sensor/pointcloud/`、`data/`、`scripts/check_deployment.py` 以及仓内 `cordvip_geometry` 的 DexMani 集成。实现前需确认当前版本仍满足这些描述；不要把历史路径表当作当前源码的替代品。

本文件集中保存任务规格及少量执行证据，不承担长期设计日志。实施者仅在下面更新实际完成的阶段、执行命令摘要、阻塞与必要偏差，避免把任务扩展成文档工程。

| 阶段 | 当前状态 | 实施证据/阻塞 |
| --- | --- | --- |
| P0 基线 | 已完成 | HEAD 1d1069e；相较设计基线仅新增任务书，原工作区干净。使用 real_robot Python 3.10.20；部署契约迁入 tests，新增数值与调度用例后 20 项全部通过。固定输入 8 组数组、标定 SHA-256 与原源码位于 /tmp/dexmani-refactor-mm0h9oo_，未连接设备。 |
| P1 几何 | 已完成 | pose/transforms、相机类型、两类 FK 与平面验证已归位，所有消费者含相对导入已更新；20 项测试通过，8 组数组与 P0 精确相等；配置导入未加载数值后端/设备/策略。 |
| P2 感知 | 已完成 | 点云配置/处理/worker 与手几何已迁入 perception；7 项数值/表示测试通过，包含真实在线 Policy 与离线 Canonical 输出精确一致；8 组 P0 数组精确一致。设备采集仍在 sensor。 |
| P3 控制与标定 | 已完成 | control 与 calibration 已按表迁移；合并 HOME，拆出等价角；相机 session 提前在 P3 迁入。8 项控制测试迁移前后通过；7 项配置/路径/CLI 检查通过；8 组数值、2 组真实 IK 命令、三份 JSON SHA-256 不变。默认配置仅批准路径变化，package-data 已更新（wheel 留 P6 核验）；旧 planning/state 已删除。 |
| P4 Runtime | 已完成 | SensorProcesses/shutdown 已分离，HOME/jog 迁入 motion，键集合和工作空间验证归位；SafetyParams 改名 RuntimeTimeouts，safety 键及数值不变。14 项状态/输入/退出/HOME/spawn 用例迁移前后通过；3 项观测/触觉用例通过；输入模块不再导入 jog/IK。 |
| P5 Deployment 与命名 | 已完成 | rollout/、policy_observation、robot/system、data/session_results、replay/consistency 已迁移，SessionResults 改名完成；所有调用方/patch/类型提示同步。11 项模型/单轮/录制/会话测试迁移前后通过；完整发现 64 项通过，无 skip。保留 run→session→episode 与各 owner。 |
| P6 离线验收与交付 | 已完成（离线） | 最终 unittest 69 项全部通过、0 skip；106 模块/652 个导入名及 10 个冷启动入口通过；8 个 CLI help、2 个 print-config、compileall、diff 检查通过。新旧 wheel 三份 JSON 逐字节一致，仓库外临时安装定位通过；README 26 个本地链接有效。真机与完整独立硬件部署未执行。 |
| 真机验收 | 未执行；不在默认执行授权内 | 由操作者另行安排 |


### P0–P6 结束时的执行证据与边界

- 最终仍基于 HEAD `1d1069eb90822f4ed7342a47e297811327386ab0`；未切分支、创建提交、修改依赖版本或执行远端 Git 操作。初始源码与设计基线一致，仅多出任务书。
- 使用 `/home/zhanghaoyang/miniconda3/envs/real_robot/bin/python`（Python 3.10.20，NumPy 1.26.4、SciPy 1.15.3、MPlib 0.2.1、fpsample 0.3.3）。系统 shell 没有 `python`，实际验证使用上述现有解释器，未安装/升级依赖。
- 最终测试 **69 项通过、0 失败、0 skip**，包括原有 5 项 Policy/Real 契约测试。固定输入的 8 组 FK/手几何/点云数组及两组真实 IK/动作映射结果与迁移前精确相等；在线 Policy 与离线 Canonical 的共同表示逐字段精确一致。真实数值后端使用仓库 URDF/mesh，设备、模型生命周期和录制边界使用替身。
- 393 个原有函数/类定义与 167 个常量完成 AST 对照，无算法差异：比较时去除导入语句并归一化两处批准的类型名；默认标定路径及键集合导出调整另行核验。所有配置默认值除批准的桌面路径外保持一致，Raw/Canonical 字段、期限比较和锁/清理代码未改写。
- 106 个源码模块导入、652 个仓内导入名解析、10 个关键入口独立新解释器导入均通过；CordViP 集成可导入。源代码、脚本和适配器无旧模块引用；历史说明和显式旧路径失败测试保留旧字符串。
- wheel 使用当前源码的任务自有临时副本构建，逐字节核对其中 106 个 Python 文件与最终工作区一致；三份 JSON 与迁移前 wheel 及工作区一致。只把本项目 wheel 安装到临时 target，使用 `python -I` 从仓库外确认包来源、三份资源和默认桌面路径，未改变全局环境。
- 目录归属没有偏离任务书；相机交互会话按允许顺序随 P3 迁移。新包初始化仅包含说明，不导入算法或会话。未保留旧路径兼容模块。

主要验证命令（均为离线）：

```bash
DEXMANI_PY=/home/zhanghaoyang/miniconda3/envs/real_robot/bin/python
MPLCONFIGDIR=/tmp/dexmani-refactor-mm0h9oo_/mpl "$DEXMANI_PY" -m unittest discover -s tests -p 'test_*.py' -v
"$DEXMANI_PY" -m compileall -q dexmani_real scripts tests
"$DEXMANI_PY" scripts/run_policy.py --print-config
"$DEXMANI_PY" scripts/collect_teleop.py --print-config
git diff --check
# cwd=/tmp/dexmani-refactor-mm0h9oo_/wheel-source
"$DEXMANI_PY" -m pip wheel --no-deps --no-build-isolation --no-index --wheel-dir /tmp/dexmani-refactor-mm0h9oo_/wheels .
```

8 个主实验脚本的 `--help` 均返回 0；键盘入口本来没有 `--print-config`，按既有 CLI 检查 `--help`，未扩充参数。测试日志、固定输入基线、AST 审计和 wheel 等临时证据保存在 `/tmp/dexmani-refactor-mm0h9oo_/`；该目录不是项目运行依赖。

三份标定文件的迁移前后 SHA-256：

```text
a97d21d566477bbcfcea18c08ef9d5455feedfbef2db5b1659ae3d6c414f6c4d  cameras.json
f6e66d2157e27fcb290a133a0379a9026b97bf70577bd763ad63ff6716d468eb  table_plane.json
2f3f53446fb0f8435dc0cacc2ede27c69e11ba9db03e42cc3f10970ac91e239e  vr_transform.json
```

**既有限制与未执行项：** 当前验证环境无未解决的依赖阻塞。新旧 wheel 均不包含机器人 `assets/`，这是既有打包限制，未扩大本次范围修复，不能声称完整独立安装可运行机器人。未连接任何硬件、未加载真实策略权重/GPU 推理、未执行全工作空间或时延性能验收。固定算例和设备替身不构成物理碰撞、停稳或到位证明。

**人工迁移与验收：** 外部代码需改用新模块路径；外部 YAML 显式引用旧标定位置时需人工迁移，缺失文件仍透明失败。配置 `safety` 键、结果字段/文件名和实验输出目录保持不变。操作者后续另行安排设备启停、取消/急停、HOME/jog、遥操作录制、sync/async/RTC 策略运行及回放；本次授权不包含这些操作。P0–P6 的代码、测试与文档工作已结束，无待续接的源码迁移阶段。


### 后续全链路审查与清理（2026-10-26）

- 保留此前全部未提交改动，HEAD 仍为 `1d1069eb90822f4ed7342a47e297811327386ab0`。本轮在迁移后的实际调用链上检查配置、感知、控制、标定、runtime、deployment、数据消费者、脚本及 CordViP 引用。
- `DEFAULT_JUMP_DEG` 与 `joint_target_rejection()` 从 `online_ik.py` 归入已有的 `control/joint_angles.py`，函数体和阈值不变；`TargetChecker` 不再反向依赖在线 IK。合并碰撞模型中重复的几何体→link 名称解析，删除旧专用方法并更新全部调用点。无新文件、兼容模块、框架或配置键。
- 整理迁移后的导入并合并同模块重复导入；区分 IK 配置元组与计算数组、手表面采样的片段列表与拼接数组；使用实际的多进程锁/信号类型。外部 OpenCV 的 CV_8U 输出及构造时已保证存在的碰撞工作区/IPC 通道用类型表达其不变量，没有添加逐帧重复验证或忽略错误的注释。
- 未发现需移除的旧格式/旧流程兼容层或运行时 SHA-256 完整性校验。点云 `spatial_hash()` 是确定性采样算法的一部分，继续保留。相机外参可选录制、碰撞诊断失败仍保留碰撞结论、HOME 等价角候选和清理错误报告均有现行消费者，未误删为历史 fallback。
- 初始 69 项测试通过；新增纯关节检查的导入边界、真实共享内存点云 worker 发布/时钟/ready 清理、真实环境碰撞诊断及详情失败仍拒绝的 3 项测试后，**72 项通过、0 失败、0 skip**。原 8 组数值数组与 P0 精确一致；本轮开始前快照与清理后在两组 IK/ActionMapper 输出及 12 份碰撞报告上精确一致。三份标定 JSON 仍逐字节不变。
- `python -m mypy --ignore-missing-imports --follow-imports silent --explicit-package-bases` 对 README 所列 53 个模块通过（初始 39 条诊断已处理）。这是默认的有标注代码检查，不覆盖未标注函数体或外部无类型 SDK。`ruff check --isolated --select E9,F` 对源码/脚本/测试/集成通过，本轮 Python 文件的 `I` 导入规则通过；未限定规则的扫描并非零告警：core/scripts/tests 从 226 条降至 177 条，无新增诊断，保留既有风格及异常清理相关告警，没有添加 suppression 或更改工具配置。
- 106 个模块和 775 个仓内导入引用（本次含测试）解析通过，CordViP 集成可导入；8 个实验脚本 `--help`、2 个 `--print-config`、compileall、diff 检查通过。使用现有工具在临时副本构建 wheel，106 个 Python 文件与工作区逐字节匹配、三份 JSON 完整；只将本项目安装到临时 target，仓库外 `python -I` 导入及默认路径读取通过。README 26 个本地链接有效。
- README 已补充当前关节检查职责、静态检查命令及范围。未升级依赖、创建提交或连接硬件。既有 wheel 缺少机器人 assets 的限制保持；真实模型/GPU 推理、物理停稳/到位和受控真机验收仍未执行。
- 本轮快照、测试日志、数值/碰撞比较、完整 Ruff 扫描和构建产物位于 `/tmp/dexmani-review-d0xa3cyr/`，不是项目运行依赖。上文 69 项测试与 393 定义 AST 一致性是前一阶段证据；本轮局部简化以本节的新比较与 72 项结果为准。
