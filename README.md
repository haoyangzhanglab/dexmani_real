# DexMani Real

基于 **xArm7 + XHand + RealSense RGB-D + VR/HTS** 的真实机器人灵巧操作实验仓库，支持遥操作示教、数据采集、策略部署与轨迹回放。

策略训练与推理见 [dexmani_policy](https://github.com/haoyangzhanglab/dexmani_policy)，仿真实验见 [dexmani_sim](https://github.com/haoyangzhanglab/dexmani_sim)。

## 安装与配置

使用 Python 3.10+，在仓库根目录安装并运行：

```bash
python -m pip install -e ".[interactive,control,geometry,visualization]"
```

设备 SDK、Pinocchio、MPlib 和手部重定向所需依赖需按实验环境另行安装；策略部署还需安装 `dexmani_policy`。机器人模型与模板从源码目录的 `assets/` 加载，运行时须保留源码 checkout。

默认配置集中在 [config/](dexmani_real/config/)，[config/real.py](dexmani_real/config/real.py) 提供 `RealConfig` 和 `load_config`，负责组合配置及合并覆盖；各子配置可独立导入。支持的入口可通过 `--config local.yaml` 覆盖，优先级为 **CLI > YAML > 默认值**；未知配置键会报错。运行前核对机械臂 IP、手部串口、工作空间和当前设备的标定参数。

| 配置文件 | 内容 |
| --- | --- |
| [xarm7.py](dexmani_real/config/xarm7.py) · [xhand.py](dexmani_real/config/xhand.py) | 机械臂与手的连接、限位、增益和回零参数 |
| [control.py](dexmani_real/config/control.py) | 运动约束、工作空间、桌面与静态障碍物 |
| [sensor.py](dexmani_real/config/sensor.py) | RealSense 与 VR 采集 |
| [perception.py](dexmani_real/config/perception.py) | 点云处理与手点云模板 |
| [teleop.py](dexmani_real/config/teleop.py) | 遥操作、键盘点动、采集和手部重定向 |
| [deployment.py](dexmani_real/config/deployment.py) | 策略执行模式、命令超时与默认实验预算 |
| [calibration_data/](dexmani_real/config/calibration_data/) | 相机外参、桌面平面与 VR 朝向标定结果 |

策略与真机配置的匹配及部署前校验位于 [deployment/preflight.py](dexmani_real/deployment/preflight.py)。

训练与真机部署应固定配套源码版本。部署使用公开 `PolicyInfo`；控制周期、depth 形状/单位及实际使用的点云、手点云和指尖配方沿 Canonical → 训练配置/checkpoint → 公开元信息传递，触觉来源来自已验证训练配置。Real 在设备启动前要求 joint/EEF 的控制动作分别为 19/21 维，并拒绝缺少的部署事实，不用当前 YAML 猜测训练配方。普通仿真训练和纯 policy inspect 不要求这些 Real 配方。

以下命令仅打印配置，不连接设备：

```bash
python scripts/collect_teleop.py --print-config
python scripts/run_policy.py --print-config
```

## 常用命令

`TASK`、`EPISODE` 和 `EXPERIMENT` 分别替换为任务名、episode 目录名和 Policy 实验选择器或路径。各脚本的完整参数见 `--help`。

```bash
# 键盘点动
python scripts/keyboard_teleop.py

# VR/HTS 示教与录制
python scripts/collect_teleop.py TASK

# 策略部署；执行模式支持 sync、async、rtc
python scripts/run_policy.py EXPERIMENT --execution-mode sync

# Raw 数据导出为 Canonical Zarr
python scripts/export_zarr.py episodes/TASK

# 离线查看 episode
python scripts/visualize_episode.py episodes/TASK/EPISODE

# 真机轨迹回放
python scripts/replay_episode.py episodes/TASK/EPISODE
```

键盘点动使用 WASD、方向键及 IJKL，R 回零，Q 退出，Esc 急停。其他入口按启动时显示的操作提示使用。真机回放以录制首帧的实测姿态为起点，默认手部准备目标用 0.5 秒插值到该实测姿态，另行等待实测到位；准备目标必须满足操作限位，之后按原记录目标序列播放。首个目标仍受原有执行约束，不再提前平滑到首个 action。

键盘和相机标定在显示操作提示前完成 jog 准备。普通按键变化只更新目标；松键时从当前伺服参考减速到零并保持，不停机重启，也不继续追赶积累的远处目标。IK 拒绝后减速保持，松开全部 jog 键才能恢复。HOME、标定采样、退出和急停仍执行真正停止，恢复连续控制时重新准备。按键在控制循环中采样，完全落在两次采样之间的极短按键不会补发运动。

机械臂连续控制采用 Mode 1 与 Ruckig 轨迹；prepare 确认模式和实测停稳后，控制端再读取有效观测。等待模式切换期间不要求父进程消费关节反馈；实际控制、策略输入和到达确认仍检查反馈时效。按 R 回零时先输出 `Returning home...`，取消或失败时保留具体原因。HOME 使用 Mode 0 执行路径并确认到达，完成后停止收尾，下次连续控制重新 prepare。

伺服默认 100 Hz，每次发送将 Ruckig 参考推进 10 ms；调度延迟后只继续下一步，并从实际发送时刻安排下一周期，因此延迟会拉长轨迹执行时间。新目标和松键制动沿用上一参考的位置、速度和加速度，参考只在准备后用实测反馈初始化。`arm.max_joint_jerk_deg_per_s3` 默认 6000 °/s³，限制加速度变化率，使启停和换向连续；平滑制动需要更长的停止时间和距离。发送前检查命令期限、反馈时效、控制器状态和新轨迹的关节位置极值（包含制动段），SDK 调用后检查返回码。速度、加速度与 jerk 约束作用于参考序列，实际跟随效果需在真机上确认。

SDK 连接与伺服由机械臂子进程持有。关节反馈默认 25 Hz，每次仅调用一次 `get_joint_states(num=2)` 获取七轴位置和真实速度，来源时间为查询开始时刻。模式、状态和错误码来自 SDK rich 上报，使用独立的接收时间检查时效；启动等待首份报告，模式切换等待命令之后的新报告，这些等待均响应取消。读取缓存不发起 SDK 查询，也不更新时间戳；上报断流或关节查询失败均终止执行。SDK 回调只替换状态快照，跨进程仍传递最新目标、关节反馈和准备身份。

HOME 保留直达、近端优先、远端优先及等价关节分支，按关节行程与逐段停稳开销选择通过校验的完整路径，终点始终为配置中的绝对 HOME 关节角。规划先检查公共端点和路径中点以快速拒绝无效路径，仍按不大于 1° 的关节步长完整检查碰撞、工作空间和桌面间隙；几何查询仅在单次规划内复用。停稳时长从首个合格样本查询完成算到后续合格样本查询开始，排除首尾查询的等待时间；超出本阶段期限的查询不能确认到达。

HOME 的 `HomePlanner` 使用 `environment.table`、标定平面和 `soft_clearance_m` 检查桌面；关闭该桌面模型时，仍用 `table_z_surface_m` 与 `hand_safety_margin_m` 检查手部 frame 高度代理。在线 `MotionConstraints` 的环境碰撞模型只加入 `environment.static_boxes`，不会自动继承 HOME 的桌面几何、软间隙或高度代理规则。`static_boxes` 同时用于 HOME 与在线目标检查；HOME 的桌面检查不能作为在线抓取具有同等桌面保护的依据。

`environment.table.plane_path` 也供点云去桌面和诊断工具使用；`pointcloud.remove_table` 只裁剪观测点云，不执行机器人碰撞检查。

键盘与标定 jog 的目标执行期限均从机械臂反馈的来源时间计算，读取缓存不会延长目标寿命。机械臂进程独立检查目标执行期限、机械臂反馈时效和控制器状态，HOME 另有执行期限。执行或反馈故障会保留原始原因，停止、断开设备并退出机械臂进程。父进程阻塞时，机械臂仍按这些条件处理停止；软件期限不代表物理停止延迟。

策略执行与轨迹逐帧回放每次提交目标时设置 2 秒执行期限；持续收到新目标就刷新期限，断流后由机械臂进程检查并停止。策略等待下一动作也使用这一预算，实验总时长仍限制执行期限。每轮首次历史准备另有 `(n_obs_steps-1)*dt + 2 秒` 的有限预算，首次提交预测后开始独立的 2 秒推理等待；实验总时长从 RUNNING 准入起计算，包含本轮历史准备。sync 收到预测后开始完整控制周期，来源时效与总期限仍生效。回放开始前的手部准备另按到位等待期限保持机械臂。

启动时模型加载与设备准备可重叠；设备就绪后先等待所需触觉和真实观测历史，再在授权运动前用同一观测构造器 warmup。热身不写 Raw，也不占用正式轮次预算；每轮重新采集历史。`session_result.json` 的 `finished` 表示会话生命周期正常结束，轮数仍统计已进入 RUNNING 的轮次；各轮终止原因、技术终止次数另行列出，不代表任务成功率。`inference` 统计模型线程内观测构造至 CPU 动作返回，`prediction_wait` 统计预测请求提交至主循环消费结果，`observation_to_submit` 从该预测最新观测快照完成时刻算至目标提交返回；后者不是曝光、相机接收或硬件 apply latency。

arm 的独立期限不能终止父进程内阻塞的 hand SDK。此次只读核对的 `xhand_controller 1.5.2` 中，`read_state`/`send_command` 的公开签名没有调用期限参数，也没有公开 timeout 设置方法；这不证明内部调用具有时间上界。hand 调用和停止延迟仍待现场测量，不新增独立 hand owner。臂手提交是顺序下发尝试，部分失败沿已有停止路径处理，不保证原子执行或同时到位。

在线几何能力是目标端点准入加 Ruckig 关节位置极值检查（含制动），未验证实际 Ruckig 中间/制动路径连续无碰撞。当前 SRDF 排除全部 435 对手内碰撞，不能据此声称有指间碰撞保护；须有可信网格、真实危险姿态和合法抓握回归后，才恢复相关非相邻对。模型和录制线程等待超时仍可能 pending，保留晚到结果隔离与禁止重启；超时不提供强制终止或进程退出上界。

## 标定与诊断

| 用途 | 入口 |
| --- | --- |
| 相机外参标定 | [calibrate_camera.py](scripts/calibrate_camera.py) |
| VR 朝向标定 | [calibrate_vr_heading.py](scripts/calibrate_vr_heading.py) |
| RealSense 预览 | [view_realsense.py](scripts/tools/view_realsense.py) |
| 点云预览与桌面标定 | [view_pointcloud.py](scripts/tools/view_pointcloud.py) |
| XHand 状态与动作诊断 | [control_xhand.py](scripts/tools/control_xhand.py) |

默认标定参数位于 [config/calibration_data/](dexmani_real/config/calibration_data/)，包含相机外参、桌面平面和 VR 朝向三个 JSON。相机标定实现位于 [utils/camera_calibrate/](dexmani_real/utils/camera_calibrate/)，桌面与 VR 标定辅助代码分别位于 [utils/table_plane.py](dexmani_real/utils/table_plane.py) 和 [utils/vr_heading.py](dexmani_real/utils/vr_heading.py)。

相机标定前须拆除 XHand，或将其固定在配置的 HOME 姿态。SDK 接受命令不代表机器人已到位或停稳。

RealSense 和点云诊断使用 OpenCV RGB-D 面板及 Open3D 点云窗口，需可用的桌面显示环境及 OpenGL 上下文；上述 `visualization` 安装项包含 Open3D 和离线查看所用的 Rerun。

点云处理、导出和预览共用 `PointCloudProcessor`，输出基座系 `float32[N,6]` XYZRGB；无有效点时返回 `(0,6)` 数组，预览清空显示，训练导出拒绝该行。桌面裁减默认保留距标定平面大于 10 mm 的点，可通过 `pointcloud.table_margin_m` 调整。

相机时间取 depth/color 各自首次主机回调接收时间中的较早者，并随帧进入单个最新帧槽；出队及另一通道更新不会刷新旧通道年龄。这只覆盖回调接收后的年龄，不是曝光时间，也未覆盖设备/SDK 到回调之前的延迟。此次环境的 `pyrealsense2` 为 2.54.2；实际设备、对齐 backend、depth→color 外参及 aligned-depth Z 近似误差尚未测量，继续保留现行离线/在线表示。需要改投影时须先满足论文精度条件并同步重新导出，不能单改部署。曝光恢复目前由无设备测试验证了请求及异常控制流，跨 pipeline 持久化效果仍待设备确认。

- RealSense：窗口内 `P` 开关点云、`S` 切换 raw/processed、`F` 仅冻结点云、`D` 切换 Jet/灰度、`R` 重置显示、`C` 打印配置、`A` 切换曝光优先级、`Q/Esc` 退出。关闭 RGB-D 窗口退出程序；关闭点云窗口仅关闭点云，可按 `P` 重开。退出时恢复原曝光优先级。
- 点云诊断：RGB-D 窗口按 Enter 继续，`Q/Esc` 或关闭窗口取消；桌面标定和发布确认在终端进行。采集、快照和 benchmark 完成后即释放相机，最终静态点云窗口显示工作空间框及基座/相机坐标轴。
- 相机标定：保留全局点动按键；Space 采样、Enter 求解并在质量合格后保存、Backspace 撤销、X 删除最差样本。预览显示检测、采样、求解和保存状态；关闭预览窗口请求退出，采样、求解/发布检查和回零等待共用同一取消入口。
- 离线 episode：左侧为基座场景/点云标签页和 RGB，中间为 Depth，右侧为 State、Action、Contact Force 曲线。曲线按列批量发送，与图像共用 `step` 和 `time` 时间轴。

## 模型与手点云

整机 FK、IK 与碰撞检查共用 [xarm7_xhand_right.urdf](assets/robots/xhand/xarm7_xhand_right.urdf)，手部安装位姿由该模型定义。TAG/DexPilot 重定向使用独立的 [xhand_right.urdf](assets/robots/xhand/xhand_right.urdf)。

手点云模板位于 `assets/pointcloud_template/hand/hand_<点数>.npz`，默认使用 1024 点。离线导出与部署共用模板及 FK；训练和部署应使用同一份模板。更改点数或手部 visual 网格、颜色、位姿后，重新生成对应模板：

```bash
python scripts/generate_hand_template.py --num-points 1024 --overwrite
```

Canonical 保留实际 `source_id`、影响数值结果的处理配方及完整 URDF。实验复现由人工记录 `source_id → 导出使用的 Real commit SHA、训练使用的 Policy commit SHA、手模板路径`，示例见 [Policy 的实验产物说明](https://github.com/haoyangzhanglab/dexmani_policy#实验产物)。Git 已跟踪的模板可按对应提交和路径恢复；不同处理阶段使用不同版本时分别记录。未提交的代码或自定义模板应随实验手动归档，或先提交再记录；已有实验缺失的来源信息须如实保留为未知。

修改加工算法或模板后须重新导出数据，不能只复用同名配置或模板文件。没有实际 sampler/点云的耗时和内存收益证据时，保留当前 Zarr chunks、邻居枚举、冻结目标几何检查及两套 FK。

## 数据与代码导航

Raw episode 默认保存在 `episodes/TASK/`，训练数据导出到 `datasets/TASK.zarr`；策略评估和回放结果分别位于 `rollouts/`、`replay_results/`。Raw 保存实测反馈与提交目标。机械臂 streaming 的新目标覆盖尚未消费的旧目标；提交成功表示目标已发布，不保证每条目标进入 SDK 或执行到位。HOME 路点通过独立通道顺序执行。

Canonical 默认保持固定周期有效前缀、完整 13 模态，并对真实 failed episode 保守丢弃末行；导出摘要给出保留数、丢弃数、截断点及原因。视频只验证实际保留范围，保留范围内坏帧仍失败；不承诺恢复索引或头部损坏的 MP4。每次新导出获得独立 `source_id`，原样搬迁保留 ID。Canonical checkpoint resume 的数据和 saved split 必须都有同一 ID；旧数据仍可读取、评价或从头训练，需要可靠 resume 的新实验先显式重新导出，不自动改旧数据或 normalizer。

`control_ns` 是采样起点，`observation_ns` 是本次观测快照完成时刻，设备来源时间独立保留；导出检查控制节拍，物理回放另按观测时钟整段准入。正常录制、可导出和可回放的集合不同，不自动重采样或改写时间。replay 指标仍是按索引比较两次实测轨迹：关节为绝对坐标差，EEF 为位置/旋转误差，不等于目标跟踪、时间对齐或模 2π 等价角误差。

遥操作的 `data.h5` 还逐行保存该次观测消费的右手 VR 输入，仅用于调试，不导出到 zarr：

| 字段 | 单行形状与含义 |
| --- | --- |
| `vr_wrist_pos` | `(3,)`，手腕位置，单位米 |
| `vr_wrist_quat_wxyz` | `(4,)`，手腕旋转，四元数顺序 wxyz |
| `vr_landmarks` | `(21, 3)`，SDK 原始顺序的手部关键点，单位米 |
| `vr_recv_ts_ns` | 手腕与关键点中较早的主机单调接收时间，单位纳秒 |
| `vr_sequence_id` | SDK 来源帧号，重复消费同一 VR 帧时保持不变 |

位姿和关键点保留接收端转换后的 FLU（前、左、上）表示，尚未进行机器人映射、平滑或手部重定向；浮点字段为 float64，时间和帧号为 uint64。录制频率跟随 Raw 控制周期，不额外保存 VR 全速流。VR 列在遥操作录制启动时确定，每行直接写入对应输入；策略录制不创建这些字段。

| 目录 | 职责 |
| --- | --- |
| [robot/](dexmani_real/robot/) · [sensor/](dexmani_real/sensor/) | 设备驱动与采集 |
| [geometry/](dexmani_real/geometry/) · [perception/](dexmani_real/perception/) | 坐标变换、FK、点云与手部表示 |
| [control/](dexmani_real/control/) · [runtime/](dexmani_real/runtime/) · [ipc/](dexmani_real/ipc/) | 动作计算、运行协调与进程通信 |
| [teleop/](dexmani_real/teleop/) · [deployment/](dexmani_real/deployment/) · [replay/](dexmani_real/replay/) | 遥操作、策略执行与回放 |
| [config/](dexmani_real/config/) | 配置定义、覆盖合并与标定参数 |
| [data/](dexmani_real/data/) | 数据记录与导出 |
| [utils/](dexmani_real/utils/) | 标定工具、可视化与通用辅助函数 |
| [scripts/](scripts/) | 实验与工具入口 |

CordVIP 对象重建与跟踪见 [scripts/cordvip/README.md](scripts/cordvip/README.md)。依赖配置见 [pyproject.toml](pyproject.toml)，开发约定见 [AGENTS.md](AGENTS.md)。
