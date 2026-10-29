# DexMani Real

基于 **xArm7 + XHand + RealSense RGB-D + VR/HTS** 的真实机器人灵巧操作实验仓库，支持遥操作示教、数据采集、策略部署与轨迹回放。

策略训练与推理见 [dexmani_policy](https://github.com/haoyangzhanglab/dexmani_policy)，仿真实验见 [dexmani_sim](https://github.com/haoyangzhanglab/dexmani_sim)。

## 安装与配置

使用 Python 3.10+，在仓库根目录安装并运行：

```bash
python -m pip install -e ".[interactive,control,geometry,visualization]"
```

设备 SDK、Pinocchio、MPlib 和手部重定向所需依赖需按实验环境另行安装；策略部署还需安装 `dexmani_policy`。机器人模型与模板从源码目录的 `assets/` 加载，运行时须保留源码 checkout。

默认配置见 [config.py](dexmani_real/config.py) 及各模块的配置类。支持的入口可通过 `--config local.yaml` 覆盖，优先级为 **CLI > YAML > 默认值**；未知配置键会报错。运行前核对机械臂 IP、手部串口、工作空间和当前设备的标定参数。

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

键盘点动使用 WASD、方向键及 IJKL，R 回零，Q 退出，Esc 急停。其他入口按启动时显示的操作提示使用。真机回放前，需将手部摆到接近录像首帧的姿态。

机械臂连续控制采用 Mode 1 与 Ruckig 轨迹；prepare 确认模式和实测停稳后，控制端再读取有效观测。等待模式切换期间不要求父进程消费关节反馈；实际控制、策略输入和到达确认仍检查反馈时效。HOME 使用 Mode 0 执行路径并确认到达，完成后停止收尾，下次连续控制重新 prepare。

机械臂进程独立检查目标执行期限、机械臂反馈时效和控制器状态，HOME 另有执行期限。父进程阻塞时，机械臂仍按这些条件处理停止；软件期限不代表物理停止延迟。

## 标定与诊断

| 用途 | 入口 |
| --- | --- |
| 相机外参标定 | [calibrate_camera.py](scripts/calibrate_camera.py) |
| VR 朝向标定 | [calibrate_vr_heading.py](scripts/calibrate_vr_heading.py) |
| RealSense 预览 | [view_realsense.py](scripts/tools/view_realsense.py) |
| 点云预览与桌面标定 | [view_pointcloud.py](scripts/tools/view_pointcloud.py) |
| XHand 状态与动作诊断 | [control_xhand.py](scripts/tools/control_xhand.py) |

默认标定参数位于 [calibration/parameters/](dexmani_real/calibration/parameters/)。相机标定前须拆除 XHand，或将其固定在配置的 HOME 姿态。SDK 接受命令不代表机器人已到位或停稳。

RealSense 和点云诊断使用 OpenCV RGB-D 面板及 Open3D 点云窗口，需可用的桌面显示环境及 OpenGL 上下文；上述 `visualization` 安装项包含 Open3D 和离线查看所用的 Rerun。

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

## 数据与代码导航

Raw episode 默认保存在 `episodes/TASK/`，训练数据导出到 `datasets/TASK.zarr`；策略评估和回放结果分别位于 `rollouts/`、`replay_results/`。Raw 保存实测反馈与提交目标，提交目标不表示硬件已执行到位。

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
| [data/](dexmani_real/data/) · [calibration/](dexmani_real/calibration/) | 数据记录、导出与标定 |
| [scripts/](scripts/) | 实验与工具入口 |

CordVIP 对象重建与跟踪见 [scripts/cordvip/README.md](scripts/cordvip/README.md)。依赖配置见 [pyproject.toml](pyproject.toml)，开发约定见 [AGENTS.md](AGENTS.md)。
