# DexMani Real

基于 **xArm7 + XHand + RealSense RGB-D + VR/HTS** 的真实机器人灵巧操作实验仓库，支持遥操作示教、数据采集、策略部署与轨迹回放。

策略训练与推理见 [dexmani_policy](https://github.com/haoyangzhanglab/dexmani_policy)，仿真实验见 [dexmani_sim](https://github.com/haoyangzhanglab/dexmani_sim)。

## 安装与配置

使用 Python 3.10+，在仓库根目录安装并运行：

```bash
python -m pip install -e ".[interactive,control,geometry,visualization]"
```

设备 SDK、Pinocchio、MPlib 和 NLopt 需按实验环境另行安装；策略部署还需安装 `dexmani_policy`。机器人模型与模板从源码目录的 `assets/` 加载，运行时须保留源码 checkout。

默认配置集中在 [config/](dexmani_real/config/)，[config/real.py](dexmani_real/config/real.py) 提供 `RealConfig` 和 `load_config`。复制 [experiments/default.py](experiments/default.py)，在可信本地 Python 文件中直接导出 `CONFIG: RealConfig`，通过 `--config experiments/my_experiment.py` 使用；省略路径采用 dataclass 默认值。本仓不再读取实验 YAML 或递归合并 CLI overrides，设备参数、执行模式、增益、频率和点数直接在实验文件修改。运行前核对机械臂 IP、手部串口、工作空间和当前设备的标定参数。

例如在实验文件中明确修改两个参数：

```python
from dataclasses import replace
from dexmani_real.config.real import RealConfig

base = RealConfig()
CONFIG = replace(base,
    execution=replace(base.execution, mode="sync"),
    teleop=replace(base.teleop, control_hz=16.0),
)
```

任务名、输入输出路径、段数/时长、是否录制仍由 CLI 选择；推理的 `--weights`、`--inference-steps`、`--seed`、`--device` 保留。`--no-hand` 只确认实验文件已显式设置 `hand.enabled=False`，不修改设备配置；遥操作的无手调试还必须关闭录制。旧本仓 YAML 请一次改写为 Python 文件，用户文件不会被自动删除。Policy 的研究配方从 checkpoint 恢复，其训练配置及运行快照可继续保存 YAML；本仓的 Python 配置迁移不改配套 Policy 的外部协议。

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

训练与真机部署应固定配套源码版本。部署使用公开 `PolicyInfo`；控制周期、depth 形状/单位及实际使用的点云、手点云和指尖配方沿 Canonical → 训练配置/checkpoint → 公开元信息传递，触觉来源来自已验证训练配置。Real 在设备启动前要求 joint/EEF 的控制动作分别为 19/21 维，并拒绝缺少的部署事实，不用当前实验配置猜测训练配方。普通仿真训练和纯 policy inspect 不要求这些 Real 配方。

以下命令只打印请求配置的 JSON，不连接设备、不读取标定或解析 checkpoint，也不代表实测相机事实：

```bash
python scripts/collect_teleop.py --config experiments/default.py --print-config
python scripts/run_policy.py --config experiments/default.py --print-config
```

## 常用命令

`TASK`、`EPISODE` 和 `EXPERIMENT` 分别替换为任务名、episode 目录名和 Policy 实验选择器或路径。各脚本的完整参数见 `--help`。

```bash
# 键盘点动
python scripts/keyboard_teleop.py --config experiments/default.py

# VR/HTS 示教与录制
python scripts/collect_teleop.py TASK --config experiments/default.py

# 策略部署；执行模式支持 sync、async、rtc
python scripts/run_policy.py EXPERIMENT --checkpoint latest --config experiments/default.py --num-episodes 5

# Raw 数据导出为 Canonical Zarr
python scripts/export_zarr.py episodes/TASK --config experiments/default.py --output datasets/TASK.zarr

# 离线查看 episode
python scripts/visualize_episode.py episodes/TASK/EPISODE --config experiments/default.py

# 真机轨迹回放
python scripts/replay_episode.py episodes/TASK/EPISODE --config experiments/default.py --output replay_results/run_001
```

键盘点动使用 WASD、方向键及 IJKL，R 回零，Q 退出，Esc 急停。其他入口按启动时显示的操作提示使用。真机回放以录制首帧的实测姿态为起点，默认手部准备目标用 0.5 秒插值到该实测姿态，另行等待实测到位；准备目标必须满足操作限位，之后按原记录目标序列播放。首个目标仍受原有执行约束，不再提前平滑到首个 action。

VR 遥操作和策略部署每次启动会话时自动将手部回到 HOME，确认实测到位后请求五个触觉传感器复位，并等待复位后的 aggregate/dense 反馈；启动前应让手部无接触。复位请求失败或反馈不可用时不进入遥操作或策略 warmup。SDK 请求成功及反馈就绪不等同于已验证触觉零点；复位元数据会随 Raw 保存。各段之间复用设备和模型，初始化与触觉复位不会随每段重做。

遥操作和策略会话使用 `B` 开始、`S` 保存并结束本段、`D` 丢弃本段、`Q` 保存并退出、`Esc` 急停。`H` 仅在空闲且没有活动段时回初始位；`T` 仅在空闲且手部无接触时请求触觉复位。已取消 C 暂停/续写、双 Q 确认和音频播放，日常提示直接输出到终端。

每段在 PREPARING 阶段先做一次不下发的目标预计算；机械臂准备完成后再用新观测重建锚点及滤波初值，开始控制与录制计时。本段先撤权并请求停止，再等待录制关闭，由监督方决定发布或丢弃，全部完成后才接收下一段。收尾期间积累的 B/H 等准备按键失效；writer 或模型任务未退出则终止会话，不再授权下一段。

腕部按每段初始观测做相对映射：位置增量通过 VR 朝向标定转入机器人基座系，再按 `vr_pos_scale` 缩放；姿态从初始实测 EEF 朝向跟随腕部相对旋转。映射与旋转 EMA 共用 `Rotation` 表示，沿最短旋转插值；相对旋转恰为 180° 时存在两条等长最短弧，不保证穿过该点的插值方向连续。平滑参考使用最近成功提交的工作空间投影目标，新段开始时重新锚定。

腕部 `ema_alpha_pos/ema_alpha_rot` 是每控制步中新目标的固定权重；在滤波计算中，0 保留参考，1 直接采用新目标。同一 VR 缓存帧可跨多个控制步复用，腕部 EMA 在每次成功提交后继续趋近该观测；手部重复读取该缓存帧则复用候选，不重复求解或滤波。

VR 手部重定向使用固定骨长标定，联合拟合 XHand 可对应的近端/末段方向、骨链/指尖位置、有意对指距离和时间阻尼。启动时采集 25 个新鲜且来源时间递增的右手样本，取骨长中位数；不要求伸直，标定在本次会话内固定，同一会话的各段之间复用标定。请保持右手清晰可见；标定 5 秒内未完成或骨长不稳定时明确退出，S/Q/Esc 可取消。

手部参数位于 `hand_retargeting`。求解范围采用 `hand.qpos_min_rad/qpos_max_rad`，另限制相邻提交目标的变化率，默认 4 rad/s；这不是实测电机速度或接触力保证。单层滤波使用时间常数 `smoothing_tau_s=0.02`，按本次控制观测与上次成功提交观测的时间差计算权重；候选只有在提交成功后才更新下一帧的初值、阻尼和滤波状态。无效输入和求解失败会保留具体原因并终止当前段。腕部位移倍率为 0.7、位置/旋转 EMA 权重为 0.95/0.85、控制与录制频率为 16 Hz。

手部 FK 和优化使用 NumPy、Pinocchio 和 NLopt，实现与机构约束见 [手部重定向说明](docs/hand_retargeting.md)。Raw 的 `meta.hand_retargeting_calibration_json` 保存标定样本数、15 段骨长和掌宽。

遥操作 Raw 的 `meta` 属性保存 `runtime_config_json`、实际使用的 `vr_to_robot_rotation` 和 `teleop_run_id`。启用录制时，会话配置在构造控制循环、连接设备前序列化，各分段复用同一快照；此步骤检查完整配置的 JSON 可序列化性，各参数的业务约束仍由其使用入口校验。

`meta/termination_details` 保存预计算耗时、`teleop_anchor` 初始观测，以及录制期间处理的按键和失败时的 `control_timing`。事件时间戳均为主机单调时钟；`submit_start_ns/submit_end_ns` 表示父进程提交调用，不代表机械臂到达目标。终端按键日志记录处理时的阶段与运行编号。

键盘和相机标定在显示操作提示前完成 jog 准备。普通按键变化只更新目标；松键时从当前伺服参考减速到零并保持，不停机重启，也不继续追赶积累的远处目标。IK 拒绝后减速保持，松开全部 jog 键才能恢复。HOME、标定采样、退出和急停仍执行真正停止，恢复连续控制时重新准备。按键在控制循环中采样，完全落在两次采样之间的极短按键不会补发运动。

机械臂连续控制采用 Mode 1 与 Ruckig 轨迹；prepare 切换模式并确认实测停稳后，控制端再读取有效观测。实际控制、策略输入和到达确认检查反馈时效。按 R 回零时先输出 `Returning home...`，取消或失败时保留具体原因。HOME 使用 Mode 0 执行路径并确认到达，完成后停止收尾，下次连续控制重新 prepare。

伺服默认 100 Hz，每次发送将 Ruckig 参考推进 10 ms；调度延迟后只继续下一步，并从实际发送时刻安排下一周期，因此延迟会拉长轨迹执行时间。新目标和松键制动沿用上一参考的位置、速度和加速度，参考只在准备后用实测反馈初始化。`arm.max_joint_jerk_deg_per_s3` 默认 12000 °/s³，限制加速度变化率，使启停和换向连续；平滑制动需要更长的停止时间和距离。该默认值由所有机械臂入口共用，速度/加速度上限仍为 150 °/s、900 °/s²。发送前检查命令期限、反馈时效、控制器状态和新轨迹的关节位置极值（包含制动段），SDK 调用后检查返回码。速度、加速度与 jerk 约束作用于参考序列，实际跟随效果需在真机上确认。

机械臂每段停止后输出 `Arm timing run=...`：伺服 SDK 调用、相邻调用开始间隔及运行期间反馈查询的次数、平均值和最大耗时。统计包含失败的调用尝试，按运行编号与遥操作 Raw 对照；准备和 HOME 不计入这些连续控制统计。

SDK 连接与伺服由机械臂子进程持有。关节反馈默认 25 Hz，每次主动查询错误码，再调用一次 `get_joint_states(num=2)` 获取七轴位置和真实速度，来源时间为这组查询开始时刻。模式与状态直接使用 SDK 的 100 Hz real 上报缓存，伺服发送前检查 Mode 1 和可运动状态；错误码由主动查询和运动命令返回码检查。模式切换直接检查 SDK 返回码，不另建回报快照、时效预算或等待报告的握手。父进程读取关节缓存不发起 SDK 查询，也不更新时间戳；查询失败或反馈过期均终止执行。设备关闭统一由运行入口的 finally 执行，只报告实际停止与断开失败。

HOME 保留直达、近端优先、远端优先及等价关节分支，按关节行程与逐段停稳开销选择通过校验的完整路径，终点始终为配置中的绝对 HOME 关节角。规划先检查公共端点和路径中点以快速拒绝无效路径，仍按不大于 1° 的关节步长完整检查碰撞、工作空间和桌面间隙；几何查询仅在单次规划内复用。桌面距离复用同次碰撞查询的距离下界，仅对可能缩小最小距离的几何对执行精确求距。停稳时长从首个合格样本查询完成算到后续合格样本查询开始，排除首尾查询的等待时间；超出本阶段期限的查询不能确认到达。

HOME 的 `HomePlanner` 使用 `environment.table`、标定平面和 `soft_clearance_m` 检查桌面；关闭该桌面模型时，仍用 `table_z_surface_m` 与 `hand_safety_margin_m` 检查手部 frame 高度代理。在线 `MotionConstraints` 的环境碰撞模型只加入 `environment.static_boxes`，不会自动继承 HOME 的桌面几何、软间隙或高度代理规则。`static_boxes` 同时用于 HOME 与在线目标检查；HOME 的桌面检查不能作为在线抓取具有同等桌面保护的依据。

工作空间和静态盒体位置均使用机器人基座系；盒体尺寸为完整边长，旋转四元数使用 WXYZ 顺序。`allowed_contact_links` 必须填写 URDF 中的机器人 link 名，未知名称或关节名会在桌面模型构造时明确报错。关闭桌面模型时不检查其休眠业务范围，完整配置仍须可严格序列化；启用时执行完整业务校验。

`environment.table_plane_path` 是 HOME 和点云共用的桌面标定来源，独立于 HOME 碰撞盒体参数；`pointcloud.remove_table` 只裁剪观测点云，不执行机器人碰撞检查。

键盘与标定 jog 的目标执行期限均从机械臂反馈的来源时间计算，读取缓存不会延长目标寿命。机械臂进程独立检查目标执行期限、机械臂反馈时效和控制器状态，HOME 另有执行期限。执行或反馈故障会保留原始原因，停止、断开设备并退出机械臂进程。父进程阻塞时，机械臂仍按这些条件处理停止；软件期限不代表物理停止延迟。

策略执行与轨迹逐帧回放每次提交目标时设置 2 秒执行期限；持续收到新目标就刷新期限，断流后由机械臂进程检查并停止。策略等待下一动作也使用这一预算，实验总时长仍限制执行期限。每轮首次历史准备另有 `(n_obs_steps-1)*dt + 2 秒` 的有限预算，首次提交预测后开始独立的 2 秒推理等待；实验总时长从 RUNNING 准入起计算，包含本轮历史准备。sync 收到预测后开始完整控制周期，来源时效与总期限仍生效。回放开始前的手部准备另按到位等待期限保持机械臂。

策略预取冻结的目标已完成端点准入且不可变，在固定环境中只复查依赖当前关节位置的等价分支；执行授权、反馈时效和命令期限仍在提交时检查。

启动时模型加载与设备准备可重叠；设备就绪后先完成手部 HOME 与触觉复位，等待所需真实观测历史，再在策略运动开始前用同一观测构造器 warmup。热身不写 Raw，也不占用正式轮次预算；每轮重新采集历史。`session_result.json` 的 `finished` 表示会话生命周期正常结束，轮数仍统计已进入 RUNNING 的轮次；各轮终止原因、技术终止次数另行列出，不代表任务成功率。`inference` 统计模型线程内观测构造至 CPU 动作返回，`prediction_wait` 统计预测请求提交至主循环消费结果，`observation_to_submit` 从该预测最新观测快照完成时刻算至目标提交返回；后者不是曝光、相机接收或硬件 apply latency。

arm 的独立期限不能终止父进程内阻塞的 hand SDK。此次只读核对的 `xhand_controller 1.5.2` 中，`read_state`/`send_command` 的公开签名没有调用期限参数，也没有公开 timeout 设置方法；这不证明内部调用具有时间上界。hand 调用和停止延迟仍待现场测量，不新增独立 hand owner。臂手提交是顺序下发尝试，部分失败沿已有停止路径处理，不保证原子执行或同时到位。

在线几何能力是目标端点准入加 Ruckig 关节位置极值检查（含制动），未验证实际 Ruckig 中间/制动路径连续无碰撞。臂、手的命令限位在目标入口校验；碰撞模型直接使用实测或已校验的目标姿态，不因手部反馈越过 URDF 名义限位而拒绝机械臂候选，也不裁剪实测反馈。当前碰撞模型统一跳过全部 435 对手内碰撞，SRDF 仅保存其余碰撞排除，不能据此声称有指间碰撞保护；须有可信网格、真实危险姿态和合法抓握回归后，才调整该规则、恢复相关非相邻对。模型和录制线程等待超时仍可能 pending，保留晚到结果隔离与禁止重启；超时不提供强制终止或进程退出上界。

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

点云处理、导出和预览共用 `PointCloudProcessor`，输出基座系 `float32[N,6]` XYZRGB；无有效点时返回 `(0,6)` 数组，预览清空显示，训练导出拒绝该行。默认工作空间为基座系米制 `x=[0.15,0.85]、y=[-0.50,0.50]、z=[0,0.80]`，由 `pointcloud.workspace` 配置。桌面裁减默认保留距标定平面大于 10 mm 的点，可通过 `pointcloud.table_margin_m` 调整。

点云在深度图中检查同表面邻域支持，体素聚合后执行半径密度与最少 10 点的连通分量过滤，再进行 FPS，抑制悬浮小点团对采样的影响。深度断层处的四邻域支持要求会收缩边缘，可能删除仅 1–2 像素宽的真实细结构。

相机时间取 depth/color 各自首次主机回调接收时间中的较早者，并随帧进入单个最新帧槽；出队及另一通道更新不会刷新旧通道年龄。这只覆盖回调接收后的年龄，不是曝光时间，也未覆盖设备/SDK 到回调之前的延迟。此次环境的 `pyrealsense2` 为 2.54.2；实际设备、对齐 backend、depth→color 外参及 aligned-depth Z 近似误差尚未测量，继续保留现行离线/在线表示。需要改投影时须先满足论文精度条件并同步重新导出，不能单改部署。曝光恢复目前由无设备测试验证了请求及异常控制流，跨 pipeline 持久化效果仍待设备确认。

- RealSense：窗口内 `P` 开关点云、`S` 切换 raw/processed、`F` 仅冻结点云、`D` 切换 Jet/灰度、`R` 重置显示、`C` 打印配置、`A` 切换曝光优先级、`Q/Esc` 退出。关闭 RGB-D 窗口退出程序；关闭点云窗口仅关闭点云，可按 `P` 重开。退出时恢复原曝光优先级。
- 点云诊断：RGB-D 窗口按 Enter 继续，`Q/Esc` 或关闭窗口取消；桌面标定和发布确认在终端进行。采集、快照和 benchmark 完成后即释放相机，最终静态点云窗口显示工作空间框及基座/相机坐标轴。
- 相机标定：保留全局点动按键；Space 采样、Enter 求解并在质量合格后保存、Backspace 撤销、X 删除最差样本。预览显示检测、采样、求解和保存状态；关闭预览窗口请求退出，采样、求解/发布检查和回零等待共用同一取消入口。
- 离线 episode：左侧为基座场景/点云标签页和 RGB，中间为 Depth，右侧为 State、Action、Contact Force 曲线。曲线按列批量发送，与图像共用 `step` 和 `time` 时间轴。

## 模型与手点云

整机 FK、IK 与碰撞检查共用 [xarm7_xhand_right.urdf](assets/robots/xhand/xarm7_xhand_right.urdf)，手部安装位姿由该模型定义。机械臂与手部 FK、IK 从该模型提取各自的运动学子树，手部重定向与数据导出、部署共用同一手部模型来源。

手点云模板位于 `assets/pointcloud_template/hand/hand_<点数>.npz`，默认使用 1024 点。离线导出与部署共用模板及 FK；训练和部署应使用同一份模板。更改点数或手部 visual 网格、颜色、位姿后，重新生成对应模板：

```bash
python scripts/generate_hand_template.py --num-points 1024 --overwrite
```

Canonical 保留实际 `source_id`、影响数值结果的处理配方及完整 URDF。部署恢复训练保存的完整点云配置，缺失或未知字段直接报错。实验复现由人工记录 `source_id → 导出使用的 Real commit SHA、训练使用的 Policy commit SHA、手模板路径`，示例见 [Policy 的实验产物说明](https://github.com/haoyangzhanglab/dexmani_policy#实验产物)。Git 已跟踪的模板可按对应提交和路径恢复；不同处理阶段使用不同版本时分别记录。未提交的代码或自定义模板应随实验手动归档，或先提交再记录；已有实验缺失的来源信息须如实保留为未知。

修改加工算法或模板后须重新导出数据，不能只复用同名配置或模板文件。

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
