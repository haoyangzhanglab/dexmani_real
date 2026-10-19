# DexMani Real

面向个人 PhD 研究的真实机器人灵巧操作仓库，当前实验系统为 **xArm7 + XHand + RealSense RGB-D + VR / HTS**。

本仓负责设备接入、标定、遥操作采集、Raw 数据、离线处理及策略真机部署。模型、Dataset 与训练位于 [dexmani_policy](https://github.com/haoyangzhanglab/dexmani_policy)，仿真位于 [dexmani_sim](https://github.com/haoyangzhanglab/dexmani_sim)。

工作流：**标定 → 遥操作采集 → Raw → Canonical Zarr → 策略训练 → 真机评估**。

## 安装与配置

使用 Python ≥ 3.10，在仓库根目录执行下列命令。机器人资产按源码 checkout 路径读取，实验环境使用 editable 安装：

```bash
python -m pip install -e ".[interactive,control]"
```

仅安装基础离线处理依赖时可用 `python -m pip install -e .`。安装声明见 [pyproject.toml](pyproject.toml)；硬件 SDK 和几何、推理后端沿用实验机环境，按需准备：

| 功能 | 额外依赖 |
| --- | --- |
| xArm、XHand、RealSense、VR / HTS | 对应 SDK、驱动及设备连接配置 |
| FK、IK、碰撞检查与 HOME 规划 | Pinocchio、MPlib |
| TAG / DexPilot 手部重定向 | NLopt 及所选后端的几何依赖 |
| 点云诊断视图 / episode 交互查看 | Open3D / Rerun |
| 策略部署 | 已安装的 `dexmani_policy` 及训练实验对应的推理依赖 |

键盘交互使用 pynput，需要桌面显示会话。安装 Python 包本身不验证设备连接或现场标定。

配置优先级为 **CLI 覆盖 > `--config` YAML > 包内默认值**。先查看完整声明配置：

```bash
python examples/collect_teleop.py --print-config
python examples/run_policy.py --print-config
```

将本次实验需要覆盖的字段保存为 `local.yaml`，例如：

```yaml
teleop:
  control_hz: 16.0
pointcloud:
  num_points: 1024
```

```bash
python examples/collect_teleop.py --config local.yaml --print-config
```

字段以当前输出及[配置加载入口](dexmani_real/config/experiment.py)为准，未知字段直接报错。运行前核对设备地址、serial、工作空间与现场标定；示例 YAML 仅展示覆盖方式。

各入口的 `--help` 不连接设备；支持 `--print-config` 的入口仅打印声明配置，不读取现场标定或加载策略，也不证明运行条件已满足。日志默认写入 `~/.dexmani/logs/`，可用 `DEXMANI_LOG_DIR` 指定位置。

## 实验工作流

以下 `test`、`episode_YYYYMMDD_HHMMSS` 和 `policy/task/experiment` 是示例名称，使用时替换为实际任务、episode 和策略实验。需要 YAML 覆盖时传入已准备的 `local.yaml`。

### 1. 标定

相机外参、桌面平面、手部安装与 VR alignment 属于当前实验状态；物理设置变化后重新标定或确认。相机采用固定安装的 eye-to-hand 外参，按设备 serial 绑定。

| 标定 | 入口与要求 |
| --- | --- |
| 相机外参 | `examples/calibrate_camera.py`；会连接并移动机械臂、采集图像，必须显式声明 `--hand-geometry absent` 或 `secured-home` |
| VR 朝向 | `examples/calibrate_vr_heading.py`；`--ref head` 面向机器人 +X，`--ref wrist` 将手腕指向机器人 +X |
| 桌面平面 | `examples/pointcloud_process_example.py`；启动后选择 table calibration，确认后发布拟合结果 |

先查看各入口的具体操作：

```bash
python examples/calibrate_camera.py --help
python examples/calibrate_vr_heading.py --help
python examples/pointcloud_process_example.py --help
```

相机标定中的 `absent` 表示手未安装，`secured-home` 表示已物理固定在配置 HOME 手形；两者均按固定 HOME 手形做碰撞检查。相机与 VR 标定支持 `--output`；桌面文件由 `environment.table.plane_path` 指定。

采集允许缺相机外参，点云重建需要数据对应的内外参与 depth scale。采集可通过 `--vr-transform`、`--camera-calibration` 选择标定，部署支持 `--camera-calibration`。VR 质量为 `poor` 的标定不能用于遥操作启动。

### 2. 遥操作与采集

```bash
python examples/collect_teleop.py test --config local.yaml
python examples/collect_teleop.py test --config local.yaml --control-hz 20
```

任务名默认 `test`，录制默认写入 `episodes/test/`，根目录由 `control.episodes_dir` 配置。启动时建立停止监听并执行手部 HOME，随后等待操作者开始。

| 按键 | 遥操作行为 |
| --- | --- |
| H | 保存并结束当前段，执行整机 HOME，完成后等待新的 B |
| B | 开始采集 |
| C | 暂停或恢复；恢复开始新的 episode |
| S | 停止并保存，回到空闲 |
| D | 丢弃当前未发布 capture，回到空闲 |
| T | 空闲且无 capture 时请求复位触觉传感器；操作前确保五指无接触 |
| Q | 暂停并进入退出确认，再按 Q 保存退出；空闲且无 capture 时直接退出 |
| ESC | 请求急停 |

`--control-hz` 同时设置目标、观测采样与录制频率，默认 16 Hz；机械臂伺服频率独立。频率须满足设备与反馈新鲜度预算，调频会改变输入 EMA 的时间响应，应记录实际配置。错过时隙或反馈过期时停止并保留真实前缀，不追赶补发。

手部默认使用 TAG，`control.hand_retargeting_type: dexpilot` 切换到 DexPilot。位置映射与 EMA 分别由 `control.vr_mapping`、`control.ema` 配置。`--no-record` 关闭录制及相机采集，遥操作仍会执行；`--no-hand` 仅用于明确无手或手已固定在 HOME 的无录制调试。

### 3. 离线查看与导出

```bash
python examples/visualize_episode.py episodes/test/episode_YYYYMMDD_HHMMSS --info
python examples/visualize_episode.py episodes/test/episode_YYYYMMDD_HHMMSS --config local.yaml --max-frames 100
python examples/export_policy_zarr.py episodes/test --config local.yaml
```

`--info` 打印数据概要，无需启动 Rerun；交互查看可用 `--no-point-cloud` 关闭点云。查看和导出使用 Raw 保存的相机几何，以及当前点云、桌面、手部安装与指尖配置。仅启用去桌时读取桌面平面。

公共导出要求每段全部 **15 字段**完整、浮点数值有限，并通过时序及动作执行证据检查。缺字段、NaN/Inf、RGB-D 行数不符、相机几何缺失或空点云均整段拒绝，记录原因并保留原 Raw；任务失败本身不排除技术有效的数据。导出不删坏帧、不填补缺测、不修改 Raw。

默认输出 `datasets/test.zarr`，目标必须不存在；`--output` 指定新路径，`--exclude` 排除指定 episode，`--chunk-frames` 调整批大小，`--pointcloud-num-points` 覆盖点数。同一缓存要求 task、dt、depth scale 和数组 shape/dtype 一致；合法 metadata 冲突会在读取后续数值前中止导出。文件系统或处理异常保留 staging 供排查。

训练 action 表示本拍提交的绝对关节目标，`action_ee` 是该目标的 FK，均不代表下一拍已到达。时序筛选同时检查逐拍间隔与累计偏离；Raw 采集正常结束不等于可导出。布局采用 `data/*` 与 `meta/episode_ends`，字段定义见 [Canonical 数据契约](dexmani_real/dataset/contracts.py)，处理入口见 [dataset/processing.py](dexmani_real/dataset/processing.py)。

在线与离线共用[点云处理](dexmani_real/sensor/pointcloud/processing.py)：去飞点、桌面过滤、workspace、体素 XYZ/RGB 均值、密度与连通分量筛选，再限制候选并做 **CPU QuickFPS**。默认输出 1024 点、最多 8192 个候选，依赖固定为 `fpsample==0.3.3`，无需 CUDA。不足点数时循环已有点，空云保留无有效输出。

点云模型使用训练 artifact 保存的 recipe，须明确包含 `sampling_method: quickfps`；Real 部署采用该 recipe 和当前现场标定。更换采样表示后需重新导出并训练，不能仅凭点数相同沿用模型。FPS 不保证邻帧选点连续或策略成功率提升。

### 4. 策略部署

```bash
python examples/run_policy.py policy/task/experiment --config local.yaml
python examples/run_policy.py policy/task/experiment --config local.yaml --execution-mode async --prefetch-steps 1 --n-action-steps 4
```

实验输入为不带 `experiments/` 前缀的 `policy/task/experiment`，也可使用有效的绝对实验目录。策略 artifact 提供模型输入、动作表示和控制 dt；Real 提供当前设备、标定与执行配置。每段执行步数按 `--n-action-steps` > YAML 的 `execution.action_steps` > 策略默认值确定。

| 执行模式 | 行为与配置 |
| --- | --- |
| `sync`（默认） | 执行一段后等待下一次推理，保留真实段间等待 |
| `async` | 当前段执行时预取下一段，需设置 `prefetch_steps` |
| `rtc` | 预取并冻结交接前已承诺的动作前缀，需设置 `prefetch_steps` 与 `rtc_guidance_cap` |

预取步数表示距交接还剩多少个控制步时启动下一次推理，必须满足策略 horizon 的容量约束。当前 RTC 支持连续动作 BaseAgent DDIM，ManiFlow 支持 sync/async；具体参数见 `--help`。

H 回 HOME、B 开始本轮、S 结束本轮；整理场景后再次按 H、B 开始下一轮。Q 提前退出，ESC 请求急停，T 在空闲且无接触时请求复位触觉。部署不使用 C/D。默认最多 20 轮，每轮运行预算 60 秒，分别用 `--num-episodes`、`--max-duration` 修改；这些预算不保证阻塞 SDK 或模型调用能立即返回。

checkpoint 默认 `latest`，也支持 `best`、已有百分比 milestone（如 `80pct`）或文件名。`--weights`、`--inference-steps` 覆盖推理设置。默认产物位于 `rollouts/policy/task/experiment/session_*/`，`--output` 可指定根目录；`run_config.yaml` 保存实际配置与模型信息，`session_result.json` 保存评估摘要。

`--no-record` 与 `--no-results` 分别关闭 Raw 和摘要，同时关闭时不创建会话产物；**它们不会关闭设备连接或动作执行**。相机按策略输入与 Raw 录制需求启动，仅使用机器人状态且关闭 Raw 时不启动相机。

Rollout Raw 保留 bootstrap/WAIT 的无动作行及终止原因，公共导出仍执行相同准入规则，通常会拒绝含无动作行的段。此类数据用于评估取证，不能通过填动作或裁行冒充当前训练格式。

### 5. 真机轨迹回放

```bash
python examples/replay_episode.py episodes/test/episode_YYYYMMDD_HHMMSS --config local.yaml --output replay_results/test_run
```

回放读取当前格式的 teleop Raw，按标称节拍执行已记录关节目标；会连接机械臂与手，不启动相机或 VR。启动先执行手部 HOME，再默认用 0.5 秒渐变到首帧手部目标；`--hand-start-duration 0` 仅跳过渐变。

输出目录须新建或为空，保存 `session_result.json`、`replay_data.npz` 和可计算的 `metrics.json`。Q 保存前缀并退出，ESC 请求急停，完成后可按 H 规划回 HOME。晚帧或发送超时会停止，输入 Raw 保持不变。评价中的轨迹帧偏移用于跨运行对齐，不表示硬件响应延迟。

## 诊断与其他入口

| 用途 | 入口 | 运行边界 |
| --- | --- | --- |
| 键盘 jog / HOME | [keyboard_teleop.py](examples/keyboard_teleop.py) | 连接机器人，不启动相机或 VR；松键停止 jog，R 执行 HOME |
| XHand 动作与反馈 | [xhand_control_example.py](examples/xhand_control_example.py) | 默认执行动作；`--read-only` 仍连接手并读取反馈 |
| RGB-D 诊断 | [realsense_record_example.py](examples/realsense_record_example.py) | 连接相机，交互查看 RGB-D 与点云 |
| 点云诊断 / 桌面标定 | [pointcloud_process_example.py](examples/pointcloud_process_example.py) | 连接相机；`--save-dir` 保存本次 RGB-D、点云及重建 metadata |
| 物体重建与跟踪 | [CORDViP 示例](examples/cordvip/README.md) | 使用独立的 `cordvip_geometry` 依赖与入口，区分离线数据和实时相机模式 |

点云诊断保存与最终显示使用同一帧，后续连续帧测速仅报告耗时和有效点云数量。关闭 `pointcloud.remove_table` 时处理不使用桌面平面；拟合结果只有确认后才发布。各入口的参数与按键以 `--help` 和交互提示为准。

## 数据与安全边界

**Raw 发布后不可变，Canonical 是可重建的训练缓存。** Raw episode 使用 `data.h5` 和 `rgb.mp4`，保留物理观测、目标、真实 dispatch、时钟和终止原因。RGB 与深度按保存行对齐，实际采样时刻以 HDF5 时间为准；视频使用标称帧率编码。历史格式使用对应版本代码读取，不改写 Raw 来满足新格式。

停止、晚帧、控制失败保留已采前缀；仅显式丢弃才删除当前 capture。写盘、编码、发布或执行结束无法确认时保留 staging，不能报告为完整 Raw。公共导出整段筛选，训练按模型需要加载字段，二者均不回写 Raw。

触觉按明确时限保存上一有效三轴向量保持后的消费值，保留原有效时间，过期停止；导出直接使用保存值，不再次修补或扣偏置。hand current / arm effort 等辅助缺测保留 NaN，这些 Raw 仍可留存，但会被完整 15 字段导出拒绝。

机械臂 SDK 由独立进程单独持有。各运动入口保留关节限位、最终 FK 工作空间、自碰撞与静态障碍检查，按入口采用相应手形和桌面接触规则；整机 HOME 额外检查路径并等待实测稳定。目标准入、SDK 接受、实测到位和真实停止是不同结果。

xArm 控制器碰撞检测默认关闭（`arm.collision_sensitivity: 0`），不能依赖它触发碰撞停机；软件几何检查仍执行。真机运行前确认安装、标定、限位和工作空间，保持操作者监督及硬件急停可用。SDK 或 CUDA 阻塞可能影响停止响应，离线检查不能证明真机安全。

## 开发

默认运行无设备副作用的检查：

```bash
python -m compileall -q dexmani_real examples
git diff --check
```

按改动补充纯逻辑、假时钟或 SDK/model 替身验证，本仓库不使用 pytest 或 Ruff。长期协作规则见 [AGENTS.md](AGENTS.md)，Claude 入口见 [CLAUDE.md](CLAUDE.md)。代码代理未经明确授权不得连接设备、执行运动、实时采集或写入现场标定。
